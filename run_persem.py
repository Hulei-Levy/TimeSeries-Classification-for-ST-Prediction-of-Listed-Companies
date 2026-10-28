"""PerSeM 2^3 ablations; reuse only the NEW project's fixed data/training utilities.

The existing baseline sources/configuration are deliberately immutable. Formal
training is gated on a complete, fingerprint-matched 70-run baseline summary.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import run as common

ROOT = Path(__file__).resolve().parent
VARIANTS = ['PerSeM', 'PerSeM_A', 'PerSeM_B', 'PerSeM_C',
            'PerSeM_AB', 'PerSeM_AC', 'PerSeM_BC', 'PerSeM_ABC']
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'mcc',
           'roc_auc', 'ap', 'brier', 'ece_10_bins', 'top_decile_capture']
DESIGN = dict(
    version=1, current_year=4, history_years=[0, 1, 2, 3],
    history_scales=[1, 2, 4], retrieval_top_k=2,
    modules={'A': 'within-modality multiscale self-history comparison',
             'B': 'within-company within-modality top-2 historical-year retrieval',
             'C': 'native-frequency temporal cross-attention and gated fusion'},
    natural_lengths={'financial': 20, 'governance': 5, 'text': 10},
    baseline_gate='all 70 baseline folds complete, with matching input/code fingerprint',
    training='same fixed configuration, BERT cache, folds, preprocessing and metrics as baselines',
    implementation='independent rewrite; full PerSeM branch retained in all eight variants',
)


def fingerprint(baseline_hash=None):
    baseline_hash = baseline_hash or common.fingerprint()
    h = hashlib.sha256(baseline_hash.encode())
    for name in ['run_persem.py', 'models_persem.py']:
        h.update(name.encode())
        h.update((ROOT/name).read_bytes())
    h.update(json.dumps(DESIGN, sort_keys=True).encode())
    return h.hexdigest()


def baseline_complete(expected_hash, fail_on_incomplete=False):
    """The baseline supervisor writes its summary AFTER both GPU workers exit."""
    path = ROOT/'results/summary.json'
    if not path.exists():
        if fail_on_incomplete:
            raise RuntimeError('The baseline supervisor has not finished. Formal PerSeM training is gated.')
        return False
    summary = json.loads(path.read_text())
    if summary.get('completed_runs') != 70 or len(summary.get('models', [])) != 14:
        raise RuntimeError('Baseline supervisor produced an incomplete summary. '
                           'Resolve baseline failures before starting PerSeM.')
    for name in common.MODELS:
        for fold in range(5):
            folder = ROOT/'results'/name/f'fold_{fold}'
            metrics = json.loads((folder/'metrics.json').read_text())
            if (metrics.get('fingerprint') != expected_hash
                    or metrics.get('model') != name or metrics.get('fold') != fold):
                raise RuntimeError(f'Baseline identity mismatch: {name}, fold {fold}')
            if not (folder/'best.pt').is_file() or not (folder/'predictions.csv').is_file():
                raise RuntimeError(f'Baseline artifacts are incomplete: {name}, fold {fold}')
    return True


def feature_layout(d, state):
    """Track one-hot expansion; never hard-code fold-specific category counts."""
    nraw = len(d['feature_names'])
    if nraw != 61:
        raise ValueError(f'Expected original 30 financial + 31 governance fields, got {nraw}')
    values, offset = [], 0
    for feature in map(str, d['feature_names']):
        category = state['categorical'].get(feature)
        width = len(category['vocabulary'])+2 if category is not None else 1
        values.append(list(range(offset, offset+width)))
        offset += width
    layout = {}
    for modality, indices in [('financial', range(30)), ('governance', range(30, 61))]:
        layout[modality] = dict(
            value_indices=[j for i in indices for j in values[i]],
            observed_indices=[offset+i for i in indices],
            age_indices=[offset+nraw+i for i in indices])
    layout.update(text_available_index=offset+2*nraw,
                  text_age_index=offset+2*nraw+1, valid_index=offset+2*nraw+2)
    assert layout['valid_index']+1 == state['base_input_dimension']
    return layout


def validate_identity(record, variant, fold, runhash, baseline_hash):
    expected = dict(model=variant, fold=fold, fingerprint=runhash,
                    baseline_fingerprint=baseline_hash)
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f'Result identity/version mismatch: {variant}, fold {fold}')


def make_model(variant, x, layout, embeddings, cfg, device):
    from models_persem import PerSeMClassifier
    return PerSeMClassifier(numeric_dim=x.shape[-1], layout=layout,
        embeddings=embeddings, variant=variant, dim=cfg['dim'], depth=cfg['depth'],
        dropout=cfg['dropout'], text_dim=cfg['text_dim']).to(device)


def train_one(variant, fold, device='cuda:0', smoke=False, sampling_ratio=None):
    import torch
    from torch.utils.data import DataLoader, TensorDataset, Subset
    cfg = common.config()
    torch.set_num_threads(min(cfg['torch_threads'], 4) if smoke else cfg['torch_threads'])
    common.seed_all(cfg['seed'])
    baseline_hash = common.fingerprint()
    runhash = fingerprint(baseline_hash)
    if not smoke:
        baseline_complete(baseline_hash, fail_on_incomplete=True)
    d = common.data()
    full_tr, va, te = common.partition(d, fold)
    tr, sampling_meta = full_tr.copy(), None
    if sampling_ratio is not None:
        from sampling import load_training_mask
        if variant not in SAMPLING_VARIANTS:
            raise ValueError('Variant outside the confirmed sampling pilot')
        tr, sampling_meta = load_training_mask(ROOT, d, fold, sampling_ratio)
        runhash = sampling_fingerprint(baseline_hash, sampling_ratio)
        assert not np.any(tr & ~full_tr)
        assert np.array_equal(tr & (d['labels'] == 1), full_tr & (d['labels'] == 1))
    x, prep = common.preprocess(d, full_tr)
    layout = feature_layout(d, prep)
    if sampling_ratio is not None:
        original = json.loads((ROOT/'results'/variant/f'fold_{fold}'/'preprocessing.json').read_text())
        if original != dict(**prep, modality_layout=layout):
            raise RuntimeError('Preprocessing differs from the preserved full-training-pool fit')
    embeddings = np.load(ROOT/'cache/bert_embeddings.npy')
    model = make_model(variant, x, layout, embeddings, cfg, device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'],
                                 weight_decay=cfg['weight_decay'])
    npositive = int(d['labels'][tr].sum())
    pos_weight = float(tr.sum()-npositive)/npositive
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(d['doc_ids'].astype('int64')),
        torch.from_numpy(d['valid_time'].astype(bool)), torch.from_numpy(d['labels'].astype('float32')))

    def loader(mask, shuffle=False):
        indices = np.flatnonzero(mask)
        if smoke:
            limit = 32 if sampling_ratio is not None else 2
            positives = indices[d['labels'][indices] == 1][:limit]
            negatives = indices[d['labels'][indices] == 0][:limit]
            indices = np.concatenate([positives, negatives])
        return DataLoader(Subset(dataset, indices.tolist()),
            batch_size=2 if smoke and sampling_ratio is None else cfg['batch_size'], shuffle=shuffle,
            num_workers=0, pin_memory=str(device).startswith('cuda'),
            generator=torch.Generator().manual_seed(cfg['seed']))

    def predict(mask):
        model.eval()
        probabilities = []
        with torch.inference_mode():
            for xb, db, vb, _ in loader(mask):
                logits = model(xb.to(device), db.to(device), vb.to(device))
                probabilities.append(logits.sigmoid().cpu().numpy())
        return np.concatenate(probabilities)

    folder = ROOT/'results'/variant/f'fold_{fold}'
    if sampling_ratio is not None:
        folder = sampling_folder(sampling_ratio, variant, fold)
    extra = {} if sampling_meta is None else dict(sampling_ratio=sampling_ratio, sampling=sampling_meta)
    if not smoke:
        folder.mkdir(parents=True, exist_ok=True)
        started_path = folder/'started.json'
        if started_path.exists():
            previous = json.loads(started_path.read_text())
            validate_identity(previous, variant, fold, runhash, baseline_hash)
            if previous.get('sampling_ratio') != sampling_ratio:
                raise RuntimeError('Sampling condition mismatch')
            if (folder/'metrics.json').exists():
                validate_identity(json.loads((folder/'metrics.json').read_text()),
                                  variant, fold, runhash, baseline_hash)
                if not all((folder/name).is_file() for name in ['best.pt','predictions.csv']):
                    raise RuntimeError('Completed fold lacks checkpoint or predictions')
                common.log('persem_fold_already_complete', model=variant, fold=fold, sampling_ratio=sampling_ratio)
                return
            # An incomplete fold may be retried from scratch; preserve its attempt.
            import zipfile
            prior = sorted(folder.iterdir())
            backup_dir = ROOT/'cache/backups'
            backup_dir.mkdir(parents=True, exist_ok=True)
            target = backup_dir/f'{variant}_r{sampling_ratio}_fold{fold}_attempt_{time.time_ns()}.zip'
            with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
                for path in prior:
                    if path.is_file():
                        archive.write(path, path.name)
        common.atomic_json(folder/'preprocessing.json', dict(**prep, modality_layout=layout))
        common.atomic_json(started_path, dict(model=variant, fold=fold, config=cfg,
            design=DESIGN, fingerprint=runhash, baseline_fingerprint=baseline_hash,
            train_n=int(tr.sum()), val_n=int(va.sum()), test_n=int(te.sum()),
            pos_weight=pos_weight, trainable_parameters=params,
            train_full_n=int(full_tr.sum()), training_seed=cfg['seed'],
            updates_per_epoch=int(np.ceil(tr.sum()/cfg['batch_size'])), **extra))
        (folder/'epochs.jsonl').write_text('')
    best_ap, best_epoch, best_state = -1.0, -1, None
    started = time.monotonic()
    train_loader = loader(tr, shuffle=True)
    optimizer_steps, examples_seen = 0, 0
    for epoch in range(1 if smoke else cfg['epochs']):
        model.train()
        total_loss, seen = 0.0, 0
        for xb, db, vb, yb in train_loader:
            xb, db, vb, yb = [v.to(device, non_blocking=True) for v in [xb, db, vb, yb]]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, db, vb)
            loss = loss_fn(logits, yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{variant} fold {fold}: nonfinite loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'],
                                                 error_if_nonfinite=True)
            if smoke:
                assert float(grad) > 0
                tg = model.text_projection.weight.grad
                assert tg is not None and torch.isfinite(tg).all() and float(tg.abs().sum()) > 0
            optimizer.step()
            optimizer_steps += 1
            examples_seen += len(yb)
            total_loss += float(loss.detach())*len(yb)
            seen += len(yb)
        probabilities = predict(va)
        assert np.isfinite(probabilities).all()
        if smoke:
            common.log('persem_integration_smoke_pass', model=variant, parameters=params,
                       input_dimension=x.shape[-1]+cfg['text_dim'], device=device,
                       sampling_ratio=sampling_ratio, pos_weight=pos_weight,
                       checked_batch_size=min(int(tr.sum()), cfg['batch_size']) if sampling_ratio else 2)
            return dict(variant=variant, parameters=params, passed=True)
        from sklearn.metrics import average_precision_score
        ap = float(average_precision_score(d['labels'][va], probabilities))
        if ap > best_ap:
            best_ap, best_epoch = ap, epoch
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            temporary = folder/'best.pt.tmp'
            torch.save(dict(state_dict=best_state, model=variant, fold=fold, config=cfg,
                design=DESIGN, epoch=epoch+1, validation_ap=ap,
                fingerprint=runhash, baseline_fingerprint=baseline_hash, **extra), temporary)
            temporary.replace(folder/'best.pt')
        row = dict(epoch=epoch+1, train_loss=total_loss/seen, validation_ap=ap,
                   best_validation_ap=best_ap, best_epoch=best_epoch+1,
                    optimizer_steps=optimizer_steps, training_examples_seen=examples_seen)
        if sampling_ratio is not None:
            row['sampling_ratio'] = sampling_ratio
        with (folder/'epochs.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row)+'\n')
        common.log('persem_epoch', model=variant, fold=fold, **row)
        if epoch-best_epoch >= cfg['patience']:
            break
    model.load_state_dict(best_state)
    if sampling_ratio is not None:
        validation_probabilities = predict(va)
        validation_metrics = common.scores(d['labels'][va], validation_probabilities, cfg['threshold'])
        assert abs(validation_metrics['ap']-best_ap) < 1e-10
        common.atomic_json(folder/'validation_metrics.json', dict(
            **validation_metrics, model=variant, fold=fold, sampling_ratio=sampling_ratio,
            fingerprint=runhash, baseline_fingerprint=baseline_hash, best_epoch=best_epoch+1))
        with (folder/'validation_predictions.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['sample_index','code','cutoff','label','probability','fold'])
            for index, probability in zip(np.flatnonzero(va), validation_probabilities):
                writer.writerow([index,str(d['codes'][index]),str(d['cutoffs'][index]),
                                 int(d['labels'][index]),float(probability),fold])
    probabilities = predict(te)
    assert np.isfinite(probabilities).all()
    result = common.scores(d['labels'][te], probabilities, cfg['threshold'])
    result.update(model=variant, fold=fold, best_epoch=best_epoch+1,
        best_validation_ap=best_ap, parameters=params,
        seconds=round(time.monotonic()-started, 2), fingerprint=runhash,
        baseline_fingerprint=baseline_hash, pos_weight=pos_weight,
        epochs_completed=epoch+1, optimizer_steps=optimizer_steps,
        training_examples_seen=examples_seen, positive_training_exposures=npositive*(epoch+1),
        **extra)
    with (folder/'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_index', 'code', 'cutoff', 'label', 'probability', 'fold'])
        for index, probability in zip(np.flatnonzero(te), probabilities):
            writer.writerow([index, str(d['codes'][index]), str(d['cutoffs'][index]),
                             int(d['labels'][index]), float(probability), fold])
    common.atomic_json(folder/'metrics.json', result)
    common.log('persem_fold_finished', **result)


def summarize():
    d, baseline_hash = common.data(), common.fingerprint()
    runhash = fingerprint(baseline_hash)
    summaries, runs = [], []
    for variant in VARIANTS:
        folds, predictions = [], []
        for fold in range(5):
            directory = ROOT/'results'/variant/f'fold_{fold}'
            if not (directory/'metrics.json').exists():
                continue
            row = json.loads((directory/'metrics.json').read_text())
            validate_identity(row, variant, fold, runhash, baseline_hash)
            folds.append(row)
            runs.append(row)
            with (directory/'predictions.csv').open() as stream:
                for item in csv.DictReader(stream):
                    idx = int(item['sample_index'])
                    assert int(item['fold']) == fold == int(d['fold'][idx])
                    assert item['code'] == str(d['codes'][idx])
                    assert item['cutoff'] == str(d['cutoffs'][idx])
                    assert int(item['label']) == int(d['labels'][idx])
                    predictions.append(item)
        if len(folds) != 5:
            continue
        indices = [int(item['sample_index']) for item in predictions]
        assert sorted(indices) == list(range(len(d['labels'])))
        probabilities = np.asarray([float(item['probability']) for item in predictions])
        labels = np.asarray([int(item['label']) for item in predictions])
        row = dict(model=variant, completed_folds=5)
        for metric in METRICS:
            numbers = np.asarray([item[metric] for item in folds])
            row[metric+'_mean'] = float(numbers.mean())
            row[metric+'_std'] = float(numbers.std(ddof=1))
        row['parameters_mean'] = float(np.mean([item['parameters'] for item in folds]))
        row['pooled_oof'] = common.scores(labels, probabilities, common.config()['threshold'])
        summaries.append(row)
    document = dict(design=DESIGN, fingerprint=runhash, baseline_fingerprint=baseline_hash,
        folds=runs, models=summaries, completed_runs=len(runs), expected_runs=40,
        std_definition='sample SD across dependent CV folds; not an initialization variance or CI')
    common.atomic_json(ROOT/'results/persem_summary.json', document)
    if summaries:
        fields = [key for key in summaries[0] if key != 'pooled_oof']
        with (ROOT/'results/persem_summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(summaries)
        if baseline_complete(baseline_hash):
            baseline = json.loads((ROOT/'results/summary.json').read_text())
            combined = []
            for source, rows in [('baseline', baseline['models']), ('ours_ablation', summaries)]:
                for item in rows:
                    result = {key: value for key,value in item.items() if key != 'pooled_oof'}
                    result['experiment'] = source
                    if source == 'baseline':
                        params = [f['parameters'] for f in baseline['folds'] if f['model'] == item['model']]
                        result['parameters_mean'] = float(np.mean(params))
                    combined.append(result)
            with (ROOT/'results/comparison.csv').open('w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=['experiment']+fields, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(combined)
    common.log('persem_summary', completed_runs=len(runs), completed_variants=len(summaries))
    return document


def worker(worker_id):
    baseline_hash = common.fingerprint()
    baseline_complete(baseline_hash, fail_on_incomplete=True)
    runhash = fingerprint(baseline_hash)
    for index, (variant, fold) in enumerate((name, f) for name in VARIANTS for f in range(5)):
        if index % 2 != worker_id:
            continue
        path = ROOT/'results'/variant/f'fold_{fold}'/'metrics.json'
        if path.exists():
            validate_identity(json.loads(path.read_text()), variant, fold, runhash, baseline_hash)
            if not (path.parent/'best.pt').is_file() or not (path.parent/'predictions.csv').is_file():
                raise RuntimeError(f'Incomplete result artifacts: {path.parent}')
            continue
        subprocess.run([sys.executable, '-u', '-B', str(ROOT/'run_persem.py'), 'train',
            '--variant', variant, '--fold', str(fold), '--device', f'cuda:{worker_id}'], check=True)


def pipeline():
    baseline_hash = common.fingerprint()
    runhash = fingerprint(baseline_hash)
    status = ROOT/'cache/persem_pipeline_status.json'
    common.atomic_json(status, dict(stage='waiting_for_baselines', expected_baseline_runs=70,
        expected_persem_runs=40, fingerprint=runhash, baseline_fingerprint=baseline_hash))
    common.log('persem_waiting_for_baselines', expected_runs=70)
    # The baseline supervisor creates summary.json only after both training workers exit.
    # This process intentionally holds no CUDA model/context while it is waiting.
    while not baseline_complete(baseline_hash):
        time.sleep(30)
    if fingerprint() != runhash:
        raise RuntimeError('Sources or data changed while waiting; explicit restart required.')
    common.atomic_json(status, dict(stage='integration_checks', fingerprint=runhash))
    common.log('persem_baseline_gate_passed')
    for variant in VARIANTS:
        subprocess.run([sys.executable, '-u', '-B', str(ROOT/'run_persem.py'), 'smoke',
                        '--variant', variant, '--device', 'cuda:0'], check=True)
    common.atomic_json(status, dict(stage='training', fingerprint=runhash, expected_runs=40))
    streams, processes = [], []
    for worker_id in range(2):
        path = ROOT/f'logs/persem_gpu{worker_id}.log'
        stream = path.open('a', buffering=1)
        streams.append(stream)
        processes.append(subprocess.Popen([sys.executable, '-u', '-B', str(ROOT/'run_persem.py'),
            'worker', '--worker', str(worker_id)], stdout=stream, stderr=subprocess.STDOUT))
    codes = [process.wait() for process in processes]
    for stream in streams:
        stream.close()
    summary = summarize()
    if any(codes) or summary['completed_runs'] != 40:
        raise RuntimeError(f'PerSeM training incomplete: workers={codes}, runs={summary["completed_runs"]}')
    common.atomic_json(status, dict(stage='complete', fingerprint=runhash, completed_runs=40))
    common.log('persem_experiment_complete', completed_runs=40)


def verify_preserved_baselines():
    """Recovery must leave the 65 completed predictions/metrics byte-identical."""
    record = json.loads((ROOT/'cache/sempo_recovery.json').read_text())
    for relative, digest in record['preserved_completed_artifacts'].items():
        if common.file_sha256(ROOT/relative) != digest:
            raise RuntimeError(f'Completed baseline artifact changed: {relative}')


def recover_sempo_fold(fold, device):
    """Only change the SDPA execution backend, inside the actual train process."""
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    baseline_hash = common.fingerprint()
    folder = ROOT/'results/SEMPO'/f'fold_{fold}'
    path = folder/'metrics.json'
    if path.exists():
        result = json.loads(path.read_text())
        if (result['model'] != 'SEMPO' or result['fold'] != fold
                or result['fingerprint'] != baseline_hash
                or result.get('attention_backend') != 'SDPBackend.MATH'):
            raise RuntimeError(f'Unexpected completed SEMPO identity/backend: fold {fold}')
        return
    started = folder/'started.json'
    if started.exists() and json.loads(started.read_text())['fingerprint'] != baseline_hash:
        raise RuntimeError('Refusing SEMPO retry under a different baseline source identity.')
    # Preserve any failed attempt before restarting it with the same seed/protocol.
    previous = [p for p in folder.glob('*') if p.is_file()]
    if previous:
        import zipfile
        backup = ROOT/f'cache/backups/SEMPO_fold{fold}_before_retry_{time.time_ns()}.zip'
        with zipfile.ZipFile(backup, 'w', zipfile.ZIP_DEFLATED) as archive:
            for p in previous:
                archive.write(p, p.name)
        epochs = folder/'epochs.jsonl'
        if epochs.exists():
            epochs.write_text('')
    common.log('sempo_math_backend', fold=fold, device=device,
               batch_size=common.config()['batch_size'], torch_version=torch.__version__)
    with sdpa_kernel(SDPBackend.MATH):
        common.train_one('SEMPO', fold, device)
    for path in [folder/'started.json', folder/'metrics.json']:
        result = json.loads(path.read_text())
        result['attention_backend'] = 'SDPBackend.MATH'
        result['execution_note'] = 'Full batch unchanged; default efficient CUDA SDPA rejects expanded batch 69376.'
        common.atomic_json(path, result)


def recover_sempo_worker(worker_id):
    for fold in range(worker_id, 5, 2):
        subprocess.run([sys.executable, '-u', '-B', str(ROOT/'run_persem.py'),
            'recover-fold', '--fold', str(fold), '--device', f'cuda:{worker_id}'], check=True)


def recover_then_pipeline():
    """Complete the five SEMPO folds, then use the unchanged PerSeM gate."""
    verify_preserved_baselines()
    common.atomic_json(ROOT/'cache/persem_pipeline_status.json',
        dict(stage='recovering_baseline_SEMPO', expected_baseline_runs=70,
             preserved_baseline_runs=65, expected_persem_runs=40,
             fingerprint=fingerprint(), baseline_fingerprint=common.fingerprint()))
    streams, processes = [], []
    for worker_id in range(2):
        stream = (ROOT/f'logs/sempo_recovery_gpu{worker_id}.log').open('a', buffering=1)
        streams.append(stream)
        processes.append(subprocess.Popen([sys.executable, '-u', '-B', str(ROOT/'run_persem.py'),
            'recover-worker', '--worker', str(worker_id)], stdout=stream, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in processes]
    for stream in streams:
        stream.close()
    verify_preserved_baselines()
    common.summarize()
    if any(codes):
        raise RuntimeError(f'SEMPO recovery failed: workers={codes}')
    baseline_complete(common.fingerprint(), fail_on_incomplete=True)
    common.log('sempo_recovery_complete', preserved_other_runs=65, recovered_runs=5)
    pipeline()


SAMPLING_VARIANTS = ['PerSeM', 'PerSeM_AB', 'PerSeM_ABC']
SAMPLING_RATIOS = [3, 5]
SAMPLING_CACHE = ROOT/'cache/sampling_v1'
SAMPLING_RESULTS = ROOT/'results/sampling_v1'


def sampling_folder(ratio, variant, fold):
    return SAMPLING_RESULTS/f'ratio_{ratio}'/variant/f'fold_{fold}'


def verify_sampling_legacy(full=True):
    record = json.loads((SAMPLING_CACHE/'legacy_identity.json').read_text())
    for name, digest in record['source_sha256'].items():
        if name in ('run_persem.py', 'README.md'):
            continue
        if common.file_sha256(ROOT/name) != digest:
            raise RuntimeError(f'Preserved source/input protocol changed: {name}')
    if full:
        for relative, identity in record['artifacts'].items():
            path = ROOT/relative
            if not path.is_file() or path.stat().st_size != identity['size']:
                raise RuntimeError(f'Preserved result changed: {relative}')
            if 'sha256' in identity:
                if common.file_sha256(path) != identity['sha256']:
                    raise RuntimeError(f'Preserved result bytes changed: {relative}')
            elif path.stat().st_mtime_ns != identity['mtime_ns']:
                raise RuntimeError(f'Preserved checkpoint timestamp changed: {relative}')
    return record


def sampling_fingerprint(baseline_hash=None, ratio=None):
    h = hashlib.sha256((baseline_hash or common.fingerprint()).encode())
    for name in ['run_persem.py', 'models_persem.py', 'sampling.py',
                 'cache/sampling_v1/legacy_identity.json']:
        h.update(name.encode()); h.update((ROOT/name).read_bytes())
    for fold in range(5):
        h.update((SAMPLING_CACHE/f'fold_{fold}.json').read_bytes())
    h.update(json.dumps(dict(variants=SAMPLING_VARIANTS, ratios=SAMPLING_RATIOS,
        condition_ratio=ratio, threshold=0.5, selection='prespecified conditions, no test-based winner',
        preprocessing='preserved full training pool', loss='selected negatives/positives'),
        sort_keys=True).encode())
    return h.hexdigest()


def sampling_prepare():
    from sampling import ensure_manifests, load_training_mask
    legacy = verify_sampling_legacy()
    bh = common.fingerprint()
    if bh != legacy['baseline_fingerprint']:
        raise RuntimeError('Baseline/data fingerprint changed')
    baseline_complete(bh, fail_on_incomplete=True)
    d = common.data()
    manifest_info = ensure_manifests(ROOT, d)
    stats = []
    for fold in range(5):
        full_tr, va, te = common.partition(d, fold)
        x, prep = common.preprocess(d, full_tr)
        layout = feature_layout(d, prep)
        for variant in SAMPLING_VARIANTS:
            old = json.loads((ROOT/'results'/variant/f'fold_{fold}'/'preprocessing.json').read_text())
            assert old == dict(**prep, modality_layout=layout)
        r3, m3 = load_training_mask(ROOT, d, fold, 3)
        r5, m5 = load_training_mask(ROOT, d, fold, 5)
        assert not np.any(r3 & ~r5)
        assert int(d['labels'][r3].sum()) == int(d['labels'][r5].sum()) == 60
        assert int(r3.sum()) == 240 and int(r5.sum()) == 360
        assert np.array_equal(r3 & (d['labels']==1), full_tr & (d['labels']==1))
        for ratio, mask, meta in [(3,r3,m3),(5,r5,m5)]:
            assert not np.any(mask & (va|te))
            assert int((mask & (d['labels']==0)).sum()) == ratio*int(d['labels'][mask].sum())
            stats.append(dict(ratio=ratio, **meta))
        del x
    runhash = sampling_fingerprint(bh)
    protocol = dict(experiment='sampling_v1', fingerprint=runhash, baseline_fingerprint=bh,
        legacy_persem_fingerprint=legacy['persem_fingerprint'], config=common.config(),
        variants=SAMPLING_VARIANTS, ratios=SAMPLING_RATIOS, expected_runs=30,
        model_design=DESIGN, manifest_info=manifest_info, fold_statistics=stats,
        matched_preprocessing=True, threshold_optimization=False, contrastive_loss=False,
        selection='Both ratios prespecified; report validation and test separately; no winner selected automatically.',
        optimization_budget='Same epochs/patience/batch size; 4 or 6 updates/epoch, not equal update budgets.',
        history='Per-window five-year histories unchanged by independent training-sample selection.')
    path = SAMPLING_CACHE/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError('Existing sampling protocol differs; refuse overwrite')
    common.atomic_json(path, protocol)
    import zipfile
    snapshot = SAMPLING_CACHE/'source_snapshot.zip'
    names = ['data.py','run.py','run_persem.py','sampling.py','models_classic.py',
             'models_recent.py','models_persem.py','config.json','README.md']
    if snapshot.exists():
        with zipfile.ZipFile(snapshot) as archive:
            if any(archive.read(name) != (ROOT/name).read_bytes() for name in names):
                raise RuntimeError('Sampling source snapshot differs')
    else:
        with zipfile.ZipFile(snapshot, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in names: archive.write(ROOT/name, name)
            for path in sorted(SAMPLING_CACHE.glob('fold_*.json')):
                archive.write(path, str(path.relative_to(ROOT)))
            archive.write(SAMPLING_CACHE/'protocol.json', 'cache/sampling_v1/protocol.json')
    common.log('sampling_prepared', fingerprint=runhash, expected_runs=30,
               per_fold_positive=60, ratio3_negative=180, ratio5_negative=300)
    return protocol


def check_sampling_predictions(path, d, fold, validation=False):
    with path.open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    expected = np.flatnonzero(d['fold'] == ((fold+1)%5 if validation else fold))
    indices = [int(row['sample_index']) for row in rows]
    if sorted(indices) != expected.tolist():
        raise RuntimeError(f'Prediction sample coverage mismatch: {path}')
    for row, index in zip(rows, indices):
        if (row['code'] != str(d['codes'][index]) or row['cutoff'] != str(d['cutoffs'][index])
                or int(row['label']) != int(d['labels'][index]) or int(row['fold']) != fold):
            raise RuntimeError(f'Prediction identity mismatch: {path}')
        if not np.isfinite(float(row['probability'])) or not 0 <= float(row['probability']) <= 1:
            raise RuntimeError(f'Invalid prediction: {path}')
    return rows


def sampling_summarize():
    from sampling import load_training_mask
    d, bh = common.data(), common.fingerprint()
    legacy = verify_sampling_legacy()
    protocol = json.loads((SAMPLING_CACHE/'protocol.json').read_text())
    if protocol['fingerprint'] != sampling_fingerprint(bh):
        raise RuntimeError('Current sampling source differs from approved run snapshot')
    completed, summaries, validation_summaries = [], [], []
    for ratio in SAMPLING_RATIOS:
        runhash = sampling_fingerprint(bh, ratio)
        for variant in SAMPLING_VARIANTS:
            folds, predictions, vals = [], [], []
            for fold in range(5):
                folder = sampling_folder(ratio, variant, fold)
                if not (folder/'metrics.json').exists(): continue
                item = json.loads((folder/'metrics.json').read_text())
                validate_identity(item, variant, fold, runhash, bh)
                mask, metadata = load_training_mask(ROOT,d,fold,ratio)
                assert item['sampling_ratio'] == ratio and item['sampling'] == metadata
                assert item['threshold'] == 0.5 and item['pos_weight'] == ratio
                assert (folder/'best.pt').is_file()
                rows = check_sampling_predictions(folder/'predictions.csv',d,fold)
                validation = json.loads((folder/'validation_metrics.json').read_text())
                validate_identity(validation,variant,fold,runhash,bh)
                assert validation['sampling_ratio']==ratio
                check_sampling_predictions(folder/'validation_predictions.csv',d,fold,True)
                assert abs(validation['ap']-item['best_validation_ap']) < 1e-10
                folds.append(item); completed.append(item); vals.append(validation)
                predictions.extend(rows)
            if len(folds) != 5: continue
            assert sorted(int(row['sample_index']) for row in predictions) == list(range(len(d['labels'])))
            row = dict(experiment=f'ratio_{ratio}', model=variant, completed_folds=5)
            vrow = dict(experiment=f'ratio_{ratio}', model=variant, completed_folds=5)
            for key in METRICS:
                numbers = np.asarray([f[key] for f in folds])
                row[key+'_mean'],row[key+'_std'] = float(numbers.mean()),float(numbers.std(ddof=1))
                vn = np.asarray([f[key] for f in vals])
                vrow[key+'_mean'],vrow[key+'_std'] = float(vn.mean()),float(vn.std(ddof=1))
            for key in ['parameters','epochs_completed','optimizer_steps','training_examples_seen']:
                row[key+'_mean'] = float(np.mean([f[key] for f in folds]))
            row['best_validation_ap_mean'] = float(np.mean([f['best_validation_ap'] for f in folds]))
            row['pooled_oof'] = common.scores(np.asarray([int(r['label']) for r in predictions]),
                np.asarray([float(r['probability']) for r in predictions]),0.5)
            summaries.append(row); validation_summaries.append(vrow)
    old = json.loads((ROOT/'results/persem_summary.json').read_text())
    assert old['fingerprint'] == legacy['persem_fingerprint']
    full_rows = [dict(experiment='full_training', **{k:v for k,v in item.items() if k!='pooled_oof'})
                 for item in old['models'] if item['model'] in SAMPLING_VARIANTS]
    document = dict(fingerprint=protocol['fingerprint'], completed_runs=len(completed), expected_runs=30,
        models=summaries, validation_models=validation_summaries, folds=completed,
        full_training_reference=full_rows, legacy_persem_fingerprint=legacy['persem_fingerprint'],
        std_definition='Sample SD across company folds; not a CI or initialization variance.',
        selection='No ratio/model chosen using test scores; both prespecified conditions reported.')
    SAMPLING_RESULTS.mkdir(parents=True,exist_ok=True)
    common.atomic_json(SAMPLING_RESULTS/'summary.json',document)
    table = full_rows+[{k:v for k,v in row.items() if k!='pooled_oof'} for row in summaries]
    fields = ['experiment','model','completed_folds']
    for row in table:
        for key in row:
            if key not in fields: fields.append(key)
    with (SAMPLING_RESULTS/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(table)
    if validation_summaries:
        with (SAMPLING_RESULTS/'validation_summary.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(validation_summaries[0])); writer.writeheader(); writer.writerows(validation_summaries)
    common.log('sampling_summary',completed_runs=len(completed),expected_runs=30)
    return document


def sampling_worker(worker_id):
    protocol = json.loads((SAMPLING_CACHE/'protocol.json').read_text())
    if protocol['fingerprint'] != sampling_fingerprint():
        raise RuntimeError('Sampling sources/configuration changed since preparation')
    tasks = [(r,m,f) for r in SAMPLING_RATIOS for m in SAMPLING_VARIANTS for f in range(5)]
    for index,(ratio,variant,fold) in enumerate(tasks):
        if index%2 != worker_id: continue
        subprocess.run([sys.executable,'-u','-B',str(ROOT/'run_persem.py'),'sampling-train',
            '--sampling-ratio',str(ratio),'--variant',variant,'--fold',str(fold),
            '--device',f'cuda:{worker_id}'],check=True)


def sampling_pipeline():
    import fcntl
    lock = (SAMPLING_CACHE/'pipeline.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol = json.loads((SAMPLING_CACHE/'protocol.json').read_text())
    if protocol['fingerprint'] != sampling_fingerprint():
        raise RuntimeError('Sampling sources/configuration changed since preparation')
    verification=json.loads((SAMPLING_CACHE/'validation.json').read_text())
    if verification.get('fingerprint') != protocol['fingerprint'] or not verification.get('passed'):
        raise RuntimeError('Sampling engineering validation is missing/stale')
    verify_sampling_legacy()
    status = SAMPLING_CACHE/'pipeline_status.json'
    common.atomic_json(status,dict(stage='training',fingerprint=protocol['fingerprint'],expected_runs=30))
    streams, processes = [], []
    try:
        for worker_id in range(2):
            stream=(ROOT/f'logs/sampling_gpu{worker_id}.log').open('a',buffering=1)
            streams.append(stream)
            processes.append(subprocess.Popen([sys.executable,'-u','-B',str(ROOT/'run_persem.py'),
                'sampling-worker','--worker',str(worker_id)],stdout=stream,stderr=subprocess.STDOUT))
        codes=[process.wait() for process in processes]
    finally:
        for stream in streams: stream.close()
    summary=sampling_summarize()
    if any(codes) or summary['completed_runs'] != 30:
        raise RuntimeError(f'Sampling pilot incomplete: workers={codes}, completed={summary["completed_runs"]}')
    verify_sampling_legacy()
    common.atomic_json(status,dict(stage='complete',fingerprint=protocol['fingerprint'],completed_runs=30))
    common.log('sampling_experiment_complete',completed_runs=30,full_training_results_preserved=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['smoke', 'train', 'worker', 'pipeline', 'summary',
                                          'recover', 'recover-worker', 'recover-fold',
                                          'sampling-prepare','sampling-smoke','sampling-train',
                                          'sampling-worker','sampling-pipeline','sampling-summary'])
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--fold', type=int, choices=range(5), default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--worker', type=int, choices=[0, 1], default=0)
    parser.add_argument('--sampling-ratio',type=int,choices=SAMPLING_RATIOS)
    args = parser.parse_args()
    if args.command.startswith('sampling-'):
        if args.command == 'sampling-prepare': sampling_prepare()
        elif args.command == 'sampling-summary': sampling_summarize()
        elif args.command == 'sampling-worker': sampling_worker(args.worker)
        elif args.command == 'sampling-pipeline': sampling_pipeline()
        else:
            if args.sampling_ratio is None or args.variant not in SAMPLING_VARIANTS:
                parser.error('Sampling train/smoke requires a ratio and a confirmed variant')
            train_one(args.variant,args.fold,args.device,smoke=args.command=='sampling-smoke',
                      sampling_ratio=args.sampling_ratio)
        return
    if args.command in ('train', 'smoke'):
        for variant in [args.variant] if args.variant else VARIANTS:
            train_one(variant, args.fold, args.device, smoke=args.command == 'smoke')
    elif args.command == 'worker':
        worker(args.worker)
    elif args.command == 'summary':
        summarize()
    elif args.command == 'recover':
        recover_then_pipeline()
    elif args.command == 'recover-worker':
        recover_sempo_worker(args.worker)
    elif args.command == 'recover-fold':
        recover_sempo_fold(args.fold, args.device)
    else:
        pipeline()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        common.log('persem_fatal', traceback=traceback.format_exc())
        is_sampling = len(sys.argv)>1 and sys.argv[1].startswith('sampling-')
        failure_path = SAMPLING_CACHE/'last_failure.json' if is_sampling else ROOT/'cache/persem_last_failure.json'
        common.atomic_json(failure_path,dict(traceback=traceback.format_exc(),utc=time.time()))
        if len(sys.argv) > 1 and sys.argv[1] in ('pipeline','recover','sampling-pipeline'):
            status_path = SAMPLING_CACHE/'pipeline_status.json' if is_sampling else ROOT/'cache/persem_pipeline_status.json'
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            status.update(stage='failed', traceback=traceback.format_exc())
            common.atomic_json(status_path, status)
        raise
