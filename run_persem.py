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


def train_one(variant, fold, device='cuda:0', smoke=False):
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
    tr, va, te = common.partition(d, fold)
    x, prep = common.preprocess(d, tr)
    layout = feature_layout(d, prep)
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
            positives = indices[d['labels'][indices] == 1][:2]
            negatives = indices[d['labels'][indices] == 0][:2]
            indices = np.concatenate([positives, negatives])
        return DataLoader(Subset(dataset, indices.tolist()),
            batch_size=2 if smoke else cfg['batch_size'], shuffle=shuffle,
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
    if not smoke:
        folder.mkdir(parents=True, exist_ok=True)
        started_path = folder/'started.json'
        if started_path.exists():
            previous = json.loads(started_path.read_text())
            validate_identity(previous, variant, fold, runhash, baseline_hash)
            if (folder/'metrics.json').exists():
                validate_identity(json.loads((folder/'metrics.json').read_text()),
                                  variant, fold, runhash, baseline_hash)
                common.log('persem_fold_already_complete', model=variant, fold=fold)
                return
            # An incomplete fold may be retried from scratch; preserve its attempt.
            import zipfile
            prior = sorted(folder.iterdir())
            backup_dir = ROOT/'cache/backups'
            backup_dir.mkdir(parents=True, exist_ok=True)
            target = backup_dir/f'{variant}_fold{fold}_attempt_{time.time_ns()}.zip'
            with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
                for path in prior:
                    if path.is_file():
                        archive.write(path, path.name)
        common.atomic_json(folder/'preprocessing.json', dict(**prep, modality_layout=layout))
        common.atomic_json(started_path, dict(model=variant, fold=fold, config=cfg,
            design=DESIGN, fingerprint=runhash, baseline_fingerprint=baseline_hash,
            train_n=int(tr.sum()), val_n=int(va.sum()), test_n=int(te.sum()),
            pos_weight=pos_weight, trainable_parameters=params))
        (folder/'epochs.jsonl').write_text('')
    best_ap, best_epoch, best_state = -1.0, -1, None
    started = time.monotonic()
    train_loader = loader(tr, shuffle=True)
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
            total_loss += float(loss.detach())*len(yb)
            seen += len(yb)
        probabilities = predict(va)
        assert np.isfinite(probabilities).all()
        if smoke:
            common.log('persem_integration_smoke_pass', model=variant, parameters=params,
                       input_dimension=x.shape[-1]+cfg['text_dim'], device=device)
            return dict(variant=variant, parameters=params, passed=True)
        from sklearn.metrics import average_precision_score
        ap = float(average_precision_score(d['labels'][va], probabilities))
        if ap > best_ap:
            best_ap, best_epoch = ap, epoch
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            temporary = folder/'best.pt.tmp'
            torch.save(dict(state_dict=best_state, model=variant, fold=fold, config=cfg,
                design=DESIGN, epoch=epoch+1, validation_ap=ap,
                fingerprint=runhash, baseline_fingerprint=baseline_hash), temporary)
            temporary.replace(folder/'best.pt')
        row = dict(epoch=epoch+1, train_loss=total_loss/seen, validation_ap=ap,
                   best_validation_ap=best_ap, best_epoch=best_epoch+1)
        with (folder/'epochs.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row)+'\n')
        common.log('persem_epoch', model=variant, fold=fold, **row)
        if epoch-best_epoch >= cfg['patience']:
            break
    model.load_state_dict(best_state)
    probabilities = predict(te)
    assert np.isfinite(probabilities).all()
    result = common.scores(d['labels'][te], probabilities, cfg['threshold'])
    result.update(model=variant, fold=fold, best_epoch=best_epoch+1,
        best_validation_ap=best_ap, parameters=params,
        seconds=round(time.monotonic()-started, 2), fingerprint=runhash,
        baseline_fingerprint=baseline_hash)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['smoke', 'train', 'worker', 'pipeline', 'summary',
                                          'recover', 'recover-worker', 'recover-fold'])
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--fold', type=int, choices=range(5), default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--worker', type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
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
        common.atomic_json(ROOT/'cache/persem_last_failure.json',
                           dict(traceback=traceback.format_exc(), utc=time.time()))
        if len(sys.argv) > 1 and sys.argv[1] in ('pipeline', 'recover'):
            status_path = ROOT/'cache/persem_pipeline_status.json'
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            status.update(stage='failed', traceback=traceback.format_exc())
            common.atomic_json(status_path, status)
        raise
