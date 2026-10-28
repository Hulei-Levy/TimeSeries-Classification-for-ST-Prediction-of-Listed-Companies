"""Frozen-manifest 1:3/1:5 experiments for the fourteen baseline architectures.

Existing sources, sampling manifests and results remain read-only. Full-pool
preprocessing, validation/test groups, model architectures and settings remain
unchanged. SEMPO uses MATH SDPA; TimeFilter accumulates microbatch16 to batch64.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager, nullcontext
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import zipfile
import numpy as np
import run as common
from sampling import load_training_mask

ROOT = Path(__file__).resolve().parent
EXPERIMENT = 'sampled_baselines_v1'
CACHE, RESULTS = ROOT/'cache'/EXPERIMENT, ROOT/'results'/EXPERIMENT
OLD_CACHE, OLD_RESULTS = ROOT/'cache/sampling_v1', ROOT/'results/sampling_v1'
MODELS, RATIOS = tuple(common.MODELS), (3, 5)
PILOT_MODELS = ('PerSeM', 'PerSeM_AB', 'PerSeM_ABC')
METRICS = ('accuracy','precision','recall','f1','specificity','mcc','roc_auc','ap',
           'brier','ece_10_bins','top_decile_capture')
EXPECTED_RUNS = 140
SOURCE_NAMES = ('run_sampled_baselines.py','run.py','data.py','models_classic.py',
                'models_recent.py','sampling.py','run_persem.py','models_persem.py',
                'config.json','README.md')
DESIGN = dict(version=1, experiment=EXPERIMENT, ratios=list(RATIOS), models=list(MODELS),
    folds=5, seed=42, threshold=.5, train_selection='unchanged cache/sampling_v1/fold_k.json manifests',
    preprocessing='original full training pool; exactly match original model/fold preprocessing',
    validation_test='complete original company groups, no resampling',
    loss='BCEWithLogitsLoss actual selected negatives/positives; original auxiliary losses',
    checkpoint='strict validation AP improvement; original patience/epochs',
    budget='same epochs/patience, 4 or 6 optimizer updates/epoch; not equal full-data update budgets',
    attention_backend={'SEMPO':'SDPBackend.MATH','others':'default'},
    TimeFilter='train batch64, microbatch16, eval batch16',
    comparison='28 new baseline and 6 original PerSeM sampling conditions; no test-based selection')


class AlreadyRunning(RuntimeError):
    """Lock rejection must not change a live experiment's status/failure files."""


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def sha_object(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


@contextmanager
def exclusive_lock(path, blocking=False):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise AlreadyRunning(f'Lock is already held: {path}') from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def immutable_json(path, obj):
    path = Path(path)
    if path.exists():
        require(read_json(path) == obj, f'Refusing to replace different immutable metadata: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as handle:
            json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)


def verify_legacy(full=False):
    legacy = read_json(CACHE/'legacy_identity.json')
    for name, digest in legacy['source_sha256'].items():
        if name == 'README.md':
            continue  # authorized documentation addition; snapshotted anew
        require(common.file_sha256(ROOT/name) == digest, f'Preserved source changed: {name}')
    if full:
        for relative, saved in legacy['artifacts'].items():
            path = ROOT/relative
            require(path.is_file() and path.stat().st_size == saved['size'], f'Preserved artifact changed: {relative}')
            if 'sha256' in saved:
                require(common.file_sha256(path) == saved['sha256'], f'Preserved bytes changed: {relative}')
            else:
                require(path.stat().st_mtime_ns == saved['mtime_ns'], f'Preserved checkpoint changed: {relative}')
    return legacy


def _pilot_fingerprint(archive, baseline_hash, ratio):
    """Reproduce OLD identity from frozen source bytes, without importing its runner."""
    h = hashlib.sha256(baseline_hash.encode())
    for name in ('run_persem.py','models_persem.py','sampling.py','cache/sampling_v1/legacy_identity.json'):
        h.update(name.encode())
        h.update((ROOT/name).read_bytes() if name.endswith('legacy_identity.json') else archive.read(name))
    for fold in range(5):
        h.update(archive.read(f'cache/sampling_v1/fold_{fold}.json'))
    original = dict(variants=list(PILOT_MODELS), ratios=list(RATIOS), condition_ratio=ratio,
        threshold=.5, selection='prespecified conditions, no test-based winner',
        preprocessing='preserved full training pool', loss='selected negatives/positives')
    h.update(json.dumps(original, sort_keys=True).encode())
    return h.hexdigest()


def pilot_identity(baseline_hash, legacy):
    protocol, summary = read_json(OLD_CACHE/'protocol.json'), read_json(OLD_RESULTS/'summary.json')
    require(protocol['baseline_fingerprint'] == baseline_hash, 'Pilot baseline identity differs')
    require(protocol['config'] == common.config(), 'Pilot configuration differs')
    with zipfile.ZipFile(OLD_CACHE/'source_snapshot.zip') as archive:
        for name in ('run_persem.py','models_persem.py','sampling.py','run.py','data.py',
                     'models_classic.py','models_recent.py','config.json'):
            require(archive.read(name) == (ROOT/name).read_bytes(), f'Pilot snapshot differs: {name}')
        for fold in range(5):
            name = f'cache/sampling_v1/fold_{fold}.json'
            require(archive.read(name) == (ROOT/name).read_bytes(), f'Pilot manifest changed: {fold}')
        require(json.loads(archive.read('cache/sampling_v1/protocol.json')) == protocol, 'Pilot protocol/snapshot differs')
        fp = _pilot_fingerprint(archive, baseline_hash, None)
        conditions = {str(r): _pilot_fingerprint(archive, baseline_hash, r) for r in RATIOS}
    require(fp == protocol['fingerprint'] == summary['fingerprint'] == legacy['sampling_fingerprint'],
            'Pilot protocol/summary/source identity mismatch')
    require(summary['completed_runs'] == 30 and len(summary['models']) == 6, 'Original sampling runs must be complete')
    return dict(fingerprint=fp, condition_fingerprints=conditions,
        summary_sha256=common.file_sha256(OLD_RESULTS/'summary.json'),
        protocol_sha256=common.file_sha256(OLD_CACHE/'protocol.json'),
        snapshot_sha256=common.file_sha256(OLD_CACHE/'source_snapshot.zip'),
        legacy_identity_sha256=common.file_sha256(OLD_CACHE/'legacy_identity.json'))


def identity(ratio=None):
    require(ratio is None or ratio in RATIOS, 'Ratio must be 3 or 5')
    legacy = verify_legacy()
    bh = common.fingerprint()
    require(bh == legacy['baseline_fingerprint'], 'Baseline/data/BERT fingerprint changed')
    payload = dict(design=DESIGN, condition_ratio=ratio, baseline_fingerprint=bh,
        config=common.config(), source_sha256={n:common.file_sha256(ROOT/n) for n in SOURCE_NAMES},
        manifest_sha256={str(f):common.file_sha256(OLD_CACHE/f'fold_{f}.json') for f in range(5)},
        preserved_identity_sha256=common.file_sha256(CACHE/'legacy_identity.json'),
        pilot_reference=pilot_identity(bh,legacy))
    return dict(fingerprint=sha_object(payload), **payload)


def folder(ratio, name, fold):
    return RESULTS/f'ratio_{ratio}'/name/f'fold_{fold}'


def validate_record(record, ratio, name, fold, runhash, baseline_hash):
    expected = dict(experiment=EXPERIMENT, model=name, fold=fold, sampling_ratio=ratio,
                    fingerprint=runhash, baseline_fingerprint=baseline_hash)
    require(all(record.get(k) == v for k,v in expected.items()), f'Run identity mismatch: {ratio}, {name}, {fold}')


def prediction_rows(path, d, fold, validation=False):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    expected = np.flatnonzero(d['fold'] == ((fold+1)%5 if validation else fold))
    indices = [int(r['sample_index']) for r in rows]
    require(indices == expected.tolist(), f'Prediction order/coverage differs: {path}')
    for row, i in zip(rows, indices):
        require(row['code'] == str(d['codes'][i]) and row['cutoff'] == str(d['cutoffs'][i])
                and int(row['label']) == int(d['labels'][i]) and int(row['fold']) == fold, f'Prediction identity: {path}')
        p = float(row['probability'])
        require(math.isfinite(p) and 0 <= p <= 1, f'Invalid probability: {path}')
    return rows


def row_scores(rows):
    return common.scores(np.array([int(r['label']) for r in rows]),
                         np.array([float(r['probability']) for r in rows]), .5)


def check_scores(record, rows, description):
    for key, value in common.scores(np.array([int(r['label']) for r in rows]),
            np.array([float(r['probability']) for r in rows],dtype=np.float32), .5).items():
        require(key in record and abs(float(record[key])-float(value)) <= 1e-10,
                f'Metrics differ from predictions: {description}, {key}')


def write_predictions(path, d, mask, probabilities, fold):
    path = Path(path)
    require(not path.exists(), f'Refusing to replace predictions: {path}')
    indices = np.flatnonzero(mask)
    require(len(indices) == len(probabilities), 'Prediction length mismatch')
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['sample_index','code','cutoff','label','probability','fold'])
        for i,p in zip(indices, probabilities):
            writer.writerow([i,str(d['codes'][i]),str(d['cutoffs'][i]),int(d['labels'][i]),float(p),fold])


def check_original_preprocessing(name, fold, prep, baseline_hash):
    original = ROOT/'results'/name/f'fold_{fold}'
    require(read_json(original/'preprocessing.json') == prep, f'Original preprocessing differs: {name} fold {fold}')
    metric = read_json(original/'metrics.json')
    require(metric.get('fingerprint') == baseline_hash and metric.get('model') == name
            and metric.get('fold') == fold, 'Original baseline result identity differs')


def audit_new_fold(path, d, ratio, name, fold, runhash, bh, meta=None, prep=None):
    item, val = read_json(path/'metrics.json'), read_json(path/'validation_metrics.json')
    started = read_json(path/'started.json')
    for record in (item,val,started): validate_record(record,ratio,name,fold,runhash,bh)
    cfg = common.config()
    require(started['config'] == cfg and started['training_seed'] == cfg['seed'], 'Run settings changed')
    require(item['threshold'] == val['threshold'] == cfg['threshold'] == .5, 'Threshold changed')
    require(item['pos_weight'] == ratio == started['pos_weight'], 'Selected class weight differs')
    if meta is None: _,meta = load_training_mask(ROOT,d,fold,ratio)
    require(item['sampling'] == started['sampling'] == meta, 'Selected sample metadata differs')
    if prep is None: prep = read_json(ROOT/'results'/name/f'fold_{fold}'/'preprocessing.json')
    require(read_json(path/'preprocessing.json') == prep, 'Fold preprocessing changed')
    backend = 'SDPBackend.MATH' if name == 'SEMPO' else 'default'
    require(item['attention_backend'] == started['attention_backend'] == backend, 'Attention backend differs')
    required = ('best.pt','started.json','preprocessing.json','epochs.jsonl',
                'predictions.csv','validation_metrics.json','validation_predictions.csv')
    require(set(item['artifact_sha256']) == set(required), 'Incomplete artifact manifest')
    for artifact in required:
        require((path/artifact).is_file() and common.file_sha256(path/artifact) == item['artifact_sha256'][artifact],
                f'Completed artifact changed: {path/artifact}')
    epochs = [json.loads(l) for l in (path/'epochs.jsonl').read_text().splitlines() if l]
    ntrain = meta['train_selected']['windows']
    require(len(epochs) == item['epochs_completed'] and [r['epoch'] for r in epochs] == list(range(1,len(epochs)+1)),
            'Epoch history differs')
    require(item['optimizer_steps'] == len(epochs)*math.ceil(ntrain/cfg['batch_size'])
            and item['training_examples_seen'] == len(epochs)*ntrain, 'Training update count differs')
    require(item['positive_training_exposures'] == len(epochs)*meta['train_selected']['positives'], 'Positive count differs')
    require(1 <= item['best_epoch'] <= len(epochs) and val['best_epoch'] == item['best_epoch'], 'Best epoch differs')
    require(abs(val['ap']-item['best_validation_ap']) <= 1e-10, 'Best validation AP differs')
    rows = prediction_rows(path/'predictions.csv',d,fold)
    vrows = prediction_rows(path/'validation_predictions.csv',d,fold,True)
    check_scores(item,rows,str(path));check_scores(val,vrows,str(path/'validation'))
    return item,val,rows,vrows


def aggregate(name, ratio, items, vals, rows, vrows, source_experiment, source_fingerprint, bh):
    row = dict(experiment=f'ratio_{ratio}', sampling_ratio=ratio, model=name,
               source_experiment=source_experiment, source_fingerprint=source_fingerprint,
               baseline_fingerprint=bh, completed_folds=5)
    vrow = dict(row)
    for key in METRICS:
        a,b = np.array([r[key] for r in items],float),np.array([r[key] for r in vals],float)
        row[key+'_mean'],row[key+'_std'] = float(a.mean()),float(a.std(ddof=1))
        vrow[key+'_mean'],vrow[key+'_std'] = float(b.mean()),float(b.std(ddof=1))
    for key in ('parameters','epochs_completed','optimizer_steps','training_examples_seen'):
        row[key+'_mean'] = float(np.mean([r[key] for r in items]))
    row['best_validation_ap_mean'] = float(np.mean([r['best_validation_ap'] for r in items]))
    row['pooled_oof'],vrow['pooled_validation'] = row_scores(rows),row_scores(vrows)
    return row,vrow


def audit_pilot(d, pilot, bh):
    """Validate original results against THEIR frozen identity, never ours."""
    saved = read_json(OLD_RESULTS/'summary.json')
    summaries, validation_summaries = [],[]
    for ratio in RATIOS:
        runhash = pilot['condition_fingerprints'][str(ratio)]
        for name in PILOT_MODELS:
            items,vals,rows,vrows = [],[],[],[]
            for fold in range(5):
                path = OLD_RESULTS/f'ratio_{ratio}'/name/f'fold_{fold}'
                item,val = read_json(path/'metrics.json'),read_json(path/'validation_metrics.json')
                for record in (item,val):
                    require(record['fingerprint']==runhash and record['baseline_fingerprint']==bh
                        and record['model']==name and record['fold']==fold and record['sampling_ratio']==ratio,
                        'Pilot fold identity differs from its original source snapshot')
                _,meta = load_training_mask(ROOT,d,fold,ratio)
                require(item['sampling']==meta and item['pos_weight']==ratio and (path/'best.pt').is_file(),
                        'Pilot selected sample/checkpoint differs')
                rr,vv=prediction_rows(path/'predictions.csv',d,fold),prediction_rows(path/'validation_predictions.csv',d,fold,True)
                check_scores(item,rr,str(path));check_scores(val,vv,str(path/'validation'))
                require(abs(item['best_validation_ap']-val['ap'])<=1e-10,'Pilot validation checkpoint differs')
                items.append(item);vals.append(val);rows.extend(rr);vrows.extend(vv)
            require(sorted(int(r['sample_index']) for r in rows)==list(range(len(d['labels']))),'Pilot OOF differs')
            row,vrow=aggregate(name,ratio,items,vals,rows,vrows,'sampling_v1',runhash,bh)
            prior=[r for r in saved['models'] if r['model']==name and r['experiment']==f'ratio_{ratio}']
            pv=[r for r in saved['validation_models'] if r['model']==name and r['experiment']==f'ratio_{ratio}']
            require(len(prior)==len(pv)==1,'Pilot summary condition missing/duplicated')
            for key in METRICS:
                for suffix in ('_mean','_std'):
                    require(abs(prior[0][key+suffix]-row[key+suffix])<=1e-10,'Pilot test summary inconsistent')
                    require(abs(pv[0][key+suffix]-vrow[key+suffix])<=1e-10,'Pilot validation summary inconsistent')
            for key,value in row['pooled_oof'].items():
                require(abs(float(prior[0]['pooled_oof'][key])-float(value))<=1e-10,'Pilot pooled summary inconsistent')
            summaries.append(row);validation_summaries.append(vrow)
    return summaries,validation_summaries


def prepare():
    with exclusive_lock(CACHE/'prepare.lock'):
        verify_legacy(full=True)
        ident, cfg, d = identity(), common.config(), common.data()
        require(len(MODELS)==14 and len(set(MODELS))==14, 'Expected original fourteen models')
        for key,value in dict(seed=42,batch_size=64,graph_microbatch_size=16,epochs=100,patience=15,threshold=.5).items():
            require(cfg[key]==value, f'Unexpected original setting: {key}')
        require(len(d['labels'])==10723 and int(d['labels'].sum())==100, 'Original cohort changed')
        baseline_summary=read_json(ROOT/'results/summary.json')
        require(baseline_summary['completed_runs']==70 and len(baseline_summary['models'])==14,
                'Original fourteen baselines must be complete')
        stats=[]
        for fold in range(5):
            full_tr,va,te=common.partition(d,fold)
            x,prep=common.preprocess(d,full_tr)
            del x
            for name in MODELS: check_original_preprocessing(name,fold,prep,ident['baseline_fingerprint'])
            selections={}
            for ratio in RATIOS:
                tr,meta=load_training_mask(ROOT,d,fold,ratio)
                require(not np.any(tr & ~full_tr) and np.all(tr[full_tr & (d['labels']==1)]), 'Selected training membership differs')
                require(int(d['labels'][tr].sum())==60 and int(tr.sum())==60*(ratio+1),'Unexpected training selection size')
                selections[ratio]=tr
                stats.append(dict(fold=fold,ratio=ratio,**{k:meta[k] for k in
                    ('train_full','train_selected','validation','test','manifest_sha256','index_sha256','stable_keys_sha256')}))
            require(not np.any(selections[3] & ~selections[5]), 'Nested sampling manifests differ')
        audit_pilot(d,ident['pilot_reference'],ident['baseline_fingerprint'])
        protocol=dict(**ident,experiment=EXPERIMENT,expected_runs=EXPECTED_RUNS,
            fold_statistics=stats,comparison_conditions=34,old_pilot_runs_reused=30,
            new_condition_fingerprints={str(r):identity(r)['fingerprint'] for r in RATIOS})
        immutable_json(CACHE/'protocol.json',protocol)
        names=list(SOURCE_NAMES)+[f'cache/sampling_v1/fold_{f}.json' for f in range(5)]+[
            f'cache/{EXPERIMENT}/legacy_identity.json',f'cache/{EXPERIMENT}/protocol.json']
        destination=CACHE/'source_snapshot.zip'
        if destination.exists():
            with zipfile.ZipFile(destination) as archive:
                require(set(archive.namelist())==set(names),'Snapshot inventory differs')
                for name in names: require(archive.read(name)==(ROOT/name).read_bytes(), f'Snapshot differs: {name}')
        else:
            with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as archive:
                for name in names: archive.write(ROOT/name,name)
        common.log('sampled_baselines_prepared',fingerprint=protocol['fingerprint'],expected_runs=EXPECTED_RUNS)
        return protocol


def load_protocol():
    protocol,ident=read_json(CACHE/'protocol.json'),identity()
    require(protocol['fingerprint']==ident['fingerprint'], 'Sources/data/protocol changed; refusing execution')
    require(protocol['new_condition_fingerprints']=={str(r):identity(r)['fingerprint'] for r in RATIOS},
            'Condition fingerprints differ')
    with zipfile.ZipFile(CACHE/'source_snapshot.zip') as archive:
        for name in archive.namelist():
            require(archive.read(name)==(ROOT/name).read_bytes(),f'Frozen source snapshot differs: {name}')
    return protocol


def require_validation(protocol):
    validation=read_json(CACHE/'validation.json')
    checks=read_json(CACHE/'smoke_checks.json')
    require(validation.get('passed') is True and validation.get('fingerprint')==protocol['fingerprint'],
            'Engineering validation has not passed for the current source identity')
    require(checks.get('fingerprint')==protocol['fingerprint'] and
            validation.get('smoke_checks_sha256')==sha_object(checks), 'Engineering evidence changed')
    expected={f'{r}:{n}' for r in RATIOS for n in MODELS}
    require(set(checks['checks'])==expected, 'Missing model/ratio engineering checks')
    for key,record in checks['checks'].items():
        require(record.get('passed') is True and record.get('checked_train_batch_size')==64
                and record.get('checked_eval_examples')==64 and record.get('device','').startswith('cuda'),
                f'Full-batch GPU engineering check missing: {key}')


def record_smoke(protocol,ratio,name,record):
    with exclusive_lock(CACHE/'smoke.lock',blocking=True):
        path=CACHE/'smoke_checks.json'
        checks=read_json(path) if path.exists() else dict(fingerprint=protocol['fingerprint'],checks={})
        require(checks['fingerprint']==protocol['fingerprint'],'Existing engineering checks belong to different source')
        checks['checks'][f'{ratio}:{name}']=record
        common.atomic_json(path,checks)
        expected={f'{r}:{n}' for r in RATIOS for n in MODELS}
        passed=set(checks['checks'])==expected and all(r.get('passed') is True and
            r.get('checked_train_batch_size')==64 and r.get('checked_eval_examples')==64
            and r.get('device','').startswith('cuda') for r in checks['checks'].values())
        common.atomic_json(CACHE/'validation.json',dict(fingerprint=protocol['fingerprint'],passed=passed,
            smoke_checks_sha256=sha_object(checks),completed_conditions=len(checks['checks']),
            expected_conditions=len(expected),evidence='full batch64 train/backward/step and 64 validation examples',checked_at=utc()))


def attention_context(name):
    if name!='SEMPO': return nullcontext()
    from torch.nn.attention import sdpa_kernel, SDPBackend
    return sdpa_kernel(SDPBackend.MATH)


def train_one(name,fold,ratio,device='cuda:0',smoke=False):
    require(name in MODELS and fold in range(5) and ratio in RATIOS,'Invalid condition')
    protocol=load_protocol()
    if not smoke: require_validation(protocol)
    with exclusive_lock(CACHE/'locks'/f'ratio_{ratio}_{name}_fold_{fold}.lock'):
        # SEMPO's actual model construction, forward/backward and prediction all use MATH.
        with attention_context(name):
            return _train_locked(name,fold,ratio,device,smoke,protocol)


def _train_locked(name,fold,ratio,device,smoke,protocol):
    import torch
    from torch.utils.data import DataLoader, TensorDataset, Subset
    from sklearn.metrics import average_precision_score
    cfg=common.config()
    torch.set_num_threads(cfg['torch_threads'])
    common.seed_all(cfg['seed'])
    d=common.data()
    full_tr,va,te=common.partition(d,fold)
    tr,meta=load_training_mask(ROOT,d,fold,ratio)
    require(not np.any(tr & ~full_tr) and np.all(tr[full_tr & (d['labels']==1)]),'Training membership changed')
    x,prep=common.preprocess(d,full_tr)
    bh,runhash=protocol['baseline_fingerprint'],protocol['new_condition_fingerprints'][str(ratio)]
    check_original_preprocessing(name,fold,prep,bh)
    path=folder(ratio,name,fold)
    if not smoke and (path/'metrics.json').exists():
        audit_new_fold(path,d,ratio,name,fold,runhash,bh,meta,prep)
        common.log('sampled_baseline_skip_verified',ratio=ratio,model=name,fold=fold)
        return
    if not smoke:
        require(not path.exists() or not any(path.iterdir()),
            f'Incomplete result preserved at {path}; archive it explicitly before retrying, never overwrite')
    require(np.load(ROOT/'cache/bert_complete.npy').all(),'BERT encoding incomplete')
    embeddings=np.load(ROOT/'cache/bert_embeddings.npy')
    model=common.build_model(name,x.shape[-1],embeddings,cfg,device)
    nparams=sum(p.numel() for p in model.parameters() if p.requires_grad)
    require(nparams==read_json(ROOT/'results'/name/f'fold_{fold}'/'metrics.json')['parameters'],
            'Architecture or input dimension differs from original baseline')
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    positive=int(d['labels'][tr].sum())
    weight=float(tr.sum()-positive)/positive
    require(weight==ratio and positive==60 and int(tr.sum())==60*(ratio+1),'Unexpected selected class balance')
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight,device=device))
    dataset=TensorDataset(torch.from_numpy(x),torch.from_numpy(d['doc_ids'].astype('int64')),
                          torch.from_numpy(d['valid_time'].astype(bool)),torch.from_numpy(d['labels'].astype('float32')))
    train_indices=np.flatnonzero(tr)
    if smoke:
        train_indices=np.concatenate([np.flatnonzero(tr & (d['labels']==1))[:32],
                                      np.flatnonzero(tr & (d['labels']==0))[:32]])
        require(len(train_indices)==64,'Full-batch smoke requires 32 positive and 32 negative examples')

    def loader(indices,training=False):
        batch=cfg['graph_microbatch_size'] if name=='TimeFilter' and not training else cfg['batch_size']
        return DataLoader(Subset(dataset,np.asarray(indices).tolist()),batch_size=batch,shuffle=training,
            num_workers=0,pin_memory=device.startswith('cuda'),generator=torch.Generator().manual_seed(cfg['seed']))

    def predict(indices):
        model.eval()
        values=[]
        with torch.inference_mode():
            for xb,db,vb,_ in loader(indices):
                values.append(model(xb.to(device),db.to(device),vb.to(device)).sigmoid().cpu().numpy())
        p=np.concatenate(values)
        require(np.isfinite(p).all(),'Nonfinite predicted probability')
        return p

    fields=dict(experiment=EXPERIMENT,model=name,fold=fold,sampling_ratio=ratio,
                fingerprint=runhash,baseline_fingerprint=bh)
    backend='SDPBackend.MATH' if name=='SEMPO' else 'default'
    micro_size=cfg['graph_microbatch_size'] if name=='TimeFilter' else cfg['batch_size']
    if not smoke:
        path.mkdir(parents=True,exist_ok=True)
        immutable_json(path/'preprocessing.json',prep)
        immutable_json(path/'started.json',dict(**fields,config=cfg,training_seed=cfg['seed'],sampling=meta,
            train_full_n=int(full_tr.sum()),train_n=int(tr.sum()),val_n=int(va.sum()),test_n=int(te.sum()),
            pos_weight=weight,trainable_parameters=nparams,attention_backend=backend,train_batch_size=cfg['batch_size'],
            train_microbatch_size=micro_size,eval_batch_size=micro_size,updates_per_epoch=math.ceil(int(tr.sum())/cfg['batch_size']),
            device=device,pid=os.getpid(),started_at=utc()))
    best_ap,best_epoch,best_state=-1.0,-1,None
    started=time.monotonic()
    train_loader=loader(train_indices,training=True)  # one persistent Generator across epochs
    steps,examples=0,0
    for epoch in range(1,(1 if smoke else cfg['epochs'])+1):
        model.train()
        total_loss,seen=0.0,0
        for xb,db,vb,yb in train_loader:
            xb,db,vb,yb=[v.to(device,non_blocking=True) for v in (xb,db,vb,yb)]
            optimizer.zero_grad(set_to_none=True)
            micro=cfg['graph_microbatch_size'] if name=='TimeFilter' else len(yb)
            for lo in range(0,len(yb),micro):
                sl=slice(lo,min(lo+micro,len(yb)))
                logits=model(xb[sl],db[sl],vb[sl])
                loss=loss_fn(logits,yb[sl])
                aux=model.auxiliary_loss()
                if aux is not None: loss=loss+cfg['auxiliary_loss_weight']*aux
                if not torch.isfinite(loss): raise FloatingPointError(f'{name} fold {fold} nonfinite loss')
                (loss*len(yb[sl])/len(yb)).backward()
                total_loss+=float(loss.detach())*len(yb[sl])
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
            if smoke:
                require(float(grad)>0,'No gradient in smoke check')
                tg=model.text_projection.weight.grad
                require(tg is not None and torch.isfinite(tg).all().item() and torch.count_nonzero(tg).item()>0,
                        'Text projection gradient missing/nonfinite/zero')
            optimizer.step()
            steps+=1;examples+=len(yb);seen+=len(yb)
        if smoke:
            pval=predict(np.flatnonzero(va)[:64])
            require(len(pval)==64 and seen==64 and steps==1,'Full batch smoke did not execute exactly one step')
            record=dict(**fields,passed=True,device=device,parameters=nparams,checked_train_batch_size=64,
                checked_eval_examples=64,checked_train_positives=32,checked_train_negatives=32,
                pos_weight=weight,attention_backend=backend,train_microbatch_size=micro_size,
                input_dimension=x.shape[-1]+cfg['text_dim'],optimizer_steps=steps,checked_at=utc())
            record_smoke(protocol,ratio,name,record)
            common.log('sampled_baseline_smoke_pass',**record)
            return record
        pval=predict(np.flatnonzero(va))
        ap=float(average_precision_score(d['labels'][va],pval))
        if ap>best_ap:
            best_ap,best_epoch=ap,epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            temporary=path/'best.pt.tmp'
            torch.save(dict(**fields,state_dict=best_state,config=cfg,epoch=epoch,validation_ap=ap,
                pos_weight=weight,attention_backend=backend,selected_index_sha256=meta['index_sha256']),temporary)
            os.replace(temporary,path/'best.pt')
        row=dict(epoch=epoch,train_loss=total_loss/seen,validation_ap=ap,best_validation_ap=best_ap,
                 best_epoch=best_epoch,optimizer_steps=steps,training_examples_seen=examples,
                 positive_training_exposures=epoch*positive)
        with (path/'epochs.jsonl').open('a',encoding='utf-8') as handle:
            handle.write(json.dumps(row,allow_nan=False)+'\n')
        common.log('sampled_baseline_epoch',ratio=ratio,model=name,fold=fold,**row)
        if epoch-best_epoch>=cfg['patience']: break
    model.load_state_dict(best_state)
    pval,p=predict(np.flatnonzero(va)),predict(np.flatnonzero(te))
    val_result=common.scores(d['labels'][va],pval,cfg['threshold'])
    require(abs(val_result['ap']-best_ap)<=1e-10,'Best checkpoint validation differs')
    val_result.update(**fields,best_epoch=best_epoch)
    write_predictions(path/'validation_predictions.csv',d,va,pval,fold)
    write_predictions(path/'predictions.csv',d,te,p,fold)
    immutable_json(path/'validation_metrics.json',val_result)
    result=common.scores(d['labels'][te],p,cfg['threshold'])
    result.update(**fields,best_epoch=best_epoch,best_validation_ap=best_ap,parameters=nparams,
        seconds=round(time.monotonic()-started,2),training_seed=cfg['seed'],pos_weight=weight,sampling=meta,
        epochs_completed=epoch,optimizer_steps=steps,training_examples_seen=examples,
        positive_training_exposures=epoch*positive,attention_backend=backend)
    result['artifact_sha256']={n:common.file_sha256(path/n) for n in ('best.pt','started.json',
        'preprocessing.json','epochs.jsonl','predictions.csv','validation_metrics.json','validation_predictions.csv')}
    immutable_json(path/'metrics.json',result)  # completion marker written last
    audit_new_fold(path,d,ratio,name,fold,runhash,bh,meta,prep)
    common.log('sampled_baseline_fold_finished',ratio=ratio,model=name,fold=fold,best_epoch=best_epoch,
               ap=result['ap'],epochs=epoch,optimizer_steps=steps)
    return result


def write_table(path,rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)
    os.replace(temporary,path)


def summarize():
    with exclusive_lock(CACHE/'summary.lock',blocking=True):
        protocol=load_protocol()
        verify_legacy(full=True)
        d,bh=common.data(),protocol['baseline_fingerprint']
        summaries,validations,folds=[],[],[]
        for ratio in RATIOS:
            runhash=protocol['new_condition_fingerprints'][str(ratio)]
            for name in MODELS:
                items,vals,rows,vrows=[],[],[],[]
                for fold in range(5):
                    path=folder(ratio,name,fold)
                    if not (path/'metrics.json').exists(): continue
                    item,val,rr,vv=audit_new_fold(path,d,ratio,name,fold,runhash,bh)
                    items.append(item);vals.append(val);rows.extend(rr);vrows.extend(vv)
                    folds.append(item)
                if len(items)!=5: continue
                for selected in (rows,vrows):
                    require(sorted(int(r['sample_index']) for r in selected)==list(range(len(d['labels']))),
                            'Complete model requires exactly one OOF/validation prediction per sample')
                row,vrow=aggregate(name,ratio,items,vals,rows,vrows,EXPERIMENT,runhash,bh)
                require(row['pooled_oof']['n']==10723 and row['pooled_oof']['positive']==100,'OOF population differs')
                summaries.append(row);validations.append(vrow)
        pilot,pilot_validation=audit_pilot(d,protocol['pilot_reference'],bh)
        comparison=summaries+pilot
        valmap={(r['sampling_ratio'],r['model']):r for r in validations+pilot_validation}
        csvrows=[]
        for row in comparison:
            val=valmap[(row['sampling_ratio'],row['model'])]
            item={k:row[k] for k in ('experiment','sampling_ratio','model','source_experiment',
                'source_fingerprint','baseline_fingerprint','completed_folds')}
            for key in METRICS:
                for suffix in ('_mean','_std'):
                    item['test_'+key+suffix]=row[key+suffix]
                    item['validation_'+key+suffix]=val[key+suffix]
            item.update(oof_n=row['pooled_oof']['n'],oof_positive=row['pooled_oof']['positive'],
                pooled_oof_ap=row['pooled_oof']['ap'],pooled_validation_ap=val['pooled_validation']['ap'])
            for key in ('parameters_mean','epochs_completed_mean','optimizer_steps_mean','training_examples_seen_mean'):
                item[key]=row[key]
            csvrows.append(item)
        full=read_json(ROOT/'results/summary.json')
        full_refs=[dict(row,source_experiment='full_baselines',source_fingerprint=bh) for row in full['models']]
        document=dict(experiment=EXPERIMENT,fingerprint=protocol['fingerprint'],baseline_fingerprint=bh,
            completed_runs=len(folds),expected_runs=EXPECTED_RUNS,models=summaries,validation_models=validations,
            folds=folds,pilot_reference=protocol['pilot_reference'],reused_pilot_models=pilot,
            reused_pilot_validation_models=pilot_validation,completed_comparison_conditions=len(comparison),
            expected_comparison_conditions=34,full_baseline_reference=full_refs,
            std_definition='sample standard deviation across five folds (ddof=1), not a confidence interval',
            validation_interpretation='used for checkpoint selection; pooled validation is separate from OOF test',
            selection='all prespecified conditions reported, no test-based winner',
            budget='same epochs/patience; sampled-data optimizer updates differ from full-data runs')
        RESULTS.mkdir(parents=True,exist_ok=True)
        common.atomic_json(RESULTS/'summary.json',document)
        write_table(RESULTS/'comparison.csv',csvrows)
        write_table(RESULTS/'validation_summary.csv',[
            {k:v for k,v in row.items() if k!='pooled_validation'} for row in validations+pilot_validation])
        common.log('sampled_baselines_summary',completed_runs=len(folds),expected_runs=EXPECTED_RUNS,
                   comparison_conditions=len(comparison))
        return document


def worker(worker_id):
    protocol=load_protocol()
    require_validation(protocol)
    require(worker_id in (0,1),'Worker must be 0 or 1')
    state_path=CACHE/f'worker_{worker_id}_status.json'
    tasks=[(r,n,f) for r in RATIOS for n in MODELS for f in range(5)]
    for index,(ratio,name,fold) in enumerate(tasks):
        if index%2!=worker_id: continue
        state=dict(status='running',worker=worker_id,pid=os.getpid(),fingerprint=protocol['fingerprint'],
                   ratio=ratio,model=name,fold=fold,updated_at=utc())
        common.atomic_json(state_path,state)
        command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'train','--model',name,'--fold',str(fold),
                 '--ratio',str(ratio),'--device',f'cuda:{worker_id}']
        child=subprocess.Popen(command,cwd=ROOT)
        state['child_pid']=child.pid
        common.atomic_json(state_path,state)
        code=child.wait()
        if code!=0:
            state.update(status='failed',returncode=code,updated_at=utc())
            common.atomic_json(state_path,state)
            raise RuntimeError(f'Worker {worker_id}: ratio {ratio} {name} fold {fold} exited {code}')
    common.atomic_json(state_path,dict(status='complete',worker=worker_id,pid=os.getpid(),
        fingerprint=protocol['fingerprint'],updated_at=utc()))


def stop_workers(processes):
    for child in processes:
        if child.poll() is None:
            try: os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError: pass
    for child in processes:
        try: child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try: os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            child.wait()


def pipeline():
    # Acquire before the exception/status writer: duplicate launchers cannot mark an active run failed.
    with exclusive_lock(CACHE/'pipeline.lock'):
        children,streams=[],[]
        status_path=CACHE/'pipeline_status.json'
        try:
            protocol=load_protocol()
            require_validation(protocol)
            verify_legacy(full=True)
            status=dict(status='starting',pid=os.getpid(),fingerprint=protocol['fingerprint'],
                        expected_runs=EXPECTED_RUNS,started_at=utc())
            common.atomic_json(status_path,status)
            (ROOT/'logs').mkdir(exist_ok=True)
            for worker_id in (0,1):
                stream=(ROOT/'logs'/f'sampled_baselines_gpu{worker_id}.log').open('a',encoding='utf-8')
                streams.append(stream)
                children.append(subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),
                    'worker','--worker',str(worker_id)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                    start_new_session=True))
            last_update=0.0
            while True:
                codes=[child.poll() for child in children]
                now=time.monotonic()
                if now-last_update>=10 or all(code is not None for code in codes):
                    status.update(status='running',workers=[dict(worker=i,pid=p.pid,running=codes[i] is None,
                        returncode=codes[i]) for i,p in enumerate(children)],updated_at=utc(),
                        completed_runs=sum(1 for _ in RESULTS.glob('ratio_*/*/fold_*/metrics.json')))
                    common.atomic_json(status_path,status)
                    last_update=now
                if any(code is not None and code!=0 for code in codes):
                    raise RuntimeError(f'Baseline workers failed: {codes}')
                if all(code==0 for code in codes): break
                time.sleep(2)
            document=summarize()
            require(document['completed_runs']==EXPECTED_RUNS and document['completed_comparison_conditions']==34,
                    'Workers ended without complete results')
            verify_legacy(full=True)
            status.update(status='complete',completed_runs=EXPECTED_RUNS,completed_comparison_conditions=34,finished_at=utc())
            common.atomic_json(status_path,status)
        except BaseException:
            stop_workers(children)
            status=read_json(status_path) if status_path.exists() else {}
            status.update(status='failed',pid=os.getpid(),failed_at=utc(),error=traceback.format_exc(),
                          workers=[dict(worker=i,pid=p.pid,running=p.poll() is None,returncode=p.poll())
                                   for i,p in enumerate(children)])
            common.atomic_json(status_path,status)
            raise
        finally:
            for stream in streams: stream.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','train','smoke','worker','pipeline','summary'))
    parser.add_argument('--model',choices=MODELS)
    parser.add_argument('--fold',type=int,choices=range(5),default=0)
    parser.add_argument('--ratio','--sampling-ratio',dest='ratio',type=int,choices=RATIOS)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--worker',type=int,choices=(0,1))
    args=parser.parse_args()
    if args.command=='prepare': prepare()
    elif args.command=='summary': summarize()
    elif args.command=='pipeline': pipeline()
    elif args.command=='worker':
        parser.error('--worker is required') if args.worker is None else worker(args.worker)
    elif args.command=='train':
        if args.model is None or args.ratio is None: parser.error('train requires --model and --ratio')
        train_one(args.model,args.fold,args.ratio,args.device)
    elif args.command=='smoke':
        for ratio in ([args.ratio] if args.ratio else RATIOS):
            for name in ([args.model] if args.model else MODELS):
                train_one(name,args.fold,ratio,args.device,smoke=True)


if __name__=='__main__':
    try:
        main()
    except AlreadyRunning as error:
        common.log('duplicate_launch_rejected',pid=os.getpid(),reason=str(error))
        sys.exit(3)
    except Exception:
        failure=dict(argv=sys.argv,pid=os.getpid(),failed_at=utc(),error=traceback.format_exc())
        failure_path=CACHE/'failures'/f'{time.time_ns()}_{os.getpid()}.json'
        common.atomic_json(failure_path,failure)
        common.log('sampled_baselines_failed',failure_file=str(failure_path),error=failure['error'])
        raise
