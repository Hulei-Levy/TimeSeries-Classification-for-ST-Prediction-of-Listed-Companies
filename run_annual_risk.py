"""Independent annual-risk experiments; existing experiments stay immutable.

FinancialLR and PerSeM/AB/ABC x full/ratio_3/ratio_4 x five company folds.
Neural budget: 10,500 optimizer steps, validation every 105, patience 15 checks.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
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
import warnings
import zipfile
import numpy as np
import run as common
from run_persem import feature_layout, make_model
import data_annual_risk as annual

ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'cache/annual_risk_v1'
RESULTS=ROOT/'results/annual_risk_v1'
MODELS=('FinancialLR','PerSeM','PerSeM_AB','PerSeM_ABC')
SCHEMES=('full','ratio_3','ratio_4')
METRICS=('accuracy','precision','recall','f1','specificity','mcc','roc_auc','ap','brier','ece_10_bins','top_decile_capture')
SOURCES=('run_annual_risk.py','data_annual_risk.py','run.py','run_persem.py','models_persem.py','data.py','config.json')
TRANSITIONS=('stable_nonrisk','risk_onset','persistent_risk','recovery','unknown')
DESIGN=dict(version=1,experiment='annual_risk_v1',models=list(MODELS),schemes=list(SCHEMES),folds=5,
    expected_fits=60,neural_fits=45,financial_lr_fits=15,threshold=.5,
    preprocessing='full annual-task training company pool, shared across the three sampling conditions',
    financial_lr_input='20 quarters x (30 standardized values + 30 observed masks + 30 report ages in years)',
    neural_input='unchanged PerSeM architecture; 20/5/10 native observations; frozen BERT plus trainable Linear768to64',
    selection='validation AP strictly improves; test never selects model, threshold or sampling condition',
    validation='five company groups; validation fold=(test fold+1)%5; full validation/test retained',
    carry_comparison='same covered sample subset; unavailable carry abstains; empty/single-class ROC/AP are null',
    transition_diagnostics='ex-post labels used only for reporting, never training features or matching')


class AlreadyRunning(RuntimeError):
    pass


def require(condition,message):
    if not condition: raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def index_hash(indices):
    return hashlib.sha256(np.asarray(indices,dtype='<i8').tobytes()).hexdigest()


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())


@contextmanager
def locked(path,blocking=False):
    import fcntl
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        try: fcntl.flock(stream.fileno(),fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc: raise AlreadyRunning(str(path)) from exc
        try: yield
        finally: fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def immutable_json(path,value):
    path=Path(path)
    if path.exists():
        require(read_json(path)==value,f'Refusing to overwrite different metadata: {path}')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream: json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)


def protected_unchanged(full=False):
    saved=read_json(CACHE/'preserved_identity.json')
    for name,record in saved['files'].items():
        if not full and name not in SOURCES: continue
        path=ROOT/name
        require(path.is_file() and path.stat().st_size==record['size'],f'Preserved file changed: {name}')
        if 'sha256' in record: require(common.file_sha256(path)==record['sha256'],f'Preserved bytes changed: {name}')
        elif 'mtime_ns' in record: require(path.stat().st_mtime_ns==record['mtime_ns'],f'Preserved file modified: {name}')
    return saved


def effective_config():
    inherited=common.config()
    neural={k:v for k,v in inherited.items() if k not in ('epochs','patience')}
    neural.update(max_optimizer_steps=10500,validate_every_steps=105,patience_checks=15)
    require(neural['seed']==42 and neural['batch_size']==64 and neural['threshold']==.5,'Unexpected original settings')
    return dict(inherited_config=inherited,neural=neural,financial_lr=dict(
        penalty='l2',C=1.0,solver='lbfgs',max_iter=10000,tol=1e-4,fit_intercept=True,
        class_weight='negative=1, positive=actual selected negative/positive ratio',seed=42,threshold=.5))


def runtime_environment():
    import importlib.metadata
    import platform
    import torch
    versions={name:importlib.metadata.version(name) for name in ('torch','numpy','scikit-learn','transformers')}
    return dict(python=sys.version,python_implementation=platform.python_implementation(),platform=platform.platform(),
        package_versions=versions,torch_cuda_version=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),
        cuda_available=torch.cuda.is_available(),gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        container_image='huawei-q2-lhx:20260923',
        container_image_id='sha256:079c1a319b52f1059cc6d2c22f3b0f41a619c7226aacdedb9a86830d6eb732cf',
        container_identity_source='host-verified image identity supplied by experiment controller')


def source_identity():
    protected_unchanged()
    payload=dict(design=DESIGN,effective_config=effective_config(),data_identity=annual.identity(require_embeddings=True),
        runtime_environment=read_json(CACHE/'runtime_environment.json'),
        source_sha256={name:common.file_sha256(ROOT/name) for name in SOURCES},
        preserved_identity_sha256=common.file_sha256(CACHE/'preserved_identity.json'),
        sampling_manifests={str(f):common.file_sha256(CACHE/f'fold_{f}.json') for f in range(5)})
    return dict(fingerprint=digest(payload),**payload)


def partition_indices(d,fold):
    masks=common.partition(d,fold)
    return tuple(np.flatnonzero(m) for m in masks)


def selected_indices(d,fold,scheme):
    indices=np.asarray(annual.training_indices(fold,scheme),dtype=np.int64)
    full,validation,test=partition_indices(d,fold)
    require(indices.ndim==1 and len(indices)>0 and np.array_equal(indices,np.unique(indices)), 'Training indices must be sorted and unique')
    require(np.isin(indices,full).all(),'Selected row outside training company groups')
    positives=full[d['labels'][full]==1]
    require(np.array_equal(indices[d['labels'][indices]==1],positives),'Training positives were dropped')
    if scheme=='full': require(np.array_equal(indices,full),'Full training pool differs')
    else:
        ratio=int(scheme.split('_')[-1])
        require(int((d['labels'][indices]==0).sum())==ratio*len(positives),'Selected ratio differs')
    return indices


def preproc_dir(fold):
    return CACHE/'preprocessing'/f'fold_{fold}'


def create_preprocessing(d,fold,identity):
    path=preproc_dir(fold)
    with locked(CACHE/'locks'/f'preprocessing_{fold}.lock',blocking=True):
        full,_,_=partition_indices(d,fold)
        expected=dict(data_identity_sha256=digest(identity['data_identity']),full_train_indices_sha256=index_hash(full),fold=fold)
        if (path/'identity.json').exists():
            record=read_json(path/'identity.json')
            require(all(record[k]==v for k,v in expected.items()),'Preprocessing belongs to another cohort')
            require(common.file_sha256(path/'numeric.npy')==record['numeric_sha256'],'Preprocessed numeric data changed')
            require(common.file_sha256(path/'preprocessing.json')==record['preprocessing_sha256'],'Preprocessing state changed')
            return record
        require(not path.exists() or not any(path.iterdir()),f'Incomplete preprocessing preserved: {path}')
        mask=np.zeros(len(d['labels']),dtype=bool);mask[full]=True
        x,state=common.preprocess(d,mask)
        layout=feature_layout(d,state)
        require(len(layout['financial']['value_indices'])==30,'Financial LR requires exactly thirty scalar financial values')
        require(x.shape[:2]==(len(d['labels']),20),'Expected five years of twenty quarterly nodes')
        state=dict(**state,modality_layout=layout,annual_fit_pool=expected)
        path.mkdir(parents=True,exist_ok=True)
        with (path/'numeric.npy').open('xb') as stream: np.save(stream,x,allow_pickle=False)
        immutable_json(path/'preprocessing.json',state)
        stat=(path/'numeric.npy').stat()
        record=dict(**expected,numeric_sha256=common.file_sha256(path/'numeric.npy'),
            preprocessing_sha256=common.file_sha256(path/'preprocessing.json'),numeric_size=stat.st_size,numeric_mtime_ns=stat.st_mtime_ns,
            numeric_shape=list(x.shape),financial_lr_dimension=1800)
        immutable_json(path/'identity.json',record)
        return record


def load_preprocessing(fold,protocol,full_verify=False):
    path=preproc_dir(fold)
    record=read_json(path/'identity.json')
    require(record==protocol['preprocessing'][str(fold)],'Frozen preprocessing identity differs')
    stat=(path/'numeric.npy').stat()
    require(stat.st_size==record['numeric_size'] and stat.st_mtime_ns==record['numeric_mtime_ns'],'Numeric preprocessing cache changed')
    require(common.file_sha256(path/'preprocessing.json')==record['preprocessing_sha256'],'Preprocessing JSON changed')
    if full_verify: require(common.file_sha256(path/'numeric.npy')==record['numeric_sha256'],'Numeric preprocessing bytes changed')
    return np.load(path/'numeric.npy',mmap_mode='r',allow_pickle=False),read_json(path/'preprocessing.json')


def prepare():
    with locked(CACHE/'prepare.lock'):
        protected_unchanged(full=True)
        immutable_json(CACHE/'runtime_environment.json',runtime_environment())
        identity=source_identity();d=annual.load_data()
        require(len(d['labels'])>0 and np.isin(d['labels'],[0,1]).all(),'Invalid labels')
        require(set(map(str,np.unique(d['transition_ex_post']))).issubset(TRANSITIONS),'Unknown transition category')
        prep={}
        for fold in range(5):
            selected={s:selected_indices(d,fold,s) for s in SCHEMES}
            require(np.isin(selected['ratio_3'],selected['ratio_4']).all(),'Ratio3 must be nested in ratio4')
            prep[str(fold)]=create_preprocessing(d,fold,identity)
        carry=annual.load_carry_baseline()
        require(all(len(v)==len(d['labels']) for v in carry.values()),'Carry baseline is not sample aligned')
        require(np.isin(carry['prediction'],[-1,0,1]).all() and np.array_equal(carry['available'],carry['prediction']>=0),
                'Invalid carry availability or abstention encoding')
        config=effective_config();immutable_json(CACHE/'effective_config.json',config)
        protocol=dict(**identity,preprocessing=prep,n_samples=len(d['labels']),n_positive=int(d['labels'].sum()))
        immutable_json(CACHE/'training_protocol.json',protocol)
        names=list(SOURCES)+['cache/annual_risk_v1/effective_config.json','cache/annual_risk_v1/training_protocol.json',
                            'cache/annual_risk_v1/runtime_environment.json',
                            'cache/annual_risk_v1/preserved_identity.json']
        names += [f'cache/annual_risk_v1/fold_{f}.json' for f in range(5)]
        target=CACHE/'training_source_snapshot.zip'
        if target.exists():
            with zipfile.ZipFile(target) as archive:
                require(set(archive.namelist())==set(names),'Snapshot inventory differs')
                for name in names: require(archive.read(name)==(ROOT/name).read_bytes(),f'Snapshot changed: {name}')
        else:
            with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as archive:
                for name in names: archive.write(ROOT/name,name)
        common.log('annual_training_prepared',fingerprint=protocol['fingerprint'],n=protocol['n_samples'],expected_fits=60)
        return protocol


def load_protocol():
    protocol=read_json(CACHE/'training_protocol.json');current=source_identity()
    frozen=current['runtime_environment'];actual=runtime_environment()
    for key in ('python','package_versions','torch_cuda_version','cudnn_version'):
        require(actual[key]==frozen[key],f'Runtime version differs: {key}')
    require(protocol['fingerprint']==current['fingerprint'],'Annual source/data/config identity changed')
    require(read_json(CACHE/'effective_config.json')==current['effective_config'],'Effective settings changed')
    with zipfile.ZipFile(CACHE/'training_source_snapshot.zip') as archive:
        for name in archive.namelist(): require(archive.read(name)==(ROOT/name).read_bytes(),f'Frozen source changed: {name}')
    return protocol


def fit_identity(protocol,d,model,scheme,fold):
    full,val,test=partition_indices(d,fold);selected=selected_indices(d,fold,scheme)
    values=dict(experiment='annual_risk_v1',model=model,scheme=scheme,fold=fold,
        protocol_fingerprint=protocol['fingerprint'],preprocessing=protocol['preprocessing'][str(fold)],
        selected_index_sha256=index_hash(selected),full_train_index_sha256=index_hash(full),
        validation_index_sha256=index_hash(val),test_index_sha256=index_hash(test),
        sampling_manifest_sha256=protocol['sampling_manifests'][str(fold)])
    return dict(fingerprint=digest(values),**values)


def result_dir(model,scheme,fold):
    return RESULTS/scheme/model/f'fold_{fold}'


def require_checks(protocol):
    checks=read_json(CACHE/'engineering_checks.json')
    require(checks['fingerprint']==protocol['fingerprint'],'Engineering checks refer to another source identity')
    require(set(checks['checks'])=={f'{s}:{m}' for s in SCHEMES for m in MODELS},'Incomplete engineering checks')
    require(all(v.get('passed') is True for v in checks['checks'].values()),'Engineering check failed')


def record_check(protocol,model,scheme,value):
    with locked(CACHE/'engineering_checks.lock',blocking=True):
        path=CACHE/'engineering_checks.json'
        record=read_json(path) if path.exists() else dict(fingerprint=protocol['fingerprint'],checks={})
        require(record['fingerprint']==protocol['fingerprint'],'Existing checks belong to another source identity')
        record['checks'][f'{scheme}:{model}']=value
        common.atomic_json(path,record)


def safe_scores(labels,probabilities):
    """Undefined empty/single-class ranking metrics stay null, never fabricated."""
    y=np.asarray(labels,dtype=np.int64);p=np.asarray(probabilities,dtype=np.float64)
    require(y.shape==p.shape and np.isfinite(p).all(),'Invalid metric inputs')
    require(np.isin(y,[0,1]).all() and ((p>=0)&(p<=1)).all(),'Invalid labels/probabilities')
    if len(y) and len(np.unique(y))==2: return common.scores(y,p,.5)
    out={k:None for k in METRICS}
    pred=p>=.5
    tp=int(((y==1)&pred).sum());fn=int(((y==1)&~pred).sum())
    tn=int(((y==0)&~pred).sum());fp=int(((y==0)&pred).sum())
    n=len(y);positive=int(y.sum())
    out.update(n=n,positive=positive,prevalence=float(y.mean()) if n else None,
        accuracy=float((pred==y).mean()) if n else None,precision=tp/(tp+fp) if tp+fp else None,
        recall=tp/(tp+fn) if tp+fn else None,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        specificity=tn/(tn+fp) if tn+fp else None,brier=float(np.mean((p-y)**2)) if n else None,
        tn=tn,fp=fp,fn=fn,tp=tp,threshold=.5,roc_auc=None,ap=None,
        metric_note='empty subset' if not n else 'single-class subset: ROC-AUC/AP are undefined for risk discrimination')
    return out


def diagnostics(d,indices,probabilities,carry):
    indices=np.asarray(indices,dtype=np.int64);p=np.asarray(probabilities)
    available=np.asarray(carry['available'])[indices].astype(bool)
    y=d['labels'][indices];covered=indices[available]
    years=np.asarray(d['years'])[covered].astype(int)
    reference=np.asarray(carry['reference_year'])[covered].astype(int)
    age=years-reference
    model=safe_scores(y[available],p[available])
    carried=safe_scores(y[available],np.asarray(carry['prediction'])[covered].astype(float))
    common_subset=dict(total_n=len(indices),covered_n=int(available.sum()),coverage=float(available.mean()) if len(indices) else 0.,
        model=model,carry=carried,carry_correct=int(carried.get('tn',0)+carried.get('tp',0)),
        state_age_years=dict(minimum=int(age.min()),maximum=int(age.max()),mean=float(age.mean())) if len(age) else None,
        interpretation='same covered rows only; empty or single-class coverage cannot establish risk discrimination')
    transitions={}
    for kind in TRANSITIONS:
        mask=np.asarray(d['transition_ex_post'])[indices]==kind
        transitions[kind]=safe_scores(y[mask],p[mask])
    return dict(carry_common_covered=common_subset,transition_diagnostics_ex_post=transitions)


def write_predictions(path,d,indices,probabilities,fold,carry):
    indices=np.asarray(indices,dtype=np.int64)
    require(len(indices)==len(probabilities),'Prediction length differs')
    with Path(path).open('x',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream)
        writer.writerow(['sample_index','code','year','cutoff','label','probability','fold','previous_risk_ex_post',
                         'transition_ex_post','carry_available','carry_prediction','carry_reference_year','carry_age_years'])
        for i,p in zip(indices,probabilities):
            available=bool(carry['available'][i]);reference=int(carry['reference_year'][i])
            writer.writerow([int(i),str(d['codes'][i]),int(d['years'][i]),str(d['cutoffs'][i]),int(d['labels'][i]),float(p),fold,
                int(d['previous_risk_ex_post'][i]),str(d['transition_ex_post'][i]),int(available),int(carry['prediction'][i]),
                reference,int(d['years'][i])-reference if available else ''])


def read_predictions(path,d,expected,fold,carry):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream: rows=list(csv.DictReader(stream))
    require([int(r['sample_index']) for r in rows]==list(map(int,expected)),f'Prediction coverage/order differs: {path}')
    for row,i in zip(rows,expected):
        require(row['code']==str(d['codes'][i]) and int(row['label'])==int(d['labels'][i]) and
                row['cutoff']==str(d['cutoffs'][i]) and int(row['fold'])==fold and int(row['year'])==int(d['years'][i]),
                f'Prediction identity differs: {path}')
        require(row['transition_ex_post']==str(d['transition_ex_post'][i]) and
                int(row['previous_risk_ex_post'])==int(d['previous_risk_ex_post'][i]),'Diagnostic labels differ')
        require(int(row['carry_available'])==int(carry['available'][i]) and
                int(row['carry_prediction'])==int(carry['prediction'][i]) and
                int(row['carry_reference_year'])==int(carry['reference_year'][i]),'Carry record differs')
    p=np.asarray([float(r['probability']) for r in rows],dtype=np.float64)
    require(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Invalid saved probabilities')
    return rows,p


def matching_scores(saved,computed):
    for key,value in computed.items():
        if value is None: require(saved.get(key) is None,f'Undefined metric must be null: {key}')
        elif isinstance(value,str): require(saved.get(key)==value,f'Metric note differs: {key}')
        else: require(key in saved and abs(float(saved[key])-float(value))<=1e-10,f'Metric differs: {key}')


def audit_fit(path,d,protocol,model,scheme,fold,carry):
    identity=fit_identity(protocol,d,model,scheme,fold)
    result=read_json(path/'metrics.json');validation=read_json(path/'validation_metrics.json')
    started=read_json(path/'started.json')
    for record in (result,validation,started): require(record['identity']==identity,'Completed fit belongs to another experiment')
    require(started['config']==protocol['effective_config'],'Saved effective configuration differs')
    _,va,te=partition_indices(d,fold);tr=selected_indices(d,fold,scheme)
    weight=float((d['labels'][tr]==0).sum())/float(d['labels'][tr].sum())
    require(result['pos_weight']==started['pos_weight']==weight,'Class weight differs')
    checkpoint='best.joblib' if model=='FinancialLR' else 'best.pt'
    required=('started.json','preprocessing.json','trace.jsonl','predictions.csv','validation_predictions.csv','validation_metrics.json',checkpoint)
    require(set(result['artifact_sha256'])==set(required),'Result artifact inventory differs')
    for name in required: require(common.file_sha256(path/name)==result['artifact_sha256'][name],f'Completed artifact changed: {path/name}')
    require(read_json(path/'preprocessing.json')==read_json(preproc_dir(fold)/'preprocessing.json'),'Fit preprocessing state differs')
    rows,p=read_predictions(path/'predictions.csv',d,te,fold,carry)
    vrows,pv=read_predictions(path/'validation_predictions.csv',d,va,fold,carry)
    matching_scores(result,safe_scores(d['labels'][te],p));matching_scores(validation,safe_scores(d['labels'][va],pv))
    require(result['diagnostics']==diagnostics(d,te,p,carry),'Saved diagnostics differ')
    trace=[json.loads(line) for line in (path/'trace.jsonl').read_text().splitlines() if line]
    require(len(trace)==result['validation_checks'],'Validation trace count differs')
    if model=='FinancialLR':
        require(result['converged'] is True and result['solver_iterations']<protocol['effective_config']['financial_lr']['max_iter'],
                'Logistic regression did not converge')
        require(result['financial_input_dimension']==1800 and result['optimizer_steps']==0,'LR input/accounting differs')
    else:
        cfg=protocol['effective_config']['neural']
        require(0<result['optimizer_steps']<=cfg['max_optimizer_steps'] and result['optimizer_steps']%cfg['validate_every_steps']==0,
                'Neural optimizer budget differs')
        require([r['optimizer_steps'] for r in trace]==list(range(cfg['validate_every_steps'],result['optimizer_steps']+1,cfg['validate_every_steps'])),
                'Validation interval differs')
        require(result['best_step'] in [r['optimizer_steps'] for r in trace] and
                abs(validation['ap']-result['best_validation_ap'])<=1e-10,'Best checkpoint differs')
        require(result['training_examples_seen']>0 and result['positive_training_exposures']>0,'Missing exposure accounting')
    return result,validation,rows,vrows


def financial_matrix(x,layout):
    section=layout['financial']
    require([len(section[k]) for k in ('value_indices','observed_indices','age_indices')]==[30,30,30],
            'Expected thirty financial fields in each input group')
    channels=section['value_indices']+section['observed_indices']+section['age_indices']
    out=np.asarray(x[:,:,channels],dtype=np.float64).reshape(len(x),1800)
    require(np.isfinite(out).all(),'Financial LR contains nonfinite values')
    return np.ascontiguousarray(out)


def fit_lr(features,labels,weight,cfg):
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    model=LogisticRegression(penalty=cfg['penalty'],C=cfg['C'],solver=cfg['solver'],max_iter=cfg['max_iter'],
        tol=cfg['tol'],fit_intercept=cfg['fit_intercept'],class_weight={0:1.,1:weight},random_state=cfg['seed'])
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter('always',ConvergenceWarning)
        model.fit(features,labels)
    require(not any(issubclass(w.category,ConvergenceWarning) for w in messages),'FinancialLR convergence warning; result rejected')
    require(int(np.max(model.n_iter_))<cfg['max_iter'],'FinancialLR exhausted max_iter; result rejected')
    require(np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all(),'Nonfinite logistic coefficients')
    return model


def train(model_name,scheme,fold,device='cuda:0',check=False):
    require(model_name in MODELS and scheme in SCHEMES and fold in range(5),'Invalid fit condition')
    protocol=load_protocol()
    if not check: require_checks(protocol)
    with locked(CACHE/'locks'/f'{scheme}_{model_name}_{fold}.lock'):
        return train_locked(model_name,scheme,fold,device,check,protocol)


def train_locked(model_name,scheme,fold,device,check,protocol):
    import torch
    d=annual.load_data();carry=annual.load_carry_baseline()
    full,va,te=partition_indices(d,fold);tr=selected_indices(d,fold,scheme)
    identity=fit_identity(protocol,d,model_name,scheme,fold)
    path=result_dir(model_name,scheme,fold)
    if not check and (path/'metrics.json').exists():
        audit_fit(path,d,protocol,model_name,scheme,fold,carry)
        common.log('annual_fit_skip_verified',model=model_name,scheme=scheme,fold=fold)
        return
    if not check: require(not path.exists() or not any(path.iterdir()),f'Incomplete fit preserved; archive explicitly before retry: {path}')
    x,state=load_preprocessing(fold,protocol)
    cfg=protocol['effective_config'];neural=cfg['neural']
    torch.set_num_threads(neural['torch_threads']);common.seed_all(neural['seed'])
    positive=int(d['labels'][tr].sum());weight=float(len(tr)-positive)/positive
    train_indices=tr
    if check:
        train_indices=np.concatenate([tr[d['labels'][tr]==1][:4],tr[d['labels'][tr]==0][:4]])
        require(len(train_indices)==8,'Engineering check needs four positives and four negatives')
    if not check:
        path.mkdir(parents=True,exist_ok=True)
        immutable_json(path/'preprocessing.json',state)
        immutable_json(path/'started.json',dict(identity=identity,config=cfg,pos_weight=weight,seed=neural['seed'],
            full_train_n=len(full),selected_train_n=len(tr),validation_n=len(va),test_n=len(te),
            sampling_statistics=read_json(CACHE/f'fold_{fold}.json')['statistics'],started_at=utc(),pid=os.getpid(),device=device))
    start=time.monotonic()
    if model_name=='FinancialLR':
        import joblib
        features=financial_matrix(x,state['modality_layout'])
        fitted=fit_lr(features[train_indices],d['labels'][train_indices],weight,cfg['financial_lr'])
        if check:
            probabilities=fitted.predict_proba(features[va[:8]])[:,1]
            require(np.isfinite(probabilities).all(),'LR check prediction is nonfinite')
            value=dict(passed=True,model=model_name,scheme=scheme,fold=fold,examples=8,input_dimension=1800,
                       converged=True,solver_iterations=int(fitted.n_iter_.max()),checked_at=utc())
            record_check(protocol,model_name,scheme,value);common.log('annual_engineering_check',**value)
            return value
        pv=fitted.predict_proba(features[va])[:,1];p=fitted.predict_proba(features[te])[:,1]
        joblib.dump(dict(model=fitted,identity=identity,config=cfg['financial_lr'],pos_weight=weight),path/'best.joblib')
        ap=safe_scores(d['labels'][va],pv)['ap']
        trace=dict(validation_check=1,optimizer_steps=0,solver_iterations=int(fitted.n_iter_.max()),validation_ap=ap)
        with (path/'trace.jsonl').open('x',encoding='utf-8') as stream: stream.write(json.dumps(trace)+'\n')
        accounting=dict(converged=True,solver_iterations=int(fitted.n_iter_.max()),optimizer_steps=0,
            optimizer_kind='lbfgs solver; iterations are not neural minibatch updates',training_examples_seen=None,
            positive_training_exposures=None,epochs_equivalent=None,completed_epochs=None,validation_checks=1,
            best_step=None,best_validation_ap=ap,parameters=int(fitted.coef_.size+fitted.intercept_.size),financial_input_dimension=1800)
    else:
        p,pv,accounting=fit_neural(model_name,scheme,fold,device,check,protocol,d,x,state,train_indices,tr,va,te,weight,path,identity)
        if check: return accounting
    result=safe_scores(d['labels'][te],p)
    validation=safe_scores(d['labels'][va],pv)
    result.update(identity=identity,model=model_name,scheme=scheme,fold=fold,pos_weight=weight,**accounting,
                  seconds=round(time.monotonic()-start,2),diagnostics=diagnostics(d,te,p,carry))
    validation.update(identity=identity,best_step=accounting['best_step'],checkpoint_selection='fixed LR fit' if model_name=='FinancialLR' else 'maximum validation AP')
    write_predictions(path/'predictions.csv',d,te,p,fold,carry)
    write_predictions(path/'validation_predictions.csv',d,va,pv,fold,carry)
    immutable_json(path/'validation_metrics.json',validation)
    checkpoint='best.joblib' if model_name=='FinancialLR' else 'best.pt'
    result['artifact_sha256']={n:common.file_sha256(path/n) for n in ('started.json','preprocessing.json','trace.jsonl',
        'predictions.csv','validation_predictions.csv','validation_metrics.json',checkpoint)}
    immutable_json(path/'metrics.json',result)
    audit_fit(path,d,protocol,model_name,scheme,fold,carry)
    common.log('annual_fit_complete',model=model_name,scheme=scheme,fold=fold,ap=result['ap'],updates=result['optimizer_steps'],seconds=result['seconds'])
    return result


def boundary_checks(model,x,doc_ids,valid,layout):
    """Ephemeral synthetic masks only: no test file or training-row mutation."""
    import torch
    numeric=x[:1].detach().clone().repeat(8,1,1)
    ids=doc_ids[:1].detach().clone().repeat(8,1)
    ipo=torch.ones((8,20),dtype=torch.bool,device=x.device)
    candidates=torch.cat((doc_ids[doc_ids>=0],torch.zeros(1,dtype=torch.long,device=x.device)))
    ids[:]=candidates[0]
    numeric[:,:,layout['text_available_index']]=1
    for modality in ('financial','governance'):
        numeric[:,:,layout[modality]['observed_indices']]=1
    # 0, 1 and 3 history years; missing current year; single-modality and all-modality absence.
    ipo[0,:16]=False;ipo[1,:12]=False;ipo[2,:4]=False;ipo[3,16:]=False
    numeric[4,:,layout['financial']['observed_indices']]=0
    numeric[5,:,layout['governance']['observed_indices']]=0
    numeric[6,:,layout['text_available_index']]=0;ids[6]=-1
    for modality in ('financial','governance'): numeric[7,:,layout[modality]['observed_indices']]=0
    numeric[7,:,layout['text_available_index']]=0;ids[7]=-1
    model.eval();model.zero_grad(set_to_none=True)
    outputs=model.forward_features(numeric,ids,ipo)
    logits=model.head(outputs['representation']).squeeze(-1)
    require(torch.isfinite(logits).all().item(),'Mask-boundary output is nonfinite')
    logits.square().mean().backward()
    require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()),'Mask-boundary gradient nonfinite')
    for letter in model.active:
        require(torch.count_nonzero(outputs[letter][7]).item()==0,'All-empty modality increment is not zero')
        if letter in ('A','B'): require(torch.count_nonzero(outputs[letter][3]).item()==0,'Missing-current-year increment is not zero')
    if 'B' in model.active:
        with torch.no_grad():
            banks=model.encode_history_banks(numeric,ids,ipo)
            compared=model.compare_history_banks(banks)['B']['modalities']
        for modality,bank in banks.items():
            selected=compared[modality]['selected_indices'];good=compared[modality]['selected_valid']
            expected=(bank['valid'][:,:4].sum(-1).clamp_max(2)*bank['valid'][:,4]).long()
            require(torch.equal(good.sum(-1),expected),'Top2 count differs from valid historical candidates')
            allowed=bank['valid'][:,:4].gather(1,selected.clamp_min(0)) & bank['valid'][:,4:5]
            require((~good | allowed).all().item() and (selected[~good]==-1).all().item(),'Top2 selected invalid history')
            require(bank['valid'][:3,:4].sum(-1).tolist()==[0,1,3],'Boundary cases did not cover 0/1/2+ history')
    model.zero_grad(set_to_none=True)
    return dict(history_candidate_counts=[0,1,3] if 'B' in model.active else None,missing_current_year=True,
                missing_modalities=['financial','governance','text','all'],finite_gradients=True)


def fit_neural(model_name,scheme,fold,device,check,protocol,d,x,state,train_indices,tr,va,te,weight,path,identity):
    import torch
    from torch.utils.data import DataLoader,TensorDataset,Subset
    from sklearn.metrics import average_precision_score
    cfg=protocol['effective_config']['neural']
    embeddings=annual.load_embeddings()
    model=make_model(model_name,x,state['modality_layout'],embeddings,cfg,device)
    parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight,device=device))
    dataset=TensorDataset(torch.from_numpy(np.array(x,copy=True)),torch.from_numpy(d['doc_ids'].astype('int64')),
        torch.from_numpy(d['valid_time'].astype(bool)),torch.from_numpy(d['labels'].astype('float32')))
    def loader(indices,training=False):
        return DataLoader(Subset(dataset,list(map(int,indices))),batch_size=8 if check else cfg['batch_size'],
            shuffle=training,num_workers=0,pin_memory=str(device).startswith('cuda'),
            generator=torch.Generator().manual_seed(cfg['seed']))
    def predict(indices):
        model.eval();values=[]
        with torch.inference_mode():
            for xb,db,vb,_ in loader(indices):
                values.append(model(xb.to(device),db.to(device),vb.to(device)).sigmoid().cpu().numpy())
        p=np.concatenate(values).astype(np.float64)
        require(np.isfinite(p).all(),'Nonfinite neural prediction')
        return p
    train_loader=loader(train_indices,training=True)
    iterator=iter(train_loader)
    best_ap,best_step,best_state=-1.,0,None
    steps,examples,positive_exposures,checks,stale=0,0,0,0,0
    interval_loss,interval_examples=0.,0
    if not check:
        with (path/'trace.jsonl').open('x',encoding='utf-8'): pass
    model.train()
    while steps<(1 if check else cfg['max_optimizer_steps']):
        try: xb,db,vb,yb=next(iterator)
        except StopIteration:
            iterator=iter(train_loader)
            xb,db,vb,yb=next(iterator)
        xb,db,vb,yb=[v.to(device,non_blocking=True) for v in (xb,db,vb,yb)]
        optimizer.zero_grad(set_to_none=True)
        loss=loss_fn(model(xb,db,vb),yb)
        aux=model.auxiliary_loss()
        if aux is not None: loss=loss+cfg['auxiliary_loss_weight']*aux
        require(torch.isfinite(loss).item(),'Nonfinite neural loss')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
        if check:
            tg=model.text_projection.weight.grad
            require(float(norm)>0 and tg is not None and torch.isfinite(tg).all().item() and torch.count_nonzero(tg).item()>0,
                    'Engineering check lacks finite, nonzero model/text gradients')
        optimizer.step()
        steps+=1;examples+=len(yb);positive_exposures+=int(yb.sum().item())
        interval_loss+=float(loss.detach())*len(yb);interval_examples+=len(yb)
        if check:
            p=predict(va[:8]);require(len(p)==8,'Engineering validation examples differ')
            with torch.no_grad():
                alone=model(xb[:1],db[:1],vb[:1]);batched=model(xb,db,vb)[:1]
            require(torch.allclose(alone,batched,atol=5e-5,rtol=5e-5),'Evaluation depends on other companies in batch')
            boundaries=boundary_checks(model,xb,db,vb,state['modality_layout'])
            value=dict(passed=True,model=model_name,scheme=scheme,fold=fold,device=device,examples=examples,
                parameters=parameters,optimizer_steps=steps,positive_exposures=positive_exposures,batch_invariance=True,
                boundary_checks=boundaries,checked_at=utc())
            record_check(protocol,model_name,scheme,value);common.log('annual_engineering_check',**value)
            return None,None,value
        if steps%cfg['validate_every_steps']!=0: continue
        pv=predict(va);ap=float(average_precision_score(d['labels'][va],pv));checks+=1
        if ap>best_ap:
            best_ap,best_step,stale=ap,steps,0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            temporary=path/'best.pt.tmp'
            torch.save(dict(state_dict=best_state,identity=identity,config=cfg,optimizer_steps=steps,
                validation_ap=ap,pos_weight=weight,parameters=parameters),temporary)
            os.replace(temporary,path/'best.pt')
        else: stale+=1
        row=dict(validation_check=checks,optimizer_steps=steps,training_examples_seen=examples,
            positive_training_exposures=positive_exposures,epochs_equivalent=examples/len(tr),
            completed_epochs=examples//len(tr),train_loss=interval_loss/interval_examples,validation_ap=ap,
            best_validation_ap=best_ap,best_step=best_step,nonimproving_checks=stale)
        with (path/'trace.jsonl').open('a',encoding='utf-8') as stream: stream.write(json.dumps(row,allow_nan=False)+'\n')
        common.log('annual_validation',model=model_name,scheme=scheme,fold=fold,**row)
        interval_loss,interval_examples=0.,0
        model.train()  # predict() set eval; validation may occur halfway through a training epoch.
        if stale>=cfg['patience_checks']: break
    require(best_state is not None,'No validated checkpoint exists')
    model.load_state_dict(best_state)
    pv,p=predict(va),predict(te)
    require(abs(float(average_precision_score(d['labels'][va],pv))-best_ap)<=1e-10,'Restored checkpoint AP differs')
    return p,pv,dict(parameters=parameters,optimizer_steps=steps,training_examples_seen=examples,
        positive_training_exposures=positive_exposures,epochs_equivalent=examples/len(tr),completed_epochs=examples//len(tr),
        validation_checks=checks,best_step=best_step,best_validation_ap=best_ap,
        stop_reason='patience_checks' if stale>=cfg['patience_checks'] else 'max_optimizer_steps',
        max_optimizer_steps=cfg['max_optimizer_steps'],validate_every_steps=cfg['validate_every_steps'],patience_checks=cfg['patience_checks'])


def write_table(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(key for row in rows for key in row))
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    os.replace(temporary,path)


def summary(final=False):
    with locked(CACHE/'summary.lock',blocking=True):
        protocol=load_protocol();d=annual.load_data();carry=annual.load_carry_baseline()
        protected_unchanged(full=final)
        if final:
            for fold in range(5): load_preprocessing(fold,protocol,full_verify=True)
        complete_fits=[];conditions=[];validation_conditions=[];table=[]
        for scheme in SCHEMES:
            for model_name in MODELS:
                fits=[];vals=[];rows=[];vrows=[]
                for fold in range(5):
                    path=result_dir(model_name,scheme,fold)
                    if not (path/'metrics.json').exists(): continue
                    fit,val,rr,vv=audit_fit(path,d,protocol,model_name,scheme,fold,carry)
                    fits.append(fit);vals.append(val);rows.extend(rr);vrows.extend(vv);complete_fits.append(fit)
                if len(fits)!=5: continue
                expected=list(range(len(d['labels'])))
                require(sorted(int(r['sample_index']) for r in rows)==expected and
                        sorted(int(r['sample_index']) for r in vrows)==expected,'OOF/validation must cover each annual row exactly once')
                require(len(rows)==len(d['labels'])==11199,'Confirmed annual cohort must contain exactly 11199 OOF rows')
                rows.sort(key=lambda r:int(r['sample_index']));vrows.sort(key=lambda r:int(r['sample_index']))
                p=np.array([float(r['probability']) for r in rows]);pv=np.array([float(r['probability']) for r in vrows])
                item=dict(model=model_name,scheme=scheme,completed_folds=5,source_experiment='annual_risk_v1',
                    protocol_fingerprint=protocol['fingerprint'],fit_fingerprints=[f['identity']['fingerprint'] for f in fits],
                    pooled_oof=safe_scores(d['labels'],p),diagnostics=diagnostics(d,np.arange(len(p)),p,carry))
                validation=dict(model=model_name,scheme=scheme,completed_folds=5,pooled_validation=safe_scores(d['labels'],pv))
                csvrow=dict(model=model_name,scheme=scheme,completed_folds=5,protocol_fingerprint=protocol['fingerprint'])
                for metric in METRICS:
                    for result,records,prefix in ((item,fits,'test'),(validation,vals,'validation')):
                        values=[r[metric] for r in records]
                        mean=float(np.mean(values)) if all(v is not None for v in values) else None
                        sd=float(np.std(values,ddof=1)) if all(v is not None for v in values) else None
                        result[metric+'_mean']=mean;result[metric+'_std']=sd
                        csvrow[prefix+'_'+metric+'_mean']=mean;csvrow[prefix+'_'+metric+'_std']=sd
                for key in ('optimizer_steps','training_examples_seen','positive_training_exposures','epochs_equivalent','parameters'):
                    values=[r[key] for r in fits]
                    item[key+'_mean']=float(np.mean(values)) if all(v is not None for v in values) else None
                    csvrow[key+'_mean']=item[key+'_mean']
                covered=item['diagnostics']['carry_common_covered']
                csvrow.update(oof_n=len(p),oof_positive=int(d['labels'].sum()),oof_ap=item['pooled_oof']['ap'],
                    carry_covered_n=covered['covered_n'],carry_coverage=covered['coverage'],
                    common_subset_model_accuracy=covered['model']['accuracy'],carry_accuracy=covered['carry']['accuracy'],
                    common_subset_model_ap=covered['model']['ap'],carry_ap=covered['carry']['ap'])
                for metric in METRICS: csvrow['oof_'+metric]=item['pooled_oof'][metric]
                strata=item['diagnostics']['transition_diagnostics_ex_post']
                for short,kind,metric in (('onset','risk_onset','recall'),('persistent','persistent_risk','recall'),
                                           ('recovery','recovery','specificity'),('stable_nonrisk','stable_nonrisk','specificity')):
                    csvrow[short+'_n']=strata[kind]['n']
                    csvrow[short+'_'+metric]=strata[kind][metric]
                csvrow['transition_unknown_n']=strata['unknown']['n']
                write_table(RESULTS/scheme/model_name/'oof_predictions.csv',rows)
                write_table(RESULTS/scheme/model_name/'validation_oof_predictions.csv',vrows)
                conditions.append(item);validation_conditions.append(validation);table.append(csvrow)
        if final: require(len(complete_fits)==60 and len(conditions)==12,'Final audit requires sixty completed fits and twelve conditions')
        available=np.asarray(carry['available']).astype(bool)
        carry_summary=dict(n=len(available),covered_n=int(available.sum()),coverage=float(available.mean()),
            scores=safe_scores(d['labels'][available],np.asarray(carry['prediction'])[available].astype(float)),
            reason_counts={str(key):int(count) for key,count in zip(*np.unique(carry['reason'],return_counts=True))},
            interpretation='strict known original-version coverage; a single-class subset is not a risk-discrimination baseline')
        report=dict(protocol_fingerprint=protocol['fingerprint'],completed_fits=len(complete_fits),expected_fits=60,
            completed_conditions=len(conditions),expected_conditions=12,models=conditions,validation_models=validation_conditions,
            folds=complete_fits,carry_baseline=carry_summary,final_audit=bool(final),
            std_definition='sample SD across five company folds (ddof=1); not a confidence interval',
            checkpoint_selection='validation AP only; validation metrics are not held-out test estimates',
            budget='neural maximum optimizer steps are equal; early stopping and tail batches change actual exposure',
            transition_note='ex-post diagnostic strata; not features or separate single-class ROC/AP tasks')
        RESULTS.mkdir(parents=True,exist_ok=True)
        common.atomic_json(RESULTS/'summary.json',report);write_table(RESULTS/'comparison.csv',table)
        common.log('annual_summary',completed_fits=len(complete_fits),conditions=len(conditions),final_audit=final)
        return report


def worker(worker_id):
    require(worker_id in (0,1),'Worker/GPU must be zero or one')
    protocol=load_protocol();require_checks(protocol)
    with locked(CACHE/f'worker_{worker_id}.lock'):
        state_path=CACHE/f'worker_{worker_id}_status.json'
        tasks=[(model,scheme,fold) for scheme in SCHEMES for model in MODELS for fold in range(5)]
        for index,(model,scheme,fold) in enumerate(tasks):
            if index%2!=worker_id: continue
            state=dict(status='running',worker=worker_id,pid=os.getpid(),protocol_fingerprint=protocol['fingerprint'],
                       model=model,scheme=scheme,fold=fold,updated_at=utc())
            common.atomic_json(state_path,state)
            child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),'train','--model',model,
                '--scheme',scheme,'--fold',str(fold),'--device',f'cuda:{worker_id}'],cwd=ROOT)
            state['child_pid']=child.pid;common.atomic_json(state_path,state)
            code=child.wait()
            if code:
                state.update(status='failed',returncode=code,updated_at=utc());common.atomic_json(state_path,state)
                raise RuntimeError(f'{model} {scheme} fold {fold} exited {code}')
        common.atomic_json(state_path,dict(status='complete',worker=worker_id,pid=os.getpid(),
            protocol_fingerprint=protocol['fingerprint'],updated_at=utc()))


def stop_workers(children):
    for child in children:
        if child.poll() is None:
            try: os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError: pass
    for child in children:
        try: child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try: os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            child.wait()


def pipeline():
    # Lock rejection stays outside the failure/status handler.
    with locked(CACHE/'pipeline.lock'):
        children,streams=[],[]
        status_path=CACHE/'pipeline_status.json'
        try:
            protocol=load_protocol();require_checks(protocol);protected_unchanged(full=True)
            status=dict(status='starting',pid=os.getpid(),protocol_fingerprint=protocol['fingerprint'],expected_fits=60,started_at=utc())
            common.atomic_json(status_path,status);(ROOT/'logs').mkdir(exist_ok=True)
            for worker_id in (0,1):
                stream=(ROOT/'logs'/f'annual_risk_gpu{worker_id}.log').open('a',encoding='utf-8');streams.append(stream)
                children.append(subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),'worker',
                    '--worker',str(worker_id)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True))
            last=0.
            while True:
                codes=[c.poll() for c in children];now=time.monotonic()
                if now-last>=10 or all(code is not None for code in codes):
                    status.update(status='running',updated_at=utc(),completed_fits=sum(1 for _ in RESULTS.glob('*/*/fold_*/metrics.json')),
                        workers=[dict(worker=i,pid=c.pid,running=codes[i] is None,returncode=codes[i]) for i,c in enumerate(children)])
                    common.atomic_json(status_path,status);last=now
                if any(code is not None and code!=0 for code in codes): raise RuntimeError(f'Annual workers failed: {codes}')
                if all(code==0 for code in codes): break
                time.sleep(2)
            report=summary(final=True)
            status.update(status='complete',completed_fits=report['completed_fits'],finished_at=utc())
            common.atomic_json(status_path,status)
        except BaseException:
            stop_workers(children)
            status=read_json(status_path) if status_path.exists() else {}
            status.update(status='failed',failed_at=utc(),pid=os.getpid(),error=traceback.format_exc(),
                workers=[dict(worker=i,pid=c.pid,running=c.poll() is None,returncode=c.poll()) for i,c in enumerate(children)])
            common.atomic_json(status_path,status)
            raise
        finally:
            for stream in streams: stream.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','check','train','worker','pipeline','summary','audit'))
    parser.add_argument('--model',choices=MODELS)
    parser.add_argument('--scheme',choices=SCHEMES)
    parser.add_argument('--fold',type=int,choices=range(5),default=0)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--worker','--gpu',dest='worker',type=int,choices=(0,1))
    args=parser.parse_args()
    if args.command=='prepare': prepare()
    elif args.command=='check':
        for scheme in ([args.scheme] if args.scheme else SCHEMES):
            for model in ([args.model] if args.model else MODELS): train(model,scheme,args.fold,args.device,check=True)
    elif args.command=='train':
        if args.model is None or args.scheme is None: parser.error('train requires --model and --scheme')
        train(args.model,args.scheme,args.fold,args.device)
    elif args.command=='worker':
        if args.worker is None: parser.error('worker requires --worker or --gpu')
        worker(args.worker)
    elif args.command=='pipeline': pipeline()
    elif args.command=='summary': summary()
    elif args.command=='audit': summary(final=True)


if __name__=='__main__':
    try:
        main()
    except AlreadyRunning as exc:
        common.log('annual_duplicate_launch_rejected',reason=str(exc),pid=os.getpid());sys.exit(3)
    except Exception:
        failure=dict(argv=sys.argv,pid=os.getpid(),failed_at=utc(),error=traceback.format_exc())
        target=CACHE/'failures'/f'{time.time_ns()}_{os.getpid()}.json'
        common.atomic_json(target,failure);common.log('annual_training_failed',failure=str(target),error=failure['error'])
        raise
