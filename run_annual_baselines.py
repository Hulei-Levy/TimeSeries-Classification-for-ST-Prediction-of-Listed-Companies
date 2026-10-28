"""Annual full-pool benchmark for the original fourteen baseline architectures.

This is a user-requested follow-up benchmark after viewing the previous sampling
conditions, not an independently prespecified confirmation experiment. Frozen
annual data, preprocessing, BERT and previous results remain read-only.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
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
import run_annual_risk as parent
import data_annual_risk as annual

ROOT=Path(__file__).resolve().parent
EXPERIMENT='annual_risk_baselines_v1'
CACHE=ROOT/'cache'/EXPERIMENT
RESULTS=ROOT/'results'/EXPERIMENT
MODELS=tuple(common.MODELS)
REFERENCE_MODELS=('FinancialLR','PerSeM','PerSeM_AB','PerSeM_ABC')
METRICS=parent.METRICS
SOURCES=('run_annual_baselines.py','run_annual_risk.py','data_annual_risk.py','run.py','run_persem.py',
         'models_classic.py','models_recent.py','models_persem.py','data.py','config.json')
require,read_json,digest,index_hash,utc=parent.require,parent.read_json,parent.digest,parent.index_hash,parent.utc
locked,immutable_json=parent.locked,parent.immutable_json
os.environ['PYTORCH_KERNEL_CACHE_PATH']=str(CACHE/'torch/kernels')
DESIGN=dict(version=1,experiment=EXPERIMENT,models=list(MODELS),scheme='full',folds=5,expected_fits=70,
    selection='user-requested subsequent full-pool benchmark after viewing prior full/ratio3/ratio4 results; not independent confirmation',
    parent_experiment='annual_risk_v1',comparison='14 new baselines plus 4 preserved annual full-pool models',
    preprocessing='read frozen annual full-training-pool preprocessing; no refit or replacement',
    initialization='all time-series models and text projection from scratch; frozen BERT is the only pretrained representation',
    attention_backend={'SEMPO':'SDPBackend.MATH','others':'default'},
    TimeFilter='batch64, microbatch16 weighted by actual batch length including tail63; eval16',
    auxiliary_loss='all models expose original auxiliary_loss; coefficient 0.01 retained',
    budget='10500 optimizer updates maximum; validation every105; patience15 checks; threshold0.5',
    diagnostics='preserve annual carry coverage and ex-post transition strata; empty/single-class ROC/AP remain null')


class ResourceBusy(RuntimeError):
    pass


def preserved_unchanged(full=False):
    saved=read_json(CACHE/'preserved_identity.json')
    for name,record in saved['files'].items():
        if not full and name not in SOURCES: continue
        path=ROOT/name
        require(path.is_file() and path.stat().st_size==record['size'],f'Preserved file changed: {name}')
        if 'sha256' in record: require(common.file_sha256(path)==record['sha256'],f'Preserved bytes changed: {name}')
        elif 'mtime_ns' in record: require(path.stat().st_mtime_ns==record['mtime_ns'],f'Preserved file modified: {name}')
    return saved


def source_identity():
    preserved=preserved_unchanged()
    annual_protocol=parent.load_protocol()
    previous=read_json(parent.RESULTS/'summary.json')
    require(previous['completed_fits']==60 and previous['completed_conditions']==12 and previous['final_audit'] is True,
            'The annual parent experiment is not complete and audited')
    require(annual_protocol['fingerprint']==preserved['parent_protocol_fingerprint']==previous['protocol_fingerprint'],
            'Annual parent protocol identity differs')
    cfg=annual_protocol['effective_config']['neural']
    require(cfg['max_optimizer_steps']==10500 and cfg['validate_every_steps']==105 and cfg['patience_checks']==15,
            'Annual training budget differs')
    require(cfg['batch_size']==64 and cfg['graph_microbatch_size']==16 and cfg['seed']==42 and
            cfg['threshold']==.5 and cfg['auxiliary_loss_weight']==.01,'Annual configuration differs')
    payload=dict(design=DESIGN,parent_protocol_fingerprint=annual_protocol['fingerprint'],config=cfg,
        source_sha256={n:common.file_sha256(ROOT/n) for n in SOURCES},data_identity=annual_protocol['data_identity'],
        preprocessing=annual_protocol['preprocessing'],runtime_environment=read_json(CACHE/'runtime_environment.json'),
        parent_protocol_sha256=common.file_sha256(parent.CACHE/'training_protocol.json'),
        parent_source_snapshot_sha256=common.file_sha256(parent.CACHE/'training_source_snapshot.zip'),
        parent_summary_sha256=common.file_sha256(parent.RESULTS/'summary.json'),
        preserved_identity_sha256=common.file_sha256(CACHE/'preserved_identity.json'))
    return dict(fingerprint=digest(payload),**payload)


def prepare():
    with locked(CACHE/'prepare.lock'):
        preserved_unchanged(full=True)
        immutable_json(CACHE/'runtime_environment.json',parent.runtime_environment())
        identity=source_identity();annual_protocol=parent.load_protocol();d=annual.load_data()
        require(len(d['labels'])==11199 and int(d['labels'].sum())==806,'Confirmed annual cohort differs')
        require(len(MODELS)==14 and len(set(MODELS))==14,'Expected fourteen original baseline names')
        for fold in range(5):
            x,_=parent.load_preprocessing(fold,annual_protocol,full_verify=True)
            require(x.shape==(11199,20,207),'Frozen annual input shape differs')
            full,_,_=parent.partition_indices(d,fold)
            require(np.array_equal(full,annual.training_indices(fold,'full')),'Annual full selection differs')
        references,_,_=collect_references(d,annual_protocol,annual.load_carry_baseline())
        require(len(references)==4,'Four preserved full-pool references are required')
        immutable_json(CACHE/'effective_config.json',identity['config'])
        immutable_json(CACHE/'protocol.json',identity)
        names=list(SOURCES)+[f'cache/{EXPERIMENT}/runtime_environment.json',f'cache/{EXPERIMENT}/effective_config.json',
            f'cache/{EXPERIMENT}/protocol.json',f'cache/{EXPERIMENT}/preserved_identity.json']
        destination=CACHE/'source_snapshot.zip'
        if destination.exists():
            with zipfile.ZipFile(destination) as archive:
                require(set(archive.namelist())==set(names),'Snapshot inventory differs')
                for name in names: require(archive.read(name)==(ROOT/name).read_bytes(),f'Snapshot differs: {name}')
        else:
            with zipfile.ZipFile(destination,'x',zipfile.ZIP_DEFLATED) as archive:
                for name in names: archive.write(ROOT/name,name)
        common.log('annual_baselines_prepared',fingerprint=identity['fingerprint'],expected_fits=70,reused_reference_fits=20)
        return identity


def load_protocol():
    protocol=read_json(CACHE/'protocol.json');current=source_identity()
    require(protocol==current,'Baseline source/data/protocol identity changed')
    require(read_json(CACHE/'effective_config.json')==protocol['config'],'Effective configuration changed')
    actual=parent.runtime_environment();frozen=protocol['runtime_environment']
    for key in ('python','package_versions','torch_cuda_version','cudnn_version'):
        require(actual[key]==frozen[key],f'Runtime differs: {key}')
    with zipfile.ZipFile(CACHE/'source_snapshot.zip') as archive:
        for name in archive.namelist(): require(archive.read(name)==(ROOT/name).read_bytes(),f'Frozen source changed: {name}')
    return protocol


def split_statistics(d,fold):
    output={}
    for name,indices in zip(('training','validation','test'),parent.partition_indices(d,fold)):
        positives=int(d['labels'][indices].sum())
        output[name]=dict(windows=len(indices),companies=len(np.unique(d['codes'][indices])),
            positives=positives,negatives=len(indices)-positives,prevalence=positives/len(indices))
    return output

def fit_identity(protocol,d,name,fold):
    full,va,te=parent.partition_indices(d,fold)
    require(np.array_equal(full,annual.training_indices(fold,'full')),'Full training indices differ')
    value=dict(experiment=EXPERIMENT,model=name,scheme='full',fold=fold,protocol_fingerprint=protocol['fingerprint'],
        parent_protocol_fingerprint=protocol['parent_protocol_fingerprint'],preprocessing=protocol['preprocessing'][str(fold)],
        train_index_sha256=index_hash(full),validation_index_sha256=index_hash(va),test_index_sha256=index_hash(te))
    return dict(fingerprint=digest(value),**value)


def result_dir(name,fold):
    return RESULTS/'full'/name/f'fold_{fold}'


def attention_context(name):
    if name!='SEMPO': return nullcontext()
    from torch.nn.attention import sdpa_kernel,SDPBackend
    return sdpa_kernel(SDPBackend.MATH)


def record_check(protocol,name,record):
    with locked(CACHE/'checks.lock',blocking=True):
        path=CACHE/'engineering_checks.json'
        saved=read_json(path) if path.exists() else dict(fingerprint=protocol['fingerprint'],checks={})
        require(saved['fingerprint']==protocol['fingerprint'],'Checks refer to another source identity')
        saved['checks'][name]=record;common.atomic_json(path,saved)


def require_checks(protocol):
    saved=read_json(CACHE/'engineering_checks.json')
    require(saved['fingerprint']==protocol['fingerprint'] and set(saved['checks'])==set(MODELS),'Incomplete engineering checks')
    for name,record in saved['checks'].items():
        require(record['passed'] is True and record['train_examples']==64 and record['optimizer_steps']==1
            and record['validation_examples']==64 and record['device'].startswith('cuda'),f'Full-batch CUDA check missing: {name}')
        require(record['required_peak_reserved_bytes']>0 and record['train_peak_allocated_bytes']>0,'Missing GPU memory profile')
        require(record['attention_backend']==('SDPBackend.MATH' if name=='SEMPO' else 'default'),'Check attention backend differs')
    return saved


def check_capacity(protocol,name,device,margin_bytes=1073741824):
    import torch
    require(str(device).startswith('cuda'),'Formal baseline training requires an explicitly assigned CUDA device')
    record=require_checks(protocol)['checks'][name]
    free,total=torch.cuda.mem_get_info(torch.device(device))
    needed=int(record['required_peak_reserved_bytes'])+int(margin_bytes)
    if free<needed:
        raise ResourceBusy(f'{name} on {device}: free={free}, profile+margin={needed}, total={total}; batch/model unchanged')
    return dict(free_bytes=int(free),total_bytes=int(total),required_bytes=needed,margin_bytes=int(margin_bytes))


def audit_fit(path,d,protocol,name,fold,carry):
    identity=fit_identity(protocol,d,name,fold)
    result=read_json(path/'metrics.json');val=read_json(path/'validation_metrics.json');started=read_json(path/'started.json')
    for record in (result,val,started): require(record['identity']==identity,'Completed baseline identity differs')
    require(started['config']==protocol['config'],'Completed baseline configuration differs')
    full,va,te=parent.partition_indices(d,fold)
    weight=float(len(full)-d['labels'][full].sum())/float(d['labels'][full].sum())
    require(result['pos_weight']==started['pos_weight']==weight,'Full-pool class weight differs')
    require(result['split_statistics']==started['split_statistics']==split_statistics(d,fold),'Split company/window/prevalence counts differ')
    backend='SDPBackend.MATH' if name=='SEMPO' else 'default'
    require(result['attention_backend']==started['attention_backend']==backend,'Attention backend differs')
    required=('started.json','preprocessing.json','trace.jsonl','predictions.csv','validation_predictions.csv','validation_metrics.json','best.pt')
    require(set(result['artifact_sha256'])==set(required),'Artifact inventory differs')
    for filename in required: require(common.file_sha256(path/filename)==result['artifact_sha256'][filename],f'Completed artifact changed: {path/filename}')
    require(read_json(path/'preprocessing.json')==read_json(parent.preproc_dir(fold)/'preprocessing.json'),'Annual preprocessing state differs')
    rows,p=parent.read_predictions(path/'predictions.csv',d,te,fold,carry)
    vrows,pv=parent.read_predictions(path/'validation_predictions.csv',d,va,fold,carry)
    parent.matching_scores(result,parent.safe_scores(d['labels'][te],p));parent.matching_scores(val,parent.safe_scores(d['labels'][va],pv))
    require(result['diagnostics']==parent.diagnostics(d,te,p,carry),'Diagnostic strata or carry comparison differs')
    trace=[json.loads(line) for line in (path/'trace.jsonl').read_text().splitlines() if line]
    cfg=protocol['config'];steps=result['optimizer_steps']
    require(0<steps<=cfg['max_optimizer_steps'] and steps%cfg['validate_every_steps']==0,'Invalid optimizer-step budget')
    require(len(trace)==result['validation_checks'] and [row['optimizer_steps'] for row in trace]==
        list(range(cfg['validate_every_steps'],steps+1,cfg['validate_every_steps'])),'Validation interval differs')
    require(result['best_step'] in [row['optimizer_steps'] for row in trace] and
        abs(val['ap']-result['best_validation_ap'])<=1e-10,'Best checkpoint differs')
    require(trace[-1]['training_examples_seen']==result['training_examples_seen'] and
        trace[-1]['positive_training_exposures']==result['positive_training_exposures'],'Exposure counts differ')
    batches=math.ceil(len(full)/cfg['batch_size']);cycles,remaining=divmod(steps,batches)
    expected_examples=cycles*len(full)+min(remaining*cfg['batch_size'],len(full))
    require(result['training_examples_seen']==expected_examples,'Actual batch/tail exposure accounting differs')
    return result,val,rows,vrows


def train(name,fold,device='cuda:1',check=False,margin_bytes=1073741824,plan_sha256=None):
    require(name in MODELS and fold in range(5),'Invalid baseline condition')
    protocol=load_protocol()
    if not check: require_checks(protocol)
    with locked(CACHE/'locks'/f'{name}_{fold}.lock'):
        with attention_context(name):
            return train_locked(name,fold,device,check,margin_bytes,plan_sha256,protocol)


def train_locked(name,fold,device,check,margin_bytes,plan_sha256,protocol):
    import torch
    from torch.utils.data import DataLoader,TensorDataset,Subset
    from sklearn.metrics import average_precision_score
    annual_protocol=parent.load_protocol();d=annual.load_data();carry=annual.load_carry_baseline()
    full,va,te=parent.partition_indices(d,fold);identity=fit_identity(protocol,d,name,fold)
    path=result_dir(name,fold)
    if not check and (path/'metrics.json').exists():
        audit_fit(path,d,protocol,name,fold,carry)
        common.log('annual_baseline_skip_verified',model=name,fold=fold)
        return
    if not check: require(not path.exists() or not any(path.iterdir()),f'Incomplete fit preserved; explicitly archive before retry: {path}')
    cfg=protocol['config'];torch.set_num_threads(cfg['torch_threads']);common.seed_all(cfg['seed'])
    capacity=None if check else check_capacity(protocol,name,device,margin_bytes)
    require(str(device).startswith('cuda'),'Engineering profiles and formal runs must use CUDA')
    torch.cuda.set_device(torch.device(device))
    if check:
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(torch.device(device))
    free_before,total_bytes=torch.cuda.mem_get_info(torch.device(device))
    x,state=parent.load_preprocessing(fold,annual_protocol)
    embeddings=annual.load_embeddings()
    model=common.build_model(name,x.shape[-1],embeddings,cfg,device)
    parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    positive=int(d['labels'][full].sum());weight=float(len(full)-positive)/positive
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight,device=device))
    dataset=TensorDataset(torch.from_numpy(np.array(x,copy=True)),torch.from_numpy(d['doc_ids'].astype('int64')),
        torch.from_numpy(d['valid_time'].astype(bool)),torch.from_numpy(d['labels'].astype('float32')))
    train_indices=full
    if check:
        complete=d['valid_time'].all(axis=1)
        positive_rows=full[(d['labels'][full]==1)&complete[full]]
        negative_rows=full[(d['labels'][full]==0)&complete[full]]
        require(len(positive_rows)>=32 and len(negative_rows)>=32,'Full-length profile requires 32 positive and 32 negative training windows')
        train_indices=np.concatenate((positive_rows[:32],negative_rows[:32]))
    def loader(indices,training=False):
        batch=cfg['graph_microbatch_size'] if name=='TimeFilter' and not training else cfg['batch_size']
        return DataLoader(Subset(dataset,list(map(int,indices))),batch_size=batch,shuffle=training,num_workers=0,
            pin_memory=True,generator=torch.Generator().manual_seed(cfg['seed']))
    def predict(indices):
        model.eval();values=[]
        with torch.inference_mode():
            for xb,db,vb,_ in loader(indices):
                values.append(model(xb.to(device),db.to(device),vb.to(device)).sigmoid().cpu().numpy())
        probabilities=np.concatenate(values).astype(np.float64)
        require(np.isfinite(probabilities).all(),'Nonfinite validation/test probability')
        return probabilities
    def optimize(xb,db,vb,yb):
        optimizer.zero_grad(set_to_none=True);loss_sum=0.
        micro=cfg['graph_microbatch_size'] if name=='TimeFilter' else len(yb)
        for lo in range(0,len(yb),micro):
            sl=slice(lo,min(lo+micro,len(yb)))
            loss=loss_fn(model(xb[sl],db[sl],vb[sl]),yb[sl])
            aux=model.auxiliary_loss()
            if aux is not None: loss=loss+cfg['auxiliary_loss_weight']*aux
            require(torch.isfinite(loss).item(),'Nonfinite baseline loss')
            (loss*len(yb[sl])/len(yb)).backward()
            loss_sum+=float(loss.detach())*len(yb[sl])
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
        if check:
            gradient=model.text_projection.weight.grad
            require(float(norm)>0 and gradient is not None and torch.isfinite(gradient).all().item()
                    and torch.count_nonzero(gradient).item()>0,'Missing/nonfinite model or text projection gradient')
        optimizer.step()
        return loss_sum
    backend='SDPBackend.MATH' if name=='SEMPO' else 'default'
    micro=cfg['graph_microbatch_size'] if name=='TimeFilter' else cfg['batch_size']
    if not check:
        path.mkdir(parents=True,exist_ok=True)
        immutable_json(path/'preprocessing.json',state)
        immutable_json(path/'started.json',dict(identity=identity,config=cfg,seed=cfg['seed'],pos_weight=weight,
            split_statistics=split_statistics(d,fold),full_train_n=len(full),validation_n=len(va),test_n=len(te),parameters=parameters,numeric_dimension=x.shape[-1],
            input_dimension=x.shape[-1]+cfg['text_dim'],attention_backend=backend,train_batch_size=64,
            train_microbatch_size=micro,eval_batch_size=micro,capacity_check=capacity,gpu_plan_sha256=plan_sha256,
            device=device,gpu_name=torch.cuda.get_device_name(torch.device(device)),pid=os.getpid(),started_at=utc()))
        with (path/'trace.jsonl').open('x',encoding='utf-8'): pass
    started=time.monotonic();train_loader=loader(train_indices,training=True);iterator=iter(train_loader)
    steps,examples,positive_exposures,checks,stale=0,0,0,0,0
    best_ap,best_step,best_state=-1.,0,None
    interval_loss,interval_examples=0.,0
    model.train()
    while steps<(1 if check else cfg['max_optimizer_steps']):
        try: xb,db,vb,yb=next(iterator)
        except StopIteration:
            iterator=iter(train_loader);xb,db,vb,yb=next(iterator)
        if check: torch.cuda.synchronize(torch.device(device));step_started=time.monotonic()
        xb,db,vb,yb=[v.to(device,non_blocking=True) for v in (xb,db,vb,yb)]
        loss_sum=optimize(xb,db,vb,yb)
        steps+=1;examples+=len(yb);positive_exposures+=int(yb.sum().item())
        interval_loss+=loss_sum;interval_examples+=len(yb)
        if check:
            torch.cuda.synchronize(torch.device(device));step_seconds=time.monotonic()-step_started
            train_allocated=int(torch.cuda.max_memory_allocated(torch.device(device)))
            train_reserved=int(torch.cuda.max_memory_reserved(torch.device(device)))
            torch.cuda.reset_peak_memory_stats(torch.device(device))
            validation_probabilities=predict(va[:64])
            torch.cuda.synchronize(torch.device(device))
            eval_allocated=int(torch.cuda.max_memory_allocated(torch.device(device)))
            eval_reserved=int(torch.cuda.max_memory_reserved(torch.device(device)))
            require(examples==64 and positive_exposures==32 and steps==1 and len(validation_probabilities)==64,
                    'Engineering check must execute one real batch64 update and 64 validation examples')
            with torch.inference_mode():
                one=model(xb[:1],db[:1],vb[:1]);batch=model(xb[:4],db[:4],vb[:4])[:1]
            require(torch.allclose(one,batch,atol=5e-5,rtol=5e-5),'Evaluation prediction depends on other batch members')
            record=dict(passed=True,model=name,fold=fold,identity=identity,device=device,
                gpu_name=torch.cuda.get_device_name(torch.device(device)),train_examples=64,train_positives=32,
                validation_examples=64,optimizer_steps=1,parameters=parameters,input_dimension=x.shape[-1]+cfg['text_dim'],
                train_microbatch_size=micro,eval_batch_size=micro,attention_backend=backend,auxiliary_loss_weight=cfg['auxiliary_loss_weight'],
                train_peak_allocated_bytes=train_allocated,train_peak_reserved_bytes=train_reserved,
                eval_peak_allocated_bytes=eval_allocated,eval_peak_reserved_bytes=eval_reserved,
                required_peak_reserved_bytes=max(train_reserved,eval_reserved),free_before_bytes=int(free_before),
                total_device_bytes=int(total_bytes),train_step_seconds=step_seconds,check_seconds=time.monotonic()-started,
                full_length_training_windows=True,batch_invariance=True,checked_at=utc())
            record_check(protocol,name,record)
            common.log('annual_baseline_check_pass',**{k:v for k,v in record.items() if k!='identity'})
            return record
        if steps%cfg['validate_every_steps']!=0: continue
        pv=predict(va);ap=float(average_precision_score(d['labels'][va],pv));checks+=1
        if ap>best_ap:
            best_ap,best_step,stale=ap,steps,0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            temporary=path/'best.pt.tmp'
            torch.save(dict(state_dict=best_state,identity=identity,config=cfg,optimizer_steps=steps,
                validation_ap=ap,pos_weight=weight,parameters=parameters,attention_backend=backend),temporary)
            os.replace(temporary,path/'best.pt')
        else: stale+=1
        row=dict(validation_check=checks,optimizer_steps=steps,training_examples_seen=examples,
            positive_training_exposures=positive_exposures,epochs_equivalent=examples/len(full),completed_epochs=examples//len(full),
            train_loss=interval_loss/interval_examples,validation_ap=ap,best_validation_ap=best_ap,best_step=best_step,nonimproving_checks=stale)
        with (path/'trace.jsonl').open('a',encoding='utf-8') as stream: stream.write(json.dumps(row,allow_nan=False)+'\n')
        common.log('annual_baseline_validation',model=name,scheme='full',fold=fold,**row)
        interval_loss,interval_examples=0.,0
        model.train()
        if stale>=cfg['patience_checks']: break
    require(best_state is not None,'No validation-selected checkpoint')
    model.load_state_dict(best_state);pv,p=predict(va),predict(te)
    require(abs(float(average_precision_score(d['labels'][va],pv))-best_ap)<=1e-10,'Restored checkpoint validation AP differs')
    result=parent.safe_scores(d['labels'][te],p);validation=parent.safe_scores(d['labels'][va],pv)
    result.update(identity=identity,model=name,scheme='full',fold=fold,pos_weight=weight,parameters=parameters,
        split_statistics=split_statistics(d,fold),
        optimizer_steps=steps,training_examples_seen=examples,positive_training_exposures=positive_exposures,
        epochs_equivalent=examples/len(full),completed_epochs=examples//len(full),validation_checks=checks,
        best_step=best_step,best_validation_ap=best_ap,attention_backend=backend,
        stop_reason='patience_checks' if stale>=cfg['patience_checks'] else 'max_optimizer_steps',
        seconds=round(time.monotonic()-started,2),diagnostics=parent.diagnostics(d,te,p,carry))
    validation.update(identity=identity,best_step=best_step,checkpoint_selection='maximum validation AP')
    parent.write_predictions(path/'predictions.csv',d,te,p,fold,carry)
    parent.write_predictions(path/'validation_predictions.csv',d,va,pv,fold,carry)
    immutable_json(path/'validation_metrics.json',validation)
    result['artifact_sha256']={n:common.file_sha256(path/n) for n in ('started.json','preprocessing.json','trace.jsonl',
        'predictions.csv','validation_predictions.csv','validation_metrics.json','best.pt')}
    immutable_json(path/'metrics.json',result)
    audit_fit(path,d,protocol,name,fold,carry)
    common.log('annual_baseline_fit_complete',model=name,fold=fold,ap=result['ap'],f1=result['f1'],updates=steps,seconds=result['seconds'])
    return result


def aggregate(d,carry,name,fits,vals,rows,vrows,source_experiment,source_fingerprint,persist=False):
    expected=list(range(len(d['labels'])))
    for selected in (rows,vrows):
        require(sorted(int(row['sample_index']) for row in selected)==expected,'OOF rows must cover each annual company-year once')
        selected.sort(key=lambda row:int(row['sample_index']))
    require(len(rows)==11199 and len(fits)==5 and int(d['labels'].sum())==806,'Confirmed annual OOF cohort differs')
    p=np.asarray([float(row['probability']) for row in rows]);pv=np.asarray([float(row['probability']) for row in vrows])
    result=dict(model=name,scheme='full',completed_folds=5,source_experiment=source_experiment,
        source_protocol_fingerprint=source_fingerprint,fit_fingerprints=[v['identity']['fingerprint'] for v in fits],
        pooled_oof=parent.safe_scores(d['labels'],p),diagnostics=parent.diagnostics(d,np.arange(len(p)),p,carry))
    validation=dict(model=name,scheme='full',completed_folds=5,source_experiment=source_experiment,
        source_protocol_fingerprint=source_fingerprint,pooled_validation=parent.safe_scores(d['labels'],pv))
    table=dict(model=name,scheme='full',completed_folds=5,source_experiment=source_experiment,source_protocol_fingerprint=source_fingerprint)
    for metric in METRICS:
        for output,records,prefix in ((result,fits,'test'),(validation,vals,'validation')):
            values=[r[metric] for r in records]
            mean=float(np.mean(values)) if all(v is not None for v in values) else None
            sd=float(np.std(values,ddof=1)) if all(v is not None for v in values) else None
            output[metric+'_mean']=mean;output[metric+'_std']=sd
            table[prefix+'_'+metric+'_mean']=mean;table[prefix+'_'+metric+'_std']=sd
        table['oof_'+metric]=result['pooled_oof'][metric]
    for key in ('optimizer_steps','training_examples_seen','positive_training_exposures','epochs_equivalent','parameters'):
        values=[r[key] for r in fits]
        result[key+'_mean']=float(np.mean(values)) if all(v is not None for v in values) else None
        table[key+'_mean']=result[key+'_mean']
    covered=result['diagnostics']['carry_common_covered'];strata=result['diagnostics']['transition_diagnostics_ex_post']
    table.update(oof_n=len(rows),oof_positive=int(d['labels'].sum()),carry_covered_n=covered['covered_n'],carry_coverage=covered['coverage'],
        common_subset_model_accuracy=covered['model']['accuracy'],carry_accuracy=covered['carry']['accuracy'],
        common_subset_model_ap=covered['model']['ap'],carry_ap=covered['carry']['ap'])
    for short,kind,metric in (('onset','risk_onset','recall'),('persistent','persistent_risk','recall'),
                            ('recovery','recovery','specificity'),('stable_nonrisk','stable_nonrisk','specificity')):
        table[short+'_n']=strata[kind]['n'];table[short+'_'+metric]=strata[kind][metric]
    table['transition_unknown_n']=strata['unknown']['n']
    if persist:
        parent.write_table(RESULTS/'full'/name/'oof_predictions.csv',rows)
        parent.write_table(RESULTS/'full'/name/'validation_oof_predictions.csv',vrows)
    return result,validation,table


def collect_references(d,annual_protocol,carry):
    saved=read_json(parent.RESULTS/'summary.json')
    results,validations,tables=[],[],[]
    for name in REFERENCE_MODELS:
        fits,vals,rows,vrows=[],[],[],[]
        for fold in range(5):
            path=parent.result_dir(name,'full',fold)
            fit,val,rr,vv=parent.audit_fit(path,d,annual_protocol,name,'full',fold,carry)
            fits.append(fit);vals.append(val);rows.extend(rr);vrows.extend(vv)
        row,valrow,table=aggregate(d,carry,name,fits,vals,rows,vrows,'annual_risk_v1',annual_protocol['fingerprint'])
        previous=[r for r in saved['models'] if r['model']==name and r['scheme']=='full']
        require(len(previous)==1,'Missing or duplicate preserved full-pool condition')
        for metric in METRICS:
            for suffix in ('_mean','_std'):
                old,new=previous[0][metric+suffix],row[metric+suffix]
                require(old==new or (old is not None and new is not None and abs(old-new)<=1e-10),'Preserved reference summary differs')
        parent.matching_scores(previous[0]['pooled_oof'],row['pooled_oof'])
        results.append(row);validations.append(valrow);tables.append(table)
    return results,validations,tables


def summary(final=False):
    with locked(CACHE/'summary.lock',blocking=True):
        protocol=load_protocol();annual_protocol=parent.load_protocol();d=annual.load_data();carry=annual.load_carry_baseline()
        preserved_unchanged(full=final)
        if final:
            for fold in range(5): parent.load_preprocessing(fold,annual_protocol,full_verify=True)
        complete_fits,conditions,validations,tables=[],[],[],[]
        for name in MODELS:
            fits,vals,rows,vrows=[],[],[],[]
            for fold in range(5):
                path=result_dir(name,fold)
                if not (path/'metrics.json').exists(): continue
                fit,val,rr,vv=audit_fit(path,d,protocol,name,fold,carry)
                fits.append(fit);vals.append(val);rows.extend(rr);vrows.extend(vv);complete_fits.append(fit)
            if len(fits)!=5: continue
            row,valrow,table=aggregate(d,carry,name,fits,vals,rows,vrows,EXPERIMENT,protocol['fingerprint'],persist=True)
            conditions.append(row);validations.append(valrow);tables.append(table)
        references,reference_validation,reference_tables=collect_references(d,annual_protocol,carry)
        if final: require(len(complete_fits)==70 and len(conditions)==14,'Final audit requires all seventy new baseline fits')
        report=dict(experiment=EXPERIMENT,protocol_fingerprint=protocol['fingerprint'],
            parent_protocol_fingerprint=protocol['parent_protocol_fingerprint'],completed_fits=len(complete_fits),expected_fits=70,
            models=conditions,validation_models=validations,folds=complete_fits,reference_models=references,split_statistics={str(f):split_statistics(d,f) for f in range(5)},
            reference_validation_models=reference_validation,reused_reference_fits=20,
            completed_comparison_conditions=len(conditions)+4,expected_comparison_conditions=18,final_audit=bool(final),
            selection=DESIGN['selection'],std_definition='sample SD across five company folds (ddof=1); not a confidence interval',
            checkpoint_selection='validation AP only; validation summaries are separate from held-out OOF test',
            adaptations='existing small from-scratch classification adaptations; not foundation-model pretraining reproduction',
            budget='same maximum optimizer updates and validation schedule; actual early stopping/exposure reported',
            carry_note='strict carry coverage is five negative rows; no valid risk-discrimination claim from this subset')
        RESULTS.mkdir(parents=True,exist_ok=True)
        common.atomic_json(RESULTS/'summary.json',report);parent.write_table(RESULTS/'comparison.csv',tables+reference_tables)
        parent.write_table(RESULTS/'validation_summary.csv',[{k:v for k,v in r.items() if k!='pooled_validation'}
            for r in validations+reference_validation])
        common.log('annual_baselines_summary',completed_fits=len(complete_fits),comparison_conditions=len(conditions)+4,final_audit=final)
        return report


def load_plan(path,protocol):
    plan=read_json(path)
    require(plan.get('protocol_fingerprint')==protocol['fingerprint'],'GPU plan belongs to another experiment')
    require(set(plan.get('workers',{}))=={'0','1'},'GPU plan must explicitly list workers 0 and 1')
    names=plan['workers']['0']+plan['workers']['1']
    require(len(names)==len(MODELS) and set(names)==set(MODELS),'GPU assignments must cover all fourteen models exactly once')
    margin=plan.get('reserve_margin_bytes',1073741824)
    require(isinstance(margin,int) and margin>=1073741824,'GPU plan must reserve at least one GiB headroom')
    return plan


def worker(worker_id,plan_path):
    require(worker_id in (0,1),'GPU worker must be zero or one')
    protocol=load_protocol();require_checks(protocol);plan=load_plan(plan_path,protocol)
    plan_hash=common.file_sha256(plan_path);margin=plan.get('reserve_margin_bytes',1073741824)
    with locked(CACHE/f'worker_{worker_id}.lock'):
        state_path=CACHE/f'worker_{worker_id}_status.json';(ROOT/'logs').mkdir(exist_ok=True)
        with (ROOT/'logs'/f'annual_baselines_gpu{worker_id}.log').open('a',encoding='utf-8') as stream:
            for name in plan['workers'][str(worker_id)]:
                for fold in range(5):
                    while True:
                        require(common.file_sha256(plan_path)==plan_hash,'Active GPU assignment changed during execution')
                        state=dict(status='running',worker=worker_id,pid=os.getpid(),protocol_fingerprint=protocol['fingerprint'],
                                   model=name,fold=fold,plan_sha256=plan_hash,updated_at=utc())
                        common.atomic_json(state_path,state)
                        command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'train','--model',name,'--fold',str(fold),
                            '--device',f'cuda:{worker_id}','--margin-bytes',str(margin),'--plan-sha256',plan_hash]
                        child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                        state['child_pid']=child.pid;common.atomic_json(state_path,state);code=child.wait()
                        if code==75:
                            state.update(status='waiting_resources',updated_at=utc(),returncode=code)
                            common.atomic_json(state_path,state);time.sleep(30);continue
                        if code:
                            state.update(status='failed',updated_at=utc(),returncode=code);common.atomic_json(state_path,state)
                            raise RuntimeError(f'{name} fold {fold} exited {code}')
                        break
        common.atomic_json(state_path,dict(status='complete',worker=worker_id,pid=os.getpid(),
            protocol_fingerprint=protocol['fingerprint'],plan_sha256=plan_hash,updated_at=utc()))


def pipeline(plan_path):
    with locked(CACHE/'pipeline.lock'):
        children=[];status_path=CACHE/'pipeline_status.json'
        try:
            protocol=load_protocol();require_checks(protocol);preserved_unchanged(full=True)
            plan=load_plan(plan_path,protocol)
            active=CACHE/'active_gpu_plan.json';common.atomic_json(active,plan)
            status=dict(status='starting',pid=os.getpid(),protocol_fingerprint=protocol['fingerprint'],expected_fits=70,
                plan_sha256=common.file_sha256(active),started_at=utc())
            common.atomic_json(status_path,status)
            for worker_id in (0,1):
                children.append(subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),'worker',
                    '--gpu',str(worker_id),'--plan',str(active)],cwd=ROOT,start_new_session=True))
            last=0.
            while True:
                codes=[child.poll() for child in children];now=time.monotonic()
                if now-last>=10 or all(code is not None for code in codes):
                    status.update(status='running',updated_at=utc(),completed_fits=sum(1 for _ in RESULTS.glob('full/*/fold_*/metrics.json')),
                        workers=[dict(worker=i,pid=c.pid,running=codes[i] is None,returncode=codes[i]) for i,c in enumerate(children)])
                    common.atomic_json(status_path,status);last=now
                if any(code is not None and code!=0 for code in codes): raise RuntimeError(f'Annual baseline workers failed: {codes}')
                if all(code==0 for code in codes): break
                time.sleep(2)
            report=summary(final=True)
            require(report['completed_comparison_conditions']==18,'Final comparison requires eighteen conditions')
            status.update(status='complete',completed_fits=70,completed_comparison_conditions=18,finished_at=utc())
            common.atomic_json(status_path,status)
        except BaseException:
            parent.stop_workers(children)
            status=read_json(status_path) if status_path.exists() else {}
            status.update(status='failed',failed_at=utc(),pid=os.getpid(),error=traceback.format_exc(),
                workers=[dict(worker=i,pid=c.pid,running=c.poll() is None,returncode=c.poll()) for i,c in enumerate(children)])
            common.atomic_json(status_path,status)
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','check','train','worker','pipeline','summary','audit'))
    parser.add_argument('--model',choices=MODELS)
    parser.add_argument('--fold',type=int,choices=range(5),default=0)
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--gpu','--worker',dest='gpu',type=int,choices=(0,1))
    parser.add_argument('--plan',type=Path,default=CACHE/'gpu_plan.json')
    parser.add_argument('--margin-bytes',type=int,default=1073741824)
    parser.add_argument('--plan-sha256')
    args=parser.parse_args()
    if args.command=='prepare': prepare()
    elif args.command=='check':
        if args.model: train(args.model,args.fold,args.device,check=True)
        else:
            # One fresh process per model: allocator history must not contaminate GPU profiles.
            for name in MODELS:
                subprocess.run([sys.executable,'-B','-u',str(Path(__file__).resolve()),'check','--model',name,
                    '--fold',str(args.fold),'--device',args.device],cwd=ROOT,check=True)
    elif args.command=='train':
        if args.model is None: parser.error('train requires --model')
        if args.margin_bytes<1073741824: parser.error('reserve at least one GiB headroom')
        train(args.model,args.fold,args.device,margin_bytes=args.margin_bytes,plan_sha256=args.plan_sha256)
    elif args.command=='worker':
        if args.gpu is None: parser.error('worker requires --gpu')
        worker(args.gpu,args.plan)
    elif args.command=='pipeline': pipeline(args.plan)
    elif args.command=='summary': summary()
    elif args.command=='audit': summary(final=True)


if __name__=='__main__':
    try: main()
    except parent.AlreadyRunning as exc:
        common.log('annual_baselines_duplicate_launch_rejected',reason=str(exc),pid=os.getpid());sys.exit(3)
    except ResourceBusy as exc:
        common.log('annual_baselines_waiting_resources',reason=str(exc),pid=os.getpid());sys.exit(75)
    except Exception:
        failure=dict(argv=sys.argv,pid=os.getpid(),failed_at=utc(),error=traceback.format_exc())
        target=CACHE/'failures'/f'{time.time_ns()}_{os.getpid()}.json'
        common.atomic_json(target,failure);common.log('annual_baselines_failed',failure=str(target),error=failure['error'])
        raise
