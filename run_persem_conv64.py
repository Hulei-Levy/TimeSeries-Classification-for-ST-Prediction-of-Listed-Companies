"""Matched small-convolution PerSeM with and without frozen-history residuals.

Ten new fits only. Original sources, caches, labels and results stay unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile
import numpy as np
import run as common
import run_annual_risk as parent
import run_annual_baselines as baseline
import data_annual_risk as annual
import data_linear_history as history
import run_linear_history as prior_history
import run_residual_mlp16 as prior_small

ROOT=Path(__file__).resolve().parent
EXPERIMENT='persem_conv64_v1'
CACHE=ROOT/'cache'/EXPERIMENT
RESULTS=ROOT/'results'/EXPERIMENT
MODELS=('PerSeM_Conv64','PerSeM_Conv64_R16')
SOURCES=('run_persem_conv64.py','run_residual_mlp16.py','run_residual_mlp.py','run_linear_history.py','data_linear_history.py','run_annual_risk.py','run_annual_baselines.py',
         'data_annual_risk.py','run.py','run_persem.py','models_persem.py','data.py','config.json')
require=parent.require
read_json=parent.read_json
digest=parent.digest
immutable_json=parent.immutable_json
locked=parent.locked
utc=parent.utc
DESIGN=dict(experiment=EXPERIMENT,version=1,models=list(MODELS),expected_fits=10,reused_reference_fits=10,
    task='June30 Y predicts annual risk Y; unchanged annual_risk_v1 cohort, full company-fold training',
    historical_periods='H=Y-10..Y-6, X=Y-5..Y-1; historical-period reference, not a claim of a forecast issued before X',
    reconstruction='54 numeric variables; finance Linear60to20, governance Linear15to5; values/masks/report ages',
    predictor='three inner company-group cross-fitted masked ridge predictors; lambda=0.0001; no classifier gradient',
    classification='both models use two period blocks with Inception64to64 GELU Inception64to64; absolute64/readout/head unchanged. R16 adds Linear2160to16 and Linear80to32 GELU Linear32to1 signed correction',
    initialization='from scratch seed42; construct original PerSeM, replace both conv sequences before optimizer, then optionally add R16. Both light backbones have identical initial parameters; subsequent dropout RNG may differ; no checkpoint warm start',
    parameter_count={'PerSeM_Conv64':795713,'PerSeM_Conv64_R16':832914},
    capacity_change='only Inception intermediate width128to64; main width64, depth2, kernels1/3/5, attention and residual16 unchanged',
    gpu_plan={'0':'PerSeM_Conv64','1':'PerSeM_Conv64_R16'},
    inference='plain model uses X only in one pass; R16 train uses own excluded inner model, validation/test mean of three probabilities from same residual cache',
    extra_missing='whole extra projection AND full MLP correction including bias zero when unavailable; current base logit retained',
    selection='unchanged validation AP, 10500 max updates / 105 validation interval / 15 stale checks, threshold0.5',
    scope='matched convolution-width comparison with and without R16; no gate, residual rescaling, new temporal encoder, generator refit or training-protocol changes; original files immutable')

class ResourceBusy(RuntimeError): pass

def preserved_unchanged(full=False):
    saved=read_json(CACHE/'preserved_identity.json')
    for name,record in saved['files'].items():
        if not full and name not in SOURCES: continue
        path=ROOT/name
        require(path.is_file() and path.stat().st_size==record['size'],'Preserved file changed: '+name)
        require(common.file_sha256(path)==record['sha256'],'Preserved bytes changed: '+name)
    return saved

def data_files():
    return prior_history.data_files()

def source_identity():
    preserved_unchanged()
    prior=parent.load_protocol();previous=prior_history.load_protocol();capacity=prior_small.load_protocol()
    payload=dict(design=DESIGN,capacity_reference_protocol_fingerprint=capacity['fingerprint'],
        capacity_reference_protocol_sha256=common.file_sha256(prior_small.CACHE/'protocol.json'),
        reference_protocol_fingerprint=previous['fingerprint'],
        reference_protocol_sha256=common.file_sha256(prior_history.CACHE/'protocol.json'),parent_protocol_fingerprint=prior['fingerprint'],
        parent_protocol_sha256=common.file_sha256(parent.CACHE/'training_protocol.json'),
        config=prior['effective_config']['neural'],preprocessing=prior['preprocessing'],
        source_sha256={n:common.file_sha256(ROOT/n) for n in SOURCES},
        history_artifacts=data_files(),data_identity=prior['data_identity'],
        runtime_environment=read_json(CACHE/'runtime_environment.json'),
        preserved_identity_sha256=common.file_sha256(CACHE/'preserved_identity.json'))
    return dict(fingerprint=digest(payload),**payload)

def prepare():
    with locked(CACHE/'prepare.lock'):
        preserved_unchanged(full=True)
        prior=parent.load_protocol();d=annual.load_data()
        require(len(d['labels'])==11199 and int(d['labels'].sum())==806,'Annual cohort changed')
        prior_history.load_protocol()
        for fold in range(5):
            arrays,meta=history.load_fold(fold)
            validate_features(arrays,d,fold)
            parent.load_preprocessing(fold,prior,full_verify=True)
        immutable_json(CACHE/'runtime_environment.json',parent.runtime_environment())
        identity=source_identity()
        immutable_json(CACHE/'protocol.json',identity)
        names=list(SOURCES)+[str((CACHE/n).relative_to(ROOT)) for n in
            ('protocol.json','runtime_environment.json','preserved_identity.json')]
        snapshot=CACHE/'source_snapshot.zip'
        if snapshot.exists():
            with zipfile.ZipFile(snapshot) as archive:
                require(set(archive.namelist())==set(names),'Source inventory changed')
                for name in names: require(archive.read(name)==(ROOT/name).read_bytes(),'Snapshot changed: '+name)
        else:
            with zipfile.ZipFile(snapshot,'x',zipfile.ZIP_DEFLATED) as archive:
                for name in names: archive.write(ROOT/name,name)
        common.log('persem_conv64_prepared',fingerprint=identity['fingerprint'],expected_fits=10)
        return identity

def load_protocol():
    saved=read_json(CACHE/'protocol.json')
    require(saved==source_identity(),'Frozen conv64 protocol/source/data changed')
    actual=parent.runtime_environment()
    for key in ('python','package_versions','torch_cuda_version','cudnn_version'):
        require(actual[key]==saved['runtime_environment'][key],'Runtime changed: '+key)
    with zipfile.ZipFile(CACHE/'source_snapshot.zip') as archive:
        for name in archive.namelist(): require(archive.read(name)==(ROOT/name).read_bytes(),'Snapshot changed: '+name)
    return saved

validate_features=prior_history.validate_features

def result_dir(model,fold): return RESULTS/'full'/model/f'fold_{fold}'

def fit_identity(protocol,d,model,fold):
    tr,va,te=parent.partition_indices(d,fold)
    payload=dict(experiment=EXPERIMENT,model=model,scheme='full',fold=fold,
        protocol_fingerprint=protocol['fingerprint'],
        preprocessing=protocol['preprocessing'][str(fold)],
        train_indices_sha256=parent.index_hash(tr),validation_indices_sha256=parent.index_hash(va),
        test_indices_sha256=parent.index_hash(te),
        history_metadata_sha256=protocol['history_artifacts'][str((history.CACHE/f'fold_{fold}'/'metadata.json').relative_to(ROOT))])
    return dict(fingerprint=digest(payload),**payload)

def make_model(x,state,embeddings,cfg,device,model_name):
    import torch
    from torch import nn
    from models_persem import PerSeMClassifier, _Inception
    require(model_name in MODELS,'Unknown light architecture')
    class SmallConvClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.base=PerSeMClassifier(x.shape[-1],state['modality_layout'],embeddings,
                variant='PerSeM',dim=cfg['dim'],depth=cfg['depth'],dropout=cfg['dropout'],text_dim=cfg['text_dim'])
            require(cfg['dim']==64 and cfg['depth']==2,'Unexpected base interface')
            for block in self.base.absolute_encoder.blocks:
                block.conv=nn.Sequential(_Inception(64,64),nn.GELU(),_Inception(64,64))
            self.has_residual=model_name=='PerSeM_Conv64_R16'
            if self.has_residual:
                self.extra_projection=nn.Linear(2160,16)
                self.correction=nn.Sequential(nn.Linear(80,32),nn.GELU(),nn.Linear(32,1))
            original=self.base.head[1]
            self.register_buffer('initial_base_weight',original.weight.detach().clone())
            self.register_buffer('initial_base_bias',original.bias.detach().clone())
        def forward(self,numeric,doc_ids,valid_time,extra,available):
            absolute=self.base.head[0](self.base.forward_features(numeric,doc_ids,valid_time)['absolute'])
            logit=self.base.head[1](absolute)
            if self.has_residual:
                mask=available.to(extra.dtype).unsqueeze(-1)
                side=self.extra_projection(extra)*mask
                logit=logit+self.correction(torch.cat((absolute,side),-1))*mask
            return logit.squeeze(-1)
        def auxiliary_loss(self): return 0.
    return SmallConvClassifier().to(device)

history_strata=prior_history.history_strata

def record_check(protocol,model,value):
    with locked(CACHE/'checks.lock',blocking=True):
        path=CACHE/'engineering_checks.json'
        saved=read_json(path) if path.exists() else dict(fingerprint=protocol['fingerprint'],checks={})
        require(saved['fingerprint']==protocol['fingerprint'],'Checks refer to another source')
        saved['checks'][model]=value
        common.atomic_json(path,saved)

def require_checks(protocol):
    record=read_json(CACHE/'engineering_checks.json')
    require(record['fingerprint']==protocol['fingerprint'] and set(record['checks'])==set(MODELS),'Incomplete engineering checks')
    require(all(v.get('passed') for v in record['checks'].values()),'Engineering check failed')
    require(len({v['initial_backbone_sha256'] for v in record['checks'].values()})==1,'Light models have different backbone initialization')
    return record

def resource_check(model,device,protocol):
    import torch
    if not str(device).startswith('cuda'): return
    profile=require_checks(protocol)['checks'][model]
    torch.cuda.set_device(device)
    free,total=torch.cuda.mem_get_info(device)
    needed=int(profile['cuda_peak_reserved_bytes'])+256*1024*1024
    if free<needed: raise ResourceBusy(f'{device}: free={free}, required={needed}; batch unchanged')

def train(model_name,fold,device='cuda:1',check=False):
    import torch
    from sklearn.metrics import average_precision_score
    require(model_name in MODELS and fold in range(5),'Unknown model/fold')
    protocol=load_protocol()
    if not check: require_checks(protocol)
    with locked(CACHE/f'{model_name}_{fold}.lock'):
        d=annual.load_data();carry=annual.load_carry_baseline();prior=parent.load_protocol()
        a,meta=history.load_fold(fold);feature_check=validate_features(a,d,fold)
        tr,va,te=parent.partition_indices(d,fold)
        identity=fit_identity(protocol,d,model_name,fold);path=result_dir(model_name,fold)
        if not check and (path/'metrics.json').exists():
            audit_fit(path,d,protocol,model_name,fold,carry)
            common.log('persem_conv64_verified_skip',model=model_name,fold=fold);return
        if not check:
            require(not path.exists() or not any(path.iterdir()),'Incomplete result retained: '+str(path))
            resource_check(model_name,device,protocol)
        cfg=protocol['config'];common.seed_all(cfg['seed']);torch.set_num_threads(cfg['torch_threads'])
        if str(device).startswith('cuda'):
            torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
        x,state=parent.load_preprocessing(fold,prior)
        model=make_model(x,state,annual.load_embeddings(),cfg,device,model_name)
        initial_backbone_sha256=None
        if check:
            initial_backbone_sha256=digest({n:hashlib.sha256(w.detach().cpu().numpy().tobytes()).hexdigest() for n,w in model.base.named_parameters()})
        parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        weight=float((d['labels'][tr]==0).sum())/float(d['labels'][tr].sum())
        loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight,device=device))
        xt=torch.from_numpy(np.array(x,copy=True));dt=torch.from_numpy(d['doc_ids'].astype('int64'))
        vt=torch.from_numpy(d['valid_time'].astype(bool));yt=torch.from_numpy(d['labels'].astype('float32'))
        ht=torch.from_numpy(a['history_features'].astype('float32',copy=False))
        rt=torch.from_numpy(a['residual_features'].astype('float32',copy=False))
        ha=torch.from_numpy(a['history_available'].astype(bool))
        ra=torch.from_numpy(a['residual_available'].astype(bool))
        assignment=torch.from_numpy(a['inner_assignment'].astype('int64'))
        def batch(indices,branch=None):
            ids=torch.as_tensor(indices,dtype=torch.long)
            if model_name=='PerSeM_Conv64':
                extra=torch.zeros((len(ids),2160),dtype=torch.float32);valid_extra=torch.zeros(len(ids),dtype=torch.bool)
            elif branch is None:
                require((assignment[ids]>=0).all().item(),'Training R rows need cross-fit assignment')
                extra,valid_extra=rt[assignment[ids],ids],ra[assignment[ids],ids]
            else: extra,valid_extra=rt[branch,ids],ra[branch,ids]
            return tuple(v.to(device,non_blocking=True) for v in (xt[ids],dt[ids],vt[ids],extra,valid_extra,yt[ids]))
        def predict(indices):
            model.eval();values=[]
            with torch.inference_mode():
                for start in range(0,len(indices),cfg['batch_size']):
                    selected=indices[start:start+cfg['batch_size']]
                    branches=(None,) if model_name=='PerSeM_Conv64' else range(3)
                    ps=[]
                    for branch in branches:
                        xb,db,vb,eb,ab,_=batch(selected,branch)
                        ps.append(model(xb,db,vb,eb,ab).sigmoid())
                    values.append(torch.stack(ps).mean(0).cpu().numpy())
            p=np.concatenate(values).astype('float64')
            require(np.isfinite(p).all(),'Nonfinite probabilities')
            return p
        training=tr
        if check:
            usable=np.ones(len(tr),dtype=bool) if model_name=='PerSeM_Conv64' else a['residual_available'][a['inner_assignment'][tr],tr].astype(bool)
            candidates=tr[usable];require(len(candidates)>=64,'Need 64 training samples with valid extra branch')
            positive=candidates[d['labels'][candidates]==1][:32]
            negative=candidates[d['labels'][candidates]==0][:64-len(positive)]
            training=np.r_[positive,negative]
            if len(training)<64: training=np.r_[training,candidates[~np.isin(candidates,training)][:64-len(training)]]
            require(len(training)==64,'Engineering batch must match formal batch64')
        from torch.utils.data import DataLoader
        train_loader=DataLoader(training.tolist(),batch_size=cfg['batch_size'],shuffle=True,num_workers=0,
                                generator=torch.Generator().manual_seed(cfg['seed']))
        iterator=iter(train_loader);model.train()
        steps=examples=positive_exposures=checks=stale=0
        interval_loss=0.;interval_examples=0;best_ap=-1.;best_step=0;best_state=None
        start_time=time.monotonic()
        if not check:
            path.mkdir(parents=True,exist_ok=True)
            immutable_json(path/'preprocessing.json',state)
            immutable_json(path/'started.json',dict(identity=identity,config=cfg,pos_weight=weight,
                split_statistics=baseline.split_statistics(d,fold),feature_check=feature_check,
                predictor_metadata_fingerprint=meta['fingerprint'],started_at=utc(),pid=os.getpid(),device=device))
            with (path/'trace.jsonl').open('x',encoding='utf-8'): pass
        while steps<(1 if check else cfg['max_optimizer_steps']):
            try: ids=next(iterator)
            except StopIteration: iterator=iter(train_loader);ids=next(iterator)
            xb,db,vb,eb,ab,yb=batch(ids)
            if check:
                model.eval()
                with torch.no_grad():
                    empty=torch.zeros_like(ab)
                    actual=model(xb,db,vb,eb,empty)
                    absolute=model.base.head[0](model.base.forward_features(xb,db,vb)['absolute'])
                    oracle=torch.nn.functional.linear(absolute,model.initial_base_weight,model.initial_base_bias).squeeze(-1)
                    require(torch.allclose(actual,oracle,atol=5e-5,rtol=5e-5),'Initial no-history fallback differs from original PerSeM')
                    different=model(xb,db,vb,eb+13.,empty)
                    require(torch.equal(actual,different),'Unavailable extra/bias influences prediction')
                    no_history=tr[~a['history_available'][tr].astype(bool)][:8]
                    require(len(no_history)>0,'Engineering check requires actual no-history training rows')
                    nx,nd,nv,ne,na,_=batch(no_history)
                    require(not na.any().item(),'No-H row has available extra branch')
                    no_history_actual=model(nx,nd,nv,ne,na)
                    no_history_abs=model.base.head[0](model.base.forward_features(nx,nd,nv)['absolute'])
                    no_history_expected=torch.nn.functional.linear(no_history_abs,model.initial_base_weight,model.initial_base_bias).squeeze(-1)
                    require(torch.allclose(no_history_actual,no_history_expected,atol=5e-5,rtol=5e-5),
                            'Real no-history rows do not recover original PerSeM')
                model.train()
            optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(model(xb,db,vb,eb,ab),yb)
            require(torch.isfinite(loss).item(),'Nonfinite classification loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
            if check:
                gradient_parts=[('text',model.base.text_projection.weight),('absolute_head',model.base.head[1].weight)]
                gradient_parts.extend((f'conv_{i}',b.conv[0].convs[0].weight) for i,b in enumerate(model.base.absolute_encoder.blocks))
                if model.has_residual:
                    gradient_parts.extend([('extra',model.extra_projection.weight),('MLP_input',model.correction[0].weight),('MLP_output',model.correction[2].weight)])
                for label,p in gradient_parts:
                    require(p.grad is not None and torch.isfinite(p.grad).all().item() and torch.count_nonzero(p.grad).item()>0,
                            'Missing finite nonzero '+label+' gradient')
                require(float(norm)>0,'Zero gradient')
                require(not ht.requires_grad and not rt.requires_grad,'Frozen cached predictor features require gradient')
            optimizer.step();steps+=1;examples+=len(yb);positive_exposures+=int(yb.sum().item())
            interval_loss+=float(loss.detach())*len(yb);interval_examples+=len(yb)
            if check:
                pv=predict(va[:64]);require(len(pv)==64,'Engineering validation length differs')
                with torch.no_grad():
                    first=model(xb[:1],db[:1],vb[:1],eb[:1],ab[:1])
                    together=model(xb,db,vb,eb,ab)[:1]
                    require(torch.allclose(first,together,atol=5e-5,rtol=5e-5),'Batch-dependent inference')
                    masked=model(xb,db,vb,eb,torch.zeros_like(ab))
                    current=model.base.head[1](model.base.head[0](model.base.forward_features(xb,db,vb)['absolute'])).squeeze(-1)
                    require(torch.equal(masked,current),'Post-update unavailable correction changed base logit')
                    interaction=None
                    if model.has_residual:
                        z=torch.zeros(4,80,device=device)
                        z[1,:64]=.25;z[2,64:]=.25;z[3,:]=.25
                        v=model.correction(z).flatten()
                        interaction=float((v[3]-v[1]-v[2]+v[0]).abs())
                        require(interaction>1e-8,'MLP has no nonlinear state-residual interaction')
                    else:
                        perturbed=model(xb,db,vb,eb+17.,torch.ones_like(ab))
                        require(torch.equal(perturbed,current),'Plain model reads historical residual features')
                        require(not hasattr(model,'extra_projection') and not hasattr(model,'correction'),'Plain model has extra parameters')
                    for block in model.base.absolute_encoder.blocks:
                        for layer in (block.conv[0],block.conv[2]):
                            require([c.kernel_size for c in layer.convs]==[(1,1),(3,3),(5,5)],'Inception kernels changed')
                            require(all(c.in_channels==64 and c.out_channels==64 for c in layer.convs),'Conv width differs')
                    require(parameters==DESIGN['parameter_count'][model_name],'Unexpected trainable parameter count')
                value=dict(passed=True,model=model_name,fold=fold,device=device,examples=examples,parameters=parameters,
                    optimizer_steps=steps,feature_contract=feature_check,batch_invariance=True,empty_history_fallback=True,
                    extra_bias_masked=True,full_correction_masked=True,nonlinear_interaction=interaction,
                    initial_backbone_sha256=initial_backbone_sha256,plain_ignores_history=not model.has_residual,conv_width64=True,
                    frozen_predictor_features=True,text_and_extra_gradients=True,
                    cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)) if str(device).startswith('cuda') else 0,
                    checked_at=utc())
                record_check(protocol,model_name,value);common.log('persem_conv64_check',**value);return value
            if steps%cfg['validate_every_steps']: continue
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
                completed_epochs=examples//len(tr),train_loss=interval_loss/interval_examples,
                validation_ap=ap,best_validation_ap=best_ap,best_step=best_step,nonimproving_checks=stale)
            with (path/'trace.jsonl').open('a',encoding='utf-8') as stream: stream.write(json.dumps(row,allow_nan=False)+'\n')
            common.log('persem_conv64_validation',model=model_name,fold=fold,**row)
            common.atomic_json(CACHE/f'{model_name}_{fold}_status.json',dict(status='training',identity=identity,updated_at=utc(),**row))
            model.train();interval_loss=0.;interval_examples=0
            if stale>=cfg['patience_checks']: break
        require(best_state is not None,'No validated checkpoint')
        model.load_state_dict(best_state)
        pv,p=predict(va),predict(te)
        require(abs(float(average_precision_score(d['labels'][va],pv))-best_ap)<1e-10,'Restored validation AP mismatch')
        result=parent.safe_scores(d['labels'][te],p);validation=parent.safe_scores(d['labels'][va],pv)
        accounting=dict(parameters=parameters,optimizer_steps=steps,training_examples_seen=examples,
            positive_training_exposures=positive_exposures,epochs_equivalent=examples/len(tr),completed_epochs=examples//len(tr),
            validation_checks=checks,best_step=best_step,best_validation_ap=best_ap,
            stop_reason='patience_checks' if stale>=cfg['patience_checks'] else 'max_optimizer_steps',
            max_optimizer_steps=cfg['max_optimizer_steps'],validate_every_steps=cfg['validate_every_steps'],patience_checks=cfg['patience_checks'])
        result.update(identity=identity,model=model_name,scheme='full',fold=fold,pos_weight=weight,**accounting,
            seconds=round(time.monotonic()-start_time,2),diagnostics=parent.diagnostics(d,te,p,carry),
            history_strata=history_strata(d,te,p,a['history_available']),
            predictor_training=('frozen masked ridge; separately recorded, not classifier updates' if model.has_residual else 'not used; original X-only classifier'))
        validation.update(identity=identity,best_step=best_step,checkpoint_selection='maximum validation AP; plain one pass; R16 mean of three probabilities')
        parent.write_predictions(path/'predictions.csv',d,te,p,fold,carry)
        parent.write_predictions(path/'validation_predictions.csv',d,va,pv,fold,carry)
        immutable_json(path/'validation_metrics.json',validation)
        result['artifact_sha256']={n:common.file_sha256(path/n) for n in
            ('started.json','preprocessing.json','trace.jsonl','best.pt','predictions.csv','validation_predictions.csv','validation_metrics.json')}
        immutable_json(path/'metrics.json',result)
        audit_fit(path,d,protocol,model_name,fold,carry)
        common.atomic_json(CACHE/f'{model_name}_{fold}_status.json',dict(status='complete',identity=identity,optimizer_steps=steps,updated_at=utc()))
        common.log('persem_conv64_fit_complete',model=model_name,fold=fold,ap=result['ap'],f1=result['f1'],updates=steps,seconds=result['seconds'])
        return result

def audit_fit(path,d,protocol,model,fold,carry):
    identity=fit_identity(protocol,d,model,fold)
    result=read_json(path/'metrics.json');validation=read_json(path/'validation_metrics.json');started=read_json(path/'started.json')
    for r in (result,validation,started): require(r['identity']==identity,'Fit identity mismatch')
    require(started['config']==protocol['config'],'Classifier configuration changed')
    tr,va,te=parent.partition_indices(d,fold);weight=float((d['labels'][tr]==0).sum())/float(d['labels'][tr].sum())
    require(result['pos_weight']==started['pos_weight']==weight,'Class weight changed')
    required={'started.json','preprocessing.json','trace.jsonl','best.pt','predictions.csv','validation_predictions.csv','validation_metrics.json'}
    require(set(result['artifact_sha256'])==required,'Artifact inventory mismatch')
    for name in required: require(common.file_sha256(path/name)==result['artifact_sha256'][name],'Artifact changed: '+name)
    require(read_json(path/'preprocessing.json')==read_json(parent.preproc_dir(fold)/'preprocessing.json'),'X preprocessing changed')
    rows,p=parent.read_predictions(path/'predictions.csv',d,te,fold,carry)
    vrows,pv=parent.read_predictions(path/'validation_predictions.csv',d,va,fold,carry)
    parent.matching_scores(result,parent.safe_scores(d['labels'][te],p));parent.matching_scores(validation,parent.safe_scores(d['labels'][va],pv))
    require(result['diagnostics']==parent.diagnostics(d,te,p,carry),'Diagnostics mismatch')
    a,_=history.load_fold(fold)
    require(result['history_strata']==history_strata(d,te,p,a['history_available']),'History strata mismatch')
    cfg=protocol['config'];trace=[json.loads(line) for line in (path/'trace.jsonl').read_text().splitlines()]
    require(0<result['optimizer_steps']<=cfg['max_optimizer_steps'],'Optimizer budget mismatch')
    require([r['optimizer_steps'] for r in trace]==list(range(cfg['validate_every_steps'],result['optimizer_steps']+1,cfg['validate_every_steps'])),
            'Validation interval mismatch')
    require(len(trace)==result['validation_checks'] and result['optimizer_steps']%cfg['validate_every_steps']==0,'Invalid update accounting')
    require(result['best_step'] in [r['optimizer_steps'] for r in trace] and abs(validation['ap']-result['best_validation_ap'])<1e-10,
            'Checkpoint selection mismatch')
    return result,validation,rows,vrows

def summary(final=False):
    with locked(CACHE/'summary.lock',blocking=True):
        protocol=load_protocol();prior=parent.load_protocol();previous=prior_history.load_protocol();capacity=prior_small.load_protocol();d=annual.load_data();carry=annual.load_carry_baseline()
        preserved_unchanged(full=final)
        completed=[];conditions=[];validations=[];tables=[]
        history_available=history.load_fold(0)[0]['history_available']
        for name in ('PerSeM','PerSeM_R_MLP16')+MODELS:
            fits=[];vals=[];rows=[];vrows=[]
            for fold in range(5):
                old=name not in MODELS
                path=parent.result_dir(name,'full',fold) if name=='PerSeM' else (prior_small.result_dir(name,fold) if old else result_dir(name,fold))
                if not (path/'metrics.json').exists(): continue
                if name=='PerSeM': fit,val,rr,vv=parent.audit_fit(path,d,prior,name,'full',fold,carry)
                elif old: fit,val,rr,vv=prior_small.audit_fit(path,d,capacity,name,fold,carry)
                else:
                    fit,val,rr,vv=audit_fit(path,d,protocol,name,fold,carry);completed.append(fit)
                fits.append(fit);vals.append(val);rows.extend(rr);vrows.extend(vv)
            if len(fits)!=5: continue
            old=name not in MODELS
            origin='annual_risk_v1' if name=='PerSeM' else ('residual_mlp16_v1' if old else EXPERIMENT)
            fingerprint=prior['fingerprint'] if name=='PerSeM' else (capacity['fingerprint'] if old else protocol['fingerprint'])
            row,valrow,table=baseline.aggregate(d,carry,name,fits,vals,rows,vrows,
                origin,fingerprint,persist=False)
            rows.sort(key=lambda r:int(r['sample_index']))
            probabilities=np.asarray([float(r['probability']) for r in rows])
            row['history_strata']=history_strata(d,np.arange(len(probabilities)),probabilities,history_available)
            for key,stats in row['history_strata'].items():
                for metric in ('n','positive','ap','precision','recall','f1','roc_auc'):
                    table[key+'_'+metric]=stats[metric]
            if not old:
                parent.write_table(RESULTS/'full'/name/'oof_predictions.csv',rows)
                parent.write_table(RESULTS/'full'/name/'validation_oof_predictions.csv',vrows)
            conditions.append(row);validations.append(valrow);tables.append(table)
        if final: require(len(completed)==10 and len(conditions)==4,'Need ten new fits and four comparison conditions')
        report=dict(experiment=EXPERIMENT,protocol_fingerprint=protocol['fingerprint'],completed_fits=len(completed),expected_fits=10,
            reused_reference_fits=10,models=conditions,validation_models=validations,folds=completed,final_audit=bool(final),
            history_coverage=dict(available=int(history_available.sum()),total=len(history_available)),
            std_definition='sample SD across five company folds; ddof=1; OOF reported separately',
            split_statistics={str(f):baseline.split_statistics(d,f) for f in range(5)},
            interpretation='Matched 2x2 original versus conv-hidden64 backbone, without versus with R16. Fixed input/data/protocol; grouped CV is not out-of-time evaluation')
        RESULTS.mkdir(parents=True,exist_ok=True)
        common.atomic_json(RESULTS/'summary.json',report)
        parent.write_table(RESULTS/'comparison.csv',tables)
        validation_table=[{k:v for k,v in row.items() if k in ('model','scheme','completed_folds','source_experiment','source_protocol_fingerprint') or k.startswith('validation_')} for row in tables]
        parent.write_table(RESULTS/'validation_summary.csv',validation_table)
        for item in conditions:
            common.log('persem_conv64_comparison',model=item['model'],ap_mean=item['ap_mean'],ap_sd=item['ap_std'],
                f1_mean=item['f1_mean'],f1_sd=item['f1_std'],pooled_oof=item['pooled_oof'],history_strata=item['history_strata'])
        return report

def worker(gpu):
    require(gpu in (0,1),'Expected GPU0/1');protocol=load_protocol();require_checks(protocol)
    with locked(CACHE/f'worker_{gpu}.lock'):
        status_path=CACHE/f'worker_{gpu}_status.json';model=DESIGN['gpu_plan'][str(gpu)]
        (ROOT/'logs').mkdir(exist_ok=True)
        with (ROOT/'logs'/f'persem_conv64_gpu{gpu}.log').open('a',encoding='utf-8') as stream:
            for fold in range(5):
                while True:
                    state=dict(status='running',model=model,fold=fold,gpu=gpu,pid=os.getpid(),updated_at=utc(),
                        protocol_fingerprint=protocol['fingerprint'])
                    common.atomic_json(status_path,state)
                    child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),'train','--model',model,
                        '--fold',str(fold),'--device',f'cuda:{gpu}'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                    state['child_pid']=child.pid;common.atomic_json(status_path,state);code=child.wait()
                    if code==75:
                        state.update(status='waiting_resources',updated_at=utc());common.atomic_json(status_path,state)
                        time.sleep(30);continue
                    if code:
                        state.update(status='failed',returncode=code,updated_at=utc());common.atomic_json(status_path,state)
                        raise RuntimeError(f'{model} fold{fold} failed with {code}; results retained')
                    break
        common.atomic_json(status_path,dict(status='complete',gpu=gpu,model=model,pid=os.getpid(),updated_at=utc()))

def pipeline():
    with locked(CACHE/'pipeline.lock'):
        children=[];status_path=CACHE/'pipeline_status.json'
        try:
            protocol=load_protocol();require_checks(protocol);preserved_unchanged(full=True)
            state=dict(status='starting',pid=os.getpid(),started_at=utc(),protocol_fingerprint=protocol['fingerprint'],expected_fits=10)
            common.atomic_json(status_path,state)
            for gpu in (0,1):
                children.append(subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),'worker','--gpu',str(gpu)],
                    cwd=ROOT,start_new_session=True))
            while True:
                codes=[c.poll() for c in children]
                state.update(status='running',updated_at=utc(),completed_fits=len(list(RESULTS.glob('full/*/fold_*/metrics.json'))),
                    workers=[dict(gpu=i,pid=c.pid,returncode=codes[i]) for i,c in enumerate(children)])
                common.atomic_json(status_path,state)
                if any(code is not None and code!=0 for code in codes): raise RuntimeError('Training worker failed: '+str(codes))
                if all(code==0 for code in codes): break
                time.sleep(10)
            summary(final=True)
            state.update(status='complete',completed_fits=10,finished_at=utc());common.atomic_json(status_path,state)
        except BaseException:
            parent.stop_workers(children)
            state=read_json(status_path) if status_path.exists() else {}
            state.update(status='failed',failed_at=utc(),error=traceback.format_exc());common.atomic_json(status_path,state)
            raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','check','train','summary','audit','worker','pipeline'))
    parser.add_argument('--model',choices=MODELS)
    parser.add_argument('--fold',type=int,choices=range(5),default=0)
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--gpu',type=int,choices=(0,1))
    args=parser.parse_args()
    if args.command=='prepare': prepare()
    elif args.command=='check':
        if args.model: train(args.model,args.fold,args.device,check=True)
        else:
            for model in MODELS:
                subprocess.run([sys.executable,'-B','-u',str(Path(__file__).resolve()),'check','--model',model,
                    '--fold',str(args.fold),'--device',args.device],cwd=ROOT,check=True)
    elif args.command=='train':
        if args.model is None: parser.error('train requires --model')
        train(args.model,args.fold,args.device)
    elif args.command=='summary': summary()
    elif args.command=='audit': summary(final=True)
    elif args.command=='worker':
        if args.gpu is None: parser.error('worker requires --gpu')
        worker(args.gpu)
    elif args.command=='pipeline': pipeline()

if __name__=='__main__':
    try: main()
    except parent.AlreadyRunning as exc:
        common.log('persem_conv64_already_running',error=str(exc));sys.exit(3)
    except ResourceBusy as exc:
        common.log('persem_conv64_waiting_resources',error=str(exc));sys.exit(75)
    except Exception:
        failure=dict(argv=sys.argv,failed_at=utc(),pid=os.getpid(),error=traceback.format_exc())
        path=CACHE/'failures'/f'{time.time_ns()}_{os.getpid()}.json'
        common.atomic_json(path,failure);common.log('persem_conv64_failed',failure=str(path),error=failure['error'])
        raise
