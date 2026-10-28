"""Second-stage residual-only tuning from joint-trained light PerSeM checkpoints.

Five new stage-two fits. Original sources, caches, checkpoints and results are immutable.
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
import run_persem_conv64 as stage_one

ROOT=Path(__file__).resolve().parent
EXPERIMENT='persem_stage2_v1'
CACHE=ROOT/'cache'/EXPERIMENT
RESULTS=ROOT/'results'/EXPERIMENT
MODELS=('PerSeM_Conv64_R16_Stage2',)
SOURCES=('run_persem_stage2.py','run_persem_conv64.py','run_residual_mlp16.py','run_residual_mlp.py','run_linear_history.py','data_linear_history.py','run_annual_risk.py','run_annual_baselines.py',
         'data_annual_risk.py','run.py','run_persem.py','models_persem.py','data.py','config.json')
require=parent.require
read_json=parent.read_json
digest=parent.digest
immutable_json=parent.immutable_json
locked=parent.locked
utc=parent.utc
STAGE1_MODEL='PerSeM_Conv64_R16'
DESIGN=dict(experiment=EXPERIMENT,version=1,models=list(MODELS),expected_fits=5,reused_reference_fits=15,
    task='Unchanged annual_risk_v1 June30Y-to-FYY, company-group five folds, complete training/validation/test pools',
    stage1='reuse each persem_conv64_v1/PerSeM_Conv64_R16 fold best validation-AP checkpoint; all classifier modules jointly trained; no rerun',
    stage2='load all stage1 weights, freeze base including text projection and original head, keep base.eval; update extra_projection and correction only',
    predictor='unchanged frozen three inner cross-fitted linear predictors and residual caches; frozen BERT embeddings',
    architecture='conv hidden64; absolute64; residual Linear2160to16; correction Linear80to32 GELU Linear32to1; no new modules',
    parameter_count=dict(total=832914,stage2_trainable=37201,frozen_base=795713),
    optimizer='new AdamW on residual parameters only; lr=0.0001, weight_decay=0.0001, seed42, batch64, clip1',
    training='same full training indices and neg/pos weight; no-H rows retained; wholly unavailable batches skip optimizer and are separately counted',
    selection='stage1 checkpoint included as step0 candidate; validation AP every105 actual updates, patience15, max10500 new updates; threshold0.5',
    inference='training uses own excluded inner reference; validation/test mean of three probabilities; absent residual gives frozen stage1 base score',
    gpu_plan={'0':[0,2,4],'1':[1,3]},
    attribution='additional stage2 budget; does not isolate freezing from equal-budget continued joint training',
    scope='no data, predictor, architecture, threshold, sampling or old result changes')

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

def stage1_dir(fold): return stage_one.result_dir(STAGE1_MODEL,fold)

def source_identity():
    preserved_unchanged()
    previous=stage_one.load_protocol();prior=parent.load_protocol()
    artifacts={str((stage1_dir(f)/name).relative_to(ROOT)):common.file_sha256(stage1_dir(f)/name)
               for f in range(5) for name in ('metrics.json','best.pt','predictions.csv','validation_predictions.csv')}
    payload=dict(design=DESIGN,stage1_protocol_fingerprint=previous['fingerprint'],
        stage1_protocol_sha256=common.file_sha256(stage_one.CACHE/'protocol.json'),stage1_artifacts=artifacts,
        parent_protocol_fingerprint=prior['fingerprint'],config=dict(previous['config'],learning_rate=0.0001),
        preprocessing=previous['preprocessing'],source_sha256={n:common.file_sha256(ROOT/n) for n in SOURCES},
        history_artifacts=data_files(),data_identity=previous['data_identity'],
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
        common.log('persem_stage2_prepared',fingerprint=identity['fingerprint'],expected_fits=5)
        return identity

def load_protocol():
    saved=read_json(CACHE/'protocol.json')
    require(saved==source_identity(),'Frozen stage2 protocol/source/data changed')
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
        history_metadata_sha256=protocol['history_artifacts'][str((history.CACHE/f'fold_{fold}'/'metadata.json').relative_to(ROOT))],
        stage1_checkpoint_sha256=protocol['stage1_artifacts'][str((stage1_dir(fold)/'best.pt').relative_to(ROOT))])
    return dict(fingerprint=digest(payload),**payload)

def make_model(x,state,embeddings,cfg,device,model_name):
    require(model_name in MODELS,'Unknown stage2 model')
    model=stage_one.make_model(x,state,embeddings,cfg,device,STAGE1_MODEL)
    model.base.requires_grad_(False)
    model.base.eval()
    return model

def base_hash(model):
    return digest({n:dict(shape=list(w.shape),dtype=str(w.dtype),sha256=hashlib.sha256(w.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
                   for n,w in model.base.state_dict().items()})

def training_mode(model):
    model.train()
    model.base.eval()

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
            audit_fit(path,d,protocol,model_name,fold,carry);return
        if not check:
            require(not path.exists() or not any(path.iterdir()),'Incomplete result retained: '+str(path))
            resource_check(model_name,device,protocol)
        cfg=protocol['config'];common.seed_all(cfg['seed']);torch.set_num_threads(cfg['torch_threads'])
        if str(device).startswith('cuda'):
            torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
        x,state=parent.load_preprocessing(fold,prior)
        model=make_model(x,state,annual.load_embeddings(),cfg,device,model_name)
        source=stage1_dir(fold);origin=read_json(source/'metrics.json')
        previous=stage_one.load_protocol()
        require(origin['identity']==stage_one.fit_identity(previous,d,STAGE1_MODEL,fold),'Stage1 identity mismatch')
        checkpoint=torch.load(source/'best.pt',map_location='cpu',weights_only=False)
        require(checkpoint['identity']==origin['identity'] and checkpoint['optimizer_steps']==origin['best_step'],'Wrong stage1 checkpoint')
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        frozen_initial=base_hash(model)
        trainable_names=[n for n,w in model.named_parameters() if w.requires_grad]
        require(all(n.startswith(('extra_projection.','correction.')) for n in trainable_names),'Unexpected trainable module')
        trainable=[w for w in model.parameters() if w.requires_grad]
        parameters=sum(w.numel() for w in model.parameters());trainable_parameters=sum(w.numel() for w in trainable)
        require(parameters==832914 and trainable_parameters==37201,'Parameter contract differs')
        optimizer=torch.optim.AdamW(trainable,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        weight=float((d['labels'][tr]==0).sum())/float(d['labels'][tr].sum())
        require(weight==origin['pos_weight'],'Stage1 class weight changed')
        loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight,device=device))
        xt=torch.from_numpy(np.array(x,copy=True));dt=torch.from_numpy(d['doc_ids'].astype('int64'))
        vt=torch.from_numpy(d['valid_time'].astype(bool));yt=torch.from_numpy(d['labels'].astype('float32'))
        rt=torch.from_numpy(a['residual_features'].astype('float32',copy=False));ra=torch.from_numpy(a['residual_available'].astype(bool))
        assignment=torch.from_numpy(a['inner_assignment'].astype('int64'))
        def batch(indices,branch=None):
            ids=torch.as_tensor(indices,dtype=torch.long)
            if branch is None:
                require((assignment[ids]>=0).all().item(),'Missing cross-fit assignment')
                extra,available=rt[assignment[ids],ids],ra[assignment[ids],ids]
            else: extra,available=rt[branch,ids],ra[branch,ids]
            return tuple(v.to(device,non_blocking=True) for v in (xt[ids],dt[ids],vt[ids],extra,available,yt[ids]))
        def predict(indices):
            model.eval();values=[]
            with torch.inference_mode():
                for start in range(0,len(indices),cfg['batch_size']):
                    ids=indices[start:start+cfg['batch_size']];ps=[]
                    for branch in range(3):
                        xb,db,vb,eb,ab,_=batch(ids,branch)
                        ps.append(model(xb,db,vb,eb,ab).sigmoid())
                    values.append(torch.stack(ps).mean(0).cpu().numpy())
            return np.concatenate(values).astype('float64')
        _,old_validation=parent.read_predictions(source/'validation_predictions.csv',d,va,fold,carry)
        initial_validation=predict(va)
        require(np.allclose(initial_validation,old_validation,atol=1e-7,rtol=1e-7),'Stage1 validation probabilities not reproduced')
        stage0_ap=float(average_precision_score(d['labels'][va],initial_validation))
        require(abs(stage0_ap-origin['best_validation_ap'])<1e-8,'Stage1 validation AP mismatch')
        training=tr
        if check:
            usable=a['residual_available'][a['inner_assignment'][tr],tr].astype(bool);candidates=tr[usable]
            positive=candidates[d['labels'][candidates]==1][:32];negative=candidates[d['labels'][candidates]==0][:64-len(positive)]
            training=np.r_[positive,negative];require(len(training)==64,'Need complete engineering batch')
        from torch.utils.data import DataLoader
        loader=DataLoader(training.tolist(),batch_size=cfg['batch_size'],shuffle=True,num_workers=0,
                          generator=torch.Generator().manual_seed(cfg['seed']))
        iterator=iter(loader);steps=examples=positive_exposures=active_examples=active_positive_exposures=checks=stale=0
        skipped_batches=skipped_examples=skipped_positive=0
        interval_loss=0.;interval_examples=0;best_ap=stage0_ap;best_step=0
        best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        stage2_best_ap=None;stage2_best_step=None;start_time=time.monotonic()
        def save_best():
            temporary=path/'best.pt.tmp'
            torch.save(dict(state_dict=best_state,identity=identity,config=cfg,optimizer_steps=best_step,
                validation_ap=best_ap,pos_weight=weight,parameters=parameters,trainable_parameters=trainable_parameters),temporary)
            os.replace(temporary,path/'best.pt')
        if not check:
            path.mkdir(parents=True,exist_ok=True);immutable_json(path/'preprocessing.json',state)
            immutable_json(path/'started.json',dict(identity=identity,config=cfg,pos_weight=weight,
                split_statistics=baseline.split_statistics(d,fold),feature_check=feature_check,
                stage1=dict(checkpoint_sha256=identity['stage1_checkpoint_sha256'],checkpoint_step=origin['best_step'],actual_updates=origin['optimizer_steps'],validation_ap=stage0_ap),
                frozen_initial_sha256=frozen_initial,trainable_names=trainable_names,trainable_parameters=trainable_parameters,
                predictor_metadata_fingerprint=meta['fingerprint'],started_at=utc(),pid=os.getpid(),device=device))
            with (path/'trace.jsonl').open('x',encoding='utf-8'): pass
            save_best()
        def update_batch(xb,db,vb,eb,ab,yb):
            if not ab.any().item(): return None
            optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(model(xb,db,vb,eb,ab),yb)
            require(torch.isfinite(loss).item(),'Nonfinite loss');loss.backward()
            require(all(w.grad is None for w in model.base.parameters()),'Frozen base received gradients')
            norm=torch.nn.utils.clip_grad_norm_(trainable,cfg['gradient_clip'],error_if_nonfinite=True)
            if check:
                for label,w in [('projection',model.extra_projection.weight),('MLP1',model.correction[0].weight),('MLP2',model.correction[2].weight)]:
                    require(w.grad is not None and torch.isfinite(w.grad).all().item() and torch.count_nonzero(w.grad).item()>0,'Invalid '+label+' gradient')
                require(float(norm)>0,'Zero gradient')
            optimizer.step()
            return loss
        training_mode(model)
        while steps<(1 if check else cfg['max_optimizer_steps']):
            try: ids=next(iterator)
            except StopIteration: iterator=iter(loader);ids=next(iterator)
            xb,db,vb,eb,ab,yb=batch(ids)
            require(not model.base.training and all(not w.requires_grad for w in model.base.parameters()),'Base is not frozen/eval')
            if check:
                model.eval()
                with torch.no_grad():
                    empty=torch.zeros_like(ab);masked_before=model(xb,db,vb,eb,empty)
                    oracle=model.base.head[1](model.base.head[0](model.base.forward_features(xb,db,vb)['absolute'])).squeeze(-1)
                    require(torch.equal(masked_before,oracle),'Stage2 fallback is not frozen stage1 base')
                    require(torch.equal(masked_before,model(xb,db,vb,eb+13,empty)),'Unavailable residual affects prediction')
                    no_h=tr[~a['history_available'][tr].astype(bool)][:8]
                    nx,nd,nv,ne,na,_=batch(no_h);require(not na.any().item(),'No-history availability differs')
                    no_h_before=model(nx,nd,nv,ne,na).clone()
                residual_before={n:w.detach().clone() for n,w in model.named_parameters() if w.requires_grad}
                training_mode(model)
            loss=update_batch(xb,db,vb,eb,ab,yb)
            if loss is None:
                skipped_batches+=1;skipped_examples+=len(yb);skipped_positive+=int(yb.sum().item());continue
            steps+=1;examples+=len(yb);positive_exposures+=int(yb.sum().item())
            active_examples+=int(ab.sum().item());active_positive_exposures+=int(yb[ab].sum().item())
            interval_loss+=float(loss.detach())*len(yb);interval_examples+=len(yb)
            if check:
                pv=predict(va[:64]);require(np.isfinite(pv).all(),'Nonfinite prediction')
                with torch.no_grad():
                    require(torch.equal(no_h_before,model(nx,nd,nv,ne,na)),'No-history output changed after update')
                    require(torch.equal(masked_before,model(xb,db,vb,eb,torch.zeros_like(ab))),'Frozen base prediction changed')
                    require(torch.allclose(model(xb[:1],db[:1],vb[:1],eb[:1],ab[:1]),model(xb,db,vb,eb,ab)[:1],atol=5e-5,rtol=5e-5),'Batch-dependent inference')
                require(base_hash(model)==frozen_initial,'Frozen base state changed')
                require(any(not torch.equal(w,residual_before[n]) for n,w in model.named_parameters() if w.requires_grad),'Residual parameters did not update')
                before_skip={n:w.detach().clone() for n,w in model.named_parameters() if w.requires_grad}
                before_optimizer=[float(v['step']) for v in optimizer.state.values()]
                training_mode(model)
                skipped=update_batch(nx,nd,nv,ne,na,torch.zeros(len(na),device=device))
                require(skipped is None,'Empty batch did not skip optimizer')
                require(before_optimizer==[float(v['step']) for v in optimizer.state.values()],'Empty batch advanced optimizer')
                require(all(torch.equal(w,before_skip[n]) for n,w in model.named_parameters() if w.requires_grad),'Empty-batch skip changed weights')
                value=dict(passed=True,model=model_name,fold=fold,device=device,parameters=parameters,
                    trainable_parameters=trainable_parameters,frozen_parameters=parameters-trainable_parameters,
                    stage0_validation_ap=stage0_ap,stage1_probabilities_reproduced=True,frozen_base_hash=frozen_initial,
                    frozen_base_unchanged=True,base_eval=True,no_history_unchanged=True,residual_gradients=True,
                    residual_updated=True,empty_batch_skips_optimizer=True,batch_invariance=True,
                    cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)) if str(device).startswith('cuda') else 0,checked_at=utc())
                record_check(protocol,model_name,value);common.log('persem_stage2_check',**value);return value
            if steps%cfg['validate_every_steps']:continue
            pv=predict(va);ap=float(average_precision_score(d['labels'][va],pv));checks+=1
            if stage2_best_ap is None or ap>stage2_best_ap: stage2_best_ap,stage2_best_step=ap,steps
            if ap>best_ap:
                best_ap,best_step,stale=ap,steps,0
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};save_best()
            else: stale+=1
            row=dict(validation_check=checks,optimizer_steps=steps,training_examples_seen=examples+skipped_examples,
                positive_training_exposures=positive_exposures+skipped_positive,active_residual_examples=active_examples,
                epochs_equivalent=(examples+skipped_examples)/len(tr),completed_epochs=(examples+skipped_examples)//len(tr),
                train_loss=interval_loss/interval_examples,validation_ap=ap,best_validation_ap=best_ap,best_step=best_step,
                stage0_validation_ap=stage0_ap,stage2_best_validation_ap=stage2_best_ap,nonimproving_checks=stale,skipped_no_history_batches=skipped_batches)
            with (path/'trace.jsonl').open('a',encoding='utf-8') as st:st.write(json.dumps(row,allow_nan=False)+'\n')
            common.log('persem_stage2_validation',model=model_name,fold=fold,**row)
            common.atomic_json(CACHE/f'{model_name}_{fold}_status.json',dict(status='training',identity=identity,updated_at=utc(),**row))
            training_mode(model);interval_loss=0.;interval_examples=0
            if stale>=cfg['patience_checks']:break
        require(base_hash(model)==frozen_initial,'Frozen base changed during stage2')
        model.load_state_dict(best_state);require(base_hash(model)==frozen_initial,'Selected frozen base changed')
        pv,pred=predict(va),predict(te)
        require(abs(float(average_precision_score(d['labels'][va],pv))-best_ap)<1e-10,'Checkpoint validation mismatch')
        _,old_test=parent.read_predictions(source/'predictions.csv',d,te,fold,carry)
        for ids,now,old in [(va,pv,old_validation),(te,pred,old_test)]:
            empty=~a['history_available'][ids].astype(bool)
            require(np.allclose(now[empty],old[empty],atol=1e-7,rtol=1e-7),'No-history predictions drifted from stage1')
        result=parent.safe_scores(d['labels'][te],pred);validation=parent.safe_scores(d['labels'][va],pv)
        result.update(identity=identity,model=model_name,scheme='full',fold=fold,pos_weight=weight,parameters=parameters,
            trainable_parameters=trainable_parameters,frozen_parameters=parameters-trainable_parameters,
            optimizer_steps=steps,training_examples_seen=examples+skipped_examples,positive_training_exposures=positive_exposures+skipped_positive,
            active_residual_examples=active_examples,active_positive_exposures=active_positive_exposures,
            skipped_no_history_batches=skipped_batches,skipped_no_history_examples=skipped_examples,
            epochs_equivalent=(examples+skipped_examples)/len(tr),completed_epochs=(examples+skipped_examples)//len(tr),
            validation_checks=checks,best_step=best_step,best_validation_ap=best_ap,stage0_validation_ap=stage0_ap,
            stage2_best_validation_ap=stage2_best_ap,stage2_best_step=stage2_best_step,selected_stage=2 if best_step else 1,
            stage1_actual_optimizer_steps=origin['optimizer_steps'],stage1_selected_step=origin['best_step'],
            total_executed_optimizer_steps=origin['optimizer_steps']+steps,selected_path_updates=origin['best_step']+best_step,
            frozen_initial_sha256=frozen_initial,frozen_final_sha256=base_hash(model),no_history_prediction_invariant=True,
            stop_reason='patience_checks' if stale>=cfg['patience_checks'] else 'max_optimizer_steps',
            max_optimizer_steps=cfg['max_optimizer_steps'],validate_every_steps=cfg['validate_every_steps'],patience_checks=cfg['patience_checks'],
            seconds=round(time.monotonic()-start_time,2),diagnostics=parent.diagnostics(d,te,pred,carry),
            history_strata=history_strata(d,te,pred,a['history_available']),predictor_training='unchanged frozen predictors; residual-only stage2')
        validation.update(identity=identity,best_step=best_step,checkpoint_selection='maximum validation AP including stage1 step0; three probabilities averaged')
        parent.write_predictions(path/'predictions.csv',d,te,pred,fold,carry);parent.write_predictions(path/'validation_predictions.csv',d,va,pv,fold,carry)
        immutable_json(path/'validation_metrics.json',validation)
        result['artifact_sha256']={n:common.file_sha256(path/n) for n in ('started.json','preprocessing.json','trace.jsonl','best.pt','predictions.csv','validation_predictions.csv','validation_metrics.json')}
        immutable_json(path/'metrics.json',result);audit_fit(path,d,protocol,model_name,fold,carry)
        common.atomic_json(CACHE/f'{model_name}_{fold}_status.json',dict(status='complete',identity=identity,optimizer_steps=steps,updated_at=utc()))
        common.log('persem_stage2_fit_complete',model=model_name,fold=fold,ap=result['ap'],f1=result['f1'],selected_stage=result['selected_stage'],updates=steps)
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
    require(result['best_step'] in [0]+[r['optimizer_steps'] for r in trace] and abs(validation['ap']-result['best_validation_ap'])<1e-10,
            'Checkpoint selection mismatch')
    require(result['trainable_parameters']==37201 and result['frozen_parameters']==795713,'Freeze count mismatch')
    require(result['frozen_initial_sha256']==result['frozen_final_sha256']==started['frozen_initial_sha256'],'Frozen state mismatch')
    require(result['best_validation_ap']>=result['stage0_validation_ap'],'Selected validation AP worse than step0')
    require(result['total_executed_optimizer_steps']==result['stage1_actual_optimizer_steps']+result['optimizer_steps'],'Cost accounting mismatch')
    candidates=[(result['stage0_validation_ap'],0)]+[(r['validation_ap'],r['optimizer_steps']) for r in trace]
    selected=max(candidates,key=lambda pair:pair[0])
    require(abs(result['best_validation_ap']-selected[0])<1e-10 and result['best_step']==selected[1],'Selection differs')
    require(result['selected_stage']==(2 if selected[1] else 1),'Selected stage differs')
    import torch
    saved_checkpoint=torch.load(path/'best.pt',map_location='cpu',weights_only=False)
    source_checkpoint=torch.load(stage1_dir(fold)/'best.pt',map_location='cpu',weights_only=False)
    require(saved_checkpoint['identity']==identity and saved_checkpoint['optimizer_steps']==result['best_step'],'Checkpoint metadata mismatch')
    for name,tensor in source_checkpoint['state_dict'].items():
        if name.startswith('base.'):
            require(torch.equal(tensor,saved_checkpoint['state_dict'][name]),'Stored base changed: '+name)
    for ids,now,filename in [(te,p,'predictions.csv'),(va,pv,'validation_predictions.csv')]:
        _,old=parent.read_predictions(stage1_dir(fold)/filename,d,ids,fold,carry)
        empty=~a['history_available'][ids].astype(bool)
        require(np.allclose(now[empty],old[empty],atol=1e-7,rtol=1e-7),'No-history score invariant failed')
    return result,validation,rows,vrows

def summary(final=False):
    with locked(CACHE/'summary.lock',blocking=True):
        protocol=load_protocol();prior=parent.load_protocol();previous=stage_one.load_protocol();d=annual.load_data();carry=annual.load_carry_baseline()
        preserved_unchanged(full=final)
        completed=[];conditions=[];validations=[];tables=[]
        history_available=history.load_fold(0)[0]['history_available']
        for name in ('PerSeM','PerSeM_Conv64','PerSeM_Conv64_R16')+MODELS:
            fits=[];vals=[];rows=[];vrows=[]
            for fold in range(5):
                old=name not in MODELS
                path=parent.result_dir(name,'full',fold) if name=='PerSeM' else (stage_one.result_dir(name,fold) if old else result_dir(name,fold))
                if not (path/'metrics.json').exists(): continue
                if name=='PerSeM': fit,val,rr,vv=parent.audit_fit(path,d,prior,name,'full',fold,carry)
                elif old: fit,val,rr,vv=stage_one.audit_fit(path,d,previous,name,fold,carry)
                else:
                    fit,val,rr,vv=audit_fit(path,d,protocol,name,fold,carry);completed.append(fit)
                fits.append(fit);vals.append(val);rows.extend(rr);vrows.extend(vv)
            if len(fits)!=5: continue
            old=name not in MODELS
            origin='annual_risk_v1' if name=='PerSeM' else ('persem_conv64_v1' if old else EXPERIMENT)
            fingerprint=prior['fingerprint'] if name=='PerSeM' else (previous['fingerprint'] if old else protocol['fingerprint'])
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
        if final: require(len(completed)==5 and len(conditions)==4,'Need five stage2 fits and four comparison conditions')
        report=dict(experiment=EXPERIMENT,protocol_fingerprint=protocol['fingerprint'],completed_fits=len(completed),expected_fits=5,
            reused_reference_fits=15,models=conditions,validation_models=validations,folds=completed,final_audit=bool(final),
            history_coverage=dict(available=int(history_available.sum()),total=len(history_available)),
            std_definition='sample SD across five company folds; ddof=1; OOF reported separately',
            split_statistics={str(f):baseline.split_statistics(d,f) for f in range(5)},
            interpretation='Residual-only stage2 from joint stage1; extra budget recorded; no equal-budget joint continuation control; grouped CV is not out-of-time evaluation')
        RESULTS.mkdir(parents=True,exist_ok=True)
        common.atomic_json(RESULTS/'summary.json',report)
        parent.write_table(RESULTS/'comparison.csv',tables)
        validation_table=[{k:v for k,v in row.items() if k in ('model','scheme','completed_folds','source_experiment','source_protocol_fingerprint') or k.startswith('validation_')} for row in tables]
        parent.write_table(RESULTS/'validation_summary.csv',validation_table)
        for item in conditions:
            common.log('persem_stage2_comparison',model=item['model'],ap_mean=item['ap_mean'],ap_sd=item['ap_std'],
                f1_mean=item['f1_mean'],f1_sd=item['f1_std'],pooled_oof=item['pooled_oof'],history_strata=item['history_strata'])
        return report

def worker(gpu):
    require(gpu in (0,1),'Expected GPU0/1');protocol=load_protocol();require_checks(protocol)
    with locked(CACHE/f'worker_{gpu}.lock'):
        status_path=CACHE/f'worker_{gpu}_status.json';model=MODELS[0]
        (ROOT/'logs').mkdir(exist_ok=True)
        with (ROOT/'logs'/f'persem_stage2_gpu{gpu}.log').open('a',encoding='utf-8') as stream:
            for fold in DESIGN['gpu_plan'][str(gpu)]:
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
            state=dict(status='starting',pid=os.getpid(),started_at=utc(),protocol_fingerprint=protocol['fingerprint'],expected_fits=5)
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
            state.update(status='complete',completed_fits=5,finished_at=utc());common.atomic_json(status_path,state)
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
        common.log('persem_stage2_already_running',error=str(exc));sys.exit(3)
    except ResourceBusy as exc:
        common.log('persem_stage2_waiting_resources',error=str(exc));sys.exit(75)
    except Exception:
        failure=dict(argv=sys.argv,failed_at=utc(),pid=os.getpid(),error=traceback.format_exc())
        path=CACHE/'failures'/f'{time.time_ns()}_{os.getpid()}.json'
        common.atomic_json(path,failure);common.log('persem_stage2_failed',failure=str(path),error=failure['error'])
        raise
