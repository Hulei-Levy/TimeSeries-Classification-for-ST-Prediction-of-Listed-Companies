"""Isolated annual-risk inputs, company folds, fixed sampling and frozen BERT.

June 30 Y predicts audited consolidated fiscal-Y risk. Inputs are reporting
quarters Y-5..Y-1, not a sequence of live historical forecasts. Feature
availability uses max(period+6 months, known publication); original historical
feature vintages remain unverified. No old cache or result is written here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

import data as legacy

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'cache/annual_risk_v1'
EMBEDDINGS_PATH = CACHE / 'bert_embeddings.npy'
LABELS = ROOT / 'datasets/annual_risk_v1/candidate_labels.parquet'
AMOUNTS = ROOT / 'datasets/annual_risk_v1/amounts.parquet'
PROTOCOL = 'annual-risk-v1:June30Y-to-FYY:5FY:company5:seed42'
POOLING = ('v1:frozen-eval-last-hidden;510-content-token-nonoverlapping-chunks;'
           'exclude-CLS-SEP-PAD;content-token-mean;token-count-weighted-document-mean;'
           'float32-accumulation;cuda-float16-forward')
SCHEMES = ('full', 'ratio_3', 'ratio_4')


def sha(path):
    return legacy.sha256(path)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def source_hashes():
    paths = [Path(__file__), ROOT/'data.py', LABELS, AMOUNTS,
             ROOT/'datasets/annual_risk_v1/audit.json',
             CACHE/'preserved_identity.json']
    paths += [ROOT/'datasets'/p for p in legacy.SOURCE_PATHS.values()]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def validate_manifest():
    manifest = read_json(CACHE/'manifest.json')
    if manifest['protocol'] != PROTOCOL:
        raise RuntimeError('Annual-risk protocol mismatch')
    # Full raw source hashes were verified at build. Recheck immutable code and
    # all cache artifacts here; expensive raw/source checking is an explicit build.
    for name in ('data_annual_risk.py', 'data.py'):
        if sha(ROOT/name) != manifest['source_sha256'][name]:
            raise RuntimeError(f'Builder source changed: {name}')
    for name, expected in manifest['artifact_sha256'].items():
        if sha(CACHE/name) != expected:
            raise RuntimeError(f'Frozen annual-risk artifact changed: {name}')
    return manifest


def load_data():
    validate_manifest()
    with np.load(CACHE/'samples.npz', allow_pickle=False) as archive:
        d = {key: archive[key] for key in archive.files}
    d.update(valid_nodes=d['valid_time'], text_ids=d['doc_ids'],
             text_available=d['doc_ids'] >= 0, text_ages=d['doc_ages'],
             y=d['labels'], folds=d['fold'], categorical=d['categorical_indices'])
    return d


def training_indices(fold, scheme):
    if int(fold) not in range(5) or scheme not in SCHEMES:
        raise ValueError('Expected fold 0..4 and full/ratio_3/ratio_4')
    manifest = validate_manifest()
    m = read_json(CACHE/f'fold_{int(fold)}.json')
    if m['dataset_sha256'] != manifest['artifact_sha256']['samples.npz']:
        raise RuntimeError('Fold manifest dataset mismatch')
    return np.asarray(m['train_indices'][scheme], dtype=np.int64)


def load_carry_baseline():
    validate_manifest()
    with np.load(CACHE/'carry_baseline.npz', allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def check_completion():
    manifest = read_json(CACHE/'bert_manifest.json')
    completion = read_json(CACHE/'bert_completion.json')
    st = EMBEDDINGS_PATH.stat()
    if (completion['manifest_identity'] != manifest['identity_sha256'] or
            completion['bytes'] != st.st_size or completion['mtime_ns'] != st.st_mtime_ns):
        raise RuntimeError('BERT completion identity or embedding file changed')
    done = np.load(CACHE/'bert_complete.npy', mmap_mode='r')
    if not bool(done.all()) or len(done) != len(manifest['documents']):
        raise RuntimeError('BERT encoding is incomplete')
    if completion['complete_sha256'] != sha(CACHE/'bert_complete.npy'):
        raise RuntimeError('BERT completion flags changed')
    return completion


def load_embeddings():
    validate_manifest()
    check_completion()
    result = np.load(EMBEDDINGS_PATH, mmap_mode='r')
    if result.shape != (len(read_json(CACHE/'bert_manifest.json')['documents']), 768):
        raise RuntimeError('Wrong embedding dimensions')
    return result


def identity(require_embeddings=True):
    manifest = validate_manifest()
    result = dict(protocol=PROTOCOL, manifest_sha256=sha(CACHE/'manifest.json'),
                  artifacts=manifest['artifact_sha256'], sources=manifest['source_sha256'])
    if require_embeddings:
        completion = check_completion()
        result['bert'] = dict(manifest_sha256=sha(CACHE/'bert_manifest.json'),
                              embeddings_sha256=completion['sha256'],
                              complete_sha256=completion['complete_sha256'])
    return result


def reconstruct(samples):
    datasets = ROOT/'datasets'
    fin, fv, fa = legacy.load_table(datasets, 'financial', legacy.FINANCIAL, 'report_period')
    gov, gv, ga = legacy.load_table(datasets, 'governance', legacy.GOVERNANCE, 'period_end', 'year')
    matched = dict(financial=legacy.add_known_dates(datasets, fin, fa, 'financial', legacy.FINANCIAL, 'report_period'),
                   governance=legacy.add_known_dates(datasets, gov, ga, 'governance', legacy.GOVERNANCE, 'period_end'))
    fi = {(c,p): i for i,(c,p) in enumerate(zip(fin.code, fin.report_period))}
    gi = {(c,int(y)): i for i,(c,y) in enumerate(zip(gov.code, gov.year))}
    raw_docs, skipped = legacy.index_documents(datasets, {s['code'] for s in samples})
    n = len(samples)
    values = np.full((n,20,61), np.nan, dtype=np.float32)
    ages = np.zeros_like(values)
    valid = np.zeros((n,20), dtype=bool)
    docids = np.full((n,20), -1, dtype=np.int32)
    docages = np.zeros((n,20), dtype=np.float32)
    periods = np.empty((n,20), dtype='<U10')
    docs, lookup, late = [], {}, Counter()
    for i,s in enumerate(samples):
        code, year = s['code'], int(s['target_year'])
        cutoff = date.fromisoformat(s['cutoff']).toordinal()
        lower = max(s['window_start'], s['listing_date'])
        for t in range(20):
            y,q = year-5+t//4, t%4+1
            p = f'{y}-{("03-31","06-30","09-30","12-31")[q-1]}'
            periods[i,t] = p
            if p < s['listing_date']:
                continue
            valid[i,t] = True
            f = fi.get((code,p))
            if f is not None:
                ok = fa[f] <= cutoff
                late['financial_cells'] += int((~ok & np.isfinite(fv[f])).sum())
                values[i,t,:30] = np.where(ok, fv[f], np.nan)
                ages[i,t,:30] = np.where(np.isfinite(values[i,t,:30]), 21-t, 0)
            gy = y if q == 4 else y-1
            if f'{gy}-12-31' >= lower:
                g = gi.get((code,gy))
                if g is not None:
                    ok = ga[g] <= cutoff
                    late['governance_cells'] += int((~ok & np.isfinite(gv[g])).sum())
                    values[i,t,30:] = np.where(ok, gv[g], np.nan)
                    ages[i,t,30:] = np.where(np.isfinite(values[i,t,30:]), (q if q<4 else 0)+21-t, 0)
            if q == 1:
                dp,da = f'{y-1}-12-31',1
            elif q in (2,3):
                dp,da = f'{y}-06-30',q-2
            else:
                dp,da = f'{y}-12-31',0
            record = raw_docs.get((code,dp)) if dp >= lower else None
            if record and record['availability_proxy'] <= s['cutoff']:
                if (code,dp) not in lookup:
                    lookup[(code,dp)] = len(docs)
                    docs.append(dict(doc_id=len(docs), **record))
                docids[i,t] = lookup[(code,dp)]
                docages[i,t] = da+21-t
    observed = np.isfinite(values)
    if np.any(~(observed.any(axis=(1,2)) | (docids>=0).any(axis=1))):
        raise RuntimeError('Known-label candidate has no usable input; review cohort')
    if observed[~valid].any() or (docids[~valid]>=0).any():
        raise RuntimeError('Pre-listing padding violated')
    assignments, fold_stats = legacy.balanced_company_folds(samples, 42)
    arrays = dict(values=values, ages=ages, observed=observed, valid_time=valid,
                  doc_ids=docids, doc_ages=docages, periods=periods,
                  labels=np.asarray([s['label'] for s in samples], dtype=np.int64),
                  codes=np.asarray([s['code'] for s in samples], dtype='<U6'),
                  cutoffs=np.asarray([s['cutoff'] for s in samples], dtype='<U10'),
                  years=np.asarray([s['target_year'] for s in samples], dtype=np.int16),
                  sample_index=np.arange(n, dtype=np.int64),
                  fold=np.asarray([assignments[s['code']] for s in samples], dtype=np.int8),
                  feature_names=np.asarray(legacy.FEATURES),
                  categorical_indices=np.asarray(legacy.CATEGORICAL, dtype=np.int32),
                  previous_risk_ex_post=np.asarray([s['previous_risk_ex_post'] for s in samples], dtype=np.int8),
                  transition_ex_post=np.asarray([s['transition_ex_post'] for s in samples]))
    return arrays, docs, dict(fold_groups=fold_stats, known_source_date_counts=matched,
                              late_masked_sample_positions=dict(late), document_exclusions=skipped)


def verify_old_intersection(d, docs):
    olddocs = [json.loads(line) for line in (ROOT/'cache/documents.jsonl').read_text().splitlines()]
    old = np.load(ROOT/'cache/samples.npz', allow_pickle=False)
    oldkeys = {(c,k):i for i,(c,k) in enumerate(zip(old['codes'],old['cutoffs']))}
    pairs = [(i,oldkeys[(c,k)]) for i,(c,k) in enumerate(zip(d['codes'],d['cutoffs'])) if (c,k) in oldkeys]
    ni,oi = np.asarray(pairs, dtype=np.int64).T
    for key in ('values','ages','observed','valid_time','doc_ages','periods'):
        a,b = d[key][ni],old[key][oi]
        equal = np.array_equal(a,b,equal_nan=True) if a.dtype.kind == 'f' else np.array_equal(a,b)
        if not equal:
            raise RuntimeError(f'Old/new intersection mismatch: {key}')
    new_to_old = np.full(len(docs)+1, -1, dtype=np.int32)
    old_lookup = {(r['code'],r['report_period']):r['doc_id'] for r in olddocs}
    for r in docs:
        new_to_old[r['doc_id']] = old_lookup.get((r['code'],r['report_period']), -2)
    if not np.array_equal(new_to_old[d['doc_ids'][ni]], old['doc_ids'][oi]):
        raise RuntimeError('Old/new semantic text mapping mismatch')
    return dict(intersection_samples=len(pairs), new_samples=len(d['labels'])-len(pairs),
                checked=['values','ages','observed','valid_time','doc_ages','periods','semantic_doc_ids'],
                exact_equal=True, old_samples_sha256=sha(ROOT/'cache/samples.npz'))


def stats(d, indices):
    idx = np.asarray(indices,dtype=np.int64)
    p = int(d['labels'][idx].sum())
    return dict(windows=len(idx), companies=len(set(d['codes'][idx])), positives=p,
                negatives=len(idx)-p, prevalence=p/len(idx) if len(idx) else None)


def make_folds(d, dataset_sha):
    j = list(d['feature_names']).index('AF_CFEATUREPROFILE-CompanySize')
    raw = d['values'][:,19,j]
    known = d['observed'][:,19,j] & d['valid_time'][:,19] & np.isfinite(raw) & (raw>0)
    logs = np.full(len(raw), np.nan)
    logs[known] = np.log(raw[known].astype(np.float64))
    result = []
    for fold in range(5):
        train = np.flatnonzero((d['fold']!=fold) & (d['fold']!=(fold+1)%5))
        val,test = np.flatnonzero(d['fold']==(fold+1)%5),np.flatnonzero(d['fold']==fold)
        pos,neg = train[d['labels'][train]==1],train[d['labels'][train]==0]
        rng = np.random.default_rng(42)
        rank = {int(i):r for r,i in enumerate(rng.permutation(neg))}
        years = sorted(set(d['years'][pos].tolist()))
        pools = {year:set(neg[d['years'][neg]==year].tolist()) for year in years}
        for year in years:
            if len(pools[year]) < 4*int((d['years'][pos]==year).sum()):
                raise RuntimeError(f'Insufficient same-year controls fold={fold}, year={year}')
        contributions, matches, fallback = Counter(), [], Counter()
        chosen3,chosen4 = [],[]
        for p in rng.permutation(pos).tolist():
            year = int(d['years'][p])
            selected = []
            for slot in range(4):
                pool = pools[year]
                count = min(contributions[str(d['codes'][i])] for i in pool)
                tier = [i for i in pool if contributions[str(d['codes'][i])]==count]
                size_tier = [i for i in tier if known[i]]
                if known[p] and size_tier:
                    n = min(size_tier, key=lambda i:(abs(logs[p]-logs[i]), rank[i]))
                    reason = 'size_matched'
                else:
                    n = min(tier,key=lambda i:rank[i])
                    reason = 'positive_size_missing' if not known[p] else 'minimum_contribution_tier_size_missing'
                    fallback[reason] += 1
                pool.remove(n)
                contributions[str(d['codes'][n])] += 1
                selected.append(dict(index=int(n), slot=slot+1, company_previous_contribution=count,
                                     positive_size_known=bool(known[p]), negative_size_known=bool(known[n]),
                                     log_size_distance=float(abs(logs[p]-logs[n])) if known[p] and known[n] else None,
                                     match_kind=reason))
            chosen4.extend(x['index'] for x in selected)
            chosen3.extend(x['index'] for x in selected[:3])
            matches.append(dict(positive_index=p, year=year, negatives=selected))
        indices = dict(full=train.tolist(), ratio_3=sorted(pos.tolist()+chosen3), ratio_4=sorted(pos.tolist()+chosen4))
        for ratio in (3,4):
            idx = np.asarray(indices[f'ratio_{ratio}'])
            if len(idx)!=len(set(idx)) or len(idx)!=(ratio+1)*len(pos) or not set(pos)<=set(idx) or not set(idx)<=set(train):
                raise RuntimeError('Sampling invariant violated')
        if not set(indices['ratio_3']) <= set(indices['ratio_4']):
            raise RuntimeError('Sampling nestedness violated')
        statistics = {key:stats(d,idx) for key,idx in indices.items()}
        statistics.update(validation=stats(d,val), test=stats(d,test))
        year_stats = {key:{str(year):stats(d,np.asarray(idx)[d['years'][idx]==year])
                          for year in sorted(set(d['years'][idx].tolist()))} for key,idx in indices.items()}
        m = dict(protocol=PROTOCOL, fold=fold, seed=42, dataset_sha256=dataset_sha,
                 train_indices=indices, val_indices=val.tolist(), test_indices=test.tolist(),
                 statistics=statistics, year_statistics=year_stats, matches=matches,
                 index_sha256={key:json_hash(idx) for key,idx in {**indices,'validation':val.tolist(),'test':test.tolist()}.items()},
                 negative_company_contribution=dict(sorted(contributions.items())),
                 size_fallback=dict(fallback), size_feature=str(d['feature_names'][j]),
                 size_period='latest prior FY Q4; observed and cutoff-screened; no extra fill',
                 sampling_rule='strict same year; lowest selected negative company count first; then known log-size distance; local RNG42 ties; first3of4 nested',
                 preprocessing_pool='full', pos_weight={key:statistics[key]['negatives']/statistics[key]['positives'] for key in SCHEMES})
        write_json(CACHE/f'fold_{fold}.json',m)
        result.append(statistics)
    return result


def risk3(np_now, np_prev, equity):
    if (equity is not None and equity<0) or (np_now is not None and np_prev is not None and np_now<0 and np_prev<0):
        return 1
    if equity is not None and equity>=0 and ((np_now is not None and np_now>=0) or (np_prev is not None and np_prev>=0)):
        return 0
    return -1


def carry_baseline(samples,d):
    amounts = pd.read_parquet(AMOUNTS)
    verified, document_hashes = {},{}
    for r in amounts.to_dict('records'):
        if r['original_disclosure_version_verified'] is not True or r['acceptance']!='accepted':
            continue
        if r['report_type']!='合并期末' or r['is_audit']!='是' or r['currency']!='CNY' or not np.isfinite(r['value']):
            continue
        period = f"{int(r['year'])}-12-31"
        if r['report_period']!=period or pd.isna(r['publish_date_as_reported']):
            continue
        source = ROOT/str(r['source_document_file'])
        digest = document_hashes.setdefault(str(source),sha(source))
        if digest!=r['response_sha256']:
            raise RuntimeError('Verified carry source document hash mismatch')
        r['available'] = max(str(r['publish_date_as_reported'])[:10],
                             (pd.Timestamp(period)+pd.DateOffset(months=6)).date().isoformat())
        key = (str(r['code']).zfill(6),int(r['year']),r['field'])
        if key in verified:
            raise RuntimeError('Ambiguous verified carry amount')
        verified[key]=r
    n=len(samples)
    prediction=np.full(n,-1,dtype=np.int8)
    reference=np.full(n,-1,dtype=np.int16)
    age=np.full(n,-1,dtype=np.int16)
    reason=np.full(n,'no_decisive_original_version_in_window',dtype='<U80')
    for i,s in enumerate(samples):
        lower=max(s['window_start'],s['listing_date'])
        def amount(year,field):
            r=verified.get((s['code'],year,field))
            if not r or not lower<=r['report_period']<=s['window_end'] or r['available']>s['cutoff']:
                return None
            return float(r['value'])
        for year in range(int(s['target_year'])-1,int(s['target_year'])-6,-1):
            state=risk3(amount(year,'NETPROFIT'),amount(year-1,'NETPROFIT'),amount(year,'RIGHAGGR'))
            if state>=0:
                prediction[i],reference[i],age[i]=state,year,int(s['target_year'])-year
                reason[i]='decisive_verified_original_report'
                break
    available=prediction>=0
    write_npz(CACHE/'carry_baseline.npz',prediction=prediction,available=available,
              reference_year=reference,reference_age_years=age,reason=reason)
    summary=dict(samples=n,coverage_count=int(available.sum()),coverage=float(available.mean()),
                 covered_positive_targets=int(d['labels'][available].sum()),
                 predicted_positive=int((prediction==1).sum()),reasons=dict(Counter(reason.tolist())),
                 reference_age_years=dict(Counter(age[available].astype(str).tolist())),
                 original_documents_sha256=document_hashes,
                 folds=[dict(fold=k,eligible=int((d['fold']==k).sum()),covered=int((available & (d['fold']==k)).sum()),
                             positive_targets=int(d['labels'][available & (d['fold']==k)].sum())) for k in range(5)],
                 interpretation='Strict version-verifiable continuation only; abstention is not nonrisk. No positive coverage means no evidence of risk detection.')
    write_json(CACHE/'carry_baseline.json',summary)
    return summary


def prepare_bert(docs):
    oldcache=ROOT/'cache'
    previous=read_json(oldcache/'bert_cache_manifest.json')
    previous_digest=read_json(oldcache/'bert_embedding_digest.json')
    checkpoint=oldcache/'bert-base-chinese'
    ident=previous['identity']
    if ident['pooling']!=POOLING or sha(checkpoint/'source.json')!=ident['checkpoint_source_sha256']:
        raise RuntimeError('Checkpoint source or pooling mismatch')
    for name,expected in ident['checkpoint_files'].items():
        if sha(checkpoint/name)!=expected:
            raise RuntimeError(f'Checkpoint changed: {name}')
    if sha(oldcache/'documents.jsonl')!=ident['documents_jsonl_sha256']:
        raise RuntimeError('Old document index changed')
    if previous_digest['manifest_identity']!=previous['identity_sha256'] or sha(oldcache/'bert_embeddings.npy')!=previous_digest['sha256']:
        raise RuntimeError('Old embedding identity mismatch')
    olddone=np.load(oldcache/'bert_complete.npy',mmap_mode='r')
    oldemb=np.load(oldcache/'bert_embeddings.npy',mmap_mode='r')
    olddocs=[json.loads(line) for line in (oldcache/'documents.jsonl').read_text().splitlines()]
    if not olddone.all() or oldemb.shape!=(len(olddocs),768) or not np.isfinite(oldemb).all():
        raise RuntimeError('Old embeddings are incomplete or invalid')
    oldlookup={(r['code'],r['report_period']):r['doc_id'] for r in olddocs}
    oldhashes={int(i):h for i,h in ident['document_content']}
    records,reused,new=[],[],[]
    for doc in docs:
        p=Path(doc['path']); st=p.stat(); digest=sha(p)
        oldid=oldlookup.get((doc['code'],doc['report_period']))
        if oldid is not None and digest!=oldhashes[oldid]:
            raise RuntimeError(f'Reused raw document changed: {p}')
        records.append(dict(doc_id=doc['doc_id'],path=str(p),bytes=st.st_size,
                            mtime_ns=st.st_mtime_ns,sha256=digest,old_doc_id=oldid))
        (reused if oldid is not None else new).append(doc['doc_id'])
    identity_data=dict(schema_version=1,documents_jsonl_sha256=sha(CACHE/'documents.jsonl'),
                       checkpoint_source_sha256=ident['checkpoint_source_sha256'],checkpoint_files=ident['checkpoint_files'],
                       pooling=POOLING,document_content=[[r['doc_id'],r['sha256']] for r in records],
                       reused_old_manifest_identity=previous['identity_sha256'],
                       reused_old_embeddings_sha256=previous_digest['sha256'],
                       reused_mapping=[[r['doc_id'],r['old_doc_id']] for r in records if r['old_doc_id'] is not None])
    manifest=dict(identity_sha256=json_hash(identity_data),identity=identity_data,documents=records,
                  reused_count=len(reused),new_count=len(new),new_doc_ids=new)
    if (CACHE/'bert_manifest.json').exists():
        prior=read_json(CACHE/'bert_manifest.json')
        if prior['identity_sha256']!=manifest['identity_sha256']:
            raise RuntimeError('Existing annual BERT cache input mismatch; refusing overwrite')
        if not EMBEDDINGS_PATH.exists() or not (CACHE/'bert_complete.npy').exists():
            raise RuntimeError('Incomplete annual BERT cache pair; explicit recovery required')
        return dict(reused=len(reused),new=len(new),total=len(docs))
    # Pair is initialized together; flags remain false until copied data is flushed.
    emb=np.lib.format.open_memmap(EMBEDDINGS_PATH,mode='w+',dtype=np.float32,shape=(len(docs),768))
    done=np.lib.format.open_memmap(CACHE/'bert_complete.npy',mode='w+',dtype=bool,shape=(len(docs),))
    done[:]=False; done.flush(); emb[:]=0
    for r in records:
        if r['old_doc_id'] is not None:
            emb[r['doc_id']]=oldemb[r['old_doc_id']]
    emb.flush(); done[reused]=True; done.flush()
    write_json(CACHE/'bert_manifest.json',manifest)
    return dict(reused=len(reused),new=len(new),total=len(docs))


def build():
    if not (CACHE/'preserved_identity.json').exists():
        raise RuntimeError('Parent preservation identity must exist before build')
    if (CACHE/'manifest.json').exists():
        m=validate_manifest()
        if m['source_sha256']!=source_hashes():
            raise RuntimeError('Frozen source identity changed; refusing rebuild')
        print(json.dumps(dict(existing=True,counts=m['counts']),ensure_ascii=False),flush=True)
        return m
    frame=pd.read_parquet(LABELS)
    known=frame[frame['risk_label'].notna()].copy().reset_index(drop=True)
    if len(frame)!=11206 or len(known)!=11199 or int(known.risk_label.sum())!=806:
        raise RuntimeError('Reviewed annual-label inventory changed')
    known['code']=known.code.astype(str).str.zfill(6)
    known['label']=known.risk_label.astype(int)
    known['previous_risk_ex_post']=known.previous_year_risk_ex_post.fillna(-1).astype(int)
    samples=known.to_dict('records')
    if known.duplicated(['code','cutoff']).any() or '601313' in set(known.code):
        raise RuntimeError('Candidate identity error')
    for s in samples:
        year=int(s['target_year'])
        if (not 2006<=year<=2024 or s['cutoff']!=f'{year}-06-30' or
                s['window_start']!=f'{year-5}-01-01' or s['window_end']!=f'{year-1}-12-31' or
                s['listing_date']>s['window_end'] or
                (pd.notna(s['delisting_date']) and s['delisting_date'] and s['delisting_date']<=s['cutoff'])):
            raise RuntimeError('Candidate eligibility or temporal definition mismatch')
    print('Reconstructing 11199 annual-risk windows',flush=True)
    d,docs,audit=reconstruct(samples)
    audit['old_intersection']=verify_old_intersection(d,docs)
    print(json.dumps(audit['old_intersection']),flush=True)
    write_npz(CACHE/'samples.npz',**d)
    (CACHE/'documents.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in docs),encoding='utf-8')
    mapping=known[['code','target_year','cutoff','listing_date','window_start','window_end','label','previous_risk_ex_post','transition_ex_post']].copy()
    mapping.insert(0,'sample_index',np.arange(len(mapping)))
    mapping['fold']=d['fold']
    mapping.to_csv(CACHE/'samples.csv',index=False,encoding='utf-8-sig')
    audit['fold_statistics']=make_folds(d,sha(CACHE/'samples.npz'))
    audit['carry']=carry_baseline(samples,d)
    print('Verifying and reusing frozen BERT cache',flush=True)
    audit['bert']=prepare_bert(docs)
    native_fin=d['observed'][:,:,:30].reshape(len(samples),5,4,30).any(axis=(2,3))
    native_gov=d['observed'][:,3::4,30:].any(axis=2)
    native_txt=(d['doc_ids'][:,1::2]>=0).reshape(len(samples),5,2).any(axis=2)
    audit['comparison_eligible']={key:dict(windows=int((a[:,-1]&a[:,:-1].any(1)).sum()),
                                         positives=int(d['labels'][a[:,-1]&a[:,:-1].any(1)].sum()))
                                  for key,a in [('financial',native_fin),('governance',native_gov),('text',native_txt)]}
    artifacts=['samples.npz','samples.csv','documents.jsonl','carry_baseline.npz','carry_baseline.json','bert_manifest.json']+[f'fold_{k}.json' for k in range(5)]
    manifest=dict(protocol=PROTOCOL,seed=42,source_sha256=source_hashes(),
                  artifact_sha256={name:sha(CACHE/name) for name in artifacts},
                  counts=dict(candidates=len(frame),samples=len(samples),positives=int(d['labels'].sum()),
                              unknown_excluded=int(frame.risk_label.isna().sum()),companies=len(set(d['codes'])),documents=len(docs)),
                  audit=audit,point_in_time_verified=False,
                  label='Annual consolidated audited RiskY=(NPY<0 and NPY-1<0) or equityY<0; three-valued unknown excluded from training; repeated risk included.',
                  target='June30Y forecasts fiscal-year Y; Y=2006..2024.',
                  sequence='20 reporting-quarter nodes in Y-5..Y-1. Node is report period, not a historic real-time forecast. No source before window or IPO.',
                  availability='max(report period +6 calendar months, known publication)<=sample cutoff. Original input vintages unverified; proxy does not establish real-time validity.',
                  ages='Quarters from source reporting period to prediction cutoff, not actual publication age. Latest prior-year Q4=2; earliest Q1=21; missing=0 with mask.',
                  missingness='No financial forward fill; normal annual governance and semiannual text holding only. Missing scheduled report/field resets. No imputation/scaling here.',
                  cross_validation='Five company groups rebalanced on annual labels; test=k, validation=(k+1)%5, training remaining3. No out-of-time claim.',
                  sampling='Full/1:3/1:4; all positives; strict same year; least-contributing negative company then size softmatch; no replacement; ratio3 first3of4; no future ST exclusion.',
                  preprocessing='Fit on original full fold training pool; actual selected N/P loss weight. Ex-post transition fields are for evaluation only.',
                  cohort='Uploaded company roster, not market-wide representative prevalence.',
                  carry='Only verifiable original reporting versions with decisive legal-window information; abstain otherwise. Five covered nonrisk rows cannot assess risk detection.')
    write_json(CACHE/'manifest.json',manifest)
    print(json.dumps(dict(counts=manifest['counts'],bert=audit['bert'],carry_coverage=audit['carry']['coverage_count'],
                          comparison_eligible=audit['comparison_eligible'],fold_groups=audit['fold_groups']),ensure_ascii=False),flush=True)
    return manifest


def finalize_bert():
    import fcntl
    with (CACHE/'bert_finalize.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        done=np.load(CACHE/'bert_complete.npy',mmap_mode='r')
        if not done.all():
            return False
        emb=np.load(EMBEDDINGS_PATH,mmap_mode='r')
        if not np.isfinite(emb).all():
            raise RuntimeError('Nonfinite completed BERT embeddings')
        manifest=read_json(CACHE/'bert_manifest.json')
        st=EMBEDDINGS_PATH.stat()
        completion=dict(manifest_identity=manifest['identity_sha256'],bytes=st.st_size,
                        mtime_ns=st.st_mtime_ns,sha256=sha(EMBEDDINGS_PATH),
                        complete_sha256=sha(CACHE/'bert_complete.npy'))
        write_json(CACHE/'bert_completion.json',completion)
        return True


def encode(worker=0,workers=2,device=None):
    import fcntl
    import torch
    from transformers import BertModel,BertTokenizerFast
    validate_manifest()
    if not 0<=worker<workers:
        raise ValueError('Worker must be in 0..workers-1')
    device=device or f'cuda:{worker}'
    lock=(CACHE/f'bert_worker_{worker}.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(6); torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    manifest=read_json(CACHE/'bert_manifest.json')
    local=ROOT/'cache/bert-base-chinese'
    for name,expected in manifest['identity']['checkpoint_files'].items():
        if sha(local/name)!=expected:
            raise RuntimeError('BERT checkpoint changed')
    if sha(local/'source.json')!=manifest['identity']['checkpoint_source_sha256']:
        raise RuntimeError('BERT checkpoint provenance changed')
    emb=np.load(EMBEDDINGS_PATH,mmap_mode='r+')
    done=np.load(CACHE/'bert_complete.npy',mmap_mode='r+')
    records=manifest['documents']
    new=set(manifest['new_doc_ids'])
    assignment,loads={},[0]*workers
    for r in sorted((r for r in records if r['doc_id'] in new),key=lambda r:(-r['bytes'],r['doc_id'])):
        owner=min(range(workers),key=lambda k:(loads[k],k))
        assignment[r['doc_id']]=owner;loads[owner]+=r['bytes']
    selected=[r for r in records if r['doc_id'] in new and assignment[r['doc_id']]==worker and not done[r['doc_id']]]
    if not selected:
        finalize_bert();return
    tokenizer=BertTokenizerFast.from_pretrained(local,local_files_only=True)
    model=BertModel.from_pretrained(local,local_files_only=True,attn_implementation='sdpa').to(device).eval()
    model.requires_grad_(False)
    batch_size=128
    pending,sums,counts,chunks_left=[],{},{},{}
    completed_count=0
    start=time.monotonic()
    def consume():
        nonlocal completed_count
        if not pending:return
        width=max(len(ids)+2 for _,ids in pending)
        ids_batch=np.zeros((len(pending),width),dtype=np.int64)
        attn=np.zeros_like(ids_batch);valid=np.zeros_like(ids_batch,dtype=bool)
        for b,(_,ids) in enumerate(pending):
            ids_batch[b,:len(ids)+2]=[tokenizer.cls_token_id]+ids+[tokenizer.sep_token_id]
            attn[b,:len(ids)+2]=1;valid[b,1:len(ids)+1]=True
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=str(device).startswith('cuda')):
            hidden=model(input_ids=torch.as_tensor(ids_batch,device=device),attention_mask=torch.as_tensor(attn,device=device)).last_hidden_state
            vectors=(hidden.float()*torch.as_tensor(valid,device=device).unsqueeze(-1)).sum(1).cpu().numpy()
        finished=[]
        for (docid,tokens),vector in zip(pending,vectors):
            sums[docid]+=vector;counts[docid]+=len(tokens);chunks_left[docid]-=1
            if chunks_left[docid]==0:
                vector=sums.pop(docid)/counts.pop(docid);del chunks_left[docid]
                if not np.isfinite(vector).all():raise RuntimeError('Invalid BERT vector')
                emb[docid]=vector;finished.append(docid);completed_count+=1
        emb.flush();done[finished]=True;done.flush();pending.clear()
        write_json(CACHE/f'bert_progress_{worker}.json',dict(worker=worker,workers=workers,selected=len(selected),
                   completed=completed_count,globally_completed=int(done.sum()),total=len(records),seconds=round(time.monotonic()-start,2)))
    for r in selected:
        raw=Path(r['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=r['sha256']:
            raise RuntimeError('Raw text changed since build')
        try:text=raw.decode('utf-8-sig')
        except UnicodeDecodeError:text=raw.decode('gb18030')
        tokens=tokenizer.encode(text,add_special_tokens=False,truncation=False,verbose=False)
        if not tokens:raise RuntimeError('Empty tokenized document')
        docid=r['doc_id'];chunks=[tokens[i:i+510] for i in range(0,len(tokens),510)]
        sums[docid]=np.zeros(768,dtype=np.float32);counts[docid]=0;chunks_left[docid]=len(chunks)
        for chunk in chunks:
            pending.append((docid,chunk))
            if len(pending)>=batch_size:consume()
    consume()
    complete=finalize_bert()
    print(json.dumps(dict(worker=worker,completed=completed_count,all_complete=complete)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['build','encode'])
    parser.add_argument('--worker',type=int,default=0)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--device')
    args=parser.parse_args()
    if args.command=='build':build()
    else:encode(args.worker,args.workers,args.device)


if __name__=='__main__':
    main()
