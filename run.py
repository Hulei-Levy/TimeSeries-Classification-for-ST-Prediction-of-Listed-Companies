"""Independent ST classification experiment runner; see README for protocol."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / 'cache/huggingface'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ['PYTORCH_KERNEL_CACHE_PATH'] = str(ROOT/'cache/torch/kernels')
(ROOT/'cache/torch/kernels').mkdir(parents=True, exist_ok=True)
MODELS = ['DLinear', 'TiRex', 'XLSTMMixer', 'LightTS', 'Pyraformer',
          'TimeFilter', 'TimesNet', 'Reformer', 'Informer', 'Autoformer',
          'ETSformer', 'FEDformer', 'TimeMoE', 'SEMPO']
DEFAULTS = dict(seed=42, epochs=100, patience=15, batch_size=64,
                learning_rate=0.001, weight_decay=0.0001, dim=64,
                depth=2, dropout=0.1, text_dim=64, threshold=0.5,
                gradient_clip=1.0, bert_batch_size=128, torch_threads=6,
                auxiliary_loss_weight=0.01, graph_microbatch_size=16)

def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                              allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def config():
    c = DEFAULTS.copy()
    p = ROOT / 'config.json'
    if p.exists():
        c.update(json.loads(p.read_text(encoding='utf-8')))
    return c

def log(event, **kw):
    print(json.dumps(dict(event=event, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',
                         time.gmtime()), **kw), ensure_ascii=False), flush=True)

def documents():
    with (ROOT / 'cache/documents.jsonl').open(encoding='utf-8') as f:
        docs = [json.loads(line) for line in f if line.strip()]
    assert all(int(d['doc_id']) == i for i, d in enumerate(docs))
    return docs

def data():
    return dict(np.load(ROOT / 'cache/samples.npz', allow_pickle=False))

def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()

def file_identity(path, old=None):
    """Reuse a content hash only when both file size and nanosecond mtime match."""
    stat = Path(path).stat()
    if (old and old.get('bytes') == stat.st_size
            and old.get('mtime_ns') == stat.st_mtime_ns):
        digest = old['sha256']
    else:
        digest = file_sha256(path)
    return dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=digest)

def current_bert_manifest(docs, previous=None):
    previous = previous or {}
    old_documents = {int(item['doc_id']): item
                     for item in previous.get('documents', [])}
    document_records = []
    for doc in docs:
        path = Path(doc['path'])
        if not path.exists():
            path = ROOT / str(path).split('The_Latest_ST_Project/', 1)[1]
        old = old_documents.get(int(doc['doc_id']))
        if old and old.get('path') != doc['path']:
            old = None
        document_records.append(dict(doc_id=int(doc['doc_id']), path=doc['path'],
                                     **file_identity(path, old)))
    source_path = ROOT / 'cache/bert-base-chinese/source.json'
    source = json.loads(source_path.read_text(encoding='utf-8'))
    checkpoint_records = {}
    for name, expected in sorted(source['files'].items()):
        record = file_identity(source_path.parent / name,
                               previous.get('checkpoint_files', {}).get(name))
        if record['sha256'] != expected['sha256']:
            raise RuntimeError(f'BERT checkpoint checksum mismatch: {name}')
        checkpoint_records[name] = record
    identity = dict(
        schema_version=1,
        documents_jsonl_sha256=file_sha256(ROOT/'cache/documents.jsonl'),
        checkpoint_source_sha256=file_sha256(source_path),
        checkpoint_files={name: item['sha256'] for name,item in checkpoint_records.items()},
        pooling='v1:frozen-eval-last-hidden;510-content-token-nonoverlapping-chunks;'
                'exclude-CLS-SEP-PAD;content-token-mean;token-count-weighted-document-mean;'
                'float32-accumulation;cuda-float16-forward',
        document_content=[[item['doc_id'], item['sha256']] for item in document_records],
    )
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True,
                                  separators=(',', ':')).encode()).hexdigest()
    return dict(identity_sha256=identity_hash, identity=identity,
                documents=document_records, checkpoint_files=checkpoint_records)

def initialize_embeddings():
    docs = documents()
    manifest_path = ROOT/'cache/bert_cache_manifest.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    manifest = current_bert_manifest(docs, previous)
    if previous and previous['identity_sha256'] != manifest['identity_sha256']:
        raise RuntimeError('BERT cache inputs changed (documents, checkpoint, or pooling). '
                           'Refusing to silently reuse or overwrite cached encodings.')
    path, status = ROOT/'cache/bert_embeddings.npy', ROOT/'cache/bert_complete.npy'
    pair_complete = path.exists() and status.exists()
    if pair_complete:
        embedded = np.load(path, mmap_mode='r')
        done = np.load(status, mmap_mode='r')
        if embedded.shape != (len(docs), 768) or embedded.dtype != np.float32:
            raise RuntimeError('BERT embeddings have an invalid shape or dtype.')
        if done.shape != (len(docs),) or done.dtype != np.bool_:
            raise RuntimeError('BERT completion markers have an invalid shape or dtype.')
        if previous is None and done.any():
            raise RuntimeError('Existing completed embeddings lack a provenance manifest; '
                               'refusing an unverifiable cache migration.')
        del embedded, done
    else:
        # Reset BOTH files. In particular, never pair new zero vectors with old True flags.
        # Clear completion first so interruption cannot expose stale completion markers.
        status_tmp = status.with_name(status.name + f'.{os.getpid()}.tmp')
        with status_tmp.open('wb') as stream:
            np.save(stream, np.zeros(len(docs), dtype='bool'), allow_pickle=False)
        status_tmp.replace(status)
        path_tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
        a = np.lib.format.open_memmap(path_tmp, mode='w+', dtype='float32',
                                      shape=(len(docs), 768))
        a[:] = 0
        a.flush()
        del a
        path_tmp.replace(path)
        log('bert_cache_pair_reset', documents=len(docs))
    if previous != manifest:
        atomic_json(manifest_path, manifest)
    return manifest

def completed_embedding_identity(manifest):
    """Persist the final vector checksum; subsequent folds only check file metadata."""
    if not np.load(ROOT/'cache/bert_complete.npy', mmap_mode='r').all():
        raise RuntimeError('BERT embeddings are incomplete; cannot fingerprint training.')
    path = ROOT/'cache/bert_embeddings.npy'
    record_path = ROOT/'cache/bert_embedding_digest.json'
    previous = json.loads(record_path.read_text()) if record_path.exists() else None
    if previous and previous.get('manifest_identity') != manifest['identity_sha256']:
        previous = None
    record = file_identity(path, previous)
    if not previous or record != {key: previous[key] for key in record}:
        vectors = np.load(path, mmap_mode='r')
        if not np.isfinite(vectors).all() or not np.any(vectors != 0, axis=1).all():
            raise RuntimeError('Completed BERT cache contains nonfinite or all-zero vectors.')
        del vectors
        atomic_json(record_path, dict(manifest_identity=manifest['identity_sha256'], **record))
    return record['sha256']

def encode(worker=0, workers=2, device='cuda:0', limit=0):
    """Each document is owned by exactly one worker; disjoint mmap writes."""
    import torch
    from transformers import BertModel, BertTokenizerFast
    c = config()
    torch.set_num_threads(c['torch_threads'])
    seed_all(c['seed'])
    docs = documents()
    emb = np.load(ROOT / 'cache/bert_embeddings.npy', mmap_mode='r+')
    done = np.load(ROOT / 'cache/bert_complete.npy', mmap_mode='r+')
    manifest = json.loads((ROOT/'cache/bert_cache_manifest.json').read_text())
    sizes = {item['doc_id']: item['bytes'] for item in manifest['documents']}
    loads, assignment = [0]*workers, {}
    for doc in sorted(docs, key=lambda d: (-sizes[d['doc_id']], d['doc_id'])):
        owner = min(range(workers), key=lambda k: (loads[k], k))
        assignment[doc['doc_id']] = owner
        loads[owner] += sizes[doc['doc_id']]
    selected = [d for d in docs if assignment[d['doc_id']] == worker
                and not done[d['doc_id']]]
    if limit:
        selected = [selected[i] for i in np.linspace(0, len(selected)-1,
                    min(limit, len(selected)), dtype=int)] if selected else []
    local = ROOT / 'cache/bert-base-chinese'
    tokenizer = BertTokenizerFast.from_pretrained(local, local_files_only=True)
    model = BertModel.from_pretrained(local, local_files_only=True,
                                     attn_implementation='sdpa').to(device).eval()
    model.requires_grad_(False)
    batch_size = c['bert_batch_size']
    pending, sums, counts, total_chunks = [], {}, {}, {}
    n_tokens = n_chunks = n_docs = 0
    started = time.monotonic()

    def consume():
        nonlocal n_tokens, n_chunks, n_docs
        if not pending:
            return
        width = max(len(ids) + 2 for _, ids in pending)
        ids_batch = np.zeros((len(pending), width), dtype='int64')
        attn = np.zeros_like(ids_batch)
        valid = np.zeros_like(ids_batch, dtype='bool')
        for b, (_, ids) in enumerate(pending):
            ids_batch[b, :len(ids)+2] = [tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id]
            attn[b, :len(ids)+2] = 1
            valid[b, 1:len(ids)+1] = True
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16,
                                                   enabled=str(device).startswith('cuda')):
            output = model(input_ids=torch.as_tensor(ids_batch, device=device),
                           attention_mask=torch.as_tensor(attn, device=device)).last_hidden_state
            # Sum content tokens only: excludes CLS/SEP/PAD. Accumulate in float32.
            sums_batch = (output.float() * torch.as_tensor(valid, device=device)
                           .unsqueeze(-1)).sum(1).cpu().numpy()
        completed = []
        for (doc_id, ids), vec in zip(pending, sums_batch):
            sums[doc_id] += vec
            counts[doc_id] += len(ids)
            total_chunks[doc_id] -= 1
            n_tokens += len(ids)
            n_chunks += 1
            if total_chunks[doc_id] == 0:
                result = sums.pop(doc_id) / counts.pop(doc_id)
                assert np.isfinite(result).all()
                emb[doc_id] = result
                del total_chunks[doc_id]
                completed.append(doc_id)
                n_docs += 1
        # Data must be durable before completion markers are durable.
        emb.flush()
        done[completed] = True
        done.flush()
        pending.clear()
        elapsed = time.monotonic() - started
        progress = dict(worker=worker, documents_completed=n_docs,
                        documents_this_run=len(selected), chunks=n_chunks,
                        tokens=n_tokens, seconds=round(elapsed, 2),
                        tokens_per_second=round(n_tokens/max(elapsed, 1), 1),
                        globally_completed=int(done.sum()), total_documents=len(docs))
        atomic_json(ROOT / f'cache/bert_progress_{worker}.json', progress)
        if n_chunks % (batch_size * 10) == 0 or n_docs == len(selected):
            log('bert_progress', **progress)

    for doc in selected:
        doc_id = int(doc['doc_id'])
        path = Path(doc['path'])
        if not path.exists():
            marker = 'The_Latest_ST_Project/'
            path = ROOT / str(path).split(marker, 1)[1]
        raw = path.read_bytes()
        try:
            text = raw.decode('utf-8-sig')
        except UnicodeDecodeError:
            text = raw.decode('gb18030')
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False,
                                  verbose=False)
        if not tokens:
            raise ValueError(f'Empty tokenized document {doc_id}: {path}')
        chunks = [tokens[i:i+510] for i in range(0, len(tokens), 510)]
        total_chunks[doc_id] = len(chunks)
        sums[doc_id] = np.zeros(768, dtype='float32')
        counts[doc_id] = 0
        for ids in chunks:
            pending.append((doc_id, ids))
            if len(pending) >= batch_size:
                consume()
    consume()
    log('bert_worker_finished', worker=worker, documents=n_docs,
        seconds=round(time.monotonic()-started, 2))

def fingerprint():
    manifest = initialize_embeddings()
    embedding_sha256 = completed_embedding_identity(manifest)
    h = hashlib.sha256()
    for name in ['config.json', 'data.py', 'run.py', 'models_classic.py',
                 'models_recent.py', 'cache/samples.npz', 'cache/documents.jsonl']:
        p = ROOT / name
        h.update(name.encode())
        if p.exists():
            with p.open('rb') as f:
                for block in iter(lambda: f.read(1024*1024), b''):
                    h.update(block)
    h.update(b'bert_cache_manifest_identity')
    h.update(manifest['identity_sha256'].encode())
    h.update(b'completed_bert_embeddings_sha256')
    h.update(embedding_sha256.encode())
    return h.hexdigest()

def partition(d, fold):
    test = d['fold'] == fold
    val = d['fold'] == (fold+1) % 5
    train = ~(test | val)
    code_sets = [set(d['codes'][i]) for i in [train, val, test]]
    assert not (code_sets[0] & code_sets[1] or code_sets[0] & code_sets[2]
                or code_sets[1] & code_sets[2])
    assert all(d['labels'][i].sum() > 0 for i in [train, val, test])
    return train, val, test

def preprocess(d, train):
    """Fold-local continuous statistics and categorical vocabulary; no imputer on held-out data."""
    x = d['values'].astype('float32', copy=True)
    valid = d['valid_time'].astype(bool)
    x[~valid] = np.nan
    cat = set(map(int, d['categorical_indices']))
    parts, state = [], {'continuous': {}, 'categorical': {}}
    for j in range(x.shape[-1]):
        observed = x[train, :, j]
        observed = observed[np.isfinite(observed)].astype('float64')
        feature = str(d['feature_names'][j])
        if j in cat:
            vocab = np.unique(observed).tolist()
            # Separate missing and unseen-category indicators.
            one_hot = np.zeros((*x.shape[:2], len(vocab)+2), dtype='float32')
            finite = np.isfinite(x[:, :, j])
            matched = np.zeros(x.shape[:2], dtype=bool)
            for k, value in enumerate(vocab):
                match = finite & (x[:, :, j] == value)
                one_hot[:, :, k] = match
                matched |= match
            one_hot[:, :, -2] = ~finite
            one_hot[:, :, -1] = finite & ~matched
            parts.append(one_hot)
            state['categorical'][feature] = {'vocabulary': vocab,
                                             'missing_index': len(vocab),
                                             'unseen_index': len(vocab)+1}
        else:
            median = float(np.median(observed)) if len(observed) else 0.0
            mean = float(observed.mean()) if len(observed) else 0.0
            scale = float(observed.std()) if len(observed) else 1.0
            if not np.isfinite(scale) or scale < 1e-8:
                scale = 1.0
            values = np.where(np.isfinite(x[:, :, j]), x[:, :, j], median)
            values = ((values.astype('float64')-mean)/scale).astype('float32')
            parts.append(values[..., None])
            state['continuous'][feature] = dict(median=median, mean=mean,
                                                scale=scale, observed_count=len(observed))
    observed_mask = np.isfinite(x).astype('float32')
    # Ages are cutoff minus source report quarter, scaled to years, not disclosure ages.
    ages = np.where(np.isfinite(d['ages']), d['ages'], 0).astype('float32')/4.0
    text_available = (d['doc_ids'] >= 0).astype('float32')
    text_age = np.where(text_available.astype(bool), d['doc_ages'], 0).astype('float32')/4.0
    parts.extend([observed_mask, ages, text_available[..., None],
                  text_age[..., None], valid.astype('float32')[..., None]])
    out = np.concatenate(parts, axis=-1)
    out[~valid] = 0
    assert np.isfinite(out).all()
    state['base_input_dimension'] = int(out.shape[-1])
    state['fit_scope'] = 'training companies only; observed valid window cells'
    return np.ascontiguousarray(out), state

def build_model(name, input_dim, embeddings, c, device):
    import torch
    from torch import nn
    from models_classic import build_backbone as classic
    from models_recent import build_backbone as recent

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('text_vectors', torch.as_tensor(embeddings,
                                 dtype=torch.float32), persistent=False)
            self.text_projection = nn.Linear(768, c['text_dim'])
            kwargs = dict(input_dim=input_dim+c['text_dim'], seq_len=20,
                          dim=c['dim'], depth=c['depth'], dropout=c['dropout'])
            try:
                backbone = classic(name, **kwargs)
            except (KeyError, ValueError):
                backbone = None
            if backbone is None:
                backbone = recent(name, **kwargs)
            if backbone is None:
                raise KeyError(name)
            self.backbone = backbone
            self.head = nn.Sequential(nn.LayerNorm(c['dim']), nn.Linear(c['dim'], 1))

        def forward(self, numeric, doc_ids, valid_time):
            available = (doc_ids >= 0).unsqueeze(-1)
            text = self.text_projection(self.text_vectors[doc_ids.clamp_min(0)])
            text = text * available * valid_time.unsqueeze(-1)
            x = torch.cat([numeric, text], dim=-1)
            representation = self.backbone(x, valid_time=valid_time)
            return self.head(representation).squeeze(-1)

        def auxiliary_loss(self):
            method = getattr(self.backbone, 'aux_loss', None)
            if callable(method):
                return method()
            return getattr(self.backbone, 'last_aux_loss', 0.0)

    return Classifier().to(device)

def scores(y, p, threshold=0.5):
    from sklearn.metrics import (accuracy_score, average_precision_score,
        roc_auc_score, precision_score, recall_score, f1_score,
        matthews_corrcoef, brier_score_loss, confusion_matrix)
    predicted = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    k = max(1, math.ceil(len(y)*0.1))
    top = np.argsort(-p, kind='stable')[:k]
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        sel = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if sel.any():
            ece += float(sel.mean()*abs(y[sel].mean()-p[sel].mean()))
    return dict(n=len(y), positive=int(y.sum()), prevalence=float(y.mean()),
                accuracy=float(accuracy_score(y, predicted)),
                precision=float(precision_score(y, predicted, zero_division=0)),
                recall=float(recall_score(y, predicted, zero_division=0)),
                f1=float(f1_score(y, predicted, zero_division=0)),
                specificity=float(tn/max(tn+fp, 1)), mcc=float(matthews_corrcoef(y, predicted)),
                roc_auc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)), ece_10_bins=ece,
                top_decile_capture=float(y[top].sum()/max(y.sum(), 1)),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), threshold=threshold)

def train_one(name, fold, device='cuda:0', smoke=False):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    c = config()
    torch.set_num_threads(c['torch_threads'])
    seed_all(c['seed'])
    d = data()
    train, val, test = partition(d, fold)
    x, prep = preprocess(d, train)
    if smoke:
        # Explicitly synthetic engineering check, never reported as a model result.
        embeddings = np.random.default_rng(c['seed']).normal(0, 0.1,
                     (max(1, len(documents())), 768)).astype('float32')
    else:
        assert np.load(ROOT/'cache/bert_complete.npy').all(), 'BERT encoding is incomplete'
        embeddings = np.load(ROOT/'cache/bert_embeddings.npy')
    model = build_model(name, x.shape[-1], embeddings, c, device)
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'],
                                 weight_decay=c['weight_decay'])
    positive = int(d['labels'][train].sum())
    weight = float(train.sum()-positive)/positive
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(d['doc_ids'].astype('int64')),
                            torch.from_numpy(d['valid_time'].astype(bool)),
                            torch.from_numpy(d['labels'].astype('float32')))

    def loader(mask, shuffle=False):
        indices = np.flatnonzero(mask)
        if smoke:
            indices = indices[:4]
        subset = torch.utils.data.Subset(dataset, indices.tolist())
        batch = c['batch_size']
        if name == 'TimeFilter' and not shuffle:
            batch = c['graph_microbatch_size']
        return DataLoader(subset, batch_size=batch if not smoke else 2,
                          shuffle=shuffle, num_workers=0, pin_memory=True,
                          generator=torch.Generator().manual_seed(c['seed']))

    def predict(mask):
        model.eval()
        values = []
        with torch.inference_mode():
            for xb, db, vb, _ in loader(mask):
                out = model(xb.to(device), db.to(device), vb.to(device))
                values.append(out.sigmoid().cpu().numpy())
        return np.concatenate(values)

    folder = ROOT / 'results' / name / f'fold_{fold}'
    if not smoke:
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(folder/'preprocessing.json', prep)
        runhash = fingerprint()
        atomic_json(folder/'started.json', dict(model=name, fold=fold, config=c,
                    fingerprint=runhash, train_n=int(train.sum()), val_n=int(val.sum()),
                    test_n=int(test.sum()), pos_weight=weight, trainable_parameters=nparams))
    best_ap, best_epoch, best_state = -1.0, -1, None
    started = time.monotonic()
    train_loader = loader(train, shuffle=True)
    for epoch in range(1 if smoke else c['epochs']):
        model.train()
        total_loss, n_seen = 0.0, 0
        for xb, db, vb, yb in train_loader:
            xb, db, vb, yb = [v.to(device, non_blocking=True) for v in [xb, db, vb, yb]]
            optimizer.zero_grad(set_to_none=True)
            micro = c['graph_microbatch_size'] if name == 'TimeFilter' else len(yb)
            loss_sum = 0.0
            for lo in range(0, len(yb), micro):
                sl = slice(lo, min(lo+micro, len(yb)))
                logits = model(xb[sl], db[sl], vb[sl])
                loss = loss_fn(logits, yb[sl])
                aux = model.auxiliary_loss()
                if aux is not None:
                    loss = loss + c['auxiliary_loss_weight']*aux
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'{name} fold={fold} nonfinite loss')
                (loss * len(yb[sl])/len(yb)).backward()
                loss_sum += float(loss.detach())*len(yb[sl])
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), c['gradient_clip'],
                                                 error_if_nonfinite=True)
            if smoke:
                assert float(grad) > 0, 'No gradient'
                assert model.text_projection.weight.grad is not None
            optimizer.step()
            total_loss += loss_sum
            n_seen += len(yb)
        pval = predict(val)
        if smoke:
            assert np.isfinite(pval).all()
            log('smoke_pass', model=name, parameters=nparams, input_dimension=x.shape[-1]+64)
            return
        from sklearn.metrics import average_precision_score
        ap = float(average_precision_score(d['labels'][val], pval))
        if ap > best_ap:
            best_ap, best_epoch = ap, epoch
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save(dict(state_dict=best_state, model=name, fold=fold,
                            config=c, epoch=epoch+1, validation_ap=ap,
                            fingerprint=runhash), folder/'best.pt')
        row = dict(epoch=epoch+1, train_loss=total_loss/n_seen, validation_ap=ap,
                   best_validation_ap=best_ap, best_epoch=best_epoch+1)
        with (folder/'epochs.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row)+'\n')
        log('epoch', model=name, fold=fold, **row)
        if epoch-best_epoch >= c['patience']:
            break
    model.load_state_dict(best_state)
    p = predict(test)
    assert np.isfinite(p).all()
    result = scores(d['labels'][test], p, c['threshold'])
    result.update(model=name, fold=fold, best_epoch=best_epoch+1,
                  best_validation_ap=best_ap, parameters=nparams,
                  seconds=round(time.monotonic()-started, 2), fingerprint=runhash)
    with (folder/'predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_index','code','cutoff','label','probability','fold'])
        for idx, prob in zip(np.flatnonzero(test), p):
            writer.writerow([idx, str(d['codes'][idx]), str(d['cutoffs'][idx]),
                             int(d['labels'][idx]), float(prob), fold])
    atomic_json(folder/'metrics.json', result)
    log('fold_finished', **result)

def summarize():
    rows, overall = [], []
    for name in MODELS:
        folds = []
        predictions = []
        for fold in range(5):
            folder = ROOT/'results'/name/f'fold_{fold}'
            if (folder/'metrics.json').exists():
                m = json.loads((folder/'metrics.json').read_text())
                folds.append(m)
                rows.append(m)
                with (folder/'predictions.csv').open() as f:
                    predictions.extend(list(csv.DictReader(f)))
        if len(folds) == 5:
            assert len({f['fingerprint'] for f in folds}) == 1, 'Mixed experiment versions'
            indices = [int(p['sample_index']) for p in predictions]
            assert len(set(indices)) == len(indices) == len(data()['labels'])
            aggregate = {'model': name, 'completed_folds': 5}
            for key in ['accuracy','precision','recall','f1','specificity','mcc',
                        'roc_auc','ap','brier','ece_10_bins','top_decile_capture']:
                numbers = np.array([f[key] for f in folds])
                aggregate[key+'_mean'] = float(numbers.mean())
                aggregate[key+'_std'] = float(numbers.std(ddof=1))
            y = np.array([int(p['label']) for p in predictions])
            p = np.array([float(p['probability']) for p in predictions])
            aggregate['pooled_oof'] = scores(y, p)
            overall.append(aggregate)
    atomic_json(ROOT/'results/summary.json', dict(folds=rows, models=overall,
                completed_runs=len(rows), expected_runs=len(MODELS)*5,
                std_definition='sample SD across dependent CV folds, not a confidence interval'))
    if overall:
        fields = [k for k in overall[0] if k != 'pooled_oof']
        with (ROOT/'results/summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(overall)
    log('summary', completed_runs=len(rows), complete_models=len(overall))

def worker_train(worker, workers=2):
    for index, (name, fold) in enumerate((m,f) for m in MODELS for f in range(5)):
        if index % workers != worker:
            continue
        path = ROOT/'results'/name/f'fold_{fold}'/'metrics.json'
        if path.exists():
            existing = json.loads(path.read_text())
            if existing['fingerprint'] != fingerprint():
                raise RuntimeError('Existing results use another data/code/config fingerprint')
            continue
        # Isolated process per fold resets all model state and releases GPU allocations.
        result = subprocess.run([sys.executable, '-u', str(ROOT/'run.py'), 'train',
                                 '--model', name, '--fold', str(fold),
                                 '--device', f'cuda:{worker}'])
        if result.returncode:
            raise RuntimeError(f'Training failed: {name} fold {fold}, exit {result.returncode}')

def supervise(skip_encode=False):
    (ROOT/'logs').mkdir(exist_ok=True)
    initialize_embeddings()
    if not skip_encode:
        processes, streams = [], []
        for worker in range(2):
            stream = (ROOT/f'logs/bert_gpu{worker}.log').open('a', buffering=1)
            streams.append(stream)
            processes.append(subprocess.Popen([sys.executable, '-u', str(ROOT/'run.py'),
                'encode', '--worker', str(worker), '--workers', '2',
                '--device', f'cuda:{worker}'], stdout=stream, stderr=subprocess.STDOUT))
        exit_codes = [p.wait() for p in processes]
        for stream in streams:
            stream.close()
        if any(exit_codes):
            raise RuntimeError(f'BERT worker failures: {exit_codes}')
    assert np.load(ROOT/'cache/bert_complete.npy').all()
    log('all_bert_complete')
    for name in MODELS:
        subprocess.run([sys.executable, '-u', str(ROOT/'run.py'), 'smoke',
                        '--model', name, '--device', 'cuda:0'], check=True)
    log('all_smoke_passed')
    processes, streams = [], []
    for worker in range(2):
        stream = (ROOT/f'logs/train_gpu{worker}.log').open('a', buffering=1)
        streams.append(stream)
        processes.append(subprocess.Popen([sys.executable, '-u', str(ROOT/'run.py'),
            'worker', '--worker', str(worker)], stdout=stream, stderr=subprocess.STDOUT))
    exit_codes = [p.wait() for p in processes]
    for stream in streams:
        stream.close()
    summarize()
    if any(exit_codes):
        raise RuntimeError(f'Training worker failures: {exit_codes}')
    log('experiment_complete')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['init','encode','smoke','train','worker','pipeline','summary'])
    p.add_argument('--worker', type=int, default=0)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--model', choices=MODELS)
    p.add_argument('--fold', type=int, choices=range(5), default=0)
    p.add_argument('--skip-encode', action='store_true')
    a = p.parse_args()
    if a.command == 'init':
        initialize_embeddings()
    elif a.command == 'encode':
        encode(a.worker, a.workers, a.device, a.limit)
    elif a.command in ('train','smoke'):
        for name in [a.model] if a.model else MODELS:
            train_one(name, a.fold, a.device, smoke=a.command=='smoke')
    elif a.command == 'worker':
        worker_train(a.worker, a.workers)
    elif a.command == 'summary':
        summarize()
    else:
        supervise(a.skip_encode)

if __name__ == '__main__':
    try:
        main()
    except Exception:
        log('fatal', traceback=traceback.format_exc())
        raise
