"""Isolated historical numeric inputs and company-cross-fitted linear residuals.

Importing this module performs no I/O and fits nothing. prepare_raw only builds
native H/X. prepare_fold explicitly fits three fixed ridge G models. Original
annual inputs, sampling manifests, preprocessing, BERT and outputs are read-only.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np

import data as legacy
import data_annual_risk as annual

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'cache/linear_history_v1'
ANNUAL_CACHE = ROOT / 'cache/annual_risk_v1'
PROTOCOL = 'linear-history-v1:Yminus10to6-to-Yminus5to1:company-inner3:ridge1e-4'
RIDGE = 1e-4
Q4 = np.asarray([3, 7, 11, 15, 19], dtype=np.int64)
FIN = np.arange(30, dtype=np.int64)
CAT = [31, 32, 51, 52, 53, 54, 58]
GOV = np.asarray([j for j in range(30, 61) if j not in CAT], dtype=np.int64)
DIMENSION = 2160


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def index_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype='<i8').tobytes()).hexdigest()


@contextmanager
def locked(name):
    import fcntl
    CACHE.mkdir(parents=True, exist_ok=True)
    with (CACHE / name).open('a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def write_json_new(path, value):
    path = Path(path)
    require(not path.exists(), f'Refusing to overwrite existing metadata: {path}')
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def write_npz_new(path, arrays):
    path = Path(path)
    require(not path.exists(), f'Refusing to overwrite existing arrays: {path}')
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def source_identity():
    """Hash actual input files; reject changed preserved originals before building."""
    preserved_path = CACHE / 'preserved_identity.json'
    preserved = read_json(preserved_path)['files']
    paths = [Path(__file__), ROOT / 'data.py', ROOT / 'data_annual_risk.py',
             ROOT / 'run.py', ROOT / 'run_persem.py', ROOT / 'run_annual_risk.py',
             ROOT / 'config.json', preserved_path, ANNUAL_CACHE / 'samples.npz',
             ANNUAL_CACHE / 'samples.csv', ANNUAL_CACHE / 'manifest.json',
             ANNUAL_CACHE / 'training_protocol.json']
    paths.extend(ROOT / 'datasets' / legacy.SOURCE_PATHS[key]
                 for key in ('financial', 'governance', 'financial_dates', 'governance_dates'))
    hashes = {}
    for path in paths:
        relative = str(path.relative_to(ROOT))
        actual = sha(path)
        if relative in preserved:
            require(actual == preserved[relative]['sha256'],
                    f'Preserved original changed: {relative}')
        hashes[relative] = actual
    return dict(protocol=PROTOCOL, source_sha256=hashes,
                financial_indices=FIN.tolist(), governance_indices=GOV.tolist(),
                categorical_indices=CAT, feature_dimension=DIMENSION,
                history='FY Y-10..Y-6; financial quarterly; governance annual Q4',
                recent='unchanged annual-risk cache; financial quarterly; governance Q4',
                availability='max(period+6 months,known publication)<=June30Y; node>=IPO',
                ages='report ages in years; financial (41-t)/4; governance (38-4*t)/4',
                ridge_lambda=RIDGE, identity_version=1)


def coverage(mask_f, mask_g, indices=None):
    if indices is not None:
        mask_f, mask_g = mask_f[indices], mask_g[indices]
    available = mask_f.any(axis=(1, 2)) | mask_g.any(axis=(1, 2))
    return dict(samples=int(len(available)), any_history=int(available.sum()),
                empty_history=int((~available).sum()),
                financial_cells=int(mask_f.sum()), governance_cells=int(mask_g.sum()),
                financial_samples=int(mask_f.any(axis=(1, 2)).sum()),
                governance_samples=int(mask_g.any(axis=(1, 2)).sum()))


def validate_raw(raw):
    n = len(raw['codes'])
    require(n == 11199, 'Annual cohort size changed; no samples may be dropped')
    require(np.array_equal(raw['sample_index'], np.arange(n)), 'Sample order changed')
    for modal, times, fields in (('financial', 20, 30), ('governance', 5, 24)):
        for prefix in ('H', 'X'):
            value = raw[f'{prefix}_{modal}']
            mask = raw[f'{prefix}_{modal}_mask']
            age = raw[f'{prefix}_{modal}_age']
            require(value.shape == mask.shape == age.shape == (n, times, fields),
                    f'Invalid native shape: {prefix}/{modal}')
            require(np.array_equal(mask, np.isfinite(value)), 'Mask/native value disagreement')
            require(np.isfinite(age).all() and (age[~mask] == 0).all(), 'Invalid age mask')
    available = raw['H_financial_mask'].any(axis=(1, 2)) | raw['H_governance_mask'].any(axis=(1, 2))
    require(int(available.sum()) == 4935 and int((~available).sum()) == 6264,
            'History coverage differs from the reviewed 4935/6264 cohort')
    require(np.array_equal(available, raw['history_available']), 'History availability changed')


def prepare_raw():
    """Build/load native H and unchanged X; no model or statistics are fitted."""
    with locked('raw.lock'):
        identity = source_identity()
        manifest_path, raw_path = CACHE / 'raw_manifest.json', CACHE / 'raw.npz'
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            require(manifest['identity'] == identity, 'Raw source/protocol identity changed')
            require(manifest['fingerprint'] == digest(identity), 'Invalid raw identity digest')
            require(sha(raw_path) == manifest['raw_sha256'], 'Raw cache bytes changed')
            raw = load_npz(raw_path)
            validate_raw(raw)
            return raw
        require(not raw_path.exists(), 'Incomplete raw cache preserved; inspect before rebuilding')
        d = annual.load_data()
        require(np.array_equal(d['categorical_indices'], CAT), 'Categorical schema changed')
        with (ANNUAL_CACHE / 'samples.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        n = len(d['codes'])
        require(len(rows) == n, 'CSV/cache cohort size mismatch')
        for i, row in enumerate(rows):
            require(int(row['sample_index']) == i and row['code'] == str(d['codes'][i])
                    and int(row['target_year']) == int(d['years'][i])
                    and row['cutoff'] == str(d['cutoffs'][i]), 'CSV/cache sample identity mismatch')
        dataset = ROOT / 'datasets'
        fin, fv, fa = legacy.load_table(dataset, 'financial', legacy.FINANCIAL, 'report_period')
        gov, gv, ga = legacy.load_table(dataset, 'governance', legacy.GOVERNANCE, 'period_end', 'year')
        matched_dates = dict(
            financial=legacy.add_known_dates(dataset, fin, fa, 'financial', legacy.FINANCIAL, 'report_period'),
            governance=legacy.add_known_dates(dataset, gov, ga, 'governance', legacy.GOVERNANCE, 'period_end'))
        fi = {(str(c), str(p)): i for i, (c, p) in enumerate(zip(fin.code, fin.report_period))}
        gi = {(str(c), int(y)): i for i, (c, y) in enumerate(zip(gov.code, gov.year))}
        hf = np.full((n, 20, 30), np.nan, dtype=np.float32)
        hg = np.full((n, 5, 24), np.nan, dtype=np.float32)
        hf_valid = np.zeros((n, 20), dtype=bool)
        hg_valid = np.zeros((n, 5), dtype=bool)
        hf_periods = np.empty((n, 20), dtype='<U10')
        hg_periods = np.empty((n, 5), dtype='<U10')
        late = dict(financial=0, governance=0)
        gov_local = GOV - 30
        for i, row in enumerate(rows):
            year, code = int(row['target_year']), row['code']
            cutoff = date.fromisoformat(row['cutoff']).toordinal()
            ipo = row['listing_date']
            for t in range(20):
                y, q = year - 10 + t // 4, t % 4
                period = f'{y}-{("03-31", "06-30", "09-30", "12-31")[q]}'
                hf_periods[i, t] = period
                if period < ipo:
                    continue
                hf_valid[i, t] = True
                index = fi.get((code, period))
                if index is not None:
                    ok = fa[index] <= cutoff
                    late['financial'] += int((~ok & np.isfinite(fv[index])).sum())
                    hf[i, t] = np.where(ok, fv[index], np.nan)
            for t in range(5):
                year_g = year - 10 + t
                period = f'{year_g}-12-31'
                hg_periods[i, t] = period
                if period < ipo:
                    continue
                hg_valid[i, t] = True
                index = gi.get((code, year_g))
                if index is not None:
                    ok = ga[index, gov_local] <= cutoff
                    native = gv[index, gov_local]
                    late['governance'] += int((~ok & np.isfinite(native)).sum())
                    hg[i, t] = np.where(ok, native, np.nan)
        hfm, hgm = np.isfinite(hf), np.isfinite(hg)
        require(not hfm[~hf_valid].any() and not hgm[~hg_valid].any(), 'Pre-IPO history observed')
        # Slice exactly the original cache; never reconstruct or alter X.
        xf = d['values'][:, :, :30].copy()
        xg = d['values'][:, Q4][:, :, GOV].copy()
        xfm = d['observed'][:, :, :30].copy()
        xgm = d['observed'][:, Q4][:, :, GOV].copy()
        raw = dict(H_financial=hf, H_governance=hg,
                   H_financial_mask=hfm, H_governance_mask=hgm,
                   H_financial_age=np.where(hfm, (41 - np.arange(20))[None, :, None] / 4., 0).astype(np.float32),
                   H_governance_age=np.where(hgm, (38 - 4 * np.arange(5))[None, :, None] / 4., 0).astype(np.float32),
                   H_financial_valid=hf_valid, H_governance_valid=hg_valid,
                   H_financial_periods=hf_periods, H_governance_periods=hg_periods,
                   X_financial=xf, X_governance=xg,
                   X_financial_mask=xfm, X_governance_mask=xgm,
                   X_financial_age=(d['ages'][:, :, :30] / 4.).astype(np.float32),
                   X_governance_age=(d['ages'][:, Q4][:, :, GOV] / 4.).astype(np.float32),
                   history_available=hfm.any(axis=(1, 2)) | hgm.any(axis=(1, 2)),
                   codes=d['codes'], cutoffs=d['cutoffs'], years=d['years'],
                   fold=d['fold'], sample_index=d['sample_index'], labels=d['labels'],
                   listing_dates=np.asarray([r['listing_date'] for r in rows], dtype='<U10'),
                   feature_names=d['feature_names'], financial_indices=FIN, governance_indices=GOV)
        validate_raw(raw)
        require(source_identity() == identity, 'Input identity changed while building history')
        write_npz_new(raw_path, raw)
        manifest = dict(identity=identity, fingerprint=digest(identity), raw_sha256=sha(raw_path),
                        coverage=coverage(hfm, hgm), matched_source_dates=matched_dates,
                        late_masked_cells=late, arrays={k: list(v.shape) for k, v in raw.items()},
                        sample_identity_sha256=digest(list(zip(d['codes'].tolist(), d['cutoffs'].tolist()))),
                        data_integrity='All 11199 annual samples retained; X copied from frozen cache')
        write_json_new(manifest_path, manifest)
        return raw


def outer_stats(raw, preprocessing):
    result = {}
    for modal, fields in (('financial', FIN), ('governance', GOV)):
        stats = [preprocessing['continuous'][str(raw['feature_names'][j])] for j in fields]
        result[modal] = {key: np.asarray([s[key] for s in stats], dtype=np.float64)
                         for key in ('mean', 'scale', 'median')}
        require((result[modal]['scale'] > 0).all(), 'Invalid frozen outer scale')
    return result


def packed(values, masks, ages):
    """All values (F then G), then all masks, then all ages; native time-major."""
    n = len(values[0])
    result = np.concatenate([part.reshape(n, -1) for group in (values, masks, ages)
                             for part in group], axis=1).astype(np.float32)
    require(result.shape == (n, DIMENSION) and np.isfinite(result).all(), 'Invalid packed features')
    return result


def history_features(raw, stats):
    values, masks, ages = [], [], []
    for modal in ('financial', 'governance'):
        h, mask = raw[f'H_{modal}'], raw[f'H_{modal}_mask']
        s = stats[modal]
        value = (np.where(mask, h, s['median']).astype(np.float64) - s['mean']) / s['scale']
        values.append(value)
        masks.append(mask)
        ages.append(raw[f'H_{modal}_age'])
    result = packed(values, masks, ages)
    # A completely unavailable branch must contribute exactly zero.
    result[~raw['history_available']] = 0
    return result


def fit_inner_stats(h, x, hm, xm, fit_indices):
    """Fit on H and native X observations of these two company groups only."""
    fields = h.shape[-1]
    means, scales, medians, counts = [], [], [], []
    for field in range(fields):
        hv, xv = h[fit_indices, :, field], x[fit_indices, :, field]
        observed = np.concatenate((hv[hm[fit_indices, :, field]], xv[xm[fit_indices, :, field]])).astype(np.float64)
        count = len(observed)
        mean = float(observed.mean()) if count else 0.
        scale = float(observed.std()) if count else 1.
        median = float(np.median(observed)) if count else 0.
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.
        means.append(mean); scales.append(scale); medians.append(median); counts.append(count)
    return dict(mean=np.asarray(means), scale=np.asarray(scales), median=np.asarray(medians),
                observed_count=np.asarray(counts, dtype=np.int64))


def generator_input(h, mask, age, stats):
    value = (np.where(mask, h, stats['median']).astype(np.float64) - stats['mean']) / stats['scale']
    return np.concatenate((value.transpose(0, 2, 1), mask.transpose(0, 2, 1),
                           age.transpose(0, 2, 1)), axis=-1).astype(np.float64)


def fit_generator(h, x, hm, xm, ha, fit_indices):
    """Shared variable-wise Linear(3*Th,Tx), exact masked ridge with free bias.

    One modal loss is SSE / total_observed_targets + lambda * sum(W**2).
    Therefore each output normal equation adds lambda*N_total to its W block,
    irrespective of the number of observed targets at that particular time.
    """
    stats = fit_inner_stats(h, x, hm, xm, fit_indices)
    z = generator_input(h[fit_indices], hm[fit_indices], ha[fit_indices], stats)
    target = (x[fit_indices].astype(np.float64) - stats['mean']) / stats['scale']
    valid = xm[fit_indices] & hm[fit_indices].any(axis=1)[:, None, :]
    support = valid.sum(axis=0).astype(np.int64)
    total = int(valid.sum())
    in_dim, out_dim = z.shape[-1], x.shape[1]
    weights = np.zeros((in_dim, out_dim), dtype=np.float64)
    bias = np.zeros(out_dim, dtype=np.float64)
    ridge = RIDGE * total
    losses, equations = [], []
    for t in range(out_dim):
        keep = valid[:, t, :].reshape(-1)
        count = int(keep.sum())
        if not count:
            losses.append(0.); equations.append(0.)
            continue
        design = z.reshape(-1, in_dim)[keep]
        y = target[:, t, :].reshape(-1)[keep]
        # Centering removes the unpenalized intercept analytically.
        zmean, ymean = design.mean(axis=0), float(y.mean())
        centered, centered_y = design - zmean, y - ymean
        gram = centered.T @ centered
        gram.flat[::in_dim + 1] += ridge
        rhs = centered.T @ centered_y
        weights[:, t] = np.linalg.solve(gram, rhs)
        bias[t] = ymean - zmean @ weights[:, t]
        error = design @ weights[:, t] + bias[t] - y
        losses.append(float(error @ error))
        equations.append(float(np.linalg.norm(gram @ weights[:, t] - rhs) / (1. + np.linalg.norm(rhs))))
    require(np.isfinite(weights).all() and np.isfinite(bias).all(), 'Nonfinite ridge fit')
    model = dict(weight=weights, bias=bias, support=support, **stats)
    report = dict(target_count=total, ridge_diagonal=ridge,
                  target_counts_by_time=support.sum(axis=1).tolist(),
                  supported_field_time_count=int((support > 0).sum()),
                  masked_mse=float(sum(losses) / total) if total else None,
                  weight_penalty=float(RIDGE * (weights ** 2).sum()),
                  normal_equation_relative_errors=equations,
                  observed_stat_counts=stats['observed_count'].tolist(),
                  input_dimension=in_dim, output_dimension=out_dim,
                  shared_across_variables=True, bias_penalized=False)
    return model, report


def residual_for_model(raw, modal, model, outer):
    h, hm = raw[f'H_{modal}'], raw[f'H_{modal}_mask']
    x, xm = raw[f'X_{modal}'], raw[f'X_{modal}_mask']
    z = generator_input(h, hm, raw[f'H_{modal}_age'], model)
    prediction_z = (z @ model['weight'] + model['bias']).transpose(0, 2, 1)
    prediction_native = prediction_z * model['scale'] + model['mean']
    mask = xm & hm.any(axis=1)[:, None, :] & (model['support'] > 0)[None, :, :]
    residual = np.zeros(x.shape, dtype=np.float64)
    np.subtract(x, prediction_native, out=residual, where=mask)
    residual /= outer['scale']
    age = np.where(mask, raw[f'X_{modal}_age'], 0.)
    require(np.isfinite(residual).all(), 'Nonfinite standardized residual')
    return residual, mask, age


def _fold_context(fold, raw, parent_protocol=None):
    import run_annual_risk as parent
    fold = int(fold)
    require(fold in range(5), 'Expected outer fold 0..4')
    protocol = parent.load_protocol() if parent_protocol is None else parent_protocol
    require(protocol == read_json(ANNUAL_CACHE / 'training_protocol.json'), 'Supplied parent protocol changed')
    _, preprocessing = parent.load_preprocessing(fold, protocol, full_verify=True)
    train = np.flatnonzero((raw['fold'] != fold) & (raw['fold'] != (fold + 1) % 5))
    val = np.flatnonzero(raw['fold'] == (fold + 1) % 5)
    test = np.flatnonzero(raw['fold'] == fold)
    old = read_json(ANNUAL_CACHE / f'fold_{fold}.json')
    require(np.array_equal(train, old['train_indices']['full'])
            and np.array_equal(val, old['val_indices']) and np.array_equal(test, old['test_indices']),
            'Outer partition differs from annual-risk protocol')
    groups = sorted(np.unique(raw['fold'][train]).tolist())
    require(len(groups) == 3, 'Outer full train must have exactly three company groups')
    require(set(raw['codes'][train]).isdisjoint(raw['codes'][val])
            and set(raw['codes'][train]).isdisjoint(raw['codes'][test])
            and set(raw['codes'][val]).isdisjoint(raw['codes'][test]), 'Company leakage in outer split')
    manifest = read_json(CACHE / 'raw_manifest.json')
    expected = dict(protocol=PROTOCOL, fold=fold, raw_fingerprint=manifest['fingerprint'],
                    raw_sha256=manifest['raw_sha256'], parent_fingerprint=protocol['fingerprint'],
                    preprocessing_sha256=sha(ANNUAL_CACHE / 'preprocessing' / f'fold_{fold}' / 'preprocessing.json'),
                    full_train_indices_sha256=index_hash(train), val_indices_sha256=index_hash(val),
                    test_indices_sha256=index_hash(test), inner_groups=groups,
                    feature_layout='720 values F20x30+G5x24,720 masks,720 ages in years',
                    dimension=DIMENSION, ridge_lambda=RIDGE)
    return expected, outer_stats(raw, preprocessing), train, val, test, groups


def validate_fold_arrays(arrays, raw, fold, groups):
    n = len(raw['codes'])
    shapes = dict(history_features=(n, DIMENSION), residual_features=(3, n, DIMENSION),
                  inner_assignment=(n,), history_available=(n,), residual_available=(3, n))
    for key, shape in shapes.items():
        require(arrays[key].shape == shape, f'Invalid fold feature shape: {key}')
        require(np.isfinite(arrays[key]).all(), f'Nonfinite fold feature: {key}')
    require(np.array_equal(arrays['history_available'], raw['history_available']), 'History availability changed')
    require((arrays['history_features'][~arrays['history_available']] == 0).all(), 'Empty history is nonzero')
    require((arrays['residual_features'][~arrays['residual_available']] == 0).all(), 'Empty residual is nonzero')
    expected_assignment = np.full(n, -1, dtype=np.int8)
    for inner, group in enumerate(groups):
        expected_assignment[raw['fold'] == group] = inner
    require(np.array_equal(arrays['inner_assignment'], expected_assignment), 'Cross-fit assignment mismatch')
    require((arrays['inner_assignment'][(raw['fold'] == fold) | (raw['fold'] == (fold + 1) % 5)] == -1).all(),
            'Validation/test assigned a training generator')


def _read_fold(fold, raw, expected, groups):
    folder = CACHE / f'fold_{int(fold)}'
    metadata = read_json(folder / 'metadata.json')
    require(metadata['identity'] == expected and metadata['fingerprint'] == digest(expected),
            'Fold source/data/protocol identity changed')
    for name, expected_sha in metadata['artifact_sha256'].items():
        require(sha(folder / name) == expected_sha, f'Fold artifact changed: {name}')
    arrays = load_npz(folder / 'features.npz')
    validate_fold_arrays(arrays, raw, int(fold), groups)
    return arrays, metadata


def prepare_fold(fold, parent_protocol=None):
    """Explicitly build one fold's H features and three inner G/residual caches."""
    raw = prepare_raw()
    fold = int(fold)
    with locked(f'fold_{fold}.lock'):
        expected, outer, train, val, test, groups = _fold_context(fold, raw, parent_protocol)
        folder = CACHE / f'fold_{fold}'
        if (folder / 'metadata.json').exists():
            return _read_fold(fold, raw, expected, groups)[1]
        require(not folder.exists() or not any(folder.iterdir()),
                'Incomplete fold cache preserved; inspect before rebuilding')
        assignment = np.full(len(raw['codes']), -1, dtype=np.int8)
        h_features = history_features(raw, outer)
        residuals, availability, inner_models, model_arrays = [], [], [], {}
        for inner, excluded_group in enumerate(groups):
            excluded = train[raw['fold'][train] == excluded_group]
            fit = train[raw['fold'][train] != excluded_group]
            assignment[excluded] = inner
            require(len(np.unique(raw['fold'][fit])) == 2, 'Generator must fit exactly two groups')
            require(set(raw['codes'][fit]).isdisjoint(raw['codes'][excluded]), 'Inner company leakage')
            require(set(fit).isdisjoint(val) and set(fit).isdisjoint(test), 'Held-out generator fit')
            values, masks, ages, reports = [], [], [], {}
            for modal in ('financial', 'governance'):
                model, report = fit_generator(raw[f'H_{modal}'], raw[f'X_{modal}'],
                                              raw[f'H_{modal}_mask'], raw[f'X_{modal}_mask'],
                                              raw[f'H_{modal}_age'], fit)
                for key, value in model.items():
                    model_arrays[f'inner_{inner}_{modal}_{key}'] = value
                residual, mask, age = residual_for_model(raw, modal, model, outer[modal])
                values.append(residual); masks.append(mask); ages.append(age)
                reports[modal] = report
            available = masks[0].any(axis=(1, 2)) | masks[1].any(axis=(1, 2))
            features = packed(values, masks, ages)
            features[~available] = 0
            residuals.append(features); availability.append(available)
            inner_models.append(dict(inner=inner, excluded_group=int(excluded_group),
                fit_groups=sorted(np.unique(raw['fold'][fit]).tolist()),
                fit_indices=fit.tolist(), excluded_indices=excluded.tolist(),
                fit_indices_sha256=index_hash(fit), excluded_indices_sha256=index_hash(excluded),
                fit_companies=sorted(set(raw['codes'][fit].tolist())),
                excluded_companies=sorted(set(raw['codes'][excluded].tolist())),
                statistics_scope='Only true native H and X cells from fit_indices; no excluded-company statistics',
                modalities=reports,
                residual_coverage={name: coverage(masks[0], masks[1], idx)
                                   for name, idx in (('fit', fit), ('excluded', excluded), ('validation', val), ('test', test))}))
        arrays = dict(history_features=h_features, residual_features=np.stack(residuals),
                      inner_assignment=assignment, history_available=raw['history_available'],
                      residual_available=np.stack(availability))
        validate_fold_arrays(arrays, raw, fold, groups)
        require(source_identity() == read_json(CACHE / 'raw_manifest.json')['identity'],
                'Input identity changed while fitting generators')
        folder.mkdir(parents=True, exist_ok=True)
        write_npz_new(folder / 'features.npz', arrays)
        write_npz_new(folder / 'models.npz', model_arrays)
        assigned_available = arrays['residual_available'][assignment[train], train]
        metadata = dict(identity=expected, fingerprint=digest(expected), fold=fold,
                        artifact_sha256={name: sha(folder / name) for name in ('features.npz', 'models.npz')},
                        inner_models=inner_models,
                        train_indices=train.tolist(), val_indices=val.tolist(), test_indices=test.tolist(),
                        outer_statistics={modal: {key: value.tolist() for key, value in stats.items()}
                                          for modal, stats in outer.items()},
                        history_coverage={name: coverage(raw['H_financial_mask'], raw['H_governance_mask'], idx)
                                          for name, idx in (('full_train', train), ('validation', val), ('test', test))},
                        assigned_train_residual_coverage=dict(samples=int(len(train)), available=int(assigned_available.sum()),
                                                             empty=int((~assigned_available).sum())),
                        prediction_rule='Train uses G that excludes its company group; validation/test classify each of three residuals then average probabilities',
                        residual_mask='field has observed H AND observed X target AND inner fit has >=1 valid H-to-X target at field/time',
                        residual_units='(native X - native inverse-standardized G prediction) / frozen outer full-train scale',
                        no_sample_dropped=True, classifier_sampling_independent=True)
        write_json_new(folder / 'metadata.json', metadata)
        return metadata


def load_fold(fold):
    """Read and fully validate an existing fold. Never fits missing generators."""
    raw = prepare_raw()
    expected, _, _, _, _, groups = _fold_context(int(fold), raw)
    return _read_fold(int(fold), raw, expected, groups)
