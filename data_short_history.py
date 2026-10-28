"""Eight-year numeric history to two-year cross-fitted residuals.

Imports perform no I/O. Preparation writes only cache/short_history_v1; old
sources, native inputs, annual folds, preprocessing and results stay read-only.
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import zipfile
import numpy as np
import data_linear_history as prior

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'cache/short_history_v1'
ANNUAL_CACHE = prior.ANNUAL_CACHE
PROTOCOL = 'short-history-v1:Yminus10to3-to-Yminus2to1:company-inner3:ridge1e-4'
RIDGE = prior.RIDGE
FIN, GOV, CAT = prior.FIN, prior.GOV, prior.CAT
DIMENSION = 864
CELLS = 288
require, read_json, digest, sha = prior.require, prior.read_json, prior.digest, prior.sha
index_hash, load_npz = prior.index_hash, prior.load_npz
write_json_new, write_npz_new = prior.write_json_new, prior.write_npz_new
coverage = prior.coverage


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


def legacy_identity():
    """Read and verify frozen predecessor identity without invoking preparation."""
    manifest_path = prior.CACHE / 'raw_manifest.json'
    protocol_path = prior.CACHE / 'protocol.json'
    manifest, protocol = read_json(manifest_path), read_json(protocol_path)
    require(manifest['identity'] == prior.source_identity(), 'Old raw source identity changed')
    require(manifest['fingerprint'] == digest(manifest['identity']), 'Invalid old raw fingerprint')
    payload = {k: v for k, v in protocol.items() if k != 'fingerprint'}
    require(protocol['fingerprint'] == digest(payload), 'Invalid old classifier protocol fingerprint')
    for name, expected in protocol['source_sha256'].items():
        require(sha(ROOT / name) == expected, 'Old frozen source changed: ' + name)
    require(protocol['parent_protocol_sha256'] == sha(ANNUAL_CACHE / 'training_protocol.json'),
            'Old parent protocol changed')
    with zipfile.ZipFile(prior.CACHE / 'source_snapshot.zip') as archive:
        for name in archive.namelist():
            require(archive.read(name) == (ROOT / name).read_bytes(), 'Old source snapshot changed: ' + name)
    paths = [prior.CACHE / 'raw.npz', manifest_path]
    paths.extend(prior.CACHE / f'fold_{f}' / 'metadata.json' for f in range(5))
    hashes = {}
    for path in paths:
        name = str(path.relative_to(ROOT)); actual = sha(path)
        require(actual == protocol['history_artifacts'][name], 'Old data artifact changed: ' + name)
        hashes[name] = actual
    require(hashes[str((prior.CACHE / 'raw.npz').relative_to(ROOT))] == manifest['raw_sha256'],
            'Old raw bytes changed')
    return dict(protocol_fingerprint=protocol['fingerprint'], protocol_sha256=sha(protocol_path),
                raw_fingerprint=manifest['fingerprint'], artifact_sha256=hashes,
                source_snapshot_sha256=sha(prior.CACHE / 'source_snapshot.zip'))


def source_identity():
    return dict(protocol=PROTOCOL, source_sha256={Path(__file__).name: sha(__file__)},
                legacy=legacy_identity(), financial_indices=FIN.tolist(), governance_indices=GOV.tolist(),
                categorical_indices=CAT, feature_dimension=DIMENSION, ridge_lambda=RIDGE,
                history='H=old H5 plus first three old X years: financial32quarters/governance8years',
                target='last two old X years: financial8quarters/governance2years',
                availability='original native values/masks/ages unchanged; available by June30Y; no strict historical vintage claim',
                temporal_interpretation='conditional historical expectation, not a forecast claimed issued before target period',
                construction='predictor conditions only on H values/masks/ages, target used only in training loss and residual',
                layout='288 residual values F8x30 then G2x24, 288 masks, 288 ages in years', identity_version=1)


def _derive_raw(old):
    """Pure slicing/concatenation, no statistics, interpolation or source rereads."""
    prior.validate_raw(old)
    raw = {key: old[key].copy() for key in ('codes', 'cutoffs', 'years', 'fold', 'sample_index',
           'labels', 'listing_dates', 'feature_names', 'financial_indices', 'governance_indices')}
    for modal, prefix_steps in (('financial', 12), ('governance', 3)):
        for suffix in ('', '_mask', '_age'):
            raw[f'H_{modal}{suffix}'] = np.concatenate((old[f'H_{modal}{suffix}'],
                                                       old[f'X_{modal}{suffix}'][:, :prefix_steps]), axis=1)
            raw[f'X_{modal}{suffix}'] = old[f'X_{modal}{suffix}'][:, prefix_steps:].copy()
    for modal, step in (('financial', 4), ('governance', 1)):
        for prefix, start, length in (('H', -10, 8), ('X', -2, 2)):
            periods = np.empty((len(raw['codes']), length * step), dtype='<U10')
            for t in range(length * step):
                ends = ('03-31', '06-30', '09-30', '12-31') if step == 4 else ('12-31',)
                periods[:, t] = [f'{int(y)+start+t//step}-{ends[t%step]}' for y in raw['years']]
            raw[f'{prefix}_{modal}_periods'] = periods
            raw[f'{prefix}_{modal}_valid'] = periods >= raw['listing_dates'][:, None]
    raw['legacy_history_available'] = old['history_available'].copy()
    raw['history_available'] = raw['H_financial_mask'].any(axis=(1, 2)) | raw['H_governance_mask'].any(axis=(1, 2))
    masks = [raw[f'X_{m}_mask'] & raw[f'H_{m}_mask'].any(axis=1)[:, None, :]
             for m in ('financial', 'governance')]
    raw['potential_residual_available'] = masks[0].any(axis=(1, 2)) | masks[1].any(axis=(1, 2))
    validate_raw(raw)
    return raw


def validate_raw(raw):
    n = len(raw['codes'])
    require(n == 11199 and int(raw['labels'].sum()) == 806, 'Annual cohort/label count changed')
    require(np.array_equal(raw['sample_index'], np.arange(n)), 'Annual sample order changed')
    for modal, htimes, xtimes, fields in (('financial', 32, 8, 30), ('governance', 8, 2, 24)):
        for prefix, times in (('H', htimes), ('X', xtimes)):
            value, mask, age = (raw[f'{prefix}_{modal}{suffix}'] for suffix in ('', '_mask', '_age'))
            require(value.shape == mask.shape == age.shape == (n, times, fields), 'Invalid native shape: ' + prefix + modal)
            require(mask.dtype == np.bool_ and np.array_equal(mask, np.isfinite(value)), 'Native/mask disagreement')
            require(np.isfinite(age).all() and (age[~mask] == 0).all() and (age >= 0).all(), 'Invalid observed ages')
            periods, valid = raw[f'{prefix}_{modal}_periods'], raw[f'{prefix}_{modal}_valid']
            require(periods.shape == valid.shape == (n, times), 'Invalid native temporal metadata')
            require(not mask[~valid].any(), 'Observed native values before IPO')
            require((periods < raw['cutoffs'][:, None]).all(), 'Period not before prediction cutoff')
        require((raw[f'H_{modal}_periods'][:, -1] < raw[f'X_{modal}_periods'][:, 0]).all(), 'H overlaps target')
    history = raw['H_financial_mask'].any(axis=(1, 2)) | raw['H_governance_mask'].any(axis=(1, 2))
    require(np.array_equal(raw['history_available'], history), 'History availability mismatch')
    require(int(history.sum()) == 8249, 'Reviewed eight-year history coverage changed')
    legacy = raw['legacy_history_available']
    require(legacy.shape == (n,) and legacy.dtype == np.bool_ and int(legacy.sum()) == 4935,
            'Old five-year history coverage changed')
    require(not (legacy & ~history).any(), 'Old history lost by concatenation')
    possible = np.zeros(n, dtype=bool)
    for modal in ('financial', 'governance'):
        possible |= (raw[f'X_{modal}_mask'] & raw[f'H_{modal}_mask'].any(axis=1)[:, None, :]).any(axis=(1, 2))
    require(np.array_equal(possible, raw['potential_residual_available']) and int(possible.sum()) == 8249,
            'Potential residual coverage changed')


def _read_raw(identity):
    manifest = read_json(CACHE / 'raw_manifest.json')
    require(manifest['identity'] == identity and manifest['fingerprint'] == digest(identity), 'New raw identity changed')
    require(sha(CACHE / 'raw.npz') == manifest['raw_sha256'], 'New raw bytes changed')
    raw = load_npz(CACHE / 'raw.npz'); validate_raw(raw)
    return raw


def load_raw():
    """Read existing cache only; never creates or prepares absent data."""
    return _read_raw(source_identity())


def prepare_raw():
    with locked('raw.lock'):
        identity = source_identity()
        if (CACHE / 'raw_manifest.json').exists(): return _read_raw(identity)
        require(not (CACHE / 'raw.npz').exists(), 'Partial new raw cache retained; inspect before rebuilding')
        old = load_npz(prior.CACHE / 'raw.npz'); raw = _derive_raw(old)
        require(source_identity() == identity, 'Sources changed during native slicing')
        write_npz_new(CACHE / 'raw.npz', raw)
        manifest = dict(identity=identity, fingerprint=digest(identity), raw_sha256=sha(CACHE / 'raw.npz'),
                        arrays={k: list(v.shape) for k, v in raw.items()},
                        coverage=coverage(raw['H_financial_mask'], raw['H_governance_mask']),
                        coverage_strata=dict(legacy_history=4935, newly_available_history=3314, unavailable_history=2950),
                        potential_residual_available=int(raw['potential_residual_available'].sum()),
                        sample_identity_sha256=digest(list(zip(raw['codes'].tolist(), raw['cutoffs'].tolist()))),
                        data_integrity='All 11199 windows retained; native values/masks/ages sliced verbatim from frozen old raw')
        write_json_new(CACHE / 'raw_manifest.json', manifest)
        return raw


def packed(values, masks, ages):
    n = len(values[0])
    result = np.concatenate([part.reshape(n, -1) for group in (values, masks, ages)
                             for part in group], axis=1).astype(np.float32)
    require(result.shape == (n, DIMENSION) and np.isfinite(result).all(), 'Invalid short residual layout')
    return result


def _fold_context(fold, raw, parent_protocol=None):
    import run_annual_risk as parent
    fold = int(fold); require(fold in range(5), 'Expected outer fold 0..4')
    protocol = parent.load_protocol() if parent_protocol is None else parent_protocol
    require(protocol == read_json(ANNUAL_CACHE / 'training_protocol.json'), 'Parent protocol changed')
    _, preprocessing = parent.load_preprocessing(fold, protocol, full_verify=True)
    train = np.flatnonzero((raw['fold'] != fold) & (raw['fold'] != (fold + 1) % 5))
    val = np.flatnonzero(raw['fold'] == (fold + 1) % 5); test = np.flatnonzero(raw['fold'] == fold)
    old_path = ANNUAL_CACHE / f'fold_{fold}.json'; old = read_json(old_path)
    require(np.array_equal(train, old['train_indices']['full']) and np.array_equal(val, old['val_indices'])
            and np.array_equal(test, old['test_indices']), 'Outer annual partition changed')
    groups = sorted(np.unique(raw['fold'][train]).tolist()); require(len(groups) == 3, 'Expected three training company groups')
    require(set(raw['codes'][train]).isdisjoint(raw['codes'][val])
            and set(raw['codes'][train]).isdisjoint(raw['codes'][test])
            and set(raw['codes'][val]).isdisjoint(raw['codes'][test]), 'Outer company leakage')
    legacy_path = prior.CACHE / f'fold_{fold}' / 'metadata.json'; previous = read_json(legacy_path)
    require(previous['identity']['inner_groups'] == groups
            and np.array_equal(previous['train_indices'], train) and np.array_equal(previous['val_indices'], val)
            and np.array_equal(previous['test_indices'], test), 'Predecessor group split changed')
    manifest = read_json(CACHE / 'raw_manifest.json')
    expected = dict(protocol=PROTOCOL, fold=fold, raw_fingerprint=manifest['fingerprint'], raw_sha256=manifest['raw_sha256'],
                    parent_fingerprint=protocol['fingerprint'], annual_fold_manifest_sha256=sha(old_path),
                    legacy_fold_metadata_sha256=sha(legacy_path),
                    preprocessing_sha256=sha(ANNUAL_CACHE / 'preprocessing' / f'fold_{fold}' / 'preprocessing.json'),
                    full_train_indices_sha256=index_hash(train), val_indices_sha256=index_hash(val), test_indices_sha256=index_hash(test),
                    inner_groups=groups, feature_layout='288 values F8x30+G2x24,288 masks,288 ages in years',
                    dimension=DIMENSION, ridge_lambda=RIDGE)
    return expected, prior.outer_stats(raw, preprocessing), train, val, test, groups


def validate_fold_arrays(arrays, raw, fold, groups):
    n = len(raw['codes'])
    shapes = dict(residual_features=(3, n, DIMENSION), inner_assignment=(n,), history_available=(n,),
                  legacy_history_available=(n,), residual_available=(3, n))
    require(set(arrays) == set(shapes), 'Unexpected fold array inventory')
    for key, shape in shapes.items():
        require(arrays[key].shape == shape and np.isfinite(arrays[key]).all(), 'Invalid fold array: ' + key)
    for key in ('history_available', 'legacy_history_available'):
        require(np.array_equal(arrays[key], raw[key]), 'Changed history coverage: ' + key)
    features = arrays['residual_features']; masks = features[:, :, CELLS:2*CELLS]
    require(((masks == 0) | (masks == 1)).all(), 'Nonbinary residual masks')
    valid = masks.astype(bool)
    require(np.array_equal(arrays['residual_available'], valid.any(axis=2)), 'Residual availability disagreement')
    require((features[:, :, :CELLS][~valid] == 0).all() and (features[:, :, 2*CELLS:][~valid] == 0).all(),
            'Missing residual value/age is nonzero')
    require((features[~arrays['residual_available']] == 0).all(), 'Unavailable branch nonzero')
    require(not arrays['residual_available'][:, ~raw['potential_residual_available']].any(), 'Impossible residual support')
    assignment = np.full(n, -1, dtype=np.int8)
    for inner, group in enumerate(groups): assignment[raw['fold'] == group] = inner
    require(np.array_equal(arrays['inner_assignment'], assignment), 'Inner company assignment changed')
    require((assignment[(raw['fold'] == fold) | (raw['fold'] == (fold+1)%5)] == -1).all(), 'Held-out inner assignment')


def _read_fold(fold, raw, expected, groups):
    folder = CACHE / f'fold_{fold}'; metadata = read_json(folder / 'metadata.json')
    require(metadata['identity'] == expected and metadata['fingerprint'] == digest(expected), 'New fold identity changed')
    require(set(metadata['artifact_sha256']) == {'features.npz', 'models.npz'}, 'Fold artifact inventory changed')
    for name, expected_sha in metadata['artifact_sha256'].items():
        require(sha(folder / name) == expected_sha, 'New fold artifact changed: ' + name)
    arrays = load_npz(folder / 'features.npz'); validate_fold_arrays(arrays, raw, fold, groups)
    return arrays, metadata


def prepare_fold(fold, parent_protocol=None):
    raw = prepare_raw(); fold = int(fold)
    with locked(f'fold_{fold}.lock'):
        expected, outer, train, val, test, groups = _fold_context(fold, raw, parent_protocol)
        folder = CACHE / f'fold_{fold}'
        if (folder / 'metadata.json').exists(): return _read_fold(fold, raw, expected, groups)[1]
        require(not folder.exists() or not any(folder.iterdir()), 'Partial fold cache retained; inspect before rebuilding')
        assignment = np.full(len(raw['codes']), -1, dtype=np.int8)
        residuals, availability, inner_models, model_arrays = [], [], [], {}
        previous = read_json(prior.CACHE / f'fold_{fold}' / 'metadata.json')
        for inner, excluded_group in enumerate(groups):
            excluded = train[raw['fold'][train] == excluded_group]; fit = train[raw['fold'][train] != excluded_group]
            assignment[excluded] = inner; old_inner = previous['inner_models'][inner]
            require(old_inner['excluded_group'] == excluded_group and np.array_equal(old_inner['fit_indices'], fit)
                    and np.array_equal(old_inner['excluded_indices'], excluded), 'Legacy inner group split changed')
            require(len(np.unique(raw['fold'][fit])) == 2 and set(raw['codes'][fit]).isdisjoint(raw['codes'][excluded]), 'Inner company leakage')
            require(set(fit).isdisjoint(val) and set(fit).isdisjoint(test), 'Held-out generator fitting')
            values, masks, ages, reports = [], [], [], {}
            for modal in ('financial', 'governance'):
                model, report = prior.fit_generator(raw[f'H_{modal}'], raw[f'X_{modal}'], raw[f'H_{modal}_mask'],
                                                   raw[f'X_{modal}_mask'], raw[f'H_{modal}_age'], fit)
                for key, value in model.items(): model_arrays[f'inner_{inner}_{modal}_{key}'] = value
                residual, mask, age = prior.residual_for_model(raw, modal, model, outer[modal])
                values.append(residual); masks.append(mask); ages.append(age); reports[modal] = report
            available = masks[0].any(axis=(1, 2)) | masks[1].any(axis=(1, 2))
            features = packed(values, masks, ages); features[~available] = 0
            residuals.append(features); availability.append(available)
            inner_models.append(dict(inner=inner, excluded_group=int(excluded_group),
                fit_groups=sorted(np.unique(raw['fold'][fit]).tolist()), fit_indices=fit.tolist(), excluded_indices=excluded.tolist(),
                fit_indices_sha256=index_hash(fit), excluded_indices_sha256=index_hash(excluded),
                fit_companies=sorted(set(raw['codes'][fit].tolist())), excluded_companies=sorted(set(raw['codes'][excluded].tolist())),
                statistics_scope='Only native H8 and X2 cells of fit_indices; no held-out-company statistics',
                modalities=reports, residual_coverage={name: coverage(masks[0], masks[1], idx)
                    for name, idx in (('fit', fit), ('excluded', excluded), ('validation', val), ('test', test))}))
        arrays = dict(residual_features=np.stack(residuals), inner_assignment=assignment,
                      history_available=raw['history_available'], legacy_history_available=raw['legacy_history_available'],
                      residual_available=np.stack(availability))
        validate_fold_arrays(arrays, raw, fold, groups)
        require(source_identity() == read_json(CACHE / 'raw_manifest.json')['identity'], 'Sources changed during generator fitting')
        folder.mkdir(parents=True, exist_ok=True)
        write_npz_new(folder / 'features.npz', arrays); write_npz_new(folder / 'models.npz', model_arrays)
        assigned = arrays['residual_available'][assignment[train], train]
        metadata = dict(identity=expected, fingerprint=digest(expected), fold=fold,
            artifact_sha256={name: sha(folder/name) for name in ('features.npz', 'models.npz')}, inner_models=inner_models,
            train_indices=train.tolist(), val_indices=val.tolist(), test_indices=test.tolist(),
            outer_statistics={modal: {key: value.tolist() for key, value in stats.items()} for modal, stats in outer.items()},
            history_coverage={name: coverage(raw['H_financial_mask'], raw['H_governance_mask'], idx)
                              for name, idx in (('full_train', train), ('validation', val), ('test', test))},
            assigned_train_residual_coverage=dict(samples=int(len(train)), available=int(assigned.sum()), empty=int((~assigned).sum())),
            coverage_strata=dict(legacy_history=4935, newly_available_history=3314, unavailable_history=2950),
            prediction_rule='Train uses own excluded-company G; validation/test classify each of three residual views and average probabilities',
            residual_mask='observed H in field AND observed X2 target AND inner fit support for field/time',
            residual_units='(native target - native inverse-standardized G prediction)/frozen outer full-train scale',
            predictor_parameters_per_inner=int(96*8+8+24*2+2),
            no_sample_dropped=True, classifier_sampling_independent=True)
        write_json_new(folder / 'metadata.json', metadata)
        return metadata


def load_fold(fold):
    """Read and validate already prepared cache, never fit missing generators."""
    raw = load_raw(); expected, _, _, _, _, groups = _fold_context(int(fold), raw)
    return _read_fold(int(fold), raw, expected, groups)


def data_files():
    paths = [CACHE / 'raw.npz', CACHE / 'raw_manifest.json']
    paths.extend(CACHE / f'fold_{f}' / name for f in range(5) for name in ('features.npz', 'models.npz', 'metadata.json'))
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}
