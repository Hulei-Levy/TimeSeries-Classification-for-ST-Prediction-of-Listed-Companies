"""Fixed, nested training-only matched-control sampling for the new ST project.

All positives are retained. Five controls per positive are assigned globally by
prediction year and log asset size, with at most one negative window per company.
The three closest controls among those five define the nested 1:3 condition.
Validation/test windows and the source sample cache are never modified here.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

__all__ = ['ensure_manifests', 'load_training_mask']
SIZE_FEATURE = 'AF_CFEATUREPROFILE-CompanySize'
SCHEMA_VERSION = 1
PROTOCOL = {
    'version': 'matched-negatives-v1', 'seed': 42, 'folds': 5,
    'ratios': [3, 5], 'retain_all_training_positives': True,
    'year_rule': 'strictly identical June-30 prediction year',
    'size_feature': SIZE_FEATURE, 'size_native_node': 19,
    'size_rule': 'abs(log(positive assets)-log(negative assets)); positive finite observed values only',
    'missing_size_rule': 'skip size distance (cost zero), retain strict year restriction',
    'negative_company_cap': 1, 'replacement': False,
    'positive_companies_may_supply_earlier_negative_windows': True,
    'assignment': 'global linear_sum_assignment; 5 slots per positive; negative-company columns',
    'ties': 'local default_rng(42), shuffled assignment rows/columns, then seeded control tie ranks',
    'nested_rule': 'ratio 3 uses the 3 nearest controls within each positive sample\'s assigned 5',
    'infeasible_rule': 'raise; never relax year, company cap, or replacement constraints',
    'preprocessing_fit_scope': 'original full training pool, unchanged from the baseline fold',
    'loss_weight_scope': 'runner recomputes negative/positive ratio from selected training windows',
    'availability_scope': 'use only the existing cutoff-screened sample cache; actual PIT vintages remain unverified',
}


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def _sha_object(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _index_sha(indices):
    return hashlib.sha256(np.asarray(indices, dtype='<i8').tobytes()).hexdigest()


def _matching_signature(d):
    names = np.asarray(d['feature_names']).astype(str)
    locations = np.flatnonzero(names == SIZE_FEATURE)
    if len(locations) != 1:
        raise ValueError(f'Exactly one {SIZE_FEATURE} field is required.')
    j = int(locations[0])
    arrays = {
        'labels': np.asarray(d['labels'], dtype='<i8'),
        'codes': np.asarray(d['codes']).astype('<U6'),
        'cutoffs': np.asarray(d['cutoffs']).astype('<U10'),
        'fold': np.asarray(d['fold'], dtype='<i8'),
        'feature_names': names,
        'size': np.asarray(d['values'])[:, 19, j].astype('<f8'),
        'size_observed': np.asarray(d['observed'])[:, 19, j].astype(bool),
        'size_age': np.asarray(d['ages'])[:, 19, j].astype('<f8'),
        'size_valid_time': np.asarray(d['valid_time'])[:, 19].astype(bool),
        'size_period': np.asarray(d['periods'])[:, 19].astype('<U10'),
    }
    digest = hashlib.sha256()
    for key, value in arrays.items():
        digest.update(key.encode())
        digest.update(str(value.shape).encode())
        if value.dtype.kind == 'U':
            digest.update(_canonical(value.tolist()))
        else:
            digest.update(value.tobytes())
    return digest.hexdigest(), j, arrays


def _context(root, d):
    root = Path(root).resolve()
    cache_path = root / 'cache/samples.npz'
    signature, size_index, a = _matching_signature(d)
    with np.load(cache_path, allow_pickle=False) as source:
        source_signature, _, _ = _matching_signature(source)
    if signature != source_signature:
        raise ValueError('Passed sample arrays differ from the immutable source cache.')
    labels, codes, cutoffs, groups = a['labels'], a['codes'], a['cutoffs'], a['fold']
    n = len(labels)
    if any(len(a[key]) != n for key in a if key != 'feature_names'):
        raise ValueError('Sample metadata lengths differ.')
    if not np.isin(labels, [0, 1]).all() or not np.isin(groups, range(5)).all():
        raise ValueError('Invalid binary labels or company-fold assignments.')
    if not all(len(c) == 6 and c.isdigit() for c in codes):
        raise ValueError('Company identifiers must be six-digit codes.')
    if not all(c.endswith('-06-30') and len(c) == 10 for c in cutoffs):
        raise ValueError('This protocol requires June-30 prediction cutoffs.')
    years = np.array([int(c[:4]) for c in cutoffs], dtype=np.int32)
    expected_period = np.array([f'{y-1}-12-31' for y in years])
    if not np.array_equal(a['size_period'], expected_period):
        raise ValueError('Size must come from the final native annual Q4 node.')
    company_groups, company_years = {}, set()
    for code, year, group in zip(codes, years, groups):
        previous = company_groups.setdefault(str(code), int(group))
        if previous != int(group):
            raise ValueError('A company appears in different CV groups.')
        key = (str(code), int(year))
        if key in company_years:
            raise ValueError('More than one candidate window exists for a company-year.')
        company_years.add(key)
    observable = a['size_observed'] & a['size_valid_time'] & np.isfinite(a['size'])
    if np.any(observable & (a['size_age'] != 2)):
        raise ValueError('Latest native annual size must be two report quarters before cutoff.')
    size_ok = observable & (a['size'] > 0)
    log_size = np.zeros(n, dtype=np.float64)
    log_size[size_ok] = np.log(a['size'][size_ok])
    return dict(root=root, n=n, labels=labels, codes=codes, cutoffs=cutoffs,
                groups=groups, years=years, size=a['size'], size_ok=size_ok,
                log_size=log_size, size_index=size_index,
                dataset_sha256=_sha_file(cache_path), matching_arrays_sha256=signature,
                protocol_sha256=_sha_object(PROTOCOL))


def _roles(ctx, fold):
    test = ctx['groups'] == fold
    validation = ctx['groups'] == (fold + 1) % 5
    train = ~(test | validation)
    return train, validation, test


def _stats(ctx, indices):
    indices = np.asarray(indices, dtype=np.int64)
    n = len(indices)
    positives = int(ctx['labels'][indices].sum())
    return dict(windows=n, companies=len(set(ctx['codes'][indices])), positives=positives,
                negatives=n-positives, prevalence=positives/n if n else None)


def _stable_key_sha(ctx, indices):
    lines = [f'{i}|{ctx["codes"][i]}|{ctx["cutoffs"][i]}|{ctx["labels"][i]}\n' for i in indices]
    return hashlib.sha256(''.join(lines).encode('utf-8')).hexdigest()


def _year_stats(ctx, indices):
    indices = np.asarray(indices, dtype=np.int64)
    result = {}
    for year in sorted(set(ctx['years'])):
        subset = indices[ctx['years'][indices] == year]
        positives = int(ctx['labels'][subset].sum())
        result[str(year)] = dict(windows=len(subset), positives=positives,
                                 negatives=len(subset)-positives)
    return result


def _distance(ctx, positive, negative):
    if not (ctx['size_ok'][positive] and ctx['size_ok'][negative]):
        return 0.0
    return float(abs(ctx['log_size'][positive] - ctx['log_size'][negative]))


def _sample_record(ctx, index):
    return dict(sample_index=int(index), code=str(ctx['codes'][index]),
                cutoff=str(ctx['cutoffs'][index]), year=int(ctx['years'][index]),
                size=float(ctx['size'][index]) if ctx['size_ok'][index] else None,
                size_available=bool(ctx['size_ok'][index]))


def _selection_metadata(ctx, fold, ratio, manifest):
    indices = manifest['indices']
    selected = indices[f'selected_{ratio}']
    negatives = indices[f'negative_{ratio}']
    positive_indices = indices['positive']
    splits = dict(train_full=indices['train_full'], train_selected=selected,
                  validation=indices['validation'], test=indices['test'])
    metadata = dict(schema_version=SCHEMA_VERSION, protocol_version=PROTOCOL['version'],
                    fold=fold, seed=PROTOCOL['seed'], ratio_requested=ratio,
                    ratio_actual=len(negatives)/len(positive_indices),
                    dataset_sha256=ctx['dataset_sha256'], protocol_sha256=ctx['protocol_sha256'],
                    **{name: _stats(ctx, rows) for name, rows in splits.items()})
    metadata['index_sha256'] = {name: _index_sha(rows) for name, rows in splits.items()}
    metadata['stable_keys_sha256'] = {name: _stable_key_sha(ctx, rows) for name, rows in splits.items()}
    metadata['year_statistics'] = {name: _year_stats(ctx, rows) for name, rows in splits.items()}
    negative_counts = Counter(str(ctx['codes'][i]) for i in negatives)
    selected_counts = Counter(str(ctx['codes'][i]) for i in selected)
    positive_codes = {str(ctx['codes'][i]) for i in positive_indices}
    metadata['company_repetition'] = dict(
        negative_company_cap=1, maximum_negative_windows_per_company=max(negative_counts.values(), default=0),
        unique_negative_companies=len(negative_counts),
        maximum_total_windows_per_company=max(selected_counts.values(), default=0),
        companies_with_multiple_selected_windows=sum(v > 1 for v in selected_counts.values()),
        positive_negative_company_overlap=len(positive_codes & set(negative_counts)),
        positive_companies_are_not_excluded_from_earlier_negative_candidates=True)
    matched = []
    fallback_pairs = 0
    for entry in manifest['matches']:
        controls = []
        for control in entry['controls']:
            if ratio == 5 or control['in_ratio3']:
                fallback_pairs += int(control['size_distance_skipped'])
                controls.append(control)
        matched.append(dict(positive=entry['positive'], controls=controls))
    metadata['matched_positives'] = matched
    metadata['size_matching'] = dict(
        field=SIZE_FEATURE, feature_index=ctx['size_index'], native_quarter_node=19,
        unit='yuan; historical opening-assets proxy, not market capitalization',
        positive_size_missing=int((~ctx['size_ok'][positive_indices]).sum()),
        selected_negative_size_missing=int((~ctx['size_ok'][negatives]).sum()),
        full_training_negative_size_missing=int((~ctx['size_ok'][indices['negative_candidates']]).sum()),
        same_year_size_distance_fallback_pairs=fallback_pairs,
        missing_positive_indices=[i for i in positive_indices if not ctx['size_ok'][i]],
        age_quarters=2, point_in_time_verified=False)
    metadata['preprocessing_fit_scope'] = PROTOCOL['preprocessing_fit_scope']
    return metadata


def _build_manifest(ctx, fold):
    from scipy.optimize import linear_sum_assignment
    train, validation, test = _roles(ctx, fold)
    positives = np.flatnonzero(train & (ctx['labels'] == 1))
    negatives = np.flatnonzero(train & (ctx['labels'] == 0))
    if not len(positives):
        raise ValueError(f'Fold {fold} has no training positives.')
    negative_companies = sorted(set(ctx['codes'][negatives]))
    company_column = {code: j for j, code in enumerate(negative_companies)}
    by_year = {}
    for i in negatives:
        by_year.setdefault(int(ctx['years'][i]), []).append(int(i))
    needed = Counter(int(ctx['years'][i]) for i in positives)
    for year, count in needed.items():
        if len(by_year.get(year, [])) < 5*count:
            raise ValueError(f'Fold {fold}, year {year}: insufficient same-year negatives for 1:5.')
    slots = np.repeat(positives, 5)
    if len(negative_companies) < len(slots):
        raise ValueError(f'Fold {fold}: insufficient negative companies for cap=1.')
    costs = np.full((len(slots), len(negative_companies)), np.inf, dtype=np.float64)
    candidate_lookup = {}
    for i in negatives:
        candidate_lookup[(int(ctx['years'][i]), str(ctx['codes'][i]))] = int(i)
    for row, positive in enumerate(slots):
        for negative in by_year[int(ctx['years'][positive])]:
            costs[row, company_column[ctx['codes'][negative]]] = _distance(ctx, int(positive), negative)
    # No call to np.random.seed/random.*: model RNG states remain untouched.
    rng = np.random.default_rng(PROTOCOL['seed'])
    row_order = rng.permutation(len(slots))
    column_order = rng.permutation(len(negative_companies))
    try:
        shuffled_rows, shuffled_columns = linear_sum_assignment(costs[row_order][:, column_order])
    except ValueError as exc:
        raise ValueError(f'Fold {fold}: global same-year assignment is infeasible under company cap=1.') from exc
    rows, columns = row_order[shuffled_rows], column_order[shuffled_columns]
    if len(rows) != len(slots) or not np.isfinite(costs[rows, columns]).all():
        raise ValueError(f'Fold {fold}: incomplete or cross-year assignment; refusing fallback.')
    tie_rank = rng.permutation(len(negatives))
    ties = {int(i): int(t) for i,t in zip(negatives, tie_rank)}
    assigned = {int(i): [] for i in positives}
    for row, column in zip(rows, columns):
        positive = int(slots[row])
        negative = candidate_lookup[(int(ctx['years'][positive]), str(negative_companies[column]))]
        assigned[positive].append(negative)
    matches, negative3, negative5 = [], [], []
    for positive in map(int, positives):
        controls = sorted(assigned[positive], key=lambda i: (_distance(ctx, positive, i), ties[i]))
        if len(controls) != 5:
            raise ValueError('Every retained positive must receive exactly five controls.')
        entries = []
        for rank, negative in enumerate(controls):
            entry = dict(**_sample_record(ctx, negative), distance=_distance(ctx, positive, negative),
                         size_distance_skipped=not bool(ctx['size_ok'][positive] and ctx['size_ok'][negative]),
                         tie_rank=ties[negative], in_ratio3=rank < 3)
            entries.append(entry)
        matches.append(dict(positive=_sample_record(ctx, positive), controls=entries))
        negative3.extend(controls[:3]); negative5.extend(controls)
    positive_list = positives.tolist()
    indices = dict(train_full=np.flatnonzero(train).tolist(), validation=np.flatnonzero(validation).tolist(),
                   test=np.flatnonzero(test).tolist(), positive=positive_list,
                   negative_candidates=negatives.tolist(), negative_3=sorted(negative3),
                   negative_5=sorted(negative5), selected_3=sorted(positive_list+negative3),
                   selected_5=sorted(positive_list+negative5))
    manifest = dict(schema_version=SCHEMA_VERSION, protocol=PROTOCOL, protocol_sha256=ctx['protocol_sha256'],
                    dataset_sha256=ctx['dataset_sha256'], matching_arrays_sha256=ctx['matching_arrays_sha256'],
                    fold=fold, seed=PROTOCOL['seed'], indices=indices, matches=matches)
    manifest['metadata'] = {str(ratio): _selection_metadata(ctx, fold, ratio, manifest) for ratio in (3,5)}
    manifest['content_sha256'] = _sha_object(manifest)
    return manifest


def _validate_manifest(ctx, fold, manifest):
    content = {k:v for k,v in manifest.items() if k != 'content_sha256'}
    if manifest.get('content_sha256') != _sha_object(content):
        raise ValueError(f'Fold {fold}: sampling manifest integrity check failed.')
    expected = dict(schema_version=SCHEMA_VERSION, protocol=PROTOCOL, protocol_sha256=ctx['protocol_sha256'],
                    dataset_sha256=ctx['dataset_sha256'], matching_arrays_sha256=ctx['matching_arrays_sha256'],
                    fold=fold, seed=PROTOCOL['seed'])
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f'Fold {fold}: manifest {key} differs; refusing silent replacement.')
    train, validation, test = _roles(ctx, fold)
    indices = manifest['indices']
    split_expected = dict(train_full=np.flatnonzero(train).tolist(), validation=np.flatnonzero(validation).tolist(),
                          test=np.flatnonzero(test).tolist(), positive=np.flatnonzero(train & (ctx['labels']==1)).tolist(),
                          negative_candidates=np.flatnonzero(train & (ctx['labels']==0)).tolist())
    for key,value in split_expected.items():
        if indices.get(key) != value:
            raise ValueError(f'Fold {fold}: {key} changes the original company-group split.')
    seen_positives, found3, found5 = [], [], []
    for match in manifest['matches']:
        positive = int(match['positive']['sample_index'])
        if positive not in split_expected['positive'] or match['positive'] != _sample_record(ctx, positive):
            raise ValueError('Invalid matched positive metadata.')
        seen_positives.append(positive)
        controls = match['controls']
        if len(controls) != 5 or sum(bool(c['in_ratio3']) for c in controls) != 3:
            raise ValueError('The nested three-of-five matching rule was violated.')
        ordered = sorted(controls, key=lambda c:(c['distance'], c['tie_rank']))
        if ordered != controls or [bool(c['in_ratio3']) for c in controls] != [True]*3+[False]*2:
            raise ValueError('Ratio 3 must use the three nearest assigned controls, with fixed tie ordering.')
        for control in controls:
            negative = int(control['sample_index'])
            if not (0 <= negative < ctx['n'] and train[negative] and ctx['labels'][negative] == 0):
                raise ValueError('A selected control is not a negative in this training fold.')
            if ctx['years'][negative] != ctx['years'][positive]:
                raise ValueError('Cross-year control assignment is forbidden.')
            for key,value in _sample_record(ctx,negative).items():
                if control.get(key) != value:
                    raise ValueError('Control metadata does not match the source sample.')
            if control['distance'] != _distance(ctx,positive,negative):
                raise ValueError('Control size distance is inconsistent.')
            skipped = not bool(ctx['size_ok'][positive] and ctx['size_ok'][negative])
            if control['size_distance_skipped'] != skipped:
                raise ValueError('Size fallback flag is inconsistent.')
            found5.append(negative)
            if control['in_ratio3']:
                found3.append(negative)
    if sorted(seen_positives) != split_expected['positive']:
        raise ValueError('All original training positives must be retained exactly once.')
    for ratio, found in [(3,found3),(5,found5)]:
        if len(found) != ratio*len(seen_positives) or len(set(found)) != len(found):
            raise ValueError('Wrong control ratio or sampling with replacement detected.')
        if len(set(ctx['codes'][found])) != len(found):
            raise ValueError('More than one selected negative window belongs to a company.')
        if indices[f'negative_{ratio}'] != sorted(found):
            raise ValueError('Negative list differs from matched controls.')
        if indices[f'selected_{ratio}'] != sorted(seen_positives+found):
            raise ValueError('Selected training list differs from retained positives and controls.')
        metadata = _selection_metadata(ctx,fold,ratio,manifest)
        if manifest['metadata'].get(str(ratio)) != metadata:
            raise ValueError('Sampling audit metadata differs from the actual selection.')
    if not set(found3).issubset(found5):
        raise ValueError('Ratio-3 controls must be nested within ratio-5 controls.')


def _read_validate(path, ctx, fold):
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    _validate_manifest(ctx,fold,manifest)
    return manifest


def ensure_manifests(root, d):
    """Create missing fold manifests, or strictly validate existing immutable manifests.

    Returns a JSON-compatible dict with dataset_sha256, protocol_sha256 and folds.
    Each folds[str(k)] has path, manifest_sha256 and ratio3/ratio5 metadata.
    Existing files are never silently rewritten, even when contents differ.
    """
    ctx = _context(root,d)
    folder = ctx['root']/'cache/sampling_v1'
    manifests = {}
    # Validate every pre-existing file before creating any missing file.
    for fold in range(5):
        path = folder/f'fold_{fold}.json'
        if path.exists():
            manifests[fold] = _read_validate(path,ctx,fold)
    for fold in range(5):
        if fold not in manifests:
            manifests[fold] = _build_manifest(ctx,fold)
            _validate_manifest(ctx,fold,manifests[fold])
    folder.mkdir(parents=True,exist_ok=True)
    result = dict(dataset_sha256=ctx['dataset_sha256'], protocol_sha256=ctx['protocol_sha256'], folds={})
    for fold in range(5):
        path = folder/f'fold_{fold}.json'
        manifest = manifests[fold]
        if not path.exists():
            # Exclusive create prevents overwrite of a concurrent or different manifest.
            try:
                with path.open('x',encoding='utf-8') as stream:
                    json.dump(manifest,stream,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
                    stream.write('\n')
            except FileExistsError:
                existing = _read_validate(path,ctx,fold)
                if existing != manifest:
                    raise ValueError(f'Fold {fold}: a different manifest appeared during construction.')
        digest = _sha_file(path)
        result['folds'][str(fold)] = dict(path=str(path), manifest_sha256=digest,
            ratio3=dict(manifest['metadata']['3'],manifest_sha256=digest),
            ratio5=dict(manifest['metadata']['5'],manifest_sha256=digest))
    return result


def load_training_mask(root, d, fold, ratio):
    """Read and validate one fixed manifest; return (bool[N] selection, JSON metadata).

    This function never constructs or writes manifests. Call ensure_manifests
    explicitly before training. The returned mask selects original training
    indices only; callers retain the original validation and test masks.
    """
    if fold not in range(5) or ratio not in (3,5):
        raise ValueError('fold must be 0..4 and ratio must be 3 or 5.')
    ctx = _context(root,d)
    path = ctx['root']/f'cache/sampling_v1/fold_{fold}.json'
    if not path.is_file():
        raise FileNotFoundError(f'Missing immutable sampling manifest: {path}; run ensure_manifests explicitly.')
    manifest = _read_validate(path,ctx,fold)
    selected = np.zeros(ctx['n'],dtype=bool)
    selected[manifest['indices'][f'selected_{ratio}']] = True
    metadata = dict(manifest['metadata'][str(ratio)],manifest_sha256=_sha_file(path),manifest_path=str(path))
    return selected,metadata
