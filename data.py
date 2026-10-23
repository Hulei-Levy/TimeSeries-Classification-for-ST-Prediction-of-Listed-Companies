"""Build auditable first-ST samples without fitting preprocessing or importing old code.

Sequence positions are financial reporting quarters, not historic live forecasts.
All information is screened at the sample's June 30 cutoff. Where publication
vintages are unavailable, period-end + six calendar months is an explicitly
exploratory availability proxy. This does NOT establish point-in-time validity.
Report ages are elapsed quarters from each source period-end to the final June 30
prediction cutoff, not elapsed time since actual publication.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

FINANCIAL = [
    'F010101A','F010701B','F010201A','F011701A','F011201A','F012301B','F010401A','F011901A',
    'F030101A','F031501A','F040201B','F040501B','F041201B','F041401B','F041701B','F041801B',
    'F040901B','F050501B','F051401B','F050101B','F050201B','F051501B','F052101B','F053201B',
    'F053401B','F080601A','F081001B','F081601B','F081201B','F082802A',
]
GOVERNANCE = [
    'AF_CFEATUREPROFILE-CompanySize','AF_CFEATUREPROFILE-Big4','CG_Ybasic-Y0501b',
    'CG_Agm-A0101b','CG_Agm-A0201b','CG_Agm-A0301b',
    'CG_CommissionSta-StrategyCommitteeTimes','CG_CommissionSta-AuditCommitteeTimes',
    'CG_CommissionSta-RemAssComMemberTimes','CG_ManagerShareSalary-Top3SumSalary',
    'CG_ManagerShareSalary-Top3DirectorSumSalary','CG_ManagerShareSalary-Top3SupervisorSumSalary',
    'CG_ManagerShareSalary-Top3ManageSumSalary','CG_ManagerShareSalary-DirectorHoldsharesNum',
    'CG_ManagerShareSalary-SupervisorHoldsharesNum','HLD_CR-Shrcr1','HLD_CR-Shrz',
    'HLD_CR-Shrhfd3','HLD_CR-Shrhfd5','HLD_CR-Shrhfd10','HLD_CR-Shrs',
    'IC_EvaluationRepInfo-IsDisclosingEvaRep','IC_EvaluationRepInfo-IsDeficiency',
    'IC_AuditRepInfo-IsDisclosingAuditRep','LA_SUMMARY-IsInvolvedLA',
    'EN_EquityNatureAll-LargestHolderRate','PT_LCRDSPENDING-RDPersonRatio',
    'PT_LCRDSPENDING-RDSpendSumRatio','BDT_ManaGovAbil-ContrshrNature',
    'BDT_ManaGovAbil-Boardsize','BDT_ManaGovAbil-IndDirectorRatio',
]
CATEGORICAL = [31,32,51,52,53,54,58]
FEATURES = FINANCIAL + GOVERNANCE
SOURCE_PATHS = {
    'coverage': 'st_event_history/company_coverage.csv',
    'financial': 'supplement_2001_2019/features/enhanced_with_official_supplement/financial_combined_2001_onward.parquet',
    'financial_dates': 'supplement_2001_2019/features/enhanced_with_official_supplement/financial_features_long.parquet',
    'governance': 'supplement_2001_2019/assembled/governance_original_columns_2001_2023.parquet',
    'governance_dates': 'supplement_2001_2019/assembled/governance_core_cell_sources.parquet',
}
TEXT_ROOT = '外网_CMDA_管理层讨论与分析_ALL'
EPOCH_ORDINAL = date(1970, 1, 1).toordinal()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def iso_date(value):
    return None if pd.isna(value) or str(value).strip() == '' else date.fromisoformat(str(value)[:10])


def ordinal_array(series):
    dates = pd.to_datetime(series.astype('string').str.slice(0, 10), format='%Y-%m-%d', errors='coerce')
    result = dates.to_numpy(dtype='datetime64[D]').astype(np.int64) + EPOCH_ORDINAL
    result[pd.isna(dates).to_numpy()] = 0
    return result.astype(np.int32)


def normalize_code(frame):
    frame['code'] = frame['code'].astype(str).str.zfill(6)
    return frame


def label_candidates(coverage, start_year=2006, end_year=2024):
    """First-event research labels; unconfirmed later events do not erase earlier negatives."""
    samples, excluded = [], Counter()
    for row in coverage.to_dict('records'):
        code = row['code']
        if code == '601313' or bool(row['economic_identity_break_requires_review']):
            excluded['identity_break_companies'] += 1
            continue
        if not row['source_quality_ok'] or not row['observations_start_at_listing']:
            excluded['insufficient_status_history_companies'] += 1
            continue
        ipo = iso_date(row['listing_date'])
        observed = iso_date(row['first_observed_ST_date'])
        event = iso_date(row['first_ST_label_candidate_date'])
        last = iso_date(row['last_observed_date'])
        through = iso_date(row['continuous_from_listing_through'])
        for year in range(start_year, end_year + 1):
            cutoff, horizon = date(year, 6, 30), date(year + 1, 6, 30)
            if ipo is None or ipo > cutoff:
                excluded['not_yet_listed_windows'] += 1
                continue
            if (observed is not None and observed <= cutoff) or (event is not None and event <= cutoff):
                excluded['already_ST_windows'] += 1
                continue
            if event is None and observed is not None and observed <= horizon:
                excluded['uncertain_event_in_horizon_windows'] += 1
                continue
            label = int(event is not None and cutoff < event <= horizon)
            needed = event if label else horizon
            if last is None or through is None or last < needed or through < needed:
                excluded['incomplete_followup_windows'] += 1
                continue
            if ipo > date(year - 1, 12, 31):
                excluded['entire_input_preIPO_windows'] += 1
                continue
            samples.append(dict(code=code, cutoff=cutoff.isoformat(), label=label,
                                listing_date=ipo.isoformat(), first_ST_date=event.isoformat() if event else '',
                                window_start=f'{year - 5}-01-01', window_end=f'{year - 1}-12-31'))
    return samples, dict(excluded)


def balanced_company_folds(samples, seed=42):
    """Fixed deterministic five-way group allocation, balanced on labels and sample counts."""
    groups = {}
    for sample in samples:
        counts = groups.setdefault(sample['code'], [0, 0])
        counts[0] += 1
        counts[1] += sample['label']
    rng = np.random.default_rng(seed)
    codes = list(groups)
    rng.shuffle(codes)
    codes.sort(key=lambda c: (-groups[c][1], -groups[c][0]))
    totals, positives, companies = [0]*5, [0]*5, [0]*5
    assignments = {}
    for code in codes:
        n, pos = groups[code]
        order = rng.permutation(5).tolist()
        if pos:
            group = min(order, key=lambda k: (positives[k], totals[k], companies[k]))
        else:
            group = min(order, key=lambda k: (totals[k], companies[k]))
        assignments[code] = group
        totals[group] += n
        positives[group] += pos
        companies[group] += 1
    return assignments, [dict(group=k, samples=totals[k], positives=positives[k], companies=companies[k]) for k in range(5)]


def load_table(dataset_root, name, features, date_field, year_field=None):
    path = dataset_root / SOURCE_PATHS[name]
    meta = ['code', date_field]
    if year_field:
        meta.append(year_field)
    frame = normalize_code(pd.read_parquet(path, columns=meta + features))
    if frame.duplicated(['code', date_field]).any():
        raise ValueError(f'Duplicate company-period rows: {path}')
    frame[date_field] = frame[date_field].astype(str).str.slice(0, 10)
    for field in features:
        values = frame[field]
        if field in ('AF_CFEATUREPROFILE-Big4', 'LA_SUMMARY-IsInvolvedLA'):
            unexpected = set(values.dropna().astype(str).unique()) - {'Y', 'N', '1', '0'}
            if unexpected:
                raise ValueError(f'Unknown category {field}: {unexpected}')
            values = values.astype('string').replace({'Y': '1', 'N': '0'})
        frame[field] = pd.to_numeric(values, errors='raise')
    values = frame[features].to_numpy(dtype=np.float32, na_value=np.nan)
    values[~np.isfinite(values)] = np.nan
    # A six-month fallback delay is deliberately not represented as an actual disclosure date.
    periods = pd.to_datetime(frame[date_field]) + pd.DateOffset(months=6)
    proxy = ordinal_array(periods.astype(str))
    availability = np.repeat(proxy[:, None], len(features), axis=1)
    return frame, values, availability


def add_known_dates(dataset_root, frame, availability, table_name, features, period_field):
    is_fin = table_name == 'financial'
    source_period = 'report_period' if is_fin else 'year'
    source_field = 'csmar_field' if is_fin else 'original_field'
    source_date = 'latest_reported_input_publish_date' if is_fin else 'source_available_date'
    columns = ['code', source_period, source_field, source_date]
    source = normalize_code(pd.read_parquet(dataset_root / SOURCE_PATHS[table_name + '_dates'], columns=columns))
    source = source[source[source_field].isin(features) & source[source_date].notna()].copy()
    source['date_ordinal'] = ordinal_array(source[source_date])
    source = source[source.date_ordinal > 0]
    if source.duplicated(['code', source_period, source_field]).any():
        raise ValueError('Ambiguous source dates; review before assigning availability.')
    target_period = frame['report_period'] if is_fin else frame['year']
    index = pd.MultiIndex.from_arrays([frame['code'], target_period])
    matched = 0
    for j, feature in enumerate(features):
        selected = source[source[source_field] == feature]
        lookup = pd.MultiIndex.from_arrays([selected['code'], selected[source_period]])
        positions = index.get_indexer(lookup)
        keep = positions >= 0
        positions = positions[keep]
        dates = selected['date_ordinal'].to_numpy()[keep]
        availability[positions, j] = np.maximum(availability[positions, j], dates)
        matched += int(keep.sum())
    return matched


def index_documents(dataset_root, roster):
    records, skipped = {}, Counter()
    pattern = re.compile(r'(\d{6})_(\d{4}-(?:06-30|12-31))\.txt$')
    for path in sorted((dataset_root / TEXT_ROOT).glob('*/文本/*.txt')):
        match = pattern.fullmatch(path.name)
        if not match:
            skipped['invalid_filename'] += 1
            continue
        code, period = match.groups()
        if code not in roster:
            skipped['outside_roster'] += 1
            continue
        if path.stat().st_size == 0:
            skipped['empty_file'] += 1
            continue
        key = (code, period)
        if key in records:
            raise ValueError(f'Duplicate raw MD&A company-period: {key}')
        records[key] = dict(path=str(path.resolve()), code=code, report_period=period,
                            availability_proxy=(pd.Timestamp(period) + pd.DateOffset(months=6)).date().isoformat())
    return records, dict(skipped)


def build(root, start_year=2006, end_year=2024, seed=42):
    root = Path(root).resolve()
    datasets, cache = root / 'datasets', root / 'cache'
    coverage = normalize_code(pd.read_csv(datasets / SOURCE_PATHS['coverage'], dtype={'code': str}))
    samples, exclusions = label_candidates(coverage, start_year, end_year)
    candidate_count = len(samples)
    candidate_positives = sum(x['label'] for x in samples)
    print(f'Date/label candidates: {candidate_count}, positive: {candidate_positives}', flush=True)
    fin, fvalues, favail = load_table(datasets, 'financial', FINANCIAL, 'report_period')
    gov, gvalues, gavail = load_table(datasets, 'governance', GOVERNANCE, 'period_end', 'year')
    date_counts = dict(financial=add_known_dates(datasets, fin, favail, 'financial', FINANCIAL, 'report_period'),
                       governance=add_known_dates(datasets, gov, gavail, 'governance', GOVERNANCE, 'period_end'))
    findex = {(c,p): i for i,(c,p) in enumerate(zip(fin['code'], fin['report_period']))}
    gindex = {(c,int(y)): i for i,(c,y) in enumerate(zip(gov['code'], gov['year']))}
    raw_docs, document_exclusions = index_documents(datasets, set(coverage['code']))
    doc_records, doc_lookup = [], {}
    n = len(samples)
    values = np.full((n, 20, 61), np.nan, dtype=np.float32)
    ages = np.zeros_like(values)
    valid_time = np.zeros((n, 20), dtype=bool)
    doc_ids = np.full((n, 20), -1, dtype=np.int32)
    doc_ages = np.zeros((n, 20), dtype=np.float32)
    periods = np.empty((n, 20), dtype='<U10')
    late_masked = Counter()
    kept = []
    for i, sample in enumerate(samples):
        code, year = sample['code'], int(sample['cutoff'][:4])
        cutoff_ord = date.fromisoformat(sample['cutoff']).toordinal()
        ipo, window_start = sample['listing_date'], sample['window_start']
        for t in range(20):
            y, q = year - 5 + t // 4, t % 4 + 1
            period = f'{y}-{("03-31", "06-30", "09-30", "12-31")[q-1]}'
            periods[i,t] = period
            if period < ipo:
                continue
            valid_time[i,t] = True
            fi = findex.get((code, period))
            if fi is not None:
                ok = favail[fi] <= cutoff_ord
                late_masked['financial_cells'] += int((~ok & np.isfinite(fvalues[fi])).sum())
                values[i,t,:30] = np.where(ok, fvalues[fi], np.nan)
                ages[i,t,:30] = np.where(np.isfinite(values[i,t,:30]), 21-t, 0)
            # Only ordinary annual update intervals hold the latest scheduled report.
            # A missing new annual value resets the field; older values are not backfilled.
            gy = y if q == 4 else y - 1
            source_period = f'{gy}-12-31'
            if source_period >= max(window_start, ipo):
                gi = gindex.get((code, gy))
                if gi is not None:
                    ok = gavail[gi] <= cutoff_ord
                    late_masked['governance_cells'] += int((~ok & np.isfinite(gvalues[gi])).sum())
                    values[i,t,30:] = np.where(ok, gvalues[gi], np.nan)
                    ages[i,t,30:] = np.where(np.isfinite(values[i,t,30:]), (q if q < 4 else 0) + 21-t, 0)
            if q == 1:
                dp, da = f'{y-1}-12-31', 1
            elif q in (2,3):
                dp, da = f'{y}-06-30', q-2
            else:
                dp, da = f'{y}-12-31', 0
            # Half-year/annual missing reports do not silently inherit an older report.
            if dp >= max(window_start, ipo):
                record = raw_docs.get((code, dp))
                if record is not None and record['availability_proxy'] <= sample['cutoff']:
                    if (code, dp) not in doc_lookup:
                        doc_lookup[(code, dp)] = len(doc_records)
                        doc_records.append(dict(doc_id=len(doc_records), **record))
                    doc_ids[i,t] = doc_lookup[(code, dp)]
                    doc_ages[i,t] = da + 21-t
        if np.isfinite(values[i]).any() or (doc_ids[i] >= 0).any():
            kept.append(i)
        else:
            exclusions['no_usable_input_windows'] = exclusions.get('no_usable_input_windows', 0) + 1
            exclusions['no_usable_input_positives'] = exclusions.get('no_usable_input_positives', 0) + sample['label']
    samples = [samples[i] for i in kept]
    values, ages, valid_time = values[kept], ages[kept], valid_time[kept]
    doc_ids, doc_ages, periods = doc_ids[kept], doc_ages[kept], periods[kept]
    assignments, fold_stats = balanced_company_folds(samples, seed)
    for sample in samples:
        sample['fold'] = assignments[sample['code']]
    observed = np.isfinite(values)
    assert not observed[~valid_time].any()
    assert not (doc_ids[~valid_time] >= 0).any()
    assert not np.any(ages < 0) and not np.any(doc_ages < 0)
    assert all(sum(s['label'] for s in samples if s['code'] == c) <= 1 for c in assignments)
    # Stable document IDs are stored with their original paths; the encoder reads bodies.
    cache.mkdir(exist_ok=True)
    arrays = dict(values=values, ages=ages, observed=observed, valid_time=valid_time,
                  doc_ids=doc_ids, doc_ages=doc_ages, labels=np.array([s['label'] for s in samples], dtype=np.int64),
                  codes=np.array([s['code'] for s in samples], dtype='<U6'),
                  cutoffs=np.array([s['cutoff'] for s in samples], dtype='<U10'),
                  fold=np.array([s['fold'] for s in samples], dtype=np.int8), periods=periods,
                  feature_names=np.array(FEATURES), categorical_indices=np.array(CATEGORICAL, dtype=np.int32))
    with (cache / 'samples.npz.tmp').open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    (cache / 'samples.npz.tmp').replace(cache / 'samples.npz')
    (cache / 'documents.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in doc_records), encoding='utf-8')
    pd.DataFrame(samples).to_csv(cache / 'samples.csv', index=False, encoding='utf-8-sig')
    features_audit = []
    denominator = int(valid_time.sum())
    for j, name in enumerate(FEATURES):
        count = int(observed[:,:,j].sum())
        features_audit.append(dict(name=name, categorical=j in CATEGORICAL,
                                   observed_positions=count, eligible_positions=denominator,
                                   missing_fraction=1 - count / denominator if denominator else None))
    audit = dict(
        schema_version=2, seed=seed, prediction_years=[start_year,end_year],
        candidate_samples=candidate_count, candidate_positives=candidate_positives,
        samples=len(samples), positives=int(arrays['labels'].sum()), companies=len(assignments),
        documents=len(doc_records), source_roster_companies=len(coverage), exclusions=exclusions,
        fold_groups=fold_stats, fold_roles='test=k; validation=(k+1)%5; training=other three groups',
        features=features_audit, categorical_indices=CATEGORICAL,
        categorical_encodings={'AF_CFEATUREPROFILE-Big4': {'N':0,'Y':1},
                               'LA_SUMMARY-IsInvolvedLA': {'N':0,'Y':1},
                               'other_categories': 'Original numeric codes; no ordinal semantic claim.'},
        known_source_date_counts=date_counts, late_masked_sample_positions=dict(late_masked),
        document_index_exclusions=document_exclusions, point_in_time_verified=False,
        label_scope='Research candidates for first ST/*ST, supported by provider history and official implementation notices; not official certification of first-ever ST.',
        availability='max(report-period end + 6 calendar months, known source publication date) <= June 30 cutoff; unknown original vintages remain unverified',
        sequence_alignment='20 report-quarter nodes in previous five complete fiscal years; nodes are not historical real-time prediction dates. No report period later than its node and no source outside the five-year window.',
        ages='Elapsed quarters from each source report-period end to the final June 30 prediction cutoff, not elapsed time since actual disclosure. Latest prior-year Q4 age=2 quarters; earliest five-year-window Q1 age=21 quarters. Missing ages=0 with availability masks.',
        missingness='No financial forward fill; annual governance held Q4 through next Q3; semiannual MD&A held Q2-Q3 and Q4-next Q1. Missing scheduled reports/fields reset; no cross-cycle fill. No imputation or scaling fitted here.',
        cohort_scope='Preserves the uploaded 1545-company roster; no claim of market-wide representative prevalence.',
        generalization_scope='Company-group cross-validation; no out-of-time holdout is implied.',
        source_sha256={key: sha256(datasets / path) for key,path in SOURCE_PATHS.items()},
        data_code_sha256=sha256(Path(__file__)),
    )
    (cache / 'data_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:audit[k] for k in ['samples','positives','companies','documents','fold_groups','exclusions']}, ensure_ascii=False, indent=2))
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['build'])
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--start-year', type=int, default=2006)
    parser.add_argument('--end-year', type=int, default=2024)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not 2006 <= args.start_year <= args.end_year <= 2024:
        parser.error('Supported prediction years are 2006 through 2024.')
    build(args.root, args.start_year, args.end_year, args.seed)


if __name__ == '__main__':
    main()
