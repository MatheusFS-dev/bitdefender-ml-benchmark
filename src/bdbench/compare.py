"""Compare compatible case results."""

from collections import defaultdict
import math
import random
from statistics import median, stdev
from bdbench.statistics import latency_overhead, throughput_loss


def _validate_runtimes(baseline: list[dict], protected: list[dict]) -> None:
    def runtimes(records):
        import json
        result = defaultdict(set)
        for record in records:
            if record.get('status') == 'completed' and record.get('runtime') is not None:
                result[record['case_id']].add(json.dumps(record['runtime'], sort_keys=True))
        return result
    baseline_runtimes, protected_runtimes = runtimes(baseline), runtimes(protected)
    baseline_completed = {item['case_id'] for item in baseline if item.get('status') == 'completed' and not item.get('contamination')}
    protected_completed = {item['case_id'] for item in protected if item.get('status') == 'completed' and not item.get('contamination')}
    for case in (baseline_runtimes.keys() | protected_runtimes.keys()) & baseline_completed & protected_completed:
        if baseline_runtimes.get(case, set()) != protected_runtimes.get(case, set()) or len(baseline_runtimes.get(case, set())) != 1:
            raise ValueError(f'Framework runtime differs or is missing for case {case}')


def compare_results(baseline_manifest: dict, protected_manifest: dict, baseline: list[dict], protected: list[dict]) -> list[dict]:
    """Compare median metrics for cases with identical frozen manifests.

    Args:
        baseline_manifest: Frozen baseline manifest.
        protected_manifest: Frozen protected manifest.
        baseline: Baseline trial records.
        protected: Protected trial records.

    Returns:
        Per-case descriptive comparison rows.

    Raises:
        ValueError: If manifests differ.
    """
    if not baseline_manifest.get('manifest_id') or baseline_manifest.get('manifest_id') != protected_manifest.get('manifest_id'):
        raise ValueError('Manifest identities do not match')
    _validate_runtimes(baseline, protected)
    def collect(records):
        result = defaultdict(list)
        for record in records:
            if record.get('status') != 'completed' or record.get('contamination') or record.get('warnings'):
                continue
            for name, item in record.get('metrics', {}).items():
                if isinstance(item, dict) and isinstance(item.get('value'), (float, int)) and not isinstance(item['value'], bool) and math.isfinite(item['value']):
                    result[(record['case_id'], name, item.get('unit'))].append(item['value'])
        return result
    before, after = collect(baseline), collect(protected)
    rows = []
    for case, metric, unit in sorted(before.keys() & after.keys()):
        before_values = before[(case, metric, unit)]
        after_values = after[(case, metric, unit)]
        x, y = median(before_values), median(after_values)
        change = throughput_loss(y, x) if 'throughput' in metric else latency_overhead(y, x)
        interval = []
        if len(before_values) >= 2 and len(after_values) >= 2:
            rng = random.Random(20261028)
            for _ in range(10000):
                b = median(rng.choices(before_values, k=len(before_values)))
                p = median(rng.choices(after_values, k=len(after_values)))
                effect = throughput_loss(p, b) if 'throughput' in metric else latency_overhead(p, b)
                if effect is not None:
                    interval.append(effect)
            interval.sort()
        low = interval[int(.025 * len(interval))] if interval else None
        high = interval[min(len(interval) - 1, int(.975 * len(interval)))] if interval else None
        rows.append({'case_id': case, 'metric': metric, 'unit': unit, 'baseline': x, 'protected': y, 'baseline_sd': stdev(before_values) if len(before_values) > 1 else None, 'protected_sd': stdev(after_values) if len(after_values) > 1 else None, 'baseline_p95': sorted(before_values)[math.ceil(.95 * len(before_values)) - 1] if len(before_values) >= 5 else None, 'protected_p95': sorted(after_values)[math.ceil(.95 * len(after_values)) - 1] if len(after_values) >= 5 else None, 'change_percent': change, 'ci95_low': low, 'ci95_high': high, 'baseline_count': len(before_values), 'protected_count': len(after_values), 'interpretation': 'throughput loss' if 'throughput' in metric else 'relative change', 'verified_attribution': False})
    return rows


def validate_environment_compatibility(baseline: dict, protected: dict) -> None:
    """Require comparable hardware and software metadata.

    Args:
        baseline: Recorded baseline environment.
        protected: Recorded protected environment.

    Raises:
        ValueError: If required environment metadata is missing or differs.
    """
    for section in ('hardware', 'software'):
        if not baseline.get(section) or not protected.get(section):
            raise ValueError(f'Missing environment: {section}')
    if not baseline['hardware'].get('cpu') or not baseline['software'].get('kernel'):
        raise ValueError('CPU or kernel identity is unavailable')
    versions = baseline['software'].get('framework_versions')
    if versions is not None and any(value is None for value in versions.values()):
        raise ValueError('Framework versions are unavailable')
    if baseline['hardware'].get('cpu') != protected['hardware'].get('cpu'):
        raise ValueError('CPU models differ')
    def gpu_specs(environment):
        return sorted(tuple(str(item).split(', ')[1:]) for item in environment['hardware'].get('gpus') or [])
    if gpu_specs(baseline) != gpu_specs(protected):
        raise ValueError('GPU specifications differ')
    if baseline['software'].get('lock_hashes') != protected['software'].get('lock_hashes') or baseline['software'].get('framework_versions') != protected['software'].get('framework_versions'):
        raise ValueError('Software dependencies differ')


def compare_session_groups(baseline_sessions: list[tuple[dict, list]], protected_sessions: list[tuple[dict, list]]) -> list[dict]:
    """Compare condition groups, resampling independent sessions when possible.

    Args:
        baseline_sessions: Pairs of frozen manifest and baseline trials.
        protected_sessions: Pairs of frozen manifest and protected trials.

    Returns:
        Comparable metric rows with bootstrap unit and session counts.

    Raises:
        ValueError: If either condition is missing or manifests differ.
    """
    if not baseline_sessions or not protected_sessions:
        raise ValueError('At least one session per condition is required')
    manifest = baseline_sessions[0][0]
    for other, _ in baseline_sessions + protected_sessions:
        if not manifest.get('manifest_id') or manifest['manifest_id'] != other.get('manifest_id'):
            raise ValueError('Manifest identities do not match')
    _validate_runtimes([record for _, records in baseline_sessions for record in records], [record for _, records in protected_sessions for record in records])
    if len(baseline_sessions) == len(protected_sessions) == 1:
        rows = compare_results(manifest, manifest, baseline_sessions[0][1], protected_sessions[0][1])
        for row in rows:
            row['bootstrap_unit'] = 'trial'
            row['baseline_sessions'] = 1
            row['protected_sessions'] = 1
        return rows
    def session_medians(sessions):
        reduced = []
        for _, records in sessions:
            groups = defaultdict(list)
            for record in records:
                if record.get('status') != 'completed' or record.get('contamination') or record.get('warnings'):
                    continue
                for name, metric in record.get('metrics', {}).items():
                    if not isinstance(metric, dict):
                        continue
                    value = metric.get('value')
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                        groups[(record['case_id'], name, metric.get('unit'))].append(value)
            for (case, name, unit), values in groups.items():
                reduced.append({'case_id': case, 'status': 'completed', 'metrics': {name: {'value': median(values), 'unit': unit}}})
        return reduced
    rows = compare_results(manifest, manifest, session_medians(baseline_sessions), session_medians(protected_sessions))
    for row in rows:
        row['bootstrap_unit'] = 'session'
        row['baseline_sessions'] = len(baseline_sessions)
        row['protected_sessions'] = len(protected_sessions)
    return rows


def validate_condition_labels(baseline: list[dict], protected: list[dict]) -> None:
    """Require consistent and distinct operator labels for two groups.

    Args:
        baseline: Baseline session environment records.
        protected: Protected session environment records.

    Raises:
        ValueError: If labels are absent, inconsistent or identical.
    """
    a = {item.get('condition') for item in baseline}
    b = {item.get('condition') for item in protected}
    if len(a) != 1 or len(b) != 1 or None in a or None in b or '' in a or '' in b or a == b:
        raise ValueError('Condition groups must have distinct, consistent, nonempty operator labels')


def excluded_cases(manifest: dict, baseline: list[dict], protected: list[dict]) -> list[dict]:
    """List planned cases lacking valid trials in either condition.

    Args:
        manifest: Frozen workload manifest.
        baseline: Baseline trial records.
        protected: Protected trial records.

    Returns:
        Case identifiers with explicit exclusion reasons.
    """
    def valid(records):
        return {record['case_id'] for record in records if record.get('status') == 'completed' and not record.get('contamination') and not record.get('warnings')}
    before, after = valid(baseline), valid(protected)
    exclusions = []
    for case in manifest.get('cases', []):
        case_id = case['case_id']
        missing = []
        if case_id not in before:
            missing.append('baseline')
        if case_id not in after:
            missing.append('protected')
        if missing:
            exclusions.append({'case_id': case_id, 'reason': 'No valid ' + ' or '.join(missing) + ' trials'})
    return exclusions
