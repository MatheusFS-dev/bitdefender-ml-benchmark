"""Calibrate and freeze work amounts before comparison sessions."""

import copy
import math
from bdbench.serialization import canonical_hash


def calibrate_cases(cases: list[dict], worker, target_seconds: float = 180) -> tuple[list[dict], float]:
    """Calibrate training steps from bounded user-invoked trial probes.

    Args:
        cases: Candidate cases.
        worker: Callable that executes one case.
        target_seconds: Target duration per primary training repetition.

    Returns:
        Calibrated cases and estimated measured seconds.

    Raises:
        RuntimeError: If a probe fails or produces an invalid duration.
    """
    prepared = []
    estimate = 0.0
    for original in cases:
        case = copy.deepcopy(original)
        if case['framework'] in {'pytorch', 'tensorflow'} and 'steps' in case and case.get('diagnostic') not in {'perf_stat', 'framework_profiler'} and not case.get('fixed_epoch'):
            probe = dict(case, steps=10)
            result = worker(probe)
            elapsed = result.get('metrics', {}).get('elapsed', {}).get('value')
            if result.get('status') != 'completed' or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
                raise RuntimeError(f"Calibration failed for {case['case_id']}: {result.get('error', 'invalid duration')}")
            case['steps'] = max(1, min(100000, round(10 * target_seconds / elapsed)))
            case['calibration_seconds_per_step'] = elapsed / 10
            estimate += case['steps'] * elapsed / 10 * case['repetitions']
            case.pop('case_id', None)
            case['case_id'] = canonical_hash(case)[:16]
        prepared.append(case)
    return prepared, estimate


def scale_training_budget(cases: list[dict], budget_seconds: float) -> tuple[list[dict], float]:
    """Scale calibrated fixed work toward a shared time budget.

    Args:
        cases: Calibrated and fixed-work case descriptors.
        budget_seconds: Target total measured seconds for calibrated cases.

    Returns:
        Copied cases and the adjusted estimated measured seconds.

    Raises:
        ValueError: If the time budget is nonpositive.
    """
    if budget_seconds <= 0:
        raise ValueError('Budget must be positive')
    result = copy.deepcopy(cases)
    current = sum(case.get('steps', 0) * case.get('calibration_seconds_per_step', 0) * case.get('repetitions', 1) for case in result)
    scale = min(1.0, budget_seconds / current) if current > 0 else 1.0
    estimate = 0.0
    for case in result:
        rate = case.get('calibration_seconds_per_step')
        if rate is None:
            continue
        if scale < 1.0:
            case['steps'] = max(1, int(case['steps'] * scale))
            case.pop('case_id', None)
            case['case_id'] = canonical_hash(case)[:16]
        estimate += case['steps'] * rate * case['repetitions']
    return result, estimate
