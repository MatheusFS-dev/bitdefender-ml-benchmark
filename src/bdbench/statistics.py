"""Descriptive statistics for independent benchmark trials."""

import math
import random
import statistics


def latency_overhead(protected: float, baseline: float) -> float | None:
    """Compute elapsed-time overhead as a percentage.

    Args:
        protected: Protected elapsed time.
        baseline: Baseline elapsed time.

    Returns:
        Percentage overhead, or None for invalid inputs.
    """
    if baseline is None or protected is None or baseline <= 0 or not math.isfinite(baseline) or not math.isfinite(protected):
        return None
    return 100 * (protected / baseline - 1)


def throughput_loss(protected: float, baseline: float) -> float | None:
    """Compute protected-condition throughput loss as a percentage.

    Args:
        protected: Protected throughput.
        baseline: Baseline throughput.

    Returns:
        Percentage loss, or None for invalid inputs.
    """
    if baseline is None or protected is None or baseline <= 0 or not math.isfinite(baseline) or not math.isfinite(protected):
        return None
    return 100 * (1 - protected / baseline)


def summarize(values: list) -> dict:
    """Summarize finite observations and report insufficient tails.

    Args:
        values: Numeric observations, possibly including missing values.

    Returns:
        Counts, summary statistics, and percentile estimates.
    """
    data = sorted(x for x in values if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x))
    if not data:
        return {'count': 0, 'mean': None, 'median': None, 'std': None, 'cv': None, 'p50': None, 'p90': None, 'p95': None, 'p99': None, 'max': None}
    def quantile(q):
        position = (len(data) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        return data[lower] + (data[upper] - data[lower]) * (position - lower)
    mean = statistics.mean(data)
    sd = statistics.stdev(data) if len(data) > 1 else None
    return {'count': len(data), 'mean': mean, 'median': statistics.median(data), 'std': sd, 'cv': sd / mean if sd is not None and mean else None, 'p50': quantile(.5), 'p90': quantile(.9) if len(data) >= 10 else None, 'p95': quantile(.95) if len(data) >= 20 else None, 'p99': quantile(.99) if len(data) >= 100 else None, 'max': data[-1]}


def bootstrap_interval(values: list[float], seed: int = 20261028, samples: int = 10000) -> tuple[float, float] | None:
    """Bootstrap a mean using independently sampled trial observations.

    Args:
        values: Independent trial or session-level observations.
        seed: Random generator seed.
        samples: Number of bootstrap resamples.

    Returns:
        The 95 percent interval, or None with fewer than two observations.
    """
    data = [x for x in values if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)]
    if len(data) < 2 or samples < 40:
        return None
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(data, k=len(data))) for _ in range(samples))
    return means[int(.025 * samples)], means[min(samples - 1, int(.975 * samples))]


def scaling_efficiency(distributed: float, single_device_rates: list[float]) -> float | None:
    """Compare distributed throughput with the sum of single-device rates.

    Args:
        distributed: Aggregate distributed samples per second.
        single_device_rates: Rates measured separately for each selected GPU.

    Returns:
        Scaling efficiency as a ratio, or None for incomplete baselines.
    """
    if not single_device_rates or any(value is None or not math.isfinite(value) or value <= 0 for value in single_device_rates):
        return None
    if distributed is None or not math.isfinite(distributed) or distributed <= 0:
        return None
    return distributed / sum(single_device_rates)
