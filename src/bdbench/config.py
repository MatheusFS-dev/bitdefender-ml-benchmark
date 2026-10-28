"""Configuration loading and validation."""

from dataclasses import dataclass
from pathlib import Path
import math
import tomllib


@dataclass(frozen=True)
class BenchmarkConfig:
    """Validated configuration sections and device selection."""

    sections: dict
    gpu_uuids: tuple[str, ...]
    cpu_only: bool


def load_config(path: Path, overrides: dict) -> BenchmarkConfig:
    """Load configuration and apply explicit device overrides.

    Args:
        path: Path to a TOML configuration.
        overrides: Optional cpu_only and gpu selections.

    Returns:
        Validated benchmark configuration.

    Raises:
        ValueError: If sections or device selections are invalid.
    """
    with path.open('rb') as file:
        sections = tomllib.load(file)
    allowed = {'run', 'devices', 'storage', 'limits', 'diagnostics'}
    unknown = set(sections) - allowed
    if unknown:
        raise ValueError(f'Unknown configuration sections: {sorted(unknown)}')
    section_keys = {'run': {'profile', 'seed', 'repetitions', 'sample_interval_s'}, 'devices': {'cpu', 'gpu_uuids'}, 'storage': {'roots'}, 'limits': {'max_cpu_percent', 'max_memory_percent', 'trial_timeout_s'}, 'diagnostics': {'perf', 'framework_profile'}}
    for name, value in sections.items():
        if not isinstance(value, dict) or set(value) - section_keys[name]:
            raise ValueError(f'Unknown or invalid configuration keys in {name}')
    if sections.get('run', {}).get('profile', 'standard') not in {'smoke', 'standard', 'complete', 'extended'}:
        raise ValueError('Unknown profile')
    repetitions = sections.get('run', {}).get('repetitions')
    if repetitions is not None and (isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1):
        raise ValueError('Repetitions must be a positive integer')
    interval = sections.get('run', {}).get('sample_interval_s', 1)
    if isinstance(interval, bool) or not isinstance(interval, (int, float)) or not math.isfinite(interval) or interval <= 0:
        raise ValueError('Sampling interval must be positive')
    limits = sections.get('limits', {})
    for name, lower, upper in [('max_cpu_percent', 0, 100), ('max_memory_percent', 0, 100), ('trial_timeout_s', 0, None)]:
        value = limits.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= lower or upper is not None and value > upper):
            raise ValueError(f'Invalid {name}')
    if set(overrides) - {'cpu_only', 'gpu'}:
        raise ValueError('Unknown device override')
    cpu_only = bool(overrides.get('cpu_only', False))
    gpu = overrides.get('gpu')
    if cpu_only and gpu:
        raise ValueError('CPU-only and GPU overrides are mutually exclusive')
    gpu_uuids = [] if cpu_only else (gpu if gpu is not None else sections.get('devices', {}).get('gpu_uuids', []))
    if not isinstance(gpu_uuids, list) or any(not isinstance(item, str) or not item.startswith('GPU-') for item in gpu_uuids):
        raise ValueError('GPU selections must be UUIDs')
    if len(set(gpu_uuids)) != len(gpu_uuids):
        raise ValueError('GPU selections must be unique')
    roots = sections.get('storage', {}).get('roots', [])
    if not isinstance(roots, list) or any(not isinstance(root, str) or not Path(root).is_absolute() for root in roots):
        raise ValueError('Storage roots must be absolute paths')
    return BenchmarkConfig(sections=sections, gpu_uuids=tuple(gpu_uuids), cpu_only=cpu_only)
