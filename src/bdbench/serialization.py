"""Deterministic encoding and basic measurement validation."""

import hashlib
import json
import math
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=None)
def _schema_validator(kind: str):
    import jsonschema
    schema_path = Path(__file__).resolve().parents[2] / 'schemas' / f'{kind}.schema.json'
    schema = json.loads(schema_path.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def canonical_hash(record: dict) -> str:
    """Return the SHA-256 of canonical JSON content.

    Args:
        record: JSON-compatible data to hash.

    Returns:
        Lowercase hexadecimal digest.
    """
    data = json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def validate_metric(metric: dict) -> None:
    """Validate a scalar metric and its availability metadata.

    Args:
        metric: Scalar value and associated metadata.

    Raises:
        ValueError: If a value or its availability is inconsistent.
    """
    required = {'value', 'unit', 'source', 'scope', 'availability', 'reason'}
    if set(metric) != required:
        raise ValueError('Invalid metric fields')
    status = metric['availability']
    if status not in {'available', 'unsupported', 'denied', 'missing', 'invalid'}:
        raise ValueError('Invalid availability')
    value = metric['value']
    if status == 'available':
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError('Available metrics must be finite numbers')
    elif value is not None or not metric['reason']:
        raise ValueError('Unavailable metrics require a null value and reason')


def validate_record(kind: str, record: dict) -> None:
    """Reject unsupported schema versions and unknown record types.

    Args:
        kind: Record category.
        record: Decoded JSON object.

    Raises:
        ValueError: If the record type or version is unsupported.
    """
    if kind not in {'manifest', 'environment', 'trial', 'telemetry', 'protection'}:
        raise ValueError('Unknown record type')
    try:
        import jsonschema
        _schema_validator(kind).validate(record)
    except jsonschema.ValidationError as exc:
        raise ValueError(f'Invalid {kind} record: {exc.message}') from exc
    if record.get('schema_version') != 1:
        raise ValueError('Unsupported schema version')
    required = {'manifest': {'manifest_id', 'settings', 'cases'}, 'environment': {'hardware', 'software', 'protection'}, 'trial': {'case_id', 'trial_id', 'status', 'metrics'}, 'telemetry': {'case_id', 'trial_id', 'monotonic_ns'}, 'protection': {'claimed_condition', 'observed_state'}}
    if not required[kind].issubset(record):
        raise ValueError(f'Missing {kind} fields')
    if kind == 'manifest' and (not isinstance(record['cases'], list) or not isinstance(record['settings'], dict)):
        raise ValueError('Invalid manifest fields')
    if kind == 'trial' and record['status'] not in {'completed', 'failed', 'skipped', 'interrupted'}:
        raise ValueError('Invalid trial status')
    try:
        json.dumps(record, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError('Record must contain finite JSON data') from exc


def verify_manifest(manifest: dict) -> None:
    """Verify that a frozen manifest has not changed since preparation.

    Args:
        manifest: Prepared manifest including its ID and creation timestamp.

    Raises:
        ValueError: If its version or digest is invalid.
    """
    validate_record('manifest', manifest)
    content = {key: value for key, value in manifest.items() if key not in {'manifest_id', 'created_utc'}}
    if manifest.get('manifest_id') != canonical_hash(content):
        raise ValueError('Manifest digest mismatch')


def normalize_metrics(metrics: dict) -> dict:
    """Normalize worker scalar metrics to explicit availability records.

    Args:
        metrics: Mapping from metric names to scalar measurement dictionaries.

    Returns:
        Mapping to validated scalar metric records.

    Raises:
        ValueError: If a metric has no unit or has an unsupported value type.
    """
    if not isinstance(metrics, dict):
        raise ValueError('Metrics must be a mapping')
    result = {}
    for name, metric in metrics.items():
        if not isinstance(name, str) or not isinstance(metric, dict) or not isinstance(metric.get('unit'), str) or not metric['unit']:
            raise ValueError('Metric name and unit must be nonempty strings')
        value = metric.get('value')
        reason = metric.get('reason')
        if value is None:
            status = 'missing'
            reason = reason or 'Measurement unavailable'
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'Metric {name} is not numeric')
        elif not math.isfinite(value):
            status = 'invalid'
            value = None
            reason = reason or 'Nonfinite measurement'
        else:
            status = 'available'
            reason = None
        normalized = {'value': value, 'unit': metric['unit'], 'source': metric.get('source') or 'worker', 'scope': metric.get('scope') or 'trial', 'availability': status, 'reason': reason}
        validate_metric(normalized)
        result[name] = normalized
    return result
