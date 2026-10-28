"""Durable JSONL session records."""

import json
import os
from pathlib import Path


def read_results(path: Path, recover_tail: bool = False) -> list[dict]:
    """Read records, optionally quarantining an incomplete final line.

    Args:
        path: Session result file.
        recover_tail: Whether to quarantine an unterminated trailing record.

    Returns:
        Complete decoded records.

    Raises:
        ValueError: If a record is malformed or tail recovery is disabled.
    """
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b'\n'):
        if not recover_tail:
            raise ValueError('Incomplete final JSONL record')
        pos = data.rfind(b'\n') + 1
        quarantine = path.with_suffix(path.suffix + '.partial')
        if quarantine.exists():
            raise ValueError('Previous quarantined tail exists')
        quarantine.write_bytes(data[pos:])
        with path.open('r+b') as file:
            file.truncate(pos)
            file.flush()
            os.fsync(file.fileno())
        data = data[:pos]
    records = []
    for line in data.splitlines():
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError('Malformed JSONL record') from exc
        if not isinstance(record, dict):
            raise ValueError('JSONL record must be an object')
        records.append(record)
    return records


def append_result(path: Path, record: dict) -> None:
    """Append a durable trial result without overwriting a completed trial.

    Args:
        path: Session result file.
        record: JSON-compatible trial result.

    Raises:
        ValueError: If the trial is already completed or record is invalid.
    """
    trial_id = record.get('trial_id')
    if not isinstance(trial_id, str) or not trial_id:
        raise ValueError('A trial ID is required')
    if any(item.get('trial_id') == trial_id and item.get('status') == 'completed' for item in read_results(path)):
        raise ValueError('Completed trial is immutable')
    payload = json.dumps(record, allow_nan=False, separators=(',', ':')) + '\n'
    with path.open('a', encoding='utf-8') as file:
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())


from contextlib import contextmanager
import fcntl


@contextmanager
def session_lock(directory: Path):
    """Acquire a nonblocking process lock for one session.

    Args:
        directory: Existing session directory.

    Yields:
        None while the session lock is held.

    Raises:
        RuntimeError: If another writer holds the lock.
    """
    with (directory / '.lock').open('a') as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Session is already running') from exc
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


def latest_trial_summary(records: list[dict]) -> dict[str, int]:
    """Count the latest recorded state for each trial.

    Args:
        records: Session results in chronological order.

    Returns:
        Counts for completed, failed, skipped and interrupted trials.
    """
    latest = {record['trial_id']: record['status'] for record in records}
    return {status: sum(value == status for value in latest.values()) for status in ('completed', 'failed', 'skipped', 'interrupted')}
