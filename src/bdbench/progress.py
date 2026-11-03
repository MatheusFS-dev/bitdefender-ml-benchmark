"""Human-readable progress and conservative elapsed-time estimates."""

import os
from pathlib import Path


def format_duration(seconds: float) -> str:
    """Format a nonnegative duration for terminal progress output.

    Args:
        seconds: Elapsed or remaining seconds.

    Returns:
        Hours, minutes and seconds as a short string.
    """
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours}h {minutes:02d}m {seconds:02d}s' if hours else f'{minutes}m {seconds:02d}s'


def remaining_estimate(elapsed: float, completed: int, total: int) -> float | None:
    """Estimate remaining seconds from completed trial count.

    Args:
        elapsed: Seconds spent on completed trials.
        completed: Number of completed trials.
        total: Number of planned trials.

    Returns:
        Estimated seconds remaining, or None before a trial completes.
    """
    if completed < 1:
        return None
    return max(0, total - completed) * elapsed / completed


def emit_progress(message: str) -> None:
    """Append a worker stage without changing its JSON standard output.

    Args:
        message: One-line stage description.
    """
    path = os.environ.get('BDBENCH_PROGRESS_PATH')
    if path:
        with Path(path).open('a', encoding='utf-8') as stream:
            stream.write(message.replace('\n', ' ') + '\n')


def read_progress(path: Path, offset: int) -> tuple[list[str], int]:
    """Read complete worker progress lines after a prior offset.

    Args:
        path: Progress log written by the current worker.
        offset: Previous byte offset.

    Returns:
        New messages and the next byte offset.
    """
    if not path.exists():
        return [], offset
    with path.open('rb') as stream:
        stream.seek(offset)
        data = stream.read()
    complete = data.rfind(b'\n') + 1
    if not complete:
        return [], offset
    return data[:complete].decode('utf-8', 'replace').splitlines(), offset + complete
