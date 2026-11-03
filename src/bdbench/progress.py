"""Human-readable progress and conservative elapsed-time estimates."""


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
