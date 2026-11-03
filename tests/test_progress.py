"""Progress estimates must use only completed work."""

from bdbench.progress import format_duration, remaining_estimate


def test_duration_and_estimate():
    assert format_duration(3661) == '1h 01m 01s'
    assert remaining_estimate(120, 2, 6) == 240
    assert remaining_estimate(0, 0, 6) is None


def test_estimate_never_reports_negative_remaining():
    assert remaining_estimate(120, 6, 6) == 0
