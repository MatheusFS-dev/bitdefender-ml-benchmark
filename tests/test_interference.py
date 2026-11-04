"""Regression checks for graphical GPU process filtering."""

import io
from unittest.mock import patch

from bdbench.interference import conflicts


def test_desktop_process_does_not_block_selected_gpu():
    process = {'gpu_uuid': 'GPU-A', 'pid': 123}
    with patch('builtins.open', return_value=io.StringIO('gnome-shell\n')), patch('os.readlink', return_value='/usr/bin/gnome-shell'):
        assert conflicts(['GPU-A'], [process]) == []


def test_unknown_compute_process_still_blocks_selected_gpu():
    process = {'gpu_uuid': 'GPU-A', 'pid': 123}
    with patch('builtins.open', return_value=io.StringIO('python\n')), patch('os.readlink', return_value='/usr/bin/python'):
        assert conflicts(['GPU-A'], [process]) == [process]


def test_inaccessible_process_is_not_silently_ignored():
    process = {'gpu_uuid': 'GPU-A', 'pid': 123}
    with patch('builtins.open', side_effect=PermissionError), patch('os.readlink', side_effect=PermissionError):
        assert conflicts(['GPU-A'], [process]) == [process]
