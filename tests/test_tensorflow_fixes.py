"""Regression checks for TensorFlow GPU discovery and checkpoints."""

from pathlib import Path

from bdbench.runner import ROOT, run_worker, tensorflow_cuda_paths


def test_tensorflow_cuda_paths_include_installed_cudnn():
    python = ROOT / 'environments/tensorflow/.venv/bin/python'
    paths = tensorflow_cuda_paths(python)
    assert any(path.endswith('/nvidia/cudnn/lib') for path in paths)
    assert all(Path(path).is_dir() for path in paths)


def test_tensorflow_cpu_checkpoint_round_trip(tmp_path):
    case = {'framework': 'tensorflow', 'model': 'model_checkpoint', 'device': 'cpu', 'storage_root': str(tmp_path), 'storage_roots': [str(tmp_path)], 'durable': False, 'seed': 1234, 'timeout_s': 60}
    result = run_worker(case)
    assert result['status'] == 'completed', result.get('error')
    assert result['metrics']['load_time']['value'] > 0
