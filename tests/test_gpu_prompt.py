"""The launcher lets the operator choose from detected GPUs."""

import os
from pathlib import Path
import subprocess


def test_launcher_lists_names_and_rejects_invalid_choice(tmp_path):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    for name, body in {
        'nvidia-smi': 'printf "GPU-one, NVIDIA RTX 3060\\nGPU-two, NVIDIA RTX 4070\\n"',
        'uv': 'exit 99',
    }.items():
        path = bin_dir / name
        path.write_text(f'#!/bin/sh\n{body}\n')
        path.chmod(0o755)
    script = Path(__file__).resolve().parents[1] / 'run-comparison-condition.sh'
    result = subprocess.run(['bash', str(script), 'protected'], input='9\n', text=True, capture_output=True, env={**os.environ, 'PATH': f'{bin_dir}:{os.environ["PATH"]}'}, timeout=5)
    assert result.returncode != 0
    assert 'NVIDIA RTX 3060' in result.stdout
    assert 'NVIDIA RTX 4070' in result.stdout
    assert 'Invalid GPU choice' in result.stderr


def test_launcher_prompts_for_dataset_source_before_setup(tmp_path):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    for name, body in {
        'nvidia-smi': 'printf "GPU-one, NVIDIA RTX 4070\\n"',
        'uv': 'exit 99',
    }.items():
        path = bin_dir / name
        path.write_text(f'#!/bin/sh\n{body}\n')
        path.chmod(0o755)
    script = Path(__file__).resolve().parents[1] / 'run-comparison-condition.sh'
    result = subprocess.run(['bash', str(script), 'protected'], input='1\n9\n', text=True, capture_output=True, env={**os.environ, 'PATH': f'{bin_dir}:{os.environ["PATH"]}'}, timeout=5)
    assert result.returncode != 0
    assert 'Hugging Face' in result.stdout
    assert 'Zenodo' in result.stdout
    assert 'Invalid dataset source' in result.stderr
