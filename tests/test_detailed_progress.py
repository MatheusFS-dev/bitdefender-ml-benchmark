"""Detailed progress remains separate from worker JSON output."""

from bdbench.progress import emit_progress, read_progress
from bdbench import runner
import subprocess


def test_worker_progress_file_is_incremental(tmp_path, monkeypatch):
    path = tmp_path / 'progress.log'
    monkeypatch.setenv('BDBENCH_PROGRESS_PATH', str(path))
    emit_progress('model initialized')
    emit_progress('warmup complete')
    messages, offset = read_progress(path, 0)
    assert messages == ['model initialized', 'warmup complete']
    assert read_progress(path, offset) == ([], offset)


def test_runner_relays_worker_stages(monkeypatch, capsys):
    def fake_process(command, payload, environment, timeout):
        path = environment['BDBENCH_PROGRESS_PATH']
        with open(path, 'w', encoding='utf-8') as stream:
            stream.write('framework imported\n')
        return subprocess.CompletedProcess(command, 0, '{"status":"completed","metrics":{}}', '')
    monkeypatch.setattr(runner, 'run_process', fake_process)
    result = runner.run_worker({'framework': 'pytorch', 'model': 'cnn', 'device': 'cpu'})
    assert result['status'] == 'completed'
    assert '[worker] framework imported' in capsys.readouterr().out
