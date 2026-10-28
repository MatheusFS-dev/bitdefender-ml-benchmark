"""Command-line interface for benchmark preparation and reporting."""

import json
import os
from pathlib import Path
import platform
import subprocess
from datetime import datetime, timezone
import click

from bdbench.catalog import build_cases
from bdbench.compare import compare_session_groups, excluded_cases, validate_condition_labels, validate_environment_compatibility
from bdbench.config import load_config
from bdbench.datasets import prepare_cifar10, prepare_wikitext, verify_dataset_files
from bdbench.interference import ensure_cpu_idle, ensure_gpus_idle, ensure_memory_available
from bdbench.inventory import inventory_environment
from bdbench.protection import capture_protection
from bdbench.reporting import write_comparison_report, write_session_report
from bdbench.runner import ROOT, run_session, run_worker
from bdbench.prepare import calibrate_cases, scale_training_budget
from bdbench.serialization import canonical_hash, verify_manifest
from bdbench.session import read_results
from bdbench.storage import check_total_budget


@click.group()
def main():
    """Prepare, run and compare machine-learning benchmark sessions."""


@main.command()
def doctor():
    """Display read-only host and protection evidence."""
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,memory.total', '--format=csv,noheader'], capture_output=True, text=True, check=False) if __import__('shutil').which('nvidia-smi') else None
    info = {'os': platform.platform(), 'python': platform.python_version(), 'cpu_count': os.cpu_count(), 'gpu_inventory': gpu.stdout.strip() if gpu and gpu.returncode == 0 else None, 'protection': capture_protection('unknown')}
    click.echo(json.dumps(info, indent=2))


@main.command()
def setup():
    """Synchronize isolated framework environments with uv."""
    for project in (ROOT, ROOT / 'environments/pytorch', ROOT / 'environments/tensorflow'):
        try:
            subprocess.run(['uv', 'sync', '--locked', '--project', str(project)], check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise click.ClickException(f'uv synchronization failed: {exc}') from exc


@main.command()
@click.option('--config', 'config_path', type=click.Path(exists=True, path_type=Path), required=True)
@click.option('--gpu', multiple=True, help='Select an NVIDIA GPU by UUID. Repeat for multiple devices.')
@click.option('--cpu-only', is_flag=True)
@click.option('--storage-root', multiple=True, type=click.Path(exists=True, file_okay=False, path_type=Path), help='Select a storage directory. Repeat to add drives.')
@click.option('--calibrate/--no-calibrate', default=True, help='Measure short probes while preparing the manifest.')
@click.option('--output', type=click.Path(path_type=Path), default=ROOT / 'prepared' / 'manifest.json')
def prepare(config_path, gpu, cpu_only, storage_root, calibrate, output):
    """Freeze the selected workload settings into a manifest."""
    if output.exists():
        raise click.ClickException('Manifest already exists, select a new output path')
    config = load_config(config_path, {'gpu': list(gpu) if gpu else None, 'cpu_only': cpu_only})
    if any(path.is_symlink() for path in storage_root):
        raise click.ClickException('Storage roots cannot be symbolic links')
    roots = [str(path.resolve()) for path in storage_root] if storage_root else config.sections.get('storage', {}).get('roots', [])
    settings = {'profile': config.sections.get('run', {}).get('profile', 'standard'), 'seed': config.sections.get('run', {}).get('seed', 1234), 'cpu': True if cpu_only else config.sections.get('devices', {}).get('cpu', True), 'gpu_uuids': list(config.gpu_uuids), 'storage_roots': roots}
    settings['repetitions'] = config.sections.get('run', {}).get('repetitions')
    settings['sample_interval_s'] = config.sections.get('run', {}).get('sample_interval_s', 1.0)
    settings['max_cpu_percent'] = config.sections.get('limits', {}).get('max_cpu_percent', 80.0)
    settings['max_memory_percent'] = config.sections.get('limits', {}).get('max_memory_percent', 90.0)
    settings['trial_timeout_s'] = config.sections.get('limits', {}).get('trial_timeout_s', 600.0)
    settings['perf'] = config.sections.get('diagnostics', {}).get('perf', False)
    settings['framework_profile'] = config.sections.get('diagnostics', {}).get('framework_profile', False)
    hardware = inventory_environment()['hardware']
    settings.update({'physical_cores': hardware['physical_cores'], 'logical_cores': hardware['logical_cpus'], 'effective_cpus': hardware['effective_cpus']})
    if calibrate:
        try:
            ensure_cpu_idle(threshold=settings['max_cpu_percent'])
            ensure_memory_available(threshold=settings['max_memory_percent'])
            ensure_gpus_idle(settings['gpu_uuids'])
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from exc
    if settings['storage_roots']:
        check_total_budget([Path(root) for root in settings['storage_roots']], 1400 * 1024 * 1024)
        settings['datasets'] = {'cifar10': prepare_cifar10(Path(settings['storage_roots'][0])), 'wikitext2': prepare_wikitext(Path(settings['storage_roots'][0]))}
    cases = build_cases(settings)
    if not cases:
        raise click.ClickException('No devices selected')
    estimate = None
    if calibrate:
        click.echo('Preparing uses short training probes to calibrate fixed work for the selected devices.')
        try:
            def probe(case):
                ensure_cpu_idle(threshold=settings['max_cpu_percent'])
                ensure_memory_available(threshold=settings['max_memory_percent'])
                if case['device'] != 'cpu':
                    ensure_gpus_idle(case.get('device_uuids', [case['device']]))
                candidate = dict(case)
                candidate.setdefault('timeout_s', settings['trial_timeout_s'])
                return run_worker(candidate)
            cases, estimate = calibrate_cases(cases, probe, {'smoke': 2, 'standard': 30, 'complete': 180, 'extended': 180}[settings['profile']])
            if settings['profile'] in {'complete', 'extended'} and estimate > 8 * 3600:
                cases, estimate = scale_training_budget(cases, 8 * 3600)
                click.echo('Calibrated step counts were scaled to an eight-hour measured-training budget; all cases remain present.')
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f'Estimated calibrated training measurement time: {estimate / 3600:.2f} hours; excludes startup, data preparation, fixed epochs, profiling, checkpoint, classical ML and filesystem cases.')
    manifest = {'schema_version': 1, 'settings': settings, 'cases': cases, 'duration_estimate_s': estimate}
    manifest['manifest_id'] = canonical_hash(manifest)
    manifest['created_utc'] = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as file:
        file.write(json.dumps(manifest, indent=2))
    click.echo(f'Created {output} with {len(cases)} cases')


@main.command()
@click.option('--manifest', type=click.Path(exists=True, path_type=Path), required=True)
@click.option('--condition', required=True)
@click.option('--session', required=True)
@click.option('--resume', is_flag=True)
def run(manifest, condition, session, resume):
    """Run frozen cases and produce a session report."""
    if not session or any(x not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for x in session):
        raise click.ClickException('Session name must contain only letters, digits, underscore or hyphen')
    try:
        frozen_manifest = json.loads(manifest.read_text())
        verify_manifest(frozen_manifest)
        for dataset in frozen_manifest.get('settings', {}).get('datasets', {}).values():
            verify_dataset_files(Path(dataset['path']), dataset['files'])
        summary = run_session(frozen_manifest, condition, ROOT / 'sessions' / session, resume)
    except (ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f'Report: {ROOT / "sessions" / session / "report" / "report.html"}')
    click.echo(json.dumps(summary))
    if summary['failed'] or summary['interrupted']:
        raise click.ClickException('Some trials failed or were interrupted. See the session report.')


@main.command()
@click.option('--session', type=click.Path(exists=True, path_type=Path), required=True)
def report(session):
    """Regenerate a report from a saved session."""
    for path in write_session_report(session, session / 'report'):
        click.echo(str(path))


@main.command()
@click.option('--baseline', type=click.Path(exists=True, path_type=Path), multiple=True, required=True)
@click.option('--protected', type=click.Path(exists=True, path_type=Path), multiple=True, required=True)
@click.option('--output', type=click.Path(path_type=Path), required=True)
def compare(baseline, protected, output):
    """Compare saved sessions sharing an identical manifest."""
    try:
        reference = json.loads((baseline[0] / 'environment.json').read_text())
        validate_condition_labels([json.loads((path / 'environment.json').read_text()) for path in baseline], [json.loads((path / 'environment.json').read_text()) for path in protected])
        def load_group(paths):
            sessions = []
            for path in paths:
                environment = json.loads((path / 'environment.json').read_text())
                validate_environment_compatibility(reference, environment)
                manifest = json.loads((path / 'manifest.json').read_text())
                verify_manifest(manifest)
                sessions.append((manifest, read_results(path / 'results.jsonl')))
            return sessions
        baseline_sessions, protected_sessions = load_group(baseline), load_group(protected)
        rows = compare_session_groups(baseline_sessions, protected_sessions)
        exclusions = excluded_cases(baseline_sessions[0][0], [record for _, records in baseline_sessions for record in records], [record for _, records in protected_sessions for record in records])
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    for path in write_comparison_report(rows, output, exclusions):
        click.echo(str(path))


if __name__ == '__main__':
    main()
