"""Run isolated workers and persist per-trial results."""

from datetime import datetime, timezone
import json
import base64
import os
from pathlib import Path
import subprocess
import shutil
import signal
import threading
import time
import tempfile
import uuid

from bdbench.protection import capture_protection, protection_changed
from bdbench.progress import format_duration, read_progress, remaining_estimate
from bdbench.inventory import inventory_environment
from bdbench.interference import ensure_cpu_idle, ensure_gpus_idle, ensure_memory_available
from bdbench.compare import validate_environment_compatibility
from bdbench.reporting import write_session_report
from bdbench.session import append_result, read_results, session_lock, latest_trial_summary
from bdbench.serialization import normalize_metrics, validate_record
from bdbench.telemetry import sample_to_file, sample_interval, trial_gpu_energy, gpu_energy_for_device, trial_rapl_energy, energy_per_work, energy_work_unit, summarize_trial_telemetry

ROOT = Path(__file__).resolve().parents[2]


def run_process(command: list[str], payload: str, environment: dict, timeout: float):
    """Run one worker in its own process group with bounded cleanup.

    Args:
        command: Worker command and arguments.
        payload: JSON request supplied on standard input.
        environment: Worker environment variables.
        timeout: Maximum time in seconds.

    Returns:
        A subprocess-style completed result.

    Raises:
        subprocess.TimeoutExpired: If the worker exceeds its timeout.
    """
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment, start_new_session=True)
    try:
        stdout, stderr = process.communicate(input=payload, timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def parse_perf_stat(output: str) -> dict:
    """Parse supported perf stat CSV counters without substituting zeros.

    Args:
        output: stderr from perf stat using comma-separated output.

    Returns:
        Counter metrics with unsupported entries marked unavailable.
    """
    result = {}
    allowed = {'cycles', 'instructions', 'cache-misses', 'context-switches'}
    for line in output.splitlines():
        fields = line.split(',')
        if len(fields) < 3 or fields[2].strip() not in allowed:
            continue
        event = fields[2].strip()
        raw = fields[0].strip()
        try:
            value = float(raw)
            reason = None
        except ValueError:
            value = None
            reason = raw or 'Counter unavailable'
        result['perf_' + event.replace('-', '_')] = {'value': value, 'unit': 'events', 'source': 'perf stat', 'reason': reason}
    return result


def run_worker(case: dict) -> dict:
    """Launch one isolated framework worker for the supplied case.

    Args:
        case: Frozen case parameters.

    Returns:
        Result or a structured failure.
    """
    python = (ROOT / '.venv' if case['framework'] in {'sklearn', 'filesystem'} else ROOT / 'environments' / case['framework'] / '.venv') / 'bin' / 'python'
    env = os.environ.copy()
    selected = case.get('device_uuids', [case['device']])
    env['CUDA_VISIBLE_DEVICES'] = '' if case['device'] == 'cpu' else ','.join(selected)
    env['PYTHONPATH'] = str(ROOT / 'src')
    command = [str(python), '-m', 'bdbench.worker']
    if case.get('diagnostic') == 'startup':
        command = [str(python), '-m', 'bdbench.startup_worker']
    if case.get('diagnostic') == 'compile':
        command = [str(python), '-m', 'bdbench.compile_worker']
        env['TORCHINDUCTOR_CACHE_DIR'] = case['profile_output']
    if case.get('distributed') and case['framework'] == 'pytorch':
        encoded = base64.urlsafe_b64encode(json.dumps(case).encode()).decode()
        command = [str(python), '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={len(selected)}', '--module', 'bdbench.dist_worker', encoded]
    if case.get('diagnostic') == 'perf_stat':
        perf = shutil.which('perf')
        if not perf:
            return {'status': 'skipped', 'error': 'perf executable not available', 'metrics': {}}
        command = [perf, 'stat', '-x,', '-e', 'cycles,instructions,cache-misses,context-switches', '--', *command]
    with tempfile.TemporaryDirectory(prefix='bdbench-progress-') as temporary:
        progress_path = Path(temporary) / 'worker.log'
        env['BDBENCH_PROGRESS_PATH'] = str(progress_path)
        progress_stop = threading.Event()
        def relay_progress():
            offset = 0
            while True:
                messages, offset = read_progress(progress_path, offset)
                for message in messages:
                    print(f'[worker] {message}', flush=True)
                if progress_stop.wait(1):
                    messages, offset = read_progress(progress_path, offset)
                    for message in messages:
                        print(f'[worker] {message}', flush=True)
                    return
        relay = threading.Thread(target=relay_progress, daemon=True)
        relay.start()
        try:
            process_start = time.perf_counter()
            completed = run_process(command, json.dumps(case), env, case.get('timeout_s', 600))
            process_elapsed = time.perf_counter() - process_start
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {'status': 'failed', 'error': str(exc), 'metrics': {}}
        finally:
            progress_stop.set()
            relay.join(timeout=2)
    if completed.returncode:
        if case.get('diagnostic') == 'perf_stat' and ('permission' in completed.stderr.lower() or 'not supported' in completed.stderr.lower()):
            return {'status': 'skipped', 'error': completed.stderr[-8192:], 'metrics': {}}
        return {'status': 'failed', 'error': completed.stderr[-8192:], 'metrics': {}}
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {'status': 'failed', 'error': 'Worker returned invalid JSON', 'metrics': {}}
    if not isinstance(result, dict) or not isinstance(result.get('metrics', {}), dict):
        return {'status': 'failed', 'error': 'Worker returned an invalid result object', 'metrics': {}}
    if case.get('diagnostic') == 'perf_stat':
        result['perf_stat_raw'] = completed.stderr[-16384:]
        result.setdefault('metrics', {}).update(parse_perf_stat(completed.stderr))
    if case.get('diagnostic') == 'startup':
        result.setdefault('metrics', {})['process_wall_time'] = {'value': process_elapsed, 'unit': 's', 'source': 'parent monotonic clock including process creation and shutdown'}
    return result


def run_session(manifest: dict, condition: str, session_dir: Path, resume: bool = False) -> dict:
    """Run frozen cases sequentially and generate a report even on failure.

    Args:
        manifest: Immutable benchmark manifest.
        condition: Operator-supplied protection label.
        session_dir: Output directory for this session.
        resume: Continue unfinished trials in an existing session.

    Returns:
        Counts of the latest recorded trial states.

    Raises:
        ValueError: If session or manifest does not match the request.
    """
    session_dir.mkdir(parents=True, exist_ok=True)
    with session_lock(session_dir):
        _run_session_locked(manifest, condition, session_dir, resume)
        return latest_trial_summary(read_results(session_dir / 'results.jsonl'))


def _run_session_locked(manifest: dict, condition: str, session_dir: Path, resume: bool) -> None:
    manifest_path = session_dir / 'manifest.json'
    if manifest_path.exists():
        if not resume:
            raise ValueError('Session already exists, use resume')
        existing = json.loads(manifest_path.read_text())
        if existing.get('manifest_id') != manifest.get('manifest_id'):
            raise ValueError('Manifest differs from existing session')
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    result_path = session_dir / 'results.jsonl'
    previous = read_results(result_path, recover_tail=resume)
    done = {r['trial_id'] for r in previous if r.get('status') == 'completed'}
    pending = sum(f"{case['case_id']}-{repetition}" not in done for case in manifest['cases'] for repetition in range(case['repetitions']))
    completed_now = 0
    completed_seconds = 0.0
    print(f'[benchmark] Session {session_dir.name}: {pending} trials remaining, {len(done)} already completed.', flush=True)
    environment_path = session_dir / 'environment.json'
    print('[benchmark] Capturing hardware, software and protection inventory.', flush=True)
    if resume and environment_path.exists():
        previous_environment = json.loads(environment_path.read_text())
        if previous_environment.get('condition') != condition:
            raise ValueError('Cannot change condition when resuming a session')
    environment = {'schema_version': 1, 'captured_utc': datetime.now(timezone.utc).isoformat(), 'protection': capture_protection(condition), 'condition': condition, **inventory_environment()}
    if resume and environment_path.exists():
        validate_environment_compatibility(previous_environment, environment)
        if protection_changed(previous_environment.get('protection', {}), environment['protection']):
            raise ValueError('Observed protection evidence changed since the session started')
    if not environment_path.exists():
        environment_path.write_text(json.dumps(environment, indent=2))
    try:
        for case in manifest['cases']:
            for repetition in range(case['repetitions']):
                trial_id = f"{case['case_id']}-{repetition}"
                if trial_id in done:
                    continue
                label = f"{case['framework']}/{case['model']} on {case['device']}"
                print(f'[benchmark] Trial {completed_now + 1}/{pending}: {label}, repetition {repetition + 1}/{case["repetitions"]}. Checking idle resources.', flush=True)
                ensure_cpu_idle(threshold=manifest.get('settings', {}).get('max_cpu_percent', 80.0))
                ensure_memory_available(threshold=manifest.get('settings', {}).get('max_memory_percent', 90.0))
                if case['device'] != 'cpu':
                    ensure_gpus_idle(case.get('device_uuids', [case['device']]))
                print(f'[benchmark] Resources idle for {trial_id}; recording protection state.', flush=True)
                protection_before = capture_protection(condition)
                attempt_id = uuid.uuid4().hex
                start = time.monotonic_ns()
                utc_anchor = {'utc': datetime.now(timezone.utc).isoformat(), 'monotonic_ns': time.monotonic_ns()}
                stop = threading.Event()
                sampling_errors = []
                sampling_period = manifest.get('settings', {}).get('sample_interval_s', 1.0)
                def sample_loop():
                    previous_ns = None
                    while not stop.is_set():
                        try:
                            current_ns = time.monotonic_ns()
                            interval_s, missed = sample_interval(previous_ns, current_ns, sampling_period)
                            sample_to_file(session_dir / 'telemetry.jsonl', case['case_id'], trial_id, case.get('device_uuids', case['device']), attempt_id=attempt_id, interval_s=interval_s, missed_samples=missed, utc_anchor=utc_anchor, monotonic_ns=current_ns)
                            previous_ns = current_ns
                        except Exception as exc:
                            sampling_errors.append(str(exc))
                            try:
                                with (session_dir / 'telemetry.jsonl').open('a', encoding='utf-8') as file:
                                    error_sample = {'schema_version': 1, 'session_id': session_dir.name, 'case_id': case['case_id'], 'trial_id': trial_id, 'attempt_id': attempt_id, 'phase': 'worker_lifecycle', 'monotonic_ns': time.monotonic_ns(), 'utc_anchor': utc_anchor, 'interval_s': None, 'missed_samples': 0, 'sampling_error': str(exc)}
                                    validate_record('telemetry', error_sample)
                                    file.write(json.dumps(error_sample) + '\n')
                            except OSError:
                                pass
                            return
                        stop.wait(sampling_period)
                sampler = threading.Thread(target=sample_loop, daemon=True)
                sampler.start()
                print(f'[benchmark] Telemetry sampler started; launching isolated worker for {trial_id}.', flush=True)
                worker_start = time.monotonic()
                heartbeat_stop = threading.Event()
                def heartbeat():
                    while not heartbeat_stop.wait(10):
                        estimate = remaining_estimate(completed_seconds, completed_now, pending)
                        eta_text = f'approximate ETA {format_duration(estimate)} based on previous trials' if estimate is not None else 'ETA available after the first trial'
                        print(f'[benchmark] Still running {trial_id}: {format_duration(time.monotonic() - worker_start)} elapsed; {eta_text}.', flush=True)
                heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
                heartbeat_thread.start()
                try:
                    runtime_case = dict(case)
                    runtime_case.setdefault('timeout_s', manifest.get('settings', {}).get('trial_timeout_s', 600.0))
                    if case.get('diagnostic') in {'framework_profiler', 'compile'}:
                        trace_dir = session_dir / 'diagnostics' / trial_id / attempt_id
                        trace_dir.mkdir(parents=True, exist_ok=False)
                        runtime_case['profile_output'] = str(trace_dir)
                    result = run_worker(runtime_case)
                finally:
                    heartbeat_stop.set()
                    heartbeat_thread.join(timeout=1)
                    stop.set()
                    sampler.join(timeout=6)
                print(f'[benchmark] Worker finished; collecting telemetry and energy for {trial_id}.', flush=True)
                if (session_dir / 'telemetry.jsonl').exists():
                    with (session_dir / 'telemetry.jsonl').open(encoding='utf-8') as telemetry:
                        samples = [json.loads(line) for line in telemetry if line.strip()]
                    result.setdefault('metrics', {}).update(summarize_trial_telemetry(samples, trial_id, attempt_id))
                    if case['device'] != 'cpu':
                        energy = trial_gpu_energy(samples, trial_id, attempt_id)
                        result.setdefault('metrics', {})['gpu_board_energy_sampled'] = {'value': energy, 'unit': 'J', 'source': 'first selected GPU board power between first and last sample'}
                        throughput_unit = result.get('metrics', {}).get('throughput', {}).get('unit', '')
                        result['metrics']['gpu_board_energy_per_work'] = {'value': energy_per_work(energy, result.get('work_completed', 0)), 'unit': energy_work_unit(throughput_unit), 'source': 'sampled GPU board energy divided by completed work'}
                        for gpu_uuid in case.get('device_uuids', [case['device']]):
                            gpu_energy = gpu_energy_for_device(samples, trial_id, attempt_id, gpu_uuid)
                            result['metrics'][f'gpu_board_energy_{gpu_uuid}'] = {'value': gpu_energy, 'unit': 'J', 'source': 'sampled GPU board power between first and last sample'}
                    for domain, energy in trial_rapl_energy(samples, trial_id, attempt_id).items():
                        result.setdefault('metrics', {})[f'rapl_energy_{domain}'] = {'value': energy, 'unit': 'J', 'source': 'Linux powercap sampled counter deltas'}
                        unit = energy_work_unit(result.get('metrics', {}).get('throughput', {}).get('unit', ''))
                        result['metrics'][f'rapl_energy_per_work_{domain}'] = {'value': energy_per_work(energy, result.get('work_completed', 0)), 'unit': unit, 'source': 'sampled RAPL domain energy divided by completed work'}
                protection_after = capture_protection(condition)
                print(f'[benchmark] Checking protection state and saving result for {trial_id}.', flush=True)
                if sampling_errors:
                    result['warnings'] = [f'Telemetry collection failed: {error}' for error in sampling_errors]
                result['protection_before'] = protection_before
                result['protection_after'] = protection_after
                result['contamination'] = ['observable protection state or component set changed'] if protection_changed(protection_before, protection_after) else []
                result.update({'schema_version': 1, 'session_id': session_dir.name, 'trial_id': trial_id, 'attempt_id': attempt_id, 'case_id': case['case_id'], 'phase': 'worker_lifecycle', 'utc_anchor': utc_anchor, 'start_mono_ns': start, 'end_mono_ns': time.monotonic_ns()})
                try:
                    result['metrics'] = normalize_metrics(result.get('metrics', {}))
                    validate_record('trial', result)
                except (ValueError, TypeError) as exc:
                    result = {'schema_version': 1, 'session_id': session_dir.name, 'trial_id': trial_id, 'attempt_id': attempt_id, 'case_id': case['case_id'], 'phase': 'worker_lifecycle', 'utc_anchor': utc_anchor, 'start_mono_ns': start, 'end_mono_ns': time.monotonic_ns(), 'status': 'failed', 'metrics': {}, 'error': f'Invalid worker result: {exc}', 'protection_before': protection_before, 'protection_after': protection_after, 'contamination': []}
                    validate_record('trial', result)
                append_result(result_path, result)
                completed_now += 1
                completed_seconds += time.monotonic() - worker_start
                remaining = remaining_estimate(completed_seconds, completed_now, pending)
                print(f'[benchmark] Finished {completed_now}/{pending}: {result["status"]}; elapsed {format_duration(completed_seconds)}; approximate ETA {format_duration(remaining)}.', flush=True)
    finally:
        print('[benchmark] Generating session report.', flush=True)
        write_session_report(session_dir, session_dir / 'report')
