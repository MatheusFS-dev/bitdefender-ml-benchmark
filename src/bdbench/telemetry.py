"""Read-only telemetry sampling."""

import subprocess
import time


def gpu_sample(uuid: str) -> dict:
    """Read basic NVIDIA telemetry for a GPU UUID.

    Args:
        uuid: NVIDIA GPU UUID.

    Returns:
        Sample with availability and a failure reason when unavailable.
    """
    command = ['nvidia-smi', '-i', uuid, '--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm', '--format=csv,noheader,nounits']
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
        fields = [item.strip() for item in result.stdout.strip().split(',')]
        if len(fields) != 5:
            raise ValueError('Unexpected GPU telemetry output')
        return {'availability': 'available', 'value': dict(zip(['utilization_percent', 'memory_used_mib', 'temperature_c', 'power_w', 'sm_clock_mhz'], [float(item) for item in fields])), 'extras': nvml_extras(uuid), 'source': 'nvidia-smi and NVML', 'monotonic_ns': time.monotonic_ns(), 'reason': None}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {'availability': 'missing', 'value': None, 'source': 'nvidia-smi', 'monotonic_ns': time.monotonic_ns(), 'reason': str(exc)}


def host_counters(psutil) -> dict:
    """Read host CPU, memory, swap and disk counters from a psutil backend.

    Args:
        psutil: psutil-compatible backend.

    Returns:
        System utilization and cumulative counters with units in field names.
    """
    times = psutil.cpu_times()
    stats = psutil.cpu_stats()
    swap = psutil.swap_memory()
    frequency = psutil.cpu_freq()
    disk = psutil.disk_io_counters()
    return {'cpu_percent': psutil.cpu_percent(), 'per_cpu_percent': psutil.cpu_percent(percpu=True), 'cpu_user_s': times.user, 'cpu_system_s': times.system, 'cpu_iowait_s': getattr(times, 'iowait', None), 'cpu_idle_s': times.idle, 'context_switches': stats.ctx_switches, 'interrupts': stats.interrupts, 'cpu_frequency_mhz': frequency.current if frequency else None, 'memory_available_bytes': psutil.virtual_memory().available, 'swap_used_bytes': swap.used, 'swap_in_bytes': swap.sin, 'swap_out_bytes': swap.sout, 'disk_io': disk._asdict() if disk else None}


def sample_host() -> dict:
    """Sample CPU, memory, swap and disk counters through psutil.

    Returns:
        Timestamped system counters or an unavailable reason.
    """
    try:
        import psutil
        return {'availability': 'available', 'value': host_counters(psutil), 'source': 'psutil', 'monotonic_ns': time.monotonic_ns(), 'reason': None}
    except (ImportError, OSError) as exc:
        return {'availability': 'missing', 'value': None, 'source': 'psutil', 'monotonic_ns': time.monotonic_ns(), 'reason': str(exc)}


def sample_to_file(path, case_id: str, trial_id: str, device: str | list[str], *, attempt_id: str | None = None, interval_s: float | None = None, missed_samples: int = 0, phase: str = 'worker_lifecycle', utc_anchor: dict | None = None, monotonic_ns: int | None = None) -> None:
    """Append a host and optional GPU sample to a JSONL file.

    Args:
        path: Destination JSONL path.
        case_id: Current case identifier.
        trial_id: Current trial identifier.
        device: CPU, selected GPU UUID, or selected GPU UUIDs.
        attempt_id: Unique trial attempt identifier.
        interval_s: Elapsed seconds since the previous sample, if known.
        missed_samples: Number of nominal sample slots missed.
        phase: Named measurement phase.
        utc_anchor: UTC and monotonic clock reference.
        monotonic_ns: Current sample timestamp if already captured.
    """
    import json
    from datetime import datetime, timezone
    from bdbench.serialization import validate_record
    devices = device if isinstance(device, list) else ([] if device == 'cpu' else [device])
    gpus = {uuid: gpu_sample(uuid) for uuid in devices}
    stamp = time.monotonic_ns() if monotonic_ns is None else monotonic_ns
    anchor = utc_anchor or {'utc': datetime.now(timezone.utc).isoformat(), 'monotonic_ns': stamp}
    sample = {'schema_version': 1, 'session_id': path.parent.name, 'case_id': case_id, 'trial_id': trial_id, 'attempt_id': attempt_id, 'phase': phase, 'monotonic_ns': stamp, 'utc_anchor': anchor, 'interval_s': interval_s, 'missed_samples': missed_samples, 'host': sample_host(), 'gpu': gpus[devices[0]] if devices else None, 'gpus': gpus, 'rapl': rapl_sample(), 'bitdefender': bitdefender_sample(), 'workers': worker_process_sample()}
    validate_record('telemetry', sample)
    with path.open('a', encoding='utf-8') as file:
        file.write(json.dumps(sample, allow_nan=False) + '\n')


def energy_delta_uj(previous: int, current: int, counter_range: int) -> int:
    """Calculate a single-wrap RAPL energy counter delta.

    Args:
        previous: Previous counter in microjoules.
        current: Current counter in microjoules.
        counter_range: Counter wrap range in microjoules.

    Returns:
        Energy delta in microjoules assuming at most one wrap.

    Raises:
        ValueError: If counters lie outside the reported range.
    """
    if counter_range <= 0 or not 0 <= previous < counter_range or not 0 <= current < counter_range:
        raise ValueError('Invalid RAPL counter values')
    return (current - previous) % counter_range


def integrate_power(samples: list[tuple[int, float | None]]) -> float | None:
    """Integrate sampled watts over monotonic timestamps with trapezoids.

    Args:
        samples: Ordered nanosecond timestamps and power in watts.

    Returns:
        Estimated joules, or None when data is incomplete.

    Raises:
        ValueError: If timestamps are not strictly increasing.
    """
    if len(samples) < 2 or any(power is None for _, power in samples):
        return None
    energy = 0.0
    for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
        if t1 <= t0:
            raise ValueError('Power sample timestamps must be increasing')
        energy += (p0 + p1) / 2 * (t1 - t0) / 1e9
    return energy


def trial_gpu_energy(records: list[dict], trial_id: str, attempt_id: str | None = None) -> float | None:
    """Estimate GPU board energy over observed samples for one trial.

    Args:
        records: Timestamped telemetry records.
        trial_id: Trial whose GPU samples should be integrated.
        attempt_id: Optional attempt identifier to isolate retries.

    Returns:
        Board joules between first and last sample, or None if unavailable.
    """
    samples = []
    for record in records:
        if record.get('trial_id') == trial_id and (attempt_id is None or record.get('attempt_id') == attempt_id):
            gpu = record.get('gpu') or {}
            value = gpu.get('value') or {}
            samples.append((record['monotonic_ns'], value.get('power_w')))
    return integrate_power(samples)


def rapl_sample(root=None) -> dict:
    """Read available Linux RAPL domain counters without elevation.

    Args:
        root: Optional powercap directory, primarily for fixture testing.

    Returns:
        Raw microjoule counters and ranges, or an unavailable reason.
    """
    from pathlib import Path
    root = Path('/sys/class/powercap') if root is None else root
    domains = []
    errors = []
    for directory in sorted(root.glob('intel-rapl:*')):
        try:
            domains.append({'id': directory.name, 'name': (directory / 'name').read_text().strip(), 'energy_uj': int((directory / 'energy_uj').read_text()), 'range_uj': int((directory / 'max_energy_range_uj').read_text())})
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    if not domains:
        return {'value': None, 'availability': 'missing', 'reason': '; '.join(errors) or 'RAPL counters not present', 'source': 'Linux powercap'}
    return {'value': domains, 'availability': 'available', 'reason': '; '.join(errors) or None, 'source': 'Linux powercap'}


def trial_rapl_energy(records: list[dict], trial_id: str, attempt_id: str | None = None) -> dict[str, float]:
    """Compute sampled RAPL domain energy deltas for a trial.

    Args:
        records: Timestamped telemetry records.
        trial_id: Trial identifier.
        attempt_id: Optional attempt identifier to isolate retries.

    Returns:
        Per-domain joules between sampled counters, without summing domains.
    """
    totals = {}
    previous = {}
    for record in records:
        if record.get('trial_id') != trial_id or (attempt_id is not None and record.get('attempt_id') != attempt_id):
            continue
        domains = (record.get('rapl') or {}).get('value') or []
        current = {item['id']: item for item in domains}
        for key in previous.keys() & current.keys():
            left, right = previous[key], current[key]
            if left['range_uj'] == right['range_uj']:
                delta = energy_delta_uj(left['energy_uj'], right['energy_uj'], left['range_uj'])
                totals[key] = totals.get(key, 0.0) + delta / 1e6
        previous = current
    return totals


def energy_per_work(energy: float | None, work: int) -> float | None:
    """Compute sampled component energy per completed unit of work.

    Args:
        energy: Estimated energy in joules, if available.
        work: Completed units such as samples or tokens.

    Returns:
        Joules per unit, or None for missing energy or no completed work.
    """
    if energy is None or work <= 0:
        return None
    return energy / work


from bdbench.protection import _processes


def bitdefender_sample() -> dict:
    """Read visible Bitdefender process counters without changing state.

    Returns:
        Per-process CPU, memory and I/O counters, or unavailable evidence.
    """
    components = _processes()
    if not components:
        return {'value': None, 'availability': 'missing', 'reason': 'No visible Bitdefender processes', 'source': 'procfs and psutil'}
    try:
        import psutil
    except ImportError as exc:
        return {'value': None, 'availability': 'missing', 'reason': str(exc), 'source': 'procfs and psutil'}
    records = []
    for item in components:
        try:
            process = psutil.Process(item['pid'])
            cpu = process.cpu_times()
            memory = process.memory_info()
            try:
                io = process.io_counters()._asdict()
            except (psutil.Error, OSError):
                io = None
            records.append({'pid': item['pid'], 'name': item['name'], 'created': process.create_time(), 'cpu_user_s': cpu.user, 'cpu_system_s': cpu.system, 'rss_bytes': memory.rss, 'io': io})
        except (psutil.Error, OSError):
            continue
    if not records:
        return {'value': None, 'availability': 'missing', 'reason': 'Visible process counters could not be read', 'source': 'procfs and psutil'}
    return {'value': records, 'availability': 'available', 'reason': None if len(records) == len(components) else 'Some process counters unavailable', 'source': 'procfs and psutil'}


def worker_process_sample() -> dict:
    """Sample the controller's visible child process tree.

    Returns:
        Per-process counters, including creation times, or missing evidence.
    """
    import os
    from pathlib import Path
    try:
        import psutil
    except ImportError as exc:
        return {'availability': 'missing', 'value': None, 'reason': str(exc), 'source': 'psutil child processes'}
    records = []
    try:
        children = psutil.Process(os.getpid()).children(recursive=True)
        for process in children:
            try:
                cpu = process.cpu_times()
                memory = process.memory_info()
                switches = process.num_ctx_switches()
                try:
                    io = process.io_counters()._asdict()
                except (psutil.Error, OSError):
                    io = None
                try:
                    pss = getattr(process.memory_full_info(), 'pss', None)
                except (psutil.Error, OSError):
                    pss = None
                try:
                    minor_faults, major_faults = parse_proc_faults((Path('/proc') / str(process.pid) / 'stat').read_text())
                except (OSError, ValueError):
                    minor_faults, major_faults = None, None
                records.append({'pid': process.pid, 'created': process.create_time(), 'name': process.name(), 'cpu_user_s': cpu.user, 'cpu_system_s': cpu.system, 'rss_bytes': memory.rss, 'pss_bytes': pss, 'minor_faults': minor_faults, 'major_faults': major_faults, 'voluntary_context_switches': switches.voluntary, 'involuntary_context_switches': switches.involuntary, 'io': io})
            except (psutil.Error, OSError):
                continue
    except (psutil.Error, OSError) as exc:
        return {'availability': 'missing', 'value': None, 'reason': str(exc), 'source': 'psutil child processes'}
    return {'availability': 'available', 'value': records, 'reason': None, 'source': 'psutil child processes'}


def summarize_trial_telemetry(records: list[dict], trial_id: str, attempt_id: str | None = None) -> dict:
    """Summarize observed resource samples for one trial.

    Args:
        records: Timestamped telemetry records.
        trial_id: Trial identifier.
        attempt_id: Optional attempt identifier to isolate retries.

    Returns:
        Sample-based mean utilization and peak memory measurements.
    """
    from statistics import mean
    records = [row for row in records if row.get('trial_id') == trial_id and (attempt_id is None or row.get('attempt_id') == attempt_id)]
    cpu, gpu, gpu_power, gpu_memory, worker_rss, security_rss = [], [], [], [], [], []
    security_previous = {}
    security_cpu_delta = []
    disk_previous = {}
    disk_deltas = {name: [] for name in ("read_bytes", "write_bytes", "read_count", "write_count", "read_time", "write_time")}
    for row in records:
        host = (row.get('host') or {}).get('value') or {}
        device = (row.get('gpu') or {}).get('value') or {}
        disk = host.get('disk_io') or {}
        for name, values in disk_deltas.items():
            if name in disk and name in disk_previous and disk[name] >= disk_previous[name]:
                values.append(disk[name] - disk_previous[name])
        disk_previous = disk
        if host.get('cpu_percent') is not None:
            cpu.append(host['cpu_percent'])
        if device.get('utilization_percent') is not None:
            gpu.append(device['utilization_percent'])
        if device.get('power_w') is not None:
            gpu_power.append(device['power_w'])
        if device.get('memory_used_mib') is not None:
            gpu_memory.append(device['memory_used_mib'])
        workers = (row.get('workers') or {}).get('value')
        if workers:
            worker_rss.append(sum(item['rss_bytes'] for item in workers if item.get('rss_bytes') is not None))
        security = (row.get('bitdefender') or {}).get('value')
        if security:
            security_rss.append(sum(item['rss_bytes'] for item in security if item.get('rss_bytes') is not None))
            current = {(item.get('pid'), item.get('created')): item['cpu_user_s'] + item['cpu_system_s'] for item in security if item.get('cpu_user_s') is not None and item.get('cpu_system_s') is not None}
            for identity in current.keys() & security_previous.keys():
                delta = current[identity] - security_previous[identity]
                if delta >= 0:
                    security_cpu_delta.append(delta)
            security_previous = current
        else:
            security_previous = {}
    def metric(values, unit, aggregate, source):
        return {'value': aggregate(values) if values else None, 'unit': unit, 'source': source, 'reason': None if values else 'No available samples'}
    result = {'cpu_utilization_mean': metric(cpu, '%', mean, 'psutil system sampling'), 'gpu_utilization_mean': metric(gpu, '%', mean, 'first selected GPU sampling'), 'gpu_power_mean': metric(gpu_power, 'W', mean, 'first selected GPU board power sampling'), 'gpu_memory_peak': metric(gpu_memory, 'MiB', max, 'first selected GPU sampling'), 'worker_rss_peak': metric(worker_rss, 'bytes', max, 'sampled sum of visible controller child RSS'), 'bitdefender_rss_peak': metric(security_rss, 'bytes', max, 'sampled sum of visible Bitdefender RSS'), 'bitdefender_cpu_time': metric(security_cpu_delta, 's', sum, 'sampled visible process CPU-time deltas')}
    for name, values in disk_deltas.items():
        unit = 'bytes' if name.endswith('bytes') else 'ms' if name.endswith('time') else 'operations'
        result['system_disk_' + name] = metric(values, unit, sum, 'sampled host-wide cumulative disk counter deltas, excludes resets')
    return result


def energy_work_unit(throughput_unit: str) -> str:
    """Map a throughput unit to the denominator of raw work counts.

    Args:
        throughput_unit: Unit attached to the case throughput.

    Returns:
        Joules-per-work-unit label consistent with raw work counts.
    """
    if throughput_unit == 'GFLOP/s':
        return 'J/FLOP'
    return 'J/' + (throughput_unit.removesuffix('/s') if throughput_unit.endswith('/s') else 'work unit')


def parse_proc_faults(stat_line: str) -> tuple[int, int]:
    """Parse Linux process minor and major page fault counters.

    Args:
        stat_line: A single `/proc/<pid>/stat` line.

    Returns:
        Minor and major page fault counts.

    Raises:
        ValueError: If required fields are absent or invalid.
    """
    closing = stat_line.rfind(')')
    if closing < 0:
        raise ValueError('Invalid process stat line')
    fields = stat_line[closing + 1:].split()
    if len(fields) < 10:
        raise ValueError('Incomplete process stat line')
    return int(fields[7]), int(fields[9])


def nvml_extras(uuid: str, backend=None) -> dict:
    """Read optional GPU memory, PCIe, clock and throttling metrics.

    Args:
        uuid: NVIDIA GPU UUID.
        backend: Optional NVML-compatible test backend.

    Returns:
        Measurements with values or explicit unavailable reasons.
    """
    names = ('memory_utilization_percent', 'memory_clock_mhz', 'pcie_tx_kb_s', 'pcie_rx_kb_s', 'throttle_reason_bits', 'power_limit_w')
    try:
        if backend is None:
            import pynvml as backend
        backend.nvmlInit()
        handle = backend.nvmlDeviceGetHandleByUUID(uuid)
    except Exception as exc:
        return {name: {'value': None, 'reason': str(exc)} for name in names}
    try:
        getters = {
            'memory_utilization_percent': lambda: backend.nvmlDeviceGetUtilizationRates(handle).memory,
            'memory_clock_mhz': lambda: backend.nvmlDeviceGetClockInfo(handle, backend.NVML_CLOCK_MEM),
            'pcie_tx_kb_s': lambda: backend.nvmlDeviceGetPcieThroughput(handle, backend.NVML_PCIE_UTIL_TX_BYTES),
            'pcie_rx_kb_s': lambda: backend.nvmlDeviceGetPcieThroughput(handle, backend.NVML_PCIE_UTIL_RX_BYTES),
            'throttle_reason_bits': lambda: backend.nvmlDeviceGetCurrentClocksThrottleReasons(handle),
            'power_limit_w': lambda: backend.nvmlDeviceGetPowerManagementLimit(handle) / 1000,
        }
        result = {}
        for name, getter in getters.items():
            try:
                result[name] = {'value': getter(), 'reason': None}
            except Exception as exc:
                result[name] = {'value': None, 'reason': str(exc)}
        return result
    finally:
        backend.nvmlShutdown()


def sample_interval(previous_ns: int | None, current_ns: int, period_s: float) -> tuple[float | None, int]:
    """Compute actual elapsed time and missed sampling intervals.

    Args:
        previous_ns: Previous sample's monotonic timestamp, if any.
        current_ns: Current sample's monotonic timestamp.
        period_s: Nominal sampling period in seconds.

    Returns:
        Elapsed seconds or None for the first sample, and missed interval count.

    Raises:
        ValueError: If the clock moves backward or the period is invalid.
    """
    if period_s <= 0 or previous_ns is not None and current_ns <= previous_ns:
        raise ValueError('Invalid sampling interval')
    if previous_ns is None:
        return None, 0
    interval = (current_ns - previous_ns) / 1e9
    return interval, max(0, int(interval // period_s) - 1)


def gpu_energy_for_device(records: list[dict], trial_id: str, attempt_id: str, uuid: str) -> float | None:
    """Integrate sampled board power for one selected GPU and attempt.

    Args:
        records: Timestamped telemetry records.
        trial_id: Trial identifier.
        attempt_id: Attempt identifier.
        uuid: Physical GPU UUID.

    Returns:
        Estimated board joules over observed samples, or None if incomplete.
    """
    samples = []
    for record in records:
        if record.get('trial_id') != trial_id or record.get('attempt_id') != attempt_id:
            continue
        gpu = (record.get('gpus') or {}).get(uuid) or {}
        value = gpu.get('value') or {}
        samples.append((record['monotonic_ns'], value.get('power_w')))
    return integrate_power(samples)
