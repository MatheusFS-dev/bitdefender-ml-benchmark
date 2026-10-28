"""Detect unrelated GPU compute workloads without changing them."""

import subprocess


def conflicts(selected: list[str], processes: list[dict]) -> list[dict]:
    """Filter running compute processes to explicitly selected GPUs.

    Args:
        selected: Selected physical GPU UUIDs.
        processes: Observed GPU process records.

    Returns:
        Processes using selected GPUs.
    """
    return [process for process in processes if process.get('gpu_uuid') in selected]


def gpu_processes() -> list[dict] | None:
    """Query NVIDIA compute processes, returning None if unavailable.

    Returns:
        Process records or None when enumeration cannot be performed.
    """
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=5)
        records = []
        for line in result.stdout.splitlines():
            fields = [value.strip() for value in line.split(',')]
            if len(fields) == 3:
                records.append({'gpu_uuid': fields[0], 'pid': int(fields[1]), 'memory_mib': fields[2]})
        return records
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def ensure_gpus_idle(selected: list[str]) -> None:
    """Reject use of GPUs with competing compute processes.

    Args:
        selected: Physical GPU UUIDs selected by the operator.

    Raises:
        RuntimeError: If selected GPUs are busy or cannot be inspected.
    """
    if not selected:
        return
    processes = gpu_processes()
    if processes is None:
        raise RuntimeError('Cannot inspect selected GPUs for competing workloads')
    busy = conflicts(selected, processes)
    if busy:
        raise RuntimeError(f'Selected GPUs have active compute processes: {busy}. Stop or wait for those workloads before retrying.')


def ensure_cpu_idle(sample=None, threshold: float = 80.0) -> None:
    """Reject trial startup when host CPU activity is sustained and high.

    Args:
        sample: Optional callable returning host CPU percentage.
        threshold: Maximum accepted CPU percentage.

    Raises:
        RuntimeError: If utilization is too high or cannot be measured.
    """
    if sample is None:
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError('Cannot measure host CPU activity') from exc
        sample = lambda: psutil.cpu_percent(interval=3)
    value = sample()
    if value > threshold:
        raise RuntimeError(f'Host CPU activity is {value:.1f}%, above {threshold:.1f}%. Wait for background workloads before retrying.')


def ensure_memory_available(sample=None, threshold: float = 90.0) -> None:
    """Reject new work when host memory utilization exceeds a limit.

    Args:
        sample: Optional callable returning host memory utilization percent.
        threshold: Maximum accepted percent of memory in use.

    Raises:
        RuntimeError: If memory pressure is high or cannot be checked.
    """
    if sample is None:
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError('Cannot measure host memory use') from exc
        sample = lambda: psutil.virtual_memory().percent
    value = sample()
    if value > threshold:
        raise RuntimeError(f'Host memory use is {value:.1f}%, above {threshold:.1f}%. Wait for memory to become available.')
