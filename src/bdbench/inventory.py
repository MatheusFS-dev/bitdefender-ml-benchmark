"""Read-only hardware and software identity capture."""

import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys
import shutil

ROOT = Path(__file__).resolve().parents[2]


def _gpu_inventory():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,driver_version,memory.total,power.limit', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=5)
        return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def inventory_environment() -> dict:
    """Capture stable hardware and software identities for comparison.

    Returns:
        Hardware and software metadata with unavailable data represented as null.
    """
    cpu = None
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                cpu = line.split(':', 1)[1].strip()
                break
    except OSError:
        pass
    locks = {}
    for path in [ROOT / 'uv.lock', ROOT / 'environments/pytorch/uv.lock', ROOT / 'environments/tensorflow/uv.lock']:
        locks[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    logical = os.cpu_count() or 1
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or logical
    except ImportError:
        physical = logical
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = logical
    try:
        cpu_max = Path('/sys/fs/cgroup/cpu.max').read_text().strip()
        effective = effective_cpu_count(logical, affinity, cpu_max)
    except (OSError, ValueError, ZeroDivisionError):
        effective = min(logical, affinity)
    governors = []
    for path in Path('/sys/devices/system/cpu').glob('cpu[0-9]*/cpufreq/scaling_governor'):
        try:
            governors.append(path.read_text().strip())
        except OSError:
            continue
    memory_bytes = None
    try:
        memory_bytes = psutil.virtual_memory().total
    except (ImportError, NameError):
        pass
    disk = shutil.disk_usage(ROOT)
    return {'hardware': {'cpu': cpu, 'logical_cpus': logical, 'physical_cores': physical, 'effective_cpus': effective, 'memory_bytes': memory_bytes, 'storage_total_bytes': disk.total, 'gpus': _gpu_inventory()}, 'software': {'system': platform.system(), 'distribution': platform.freedesktop_os_release().get('PRETTY_NAME') if hasattr(platform, 'freedesktop_os_release') else None, 'kernel': platform.release(), 'architecture': platform.machine(), 'python': sys.version, 'framework_versions': framework_versions(), 'cpu_governors': sorted(set(governors)) or None, 'lock_hashes': locks}}


def effective_cpu_count(logical: int, affinity: int, cpu_max: str) -> int:
    """Compute effective CPU thread capacity from affinity and cgroup quota.

    Args:
        logical: Logical processors detected by the OS.
        affinity: Process CPU affinity count.
        cpu_max: Linux cgroup v2 cpu.max content.

    Returns:
        Conservative positive integer CPU capacity.

    Raises:
        ValueError: If input values are invalid.
    """
    if logical < 1 or affinity < 1:
        raise ValueError('CPU counts must be positive')
    quota, period = cpu_max.split()
    capacity = min(logical, affinity)
    if quota != 'max':
        capacity = min(capacity, max(1, int(quota) // int(period)))
    return capacity


def framework_versions() -> dict:
    """Read installed framework package versions without importing frameworks.

    Returns:
        Per-environment package versions, or None if a venv is unavailable.
    """
    import json
    result = {}
    program = "import importlib.metadata as m,json,sys; print(json.dumps({name:m.version(name) for name in sys.argv[1:]}))"
    for framework, packages in [('pytorch', ['torch', 'torchvision']), ('tensorflow', ['tensorflow'])]:
        python = ROOT / 'environments' / framework / '.venv/bin/python'
        if not python.is_file():
            result[framework] = None
            continue
        try:
            completed = subprocess.run([str(python), '-c', program, *packages], capture_output=True, text=True, check=True, timeout=10)
            result[framework] = json.loads(completed.stdout)
        except (OSError, subprocess.SubprocessError, ValueError):
            result[framework] = None
    return result
