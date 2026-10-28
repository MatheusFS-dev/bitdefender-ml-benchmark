"""Read-only protection process evidence."""

from datetime import datetime, timezone
from pathlib import Path
import subprocess


def parse_product_status(output: str) -> dict:
    """Extract allowlisted product status fields from bduitool output.

    Args:
        output: Text returned by the documented product status command.

    Returns:
        Product version, signatures version and antimalware module state.
    """
    result = {'product_version': None, 'signatures_version': None, 'antimalware_module': 'unknown'}
    for line in output.splitlines():
        line = line.strip().lstrip('-').strip()
        key, separator, value = line.partition(':')
        if not separator:
            continue
        key, value = key.strip().lower(), value.strip()
        if key == 'product version':
            result['product_version'] = value
        elif key == 'signatures version':
            result['signatures_version'] = value
        elif key == 'antimalware status' and value.lower() in {'on', 'off'}:
            result['antimalware_module'] = value.lower()
    return result


def _processes() -> list[dict]:
    processes = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / 'cmdline').read_bytes().decode('utf-8', 'replace').split('\0')
            if command and command[0].startswith('/opt/bitdefender-security-tools/'):
                name = Path(command[0]).name
                if '-m' in command and command.index('-m') + 1 < len(command):
                    name += ':' + command[command.index('-m') + 1]
                processes.append({'pid': int(entry.name), 'name': name})
        except (OSError, IndexError):
            continue
    return processes


def capture_protection(claimed_condition: str) -> dict:
    """Capture available process evidence without inferring policy state.

    Args:
        claimed_condition: Operator supplied condition label.

    Returns:
        Timestamped evidence with an explicitly unknown observed policy state.
    """
    status = parse_product_status('')
    reason = None
    try:
        result = subprocess.run(['/opt/bitdefender-security-tools/bin/bduitool', 'get', 'ps'], capture_output=True, text=True, timeout=5, check=True)
        status = parse_product_status(result.stdout)
    except (OSError, subprocess.SubprocessError) as exc:
        reason = str(exc)
    observed = {'on': 'on_access_reported_on', 'off': 'on_access_reported_off'}.get(status['antimalware_module'], 'unknown')
    return {'schema_version': 1, 'captured_utc': datetime.now(timezone.utc).isoformat(), 'claimed_condition': claimed_condition, 'observed_state': observed, 'product_status': status, 'status_error': reason, 'components': _processes(), 'source': 'bduitool get ps and procfs', 'visibility': 'partial'}


def protection_changed(before: dict, after: dict) -> bool:
    """Detect changed observable state or component names.

    Args:
        before: Evidence before a trial.
        after: Evidence after a trial.

    Returns:
        Whether observed state or component membership changed.
    """
    names_before = {item.get('name') for item in before.get('components', [])}
    names_after = {item.get('name') for item in after.get('components', [])}
    return before.get('observed_state') != after.get('observed_state') or before.get('product_status') != after.get('product_status') or names_before != names_after
