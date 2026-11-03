"""Map machine-local case identifiers to comparable workload identities."""

from copy import deepcopy
from bdbench.serialization import canonical_hash


def case_key(case: dict, settings: dict) -> str:
    """Return a hardware-neutral key for one frozen case.

    Args:
        case: Frozen case descriptor.
        settings: Manifest settings with local device and storage bindings.

    Returns:
        Canonical comparable workload key.
    """
    item = deepcopy(case)
    item.pop('case_id', None)
    for field in ('calibration_seconds_per_step', 'profile_output'):
        item.pop(field, None)
    uuids = settings.get('gpu_uuids', [])
    roots = settings.get('storage_roots', [])
    if item.get('device') in uuids:
        item['device'] = f'gpu:{uuids.index(item["device"])}'
    if 'device_uuids' in item:
        item['device_uuids'] = [f'gpu:{uuids.index(value)}' for value in item['device_uuids']]
    if 'storage_root' in item:
        item['storage_root'] = f'storage:{roots.index(item["storage_root"])}'
    if 'storage_roots' in item:
        item['storage_roots'] = [f'storage:{roots.index(value)}' for value in item['storage_roots']]
    if 'dataset_path' in item:
        item['dataset_path'] = item.get('dataset')
    return canonical_hash(item)[:16]


def normalize_sessions(sessions: list[tuple[dict, list[dict]]]) -> list[tuple[dict, list[dict]]]:
    """Replace local case IDs with common workload keys for comparison.

    Args:
        sessions: Manifest and result pairs.

    Returns:
        Copies with portable case IDs and a shared logical manifest identity.

    Raises:
        ValueError: If cases or relevant data contents differ.
    """
    output = []
    reference = None
    for manifest, records in sessions:
        settings = manifest['settings']
        datasets = {name: data.get('sha256') for name, data in settings.get('datasets', {}).items()}
        mapping = {case['case_id']: case_key(case, settings) for case in manifest['cases']}
        keys = sorted(mapping.values())
        identity = (keys, datasets, settings.get('profile'), settings.get('seed'), settings.get('repetitions'))
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError('Workload cases, dataset hashes or fixed settings differ between machines')
        copied = deepcopy(manifest)
        copied['manifest_id'] = canonical_hash(identity)
        for case in copied['cases']:
            case['case_id'] = mapping[case['case_id']]
        results = []
        for record in records:
            item = deepcopy(record)
            if item.get('case_id') in mapping:
                item['case_id'] = mapping[item['case_id']]
            results.append(item)
        output.append((copied, results))
    return output
