"""Regression checks for comparing local bindings across two PCs."""

from bdbench.compare import compare_session_groups, validate_environment_compatibility
from bdbench.portable import normalize_sessions
import pytest


def _session(uuid, root, value):
    case = {'case_id': uuid, 'framework': 'pytorch', 'model': 'cnn', 'device': uuid, 'dataset': 'generated', 'steps': 20, 'repetitions': 3, 'storage_root': root, 'storage_roots': [root]}
    manifest = {'manifest_id': uuid, 'settings': {'gpu_uuids': [uuid], 'storage_roots': [root], 'profile': 'standard', 'seed': 1234, 'repetitions': 3}, 'cases': [case]}
    records = [{'case_id': uuid, 'status': 'completed', 'runtime': {'framework': 'pytorch', 'version': '1'}, 'metrics': {'throughput': {'value': value, 'unit': 'samples/s'}}}]
    return manifest, records


def test_local_uuid_and_root_have_same_comparison_identity():
    before, after = normalize_sessions([_session('GPU-AAA', '/data/a', 100), _session('GPU-BBB', '/mnt/b', 90)])
    rows = compare_session_groups([before], [after])
    assert len(rows) == 1
    assert rows[0]['change_percent'] == pytest.approx(10)


def test_changed_workload_is_rejected():
    first = _session('GPU-AAA', '/data/a', 100)
    second = _session('GPU-BBB', '/mnt/b', 90)
    second[0]['cases'][0]['steps'] = 30
    try:
        normalize_sessions([first, second])
    except ValueError as exc:
        assert 'Workload cases' in str(exc)
    else:
        raise AssertionError('Changed work was accepted')


def test_physical_gpu_uuid_difference_is_allowed():
    def environment(uuid):
        return {'hardware': {'cpu': 'same CPU', 'gpus': [f'{uuid}, NVIDIA RTX 4070, 580.0, 12000, 200']}, 'software': {'kernel': 'same', 'lock_hashes': {'a': 'x'}, 'framework_versions': {'pytorch': {'torch': '1'}}}}
    validate_environment_compatibility(environment('GPU-A'), environment('GPU-B'))
