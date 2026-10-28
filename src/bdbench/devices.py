"""Stable device selection before framework import."""


def resolve_device_selection(selected_uuids: list[str], inventory: list[dict]) -> dict:
    """Resolve explicitly selected UUIDs to worker-local CUDA indices.

    Args:
        selected_uuids: Ordered UUIDs selected by the operator.
        inventory: Discovered devices containing UUID fields.

    Returns:
        CUDA visibility value and local UUID-to-index mapping.

    Raises:
        ValueError: If a UUID is duplicated or unavailable.
    """
    available = {item['uuid'] for item in inventory}
    if len(set(selected_uuids)) != len(selected_uuids):
        raise ValueError('GPU selections must be unique')
    missing = set(selected_uuids) - available
    if missing:
        raise ValueError(f'Selected GPU UUIDs are unavailable: {sorted(missing)}')
    return {'cuda_visible_devices': ','.join(selected_uuids), 'local_indices': {uuid: index for index, uuid in enumerate(selected_uuids)}}
