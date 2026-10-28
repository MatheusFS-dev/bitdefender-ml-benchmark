"""Bounded benchmark-owned storage directories."""

from pathlib import Path
import shutil

_MARKER = '.bdbench-owner'
_MAGIC = 'bdbench-owned-directory-v1\n'


def claim_storage(root: Path) -> Path:
    """Create or open the benchmark-owned directory under a selected root.

    Args:
        root: Existing directory explicitly selected by the operator.

    Returns:
        Owned directory path.

    Raises:
        ValueError: If the path is unsafe or already owned by something else.
    """
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Storage root must be an existing non-symlink directory')
    root = root.resolve(strict=True)
    owned = root / '.bdbench-data'
    if owned.is_symlink():
        raise ValueError('Owned directory cannot be a symlink')
    if owned.exists():
        marker = owned / _MARKER
        if not owned.is_dir() or marker.is_symlink() or not marker.is_file() or marker.read_text() != _MAGIC:
            raise ValueError('Existing directory is not owned by this benchmark')
    else:
        owned.mkdir(mode=0o700)
        (owned / _MARKER).write_text(_MAGIC)
    return owned


def cleanup_owned(owned: Path) -> None:
    """Remove only a marked benchmark directory.

    Args:
        owned: Directory previously returned by claim_storage.

    Raises:
        ValueError: If ownership cannot be verified.
    """
    marker = owned / _MARKER
    if owned.name != '.bdbench-data' or owned.is_symlink() or marker.is_symlink() or not marker.is_file() or marker.read_text() != _MAGIC:
        raise ValueError('Refusing to remove an unowned directory')
    shutil.rmtree(owned)


def check_budget(root: Path, used_bytes: int, additional_bytes: int, limit_bytes: int = 20 * 1024**3, minimum_free_fraction: float = .15) -> None:
    """Check per-root size and remaining free-space limits.

    Args:
        root: Existing storage root.
        used_bytes: Bytes already attributed to the benchmark.
        additional_bytes: Proposed additional allocation.
        limit_bytes: Maximum benchmark bytes for this root.
        minimum_free_fraction: Minimum fraction of filesystem capacity remaining.

    Raises:
        ValueError: If any limit would be exceeded.
    """
    if min(used_bytes, additional_bytes, limit_bytes) < 0 or not 0 <= minimum_free_fraction < 1:
        raise ValueError('Invalid storage budget')
    disk = shutil.disk_usage(root)
    if used_bytes + additional_bytes > limit_bytes or disk.free - additional_bytes < disk.total * minimum_free_fraction:
        raise ValueError('Storage budget or free-space reserve exceeded')


def owned_bytes(directory: Path) -> int:
    """Count bytes in an owned directory while rejecting symlinks.

    Args:
        directory: Existing benchmark-owned directory.

    Returns:
        Total regular-file bytes.

    Raises:
        ValueError: If a symbolic link occurs within the directory.
    """
    total = 0
    for path in directory.rglob('*'):
        if path.is_symlink():
            raise ValueError(f'Symbolic link in benchmark directory: {path}')
        if path.is_file():
            total += path.stat().st_size
    return total


def check_total_budget(roots: list[Path], additional_bytes: int, limit_bytes: int = 40 * 1024**3) -> None:
    """Enforce the aggregate storage limit across selected roots.

    Args:
        roots: Explicit storage directories.
        additional_bytes: Proposed additional allocation.
        limit_bytes: Maximum combined benchmark bytes.

    Raises:
        ValueError: If the aggregate limit or input validity is violated.
    """
    if additional_bytes < 0 or limit_bytes < 0:
        raise ValueError('Invalid aggregate storage budget')
    if any(root.is_symlink() for root in roots):
        raise ValueError('Storage roots cannot be symbolic links')
    unique = {root.resolve(strict=True) for root in roots}
    used = 0
    for root in unique:
        owned = root / '.bdbench-data'
        if owned.exists() or owned.is_symlink():
            owned = claim_storage(root)
            used += owned_bytes(owned)
    if used + additional_bytes > limit_bytes:
        raise ValueError('Aggregate storage budget exceeded')
