"""Dataset integrity checks and safe archive extraction."""

import hashlib
from pathlib import Path
import tarfile


def verify_sha256(path: Path, expected: str) -> bool:
    """Compare a file with its expected SHA-256 digest.

    Args:
        path: File to hash.
        expected: Expected lowercase hexadecimal digest.

    Returns:
        Whether the digest matches.
    """
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest() == expected.lower()


def extract_tar(archive: Path, destination: Path, max_bytes: int = 1024**3) -> None:
    """Extract regular files and directories from a validated tar archive.

    Args:
        archive: Tar archive to read.
        destination: New directory to populate.
        max_bytes: Maximum sum of uncompressed file sizes.

    Raises:
        ValueError: If a member escapes the destination or is a link/device.
    """
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve(strict=True)
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        if sum(member.size for member in members if member.isfile()) > max_bytes:
            raise ValueError('Archive exceeds extraction size limit')
        for member in members:
            target = base / member.name
            if not (member.isdir() or member.isfile()) or not target.resolve().is_relative_to(base):
                raise ValueError(f'Unsafe archive member: {member.name}')
        for member in members:
            target = base / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise ValueError(f'Cannot read archive member: {member.name}')
                with source, target.open('xb') as output:
                    while block := source.read(1024 * 1024):
                        output.write(block)


def download_checked(url: str, destination: Path, expected_md5: str, fetch=None, max_bytes: int = 250 * 1024 * 1024) -> dict:
    """Download an archive atomically and validate its published checksum.

    Args:
        url: Dataset publisher URL.
        destination: Output archive path.
        expected_md5: Publisher's expected MD5 for corruption detection.
        fetch: Optional injected downloader accepting URL and temporary path.
        max_bytes: Maximum allowed archive size.

    Returns:
        Provenance and computed SHA-256 digest.

    Raises:
        ValueError: If the existing or downloaded archive fails validation.
    """
    import os
    import tempfile
    import urllib.request
    def checksum(path, algorithm):
        digest = hashlib.new(algorithm)
        with path.open('rb') as file:
            for block in iter(lambda: file.read(1048576), b''):
                digest.update(block)
        return digest.hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size > max_bytes:
            raise ValueError('Existing dataset archive exceeds the size limit')
        if checksum(destination, 'md5') != expected_md5:
            raise ValueError('Existing dataset checksum does not match')
    else:
        fd, name = tempfile.mkstemp(prefix='.download-', dir=destination.parent)
        os.close(fd)
        temporary = Path(name)
        try:
            if fetch:
                fetch(url, temporary)
            else:
                with urllib.request.urlopen(url, timeout=60) as source, temporary.open('wb') as target:
                    written = 0
                    while block := source.read(1048576):
                        written += len(block)
                        if written > max_bytes:
                            raise ValueError('Downloaded dataset archive exceeds the size limit')
                        target.write(block)
            if temporary.stat().st_size > max_bytes:
                raise ValueError('Downloaded dataset archive exceeds the size limit')
            if checksum(temporary, 'md5') != expected_md5:
                raise ValueError('Downloaded dataset checksum does not match')
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return {'url': url, 'md5': expected_md5, 'sha256': checksum(destination, 'sha256'), 'path': str(destination)}


def prepare_cifar10(root: Path) -> dict:
    """Prepare the published CIFAR-10 archive in an owned storage root.

    Args:
        root: Existing user-selected storage directory.

    Returns:
        Archive provenance and extracted dataset directory.
    """
    import json
    import tempfile
    from bdbench.storage import claim_storage, check_budget, owned_bytes
    owned = claim_storage(root)
    check_budget(root, owned_bytes(owned), 1200 * 1024 * 1024)
    datasets = owned / 'datasets'
    datasets.mkdir(exist_ok=True)
    archive = datasets / 'cifar-10-python.tar.gz'
    info = download_checked('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', archive, 'c58f30108f718f92721af3b95e74349a')
    target = datasets / 'cifar10'
    if not target.exists():
        with tempfile.TemporaryDirectory(prefix='.extract-', dir=datasets) as tmp:
            extracted = Path(tmp) / 'data'
            extract_tar(archive, extracted)
            extracted.rename(target)
    prepare_cifar_images(target)
    prepare_cifar_layouts(target)
    prepare_tfrecord(target)
    info['path'] = str(target)
    info['files'] = dataset_files(target)
    info['license'] = 'See https://www.cs.toronto.edu/~kriz/cifar.html'
    (datasets / 'cifar10-provenance.json').write_text(json.dumps(info, indent=2))
    return info


def extract_zip(archive: Path, destination: Path, max_bytes: int = 1024**3) -> None:
    """Extract a ZIP archive after rejecting paths and symbolic links.

    Args:
        archive: ZIP file to extract.
        destination: Destination directory.
        max_bytes: Maximum sum of uncompressed file sizes.

    Raises:
        ValueError: If an archive member is unsafe.
    """
    import stat
    import zipfile
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve(strict=True)
    with zipfile.ZipFile(archive) as source:
        if sum(info.file_size for info in source.infolist()) > max_bytes:
            raise ValueError('Archive exceeds extraction size limit')
        for info in source.infolist():
            mode = info.external_attr >> 16
            target = base / info.filename
            if stat.S_ISLNK(mode) or not target.resolve().is_relative_to(base):
                raise ValueError(f'Unsafe archive member: {info.filename}')
        for info in source.infolist():
            target = base / info.filename
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(info) as inp, target.open('xb') as out:
                    while block := inp.read(1048576):
                        out.write(block)


def prepare_wikitext(root: Path) -> dict:
    """Prepare WikiText-2 with publisher checksum and deterministic tokens.

    Args:
        root: Existing user-selected storage directory.

    Returns:
        Provenance and path to prepared tokenized data.
    """
    import collections
    import json
    import tempfile
    from bdbench.storage import claim_storage, check_budget, owned_bytes
    owned = claim_storage(root)
    check_budget(root, owned_bytes(owned), 200 * 1024 * 1024)
    base = owned / 'datasets'
    base.mkdir(exist_ok=True)
    archive = base / 'wikitext-2-v1.zip'
    info = download_checked('https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip', archive, '542ccefacc6c27f945fb54453812b3cd')
    target = base / 'wikitext2'
    if not target.exists():
        with tempfile.TemporaryDirectory(prefix='.wiki-', dir=base) as tmp:
            extracted = Path(tmp) / 'data'
            extract_zip(archive, extracted)
            train = (extracted / 'wikitext-2/wiki.train.tokens').read_text(encoding='utf-8').split()
            vocab = ['<unk>'] + [word for word, _ in sorted(collections.Counter(train).items(), key=lambda pair: (-pair[1], pair[0]))[:31999] if word != '<unk>']
            lookup = {word: i for i, word in enumerate(vocab)}
            for split in ('train', 'valid', 'test'):
                words = (extracted / f'wikitext-2/wiki.{split}.tokens').read_text(encoding='utf-8').split()
                (extracted / f'{split}.json').write_text(json.dumps([lookup.get(word, 0) for word in words]))
            (extracted / 'vocab.json').write_text(json.dumps(vocab, ensure_ascii=False))
            extracted.rename(target)
    info['path'] = str(target)
    info['files'] = dataset_files(target)
    info['tokenizer'] = 'whitespace, top 32000 tokens by frequency then lexical order, unknown=0'
    info['license'] = 'See https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/'
    (base / 'wikitext2-provenance.json').write_text(json.dumps(info, indent=2))
    return info


def dataset_files(root: Path) -> dict[str, str]:
    """Hash all regular dataset files, rejecting symbolic links.

    Args:
        root: Prepared dataset directory.

    Returns:
        Relative file names mapped to SHA-256 digests.

    Raises:
        ValueError: If a symlink appears inside the dataset.
    """
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Dataset symlink is not allowed: {path}')
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as file:
                for block in iter(lambda: file.read(1048576), b''):
                    digest.update(block)
            result[str(path.relative_to(root))] = digest.hexdigest()
    return result


def verify_dataset_files(root: Path, expected: dict[str, str]) -> None:
    """Verify prepared dataset files against a frozen manifest.

    Args:
        root: Dataset directory.
        expected: Relative paths and expected SHA-256 digests.

    Raises:
        ValueError: If a file is missing, added, modified or is a symlink.
    """
    if not root.is_dir() or root.is_symlink() or dataset_files(root) != expected:
        raise ValueError(f'Prepared dataset differs from frozen manifest: {root}')


def write_image_layout(root: Path, samples, encoder) -> int:
    """Write deterministically named images in class-specific directories.

    Args:
        root: Destination directory.
        samples: Iterable of image data and integer class labels.
        encoder: Callable accepting image data and output path.

    Returns:
        Number of images written.

    Raises:
        ValueError: If a label is outside the CIFAR-10 class range.
    """
    count = 0
    for count, (image, label) in enumerate(samples, start=1):
        if not isinstance(label, int) or not 0 <= label <= 9:
            raise ValueError('Invalid CIFAR-10 class label')
        directory = root / str(label)
        directory.mkdir(parents=True, exist_ok=True)
        encoder(image, directory / f'{count - 1:08d}.jpg')
    return count


def prepare_cifar_images(dataset: Path) -> None:
    """Convert prepared CIFAR-10 training batches into individual JPEGs.

    Args:
        dataset: Extracted CIFAR-10 dataset directory.
    """
    import pickle
    import tempfile
    from PIL import Image
    target = dataset / 'images'
    if target.exists():
        return
    def samples():
        for index in range(1, 6):
            with (dataset / 'cifar-10-batches-py' / f'data_batch_{index}').open('rb') as file:
                batch = pickle.load(file, encoding='bytes')
            images = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            for image, label in zip(images, batch[b'labels']):
                yield image, int(label)
    def encode(image, path):
        Image.fromarray(image).save(path, format='JPEG', quality=90)
    with tempfile.TemporaryDirectory(prefix='.images-', dir=dataset) as tmp:
        images = Path(tmp) / 'images'
        images.mkdir()
        if write_image_layout(images, samples(), encode) != 50000:
            raise ValueError('Unexpected CIFAR-10 training sample count')
        images.rename(target)


def image_paths(root: Path) -> tuple[list[str], list[int]]:
    """List prepared JPEG files and their class labels deterministically.

    Args:
        root: Prepared per-class image directory.

    Returns:
        Parallel lists of paths and integer labels.

    Raises:
        ValueError: If the image layout is empty or unsafe.
    """
    paths, labels = [], []
    for label in range(10):
        directory = root / str(label)
        if directory.is_symlink():
            raise ValueError('Image directory cannot be a symlink')
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob('*.jpg')):
            if path.is_symlink():
                raise ValueError('Image file cannot be a symlink')
            paths.append(str(path))
            labels.append(label)
    if not paths:
        raise ValueError('No prepared images found')
    return paths, labels


def shard_paths(root: Path) -> list[Path]:
    """List prepared array shards in deterministic order.

    Args:
        root: Directory containing NumPy shards.

    Returns:
        Sorted shard paths.

    Raises:
        ValueError: If no shards exist or any shard is a symbolic link.
    """
    paths = sorted(root.glob('*.npz'))
    if not paths or root.is_symlink() or any(path.is_symlink() for path in paths):
        raise ValueError('No safe prepared array shards found')
    return paths


def prepare_cifar_layouts(dataset: Path) -> None:
    """Prepare memory-mapped arrays and uncompressed NumPy shards.

    Args:
        dataset: Extracted CIFAR-10 dataset directory.
    """
    import pickle
    import tempfile
    import numpy as np
    target = dataset / 'array_layouts'
    if target.exists():
        return
    with tempfile.TemporaryDirectory(prefix='.arrays-', dir=dataset) as tmp:
        staging = Path(tmp) / 'layouts'
        staging.mkdir()
        shards = staging / 'shards'
        shards.mkdir()
        images = np.lib.format.open_memmap(staging / 'images.npy', mode='w+', dtype=np.uint8, shape=(50000, 32, 32, 3))
        labels = np.lib.format.open_memmap(staging / 'labels.npy', mode='w+', dtype=np.int64, shape=(50000,))
        offset = 0
        for index in range(1, 6):
            with (dataset / 'cifar-10-batches-py' / f'data_batch_{index}').open('rb') as file:
                batch = pickle.load(file, encoding='bytes')
            array = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy()
            y = np.asarray(batch[b'labels'], dtype=np.int64)
            end = offset + len(y)
            images[offset:end] = array
            labels[offset:end] = y
            np.savez(shards / f'{index:02d}.npz', images=array, labels=y)
            offset = end
        if offset != 50000:
            raise ValueError('Unexpected CIFAR-10 training sample count')
        images.flush()
        labels.flush()
        del images, labels
        staging.rename(target)


def iter_array_samples(dataset: str, layout: str):
    """Iterate memory-mapped arrays or packed shards without eager caching.

    Args:
        dataset: Prepared CIFAR-10 directory.
        layout: Either cifar10_mmap or cifar10_shards.

    Yields:
        Image arrays and integer labels in deterministic order.

    Raises:
        ValueError: If the layout is unsupported.
    """
    if layout not in {'cifar10_mmap', 'cifar10_shards'}:
        raise ValueError('Unsupported array layout')
    import numpy as np
    root = Path(dataset) / 'array_layouts'
    if layout == 'cifar10_mmap':
        images = np.load(root / 'images.npy', mmap_mode='r', allow_pickle=False)
        labels = np.load(root / 'labels.npy', mmap_mode='r', allow_pickle=False)
        for image, label in zip(images, labels):
            yield image, int(label)
    else:
        for path in shard_paths(root / 'shards'):
            with np.load(path, allow_pickle=False) as shard:
                for image, label in zip(shard['images'], shard['labels']):
                    yield image, int(label)


import os
import subprocess


def prepare_tfrecord(dataset: Path) -> None:
    """Create TFRecord shards using the isolated TensorFlow environment.

    Args:
        dataset: Prepared CIFAR-10 directory.

    Raises:
        subprocess.SubprocessError: If the conversion worker fails.
    """
    if (dataset / 'tfrecord').exists():
        return
    root = Path(__file__).resolve().parents[2]
    python = root / 'environments/tensorflow/.venv/bin/python'
    environment = os.environ.copy()
    environment['CUDA_VISIBLE_DEVICES'] = ''
    environment['PYTHONPATH'] = str(root / 'src')
    subprocess.run([str(python), '-m', 'bdbench.tfrecord_prepare', str(dataset)], env=environment, check=True, timeout=1800)
