"""Convert prepared CIFAR arrays to TensorFlow record shards."""

from pathlib import Path
import sys
import tempfile


def prepare(dataset: Path) -> None:
    """Write deterministic TFRecord shards from prepared CIFAR arrays.

    Args:
        dataset: Prepared CIFAR dataset directory.
    """
    import tensorflow as tf
    from bdbench.datasets import iter_array_samples
    target = dataset / 'tfrecord'
    if target.exists():
        return
    with tempfile.TemporaryDirectory(prefix='.tfrecord-', dir=dataset) as tmp:
        staging = Path(tmp) / 'records'
        staging.mkdir()
        writer = None
        count = 0
        try:
            for index, (image, label) in enumerate(iter_array_samples(str(dataset), 'cifar10_mmap')):
                if index % 10000 == 0:
                    if writer:
                        writer.close()
                    writer = tf.io.TFRecordWriter(str(staging / f'{index // 10000:02d}.tfrecord'))
                example = tf.train.Example(features=tf.train.Features(feature={'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])), 'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}))
                writer.write(example.SerializeToString())
                count += 1
        finally:
            if writer:
                writer.close()
        if count != 50000:
            raise ValueError('Unexpected TFRecord example count')
        staging.rename(target)


if __name__ == '__main__':
    prepare(Path(sys.argv[1]))
