"""Isolated framework training worker."""

import json
import math
import sys
import time
from bdbench.progress import emit_progress


def matmul_flops(size: int, steps: int) -> int:
    """Count conventional floating-point operations for square matrix products.

    Args:
        size: Matrix dimension.
        steps: Number of products.

    Returns:
        Conventional operation count using two operations per multiply-add.
    """
    if size <= 0 or steps <= 0:
        raise ValueError('Matrix size and steps must be positive')
    return 2 * size ** 3 * steps


def time_to_target(accuracy: float, target: float, elapsed: float) -> float | None:
    """Report checkpoint time only when the validation target is reached.

    Args:
        accuracy: Held-out accuracy at the checkpoint.
        target: Predefined accuracy threshold.
        elapsed: Elapsed training time to the checkpoint.

    Returns:
        Elapsed seconds when the target is reached, otherwise None.
    """
    return elapsed if math.isfinite(accuracy) and accuracy >= target else None


def checkpoint_due(completed_steps: int, interval: int | None) -> bool:
    """Determine whether a periodic checkpoint is due.

    Args:
        completed_steps: Number of training steps completed.
        interval: Checkpoint interval, or None when disabled.

    Returns:
        Whether a checkpoint should be written.
    """
    return interval is not None and interval > 0 and completed_steps > 0 and completed_steps % interval == 0


def latency_metrics(observations: list[float]) -> dict:
    """Summarize instrumented step latency observations.

    Args:
        observations: Completed-step wall-clock durations in seconds.

    Returns:
        Count and percentile measurements with explicit units.
    """
    from bdbench.statistics import summarize
    summary = summarize(observations)
    metrics = {f'step_latency_{key}': {'value': summary[key], 'unit': 's'} for key in ('p50', 'p90', 'p95', 'p99', 'max')}
    metrics['step_latency_count'] = {'value': summary['count'], 'unit': 'steps'}
    return metrics


def perplexity(loss: float) -> float | None:
    """Convert finite mean token loss in nats to perplexity.

    Args:
        loss: Mean cross-entropy in nats per token.

    Returns:
        Perplexity, or None when it cannot be represented.
    """
    if not math.isfinite(loss) or loss > 700:
        return None
    return math.exp(loss)


def timed_call(function, clock=time.perf_counter):
    """Execute an operation and measure its elapsed monotonic duration.

    Args:
        function: Operation to invoke without arguments.
        clock: Monotonic clock returning seconds.

    Returns:
        The operation result and elapsed seconds.
    """
    start = clock()
    result = function()
    return result, clock() - start


def causal_mask(length: int) -> list[list[bool]]:
    """Return a boolean attention mask blocking access to future tokens.

    Args:
        length: Number of positions in the sequence.

    Returns:
        Square mask where true denotes blocked attention.
    """
    return [[column > row for column in range(length)] for row in range(length)]


def token_batch(tokens: list[int], batch_size: int, length: int, offset: int) -> tuple[list, list]:
    """Build fixed-length next-token inputs and targets.

    Args:
        tokens: Prepared token IDs.
        batch_size: Number of sequences.
        length: Tokens per sequence.
        offset: Starting token position.

    Returns:
        Two lists of input and target token sequences.

    Raises:
        ValueError: If the dataset cannot supply one batch.
    """
    width = batch_size * length
    if batch_size <= 0 or length <= 0 or len(tokens) <= width:
        raise ValueError('Not enough tokens for a batch')
    start = offset % (len(tokens) - width)
    return ([tokens[start + i * length:start + (i + 1) * length] for i in range(batch_size)], [tokens[start + i * length + 1:start + (i + 1) * length + 1] for i in range(batch_size)])


def dataset_kind(case: dict) -> str:
    """Validate the configured training dataset.

    Args:
        case: Frozen case descriptor.

    Returns:
        The dataset identifier.

    Raises:
        ValueError: If the dataset is unsupported or its path is missing.
    """
    kind = case.get('dataset', 'generated')
    if kind not in {'generated', 'cifar10', 'cifar10_jpeg', 'cifar10_mmap', 'cifar10_shards', 'cifar10_tfrecord'}:
        raise ValueError('Unsupported training dataset')
    if kind in {'cifar10', 'cifar10_jpeg', 'cifar10_mmap', 'cifar10_shards', 'cifar10_tfrecord'} and not case.get('dataset_path'):
        raise ValueError('CIFAR-10 requires a prepared dataset path')
    return kind


def train_torch(case: dict) -> dict:
    """Run a PyTorch training case.

    Args:
        case: Fixed model and training parameters.

    Returns:
        Trial metrics from the requested device.
    """
    emit_progress('Importing PyTorch')
    import torch
    emit_progress('PyTorch imported; selecting device and building model')
    from torch import nn
    if case.get('cpu_threads'):
        torch.set_num_threads(case['cpu_threads'])
    if case['model'] == 'model_checkpoint':
        emit_progress('Preparing PyTorch model and optimizer checkpoint')
        return train_torch_checkpoint(case)
    if case['model'] == 'matmul':
        emit_progress('Preparing PyTorch matrix operands and warmup')
        return train_torch_matmul(case)
    if case['model'] == 'transformer_lm':
        emit_progress('Reading WikiText tokens and building PyTorch language model')
        return train_torch_language(case)
    torch.manual_seed(case['seed'])
    gpu = case['device'] != 'cpu'
    if gpu and not torch.cuda.is_available():
        raise RuntimeError('Selected CUDA device is unavailable to PyTorch')
    device = torch.device('cuda:0' if gpu else 'cpu')
    batch = case['batch_size']
    model_name = case['model']
    if model_name == 'mlp':
        x = torch.randn(batch, 256, device=device)
        model = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 10))
    elif model_name == 'resnet18':
        from torchvision.models import resnet18
        x = torch.randn(batch, 3, 32, 32, device=device)
        model = resnet18(weights=None, num_classes=10)
    elif model_name == 'cnn':
        x = torch.randn(batch, 3, 32, 32, device=device)
        model = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10))
    elif model_name == 'transformer':
        x = torch.randn(batch, 32, 64, device=device)
        model = nn.Sequential(nn.TransformerEncoder(nn.TransformerEncoderLayer(64, 4, 128, batch_first=True), 2), nn.Flatten(), nn.Linear(32 * 64, 10))
    else:
        raise ValueError('Unknown model')
    model = model.to(device)
    emit_progress(f'Model {model_name} initialized on {device}')
    y = torch.randint(0, 10, (batch,), device=device)
    batches = None
    dataset_load_seconds = None
    if dataset_kind(case) in {'cifar10', 'cifar10_jpeg', 'cifar10_mmap', 'cifar10_shards'}:
        emit_progress(f'Loading {case["dataset"]} input data')
        from torchvision.datasets import CIFAR10, ImageFolder
        from torchvision.transforms import ToTensor
        if case['dataset'] == 'cifar10_jpeg':
            from pathlib import Path
            dataset, dataset_load_seconds = timed_call(lambda: ImageFolder(str(Path(case['dataset_path']) / 'images'), transform=ToTensor()))
        elif case['dataset'] in {'cifar10_mmap', 'cifar10_shards'}:
            from bdbench.datasets import iter_array_samples
            class ArrayDataset(torch.utils.data.IterableDataset):
                def __iter__(self):
                    for image, label in iter_array_samples(case['dataset_path'], case['dataset']):
                        yield ToTensor()(image.copy()), label
            dataset, dataset_load_seconds = timed_call(ArrayDataset)
        else:
            dataset, dataset_load_seconds = timed_call(lambda: CIFAR10(case['dataset_path'], train=True, download=False, transform=ToTensor()))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False, drop_last=True, num_workers=case.get('input_workers', 0), pin_memory=case.get('pin_memory', False))
        batches = iter(loader)
        emit_progress('Input data loader ready')
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loss_fn = nn.CrossEntropyLoss()
    mixed = case.get('precision') == 'mixed_float16'
    scaler = torch.amp.GradScaler('cuda', enabled=mixed)
    def step():
        nonlocal batches
        inputs, targets = x, y
        if batches is not None:
            try:
                inputs, targets = next(batches)
            except StopIteration:
                batches = iter(loader)
                inputs, targets = next(batches)
            inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=mixed):
            loss = loss_fn(model(inputs), targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    emit_progress(f'Warming up {case["warmup_steps"]} steps')
    for _ in range(case['warmup_steps']):
        step()
    emit_progress(f'Warmup complete; measuring {case["steps"]} training steps')
    if gpu:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    from contextlib import ExitStack
    import tempfile
    from pathlib import Path
    checkpoint_times = []
    profiler = None
    with ExitStack() as stack:
        checkpoint_path = None
        if case.get('checkpoint_interval'):
            from bdbench.storage import claim_storage, check_budget, owned_bytes, check_total_budget
            root = Path(case['storage_root'])
            owned = claim_storage(root)
            check_budget(root, owned_bytes(owned), 128 * 1024 * 1024)
            check_total_budget([Path(item) for item in case.get('storage_roots', [str(root)])], 128 * 1024 * 1024)
            checkpoint_path = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='training-', dir=owned))) / 'model.pt'
        if case.get('diagnostic') == 'framework_profiler':
            activities = [torch.profiler.ProfilerActivity.CPU]
            if gpu:
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            profiler = stack.enter_context(torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True))
        start = time.perf_counter()
        latencies = []
        for index in range(case['steps']):
            step_start = time.perf_counter() if case.get('diagnostic') == 'step_latency' else None
            step()
            if profiler is not None:
                profiler.step()
            if checkpoint_path and checkpoint_due(index + 1, case['checkpoint_interval']):
                checkpoint_start = time.perf_counter()
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint_path)
                checkpoint_times.append(time.perf_counter() - checkpoint_start)
            if step_start is not None:
                if gpu:
                    torch.cuda.synchronize()
                latencies.append(time.perf_counter() - step_start)
        if gpu:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    emit_progress(f'Timed training complete in {elapsed:.2f} s; summarizing metrics')
    if profiler is not None:
        profiler.export_chrome_trace(str(Path(case['profile_output']) / 'torch-trace.json'))
    metrics = {'elapsed': {'value': elapsed, 'unit': 's'}, 'throughput': {'value': batch * case['steps'] / elapsed, 'unit': 'samples/s'}}
    if latencies:
        metrics.update(latency_metrics(latencies))
    if checkpoint_times:
        metrics['checkpoint_stall_time'] = {'value': sum(checkpoint_times), 'unit': 's'}
        metrics['checkpoint_count'] = {'value': len(checkpoint_times), 'unit': 'checkpoints'}
    if dataset_load_seconds is not None:
        emit_progress('Running held-out validation')
        metrics['dataset_load_time'] = {'value': dataset_load_seconds, 'unit': 's'}
        from torchvision.datasets import CIFAR10
        from torchvision.transforms import ToTensor
        validation = torch.utils.data.DataLoader(CIFAR10(case['dataset_path'], train=False, download=False, transform=ToTensor()), batch_size=batch, shuffle=False)
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for valid_x, valid_y in validation:
                correct += int((model(valid_x.to(device)).argmax(dim=1) == valid_y.to(device)).sum().cpu())
                total += len(valid_y)
                if not case.get('fixed_epoch'):
                    break
        accuracy = correct / total
        accuracy_metric = 'validation_accuracy' if case.get('fixed_epoch') else 'validation_batch_accuracy'
        metrics[accuracy_metric] = {'value': accuracy, 'unit': 'fraction'}
        if case.get('fixed_epoch'):
            metrics['training_elapsed_at_validation_target'] = {'value': time_to_target(accuracy, case['quality_target'], elapsed), 'unit': 's', 'reason': None if accuracy >= case['quality_target'] else 'Target not reached at the epoch checkpoint'}
    if gpu:
        metrics['peak_allocated_memory'] = {'value': torch.cuda.max_memory_allocated(), 'unit': 'bytes'}
        metrics['peak_reserved_memory'] = {'value': torch.cuda.max_memory_reserved(), 'unit': 'bytes'}
    return {'status': 'completed', 'metrics': metrics, 'work_completed': batch * case['steps']}


def train_tensorflow(case: dict) -> dict:
    """Run a TensorFlow training case.

    Args:
        case: Fixed model and training parameters.

    Returns:
        Trial metrics from the requested device.
    """
    emit_progress('Importing TensorFlow')
    import tensorflow as tf
    emit_progress('TensorFlow imported; selecting device and building model')
    if case.get('distributed'):
        emit_progress('Preparing distributed TensorFlow workers')
        return train_tensorflow_distributed(case)
    if case.get('cpu_threads'):
        tf.config.threading.set_intra_op_parallelism_threads(case['cpu_threads'])
    if case['model'] == 'model_checkpoint':
        emit_progress('Preparing TensorFlow model and optimizer checkpoint')
        return train_tensorflow_checkpoint(case)
    if case['model'] == 'matmul':
        emit_progress('Preparing TensorFlow matrix operands and warmup')
        return train_tensorflow_matmul(case)
    if case['model'] == 'transformer_lm':
        emit_progress('Reading WikiText tokens and building TensorFlow language model')
        return train_tensorflow_language(case)
    tf.random.set_seed(case['seed'])
    gpu = case['device'] != 'cpu'
    if gpu and not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('Selected CUDA device is unavailable to TensorFlow')
    target = '/GPU:0' if gpu else '/CPU:0'
    mixed = case.get('precision') == 'mixed_float16'
    if mixed:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    batch = case['batch_size']
    with tf.device(target):
        if case['model'] == 'mlp':
            x = tf.random.normal([batch, 256])
            model = tf.keras.Sequential([tf.keras.layers.Input((256,)), tf.keras.layers.Dense(512, activation='relu'), tf.keras.layers.Dense(10)])
        elif case['model'] == 'cnn':
            x = tf.random.normal([batch, 32, 32, 3])
            model = tf.keras.Sequential([tf.keras.layers.Input((32, 32, 3)), tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'), tf.keras.layers.MaxPool2D(), tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'), tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(10)])
        elif case['model'] == 'transformer':
            x = tf.random.normal([batch, 32, 64])
            inputs = tf.keras.Input((32, 64))
            attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(inputs, inputs)
            output = tf.keras.layers.Dense(10)(tf.keras.layers.Flatten()(attention))
            model = tf.keras.Model(inputs, output)
        else:
            raise ValueError('Unknown model')
        y = tf.random.uniform([batch], maxval=10, dtype=tf.int32)
        emit_progress(f'Model {case["model"]} initialized on {target}')
        batches = None
        dataset_load_seconds = None
        if dataset_kind(case) in {'cifar10', 'cifar10_jpeg', 'cifar10_mmap', 'cifar10_shards', 'cifar10_tfrecord'}:
            emit_progress(f'Loading {case["dataset"]} input data')
            import pickle
            import numpy as np
            from pathlib import Path
            load_start = time.perf_counter()
            if case['dataset'] == 'cifar10_jpeg':
                from bdbench.datasets import image_paths
                paths, labels = image_paths(Path(case['dataset_path']) / 'images')
                dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
                dataset = dataset.map(lambda path, label: (tf.cast(tf.io.decode_jpeg(tf.io.read_file(path), channels=3), tf.float32) / 255.0, label))
            elif case['dataset'] in {'cifar10_mmap', 'cifar10_shards'}:
                from bdbench.datasets import iter_array_samples
                dataset = tf.data.Dataset.from_generator(lambda: iter_array_samples(case['dataset_path'], case['dataset']), output_signature=(tf.TensorSpec((32, 32, 3), tf.uint8), tf.TensorSpec((), tf.int64)))
                dataset = dataset.map(lambda image, label: (tf.cast(image, tf.float32) / 255.0, label))
            elif case['dataset'] == 'cifar10_tfrecord':
                paths = sorted(str(path) for path in (Path(case['dataset_path']) / 'tfrecord').glob('*.tfrecord'))
                if not paths:
                    raise ValueError('No prepared TFRecord shards found')
                dataset = tf.data.TFRecordDataset(paths)
                def parse(record):
                    fields = tf.io.parse_single_example(record, {'image': tf.io.FixedLenFeature([], tf.string), 'label': tf.io.FixedLenFeature([], tf.int64)})
                    image = tf.reshape(tf.io.decode_raw(fields['image'], tf.uint8), (32, 32, 3))
                    return tf.cast(image, tf.float32) / 255.0, fields['label']
                dataset = dataset.map(parse)
            else:
                images, labels = [], []
                for index in range(1, 6):
                    with (Path(case['dataset_path']) / 'cifar-10-batches-py' / f'data_batch_{index}').open('rb') as file:
                        item = pickle.load(file, encoding='bytes')
                    images.append(item[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
                    labels.extend(item[b'labels'])
                dataset = tf.data.Dataset.from_tensor_slices((np.concatenate(images).astype('float32') / 255.0, np.asarray(labels, dtype='int32')))
            dataset = dataset.batch(batch, drop_remainder=True).repeat()
            if case.get('input_workers', 0):
                options = tf.data.Options()
                options.threading.private_threadpool_size = case['input_workers']
                dataset = dataset.with_options(options).prefetch(case['input_workers'])
            batches = iter(dataset)
            dataset_load_seconds = time.perf_counter() - load_start
            emit_progress('Input data loader ready')
        optimizer = tf.keras.optimizers.Adam(.001)
        if mixed:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        @tf.function
        def step(inputs, targets):
            with tf.GradientTape() as tape:
                loss = loss_fn(targets, model(inputs, training=True))
                scaled_loss = optimizer.scale_loss(loss) if mixed else loss
            gradients = tape.gradient(scaled_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss
        emit_progress(f'Warming up {case["warmup_steps"]} steps')
        for _ in range(case['warmup_steps']):
            inputs, targets = next(batches) if batches is not None else (x, y)
            loss = step(inputs, targets)
        loss.numpy()
        emit_progress(f'Warmup complete; measuring {case["steps"]} training steps')
        from contextlib import ExitStack
        import tempfile
        from pathlib import Path
        checkpoint_times = []
        with ExitStack() as stack:
            checkpoint_prefix = None
            if case.get('checkpoint_interval'):
                from bdbench.storage import claim_storage, check_budget, owned_bytes, check_total_budget
                root = Path(case['storage_root'])
                owned = claim_storage(root)
                check_budget(root, owned_bytes(owned), 128 * 1024 * 1024)
                check_total_budget([Path(item) for item in case.get('storage_roots', [str(root)])], 128 * 1024 * 1024)
                checkpoint_prefix = str(Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='training-', dir=owned))) / 'model')
                checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
            if case.get('diagnostic') == 'framework_profiler':
                tf.profiler.experimental.start(case['profile_output'])
                stack.callback(tf.profiler.experimental.stop)
            start = time.perf_counter()
            latencies = []
            for index in range(case['steps']):
                step_start = time.perf_counter() if case.get('diagnostic') == 'step_latency' else None
                inputs, targets = next(batches) if batches is not None else (x, y)
                loss = step(inputs, targets)
                if checkpoint_prefix and checkpoint_due(index + 1, case['checkpoint_interval']):
                    checkpoint_start = time.perf_counter()
                    checkpoint.write(checkpoint_prefix)
                    checkpoint_times.append(time.perf_counter() - checkpoint_start)
                if step_start is not None:
                    loss.numpy()
                    latencies.append(time.perf_counter() - step_start)
            loss.numpy()
            elapsed = time.perf_counter() - start
    emit_progress(f'Timed training complete in {elapsed:.2f} s; summarizing metrics')
    metrics = {'elapsed': {'value': elapsed, 'unit': 's'}, 'throughput': {'value': batch * case['steps'] / elapsed, 'unit': 'samples/s'}}
    if latencies:
        metrics.update(latency_metrics(latencies))
    if checkpoint_times:
        metrics['checkpoint_stall_time'] = {'value': sum(checkpoint_times), 'unit': 's'}
        metrics['checkpoint_count'] = {'value': len(checkpoint_times), 'unit': 'checkpoints'}
    if dataset_load_seconds is not None:
        emit_progress('Running held-out validation')
        metrics['dataset_load_time'] = {'value': dataset_load_seconds, 'unit': 's'}
        import pickle
        from pathlib import Path
        with (Path(case['dataset_path']) / 'cifar-10-batches-py' / 'test_batch').open('rb') as file:
            validation = pickle.load(file, encoding='bytes')
        valid_size = len(validation[b'labels']) if case.get('fixed_epoch') else batch
        correct = 0
        for start_index in range(0, valid_size, batch):
            stop_index = min(valid_size, start_index + batch)
            valid_x = validation[b'data'][start_index:stop_index].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype('float32') / 255.0
            valid_y = tf.constant(validation[b'labels'][start_index:stop_index], dtype=tf.int32)
            predicted = tf.argmax(model(valid_x, training=False), axis=1, output_type=tf.int32)
            correct += int(tf.reduce_sum(tf.cast(predicted == valid_y, tf.int32)).numpy())
        accuracy = correct / valid_size
        accuracy_metric = 'validation_accuracy' if case.get('fixed_epoch') else 'validation_batch_accuracy'
        metrics[accuracy_metric] = {'value': accuracy, 'unit': 'fraction'}
        if case.get('fixed_epoch'):
            metrics['training_elapsed_at_validation_target'] = {'value': time_to_target(accuracy, case['quality_target'], elapsed), 'unit': 's', 'reason': None if accuracy >= case['quality_target'] else 'Target not reached at the epoch checkpoint'}
    return {'status': 'completed', 'metrics': metrics, 'work_completed': batch * case['steps']}


def train_torch_language(case: dict) -> dict:
    """Train a next-token PyTorch transformer on prepared WikiText.

    Args:
        case: Frozen sequence training case.

    Returns:
        Token throughput, elapsed time and final training loss.
    """
    from pathlib import Path
    import torch
    from torch import nn
    tokens = json.loads((Path(case['dataset_path']) / 'train.json').read_text())
    vocab = len(json.loads((Path(case['dataset_path']) / 'vocab.json').read_text()))
    torch.manual_seed(case['seed'])
    gpu = case['device'] != 'cpu'
    if gpu and not torch.cuda.is_available():
        raise RuntimeError('Selected CUDA device is unavailable to PyTorch')
    device = torch.device('cuda:0' if gpu else 'cpu')
    class LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab, 64)
            self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(64, 4, 128, batch_first=True), 2)
            self.output = nn.Linear(64, vocab)
            self.register_buffer('mask', torch.tensor(causal_mask(32), dtype=torch.bool))
        def forward(self, value):
            return self.output(self.encoder(self.embedding(value), mask=self.mask))
    model = LanguageModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loss_fn = nn.CrossEntropyLoss()
    def step(index):
        xs, ys = token_batch(tokens, case['batch_size'], 32, index * case['batch_size'] * 32)
        inputs = torch.tensor(xs, dtype=torch.long, device=device)
        targets = torch.tensor(ys, dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = loss_fn(logits.reshape(-1, vocab), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        return loss
    for i in range(case['warmup_steps']):
        step(i)
    if gpu:
        torch.cuda.synchronize()
    emit_progress(f'Warmup complete; measuring {case["steps"]} language-model steps')
    start = time.perf_counter()
    for i in range(case['steps']):
        loss = step(i + case['warmup_steps'])
    if gpu:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    emit_progress(f'Timed language-model training complete in {elapsed:.2f} s; validating')
    count = case['batch_size'] * 32 * case['steps']
    validation_tokens = json.loads((Path(case['dataset_path']) / 'valid.json').read_text())
    valid_x, valid_y = token_batch(validation_tokens, case['batch_size'], 32, 0)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(valid_x, dtype=torch.long, device=device))
        valid_loss = float(loss_fn(logits.reshape(-1, vocab), torch.tensor(valid_y, dtype=torch.long, device=device).reshape(-1)).cpu())
    return {'status': 'completed', 'metrics': {'elapsed': {'value': elapsed, 'unit': 's'}, 'throughput': {'value': count / elapsed, 'unit': 'tokens/s'}, 'training_loss': {'value': float(loss.detach().cpu()), 'unit': 'nats/token'}, 'validation_batch_loss': {'value': valid_loss, 'unit': 'nats/token'}, 'validation_batch_perplexity': {'value': perplexity(valid_loss), 'unit': 'ratio'}}, 'work_completed': count}


def train_tensorflow_language(case: dict) -> dict:
    """Train a next-token TensorFlow transformer on prepared WikiText.

    Args:
        case: Frozen sequence training case.

    Returns:
        Token throughput, elapsed time and final training loss.
    """
    from pathlib import Path
    import tensorflow as tf
    tokens = json.loads((Path(case['dataset_path']) / 'train.json').read_text())
    vocab = len(json.loads((Path(case['dataset_path']) / 'vocab.json').read_text()))
    tf.random.set_seed(case['seed'])
    gpu = case['device'] != 'cpu'
    if gpu and not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('Selected CUDA device is unavailable to TensorFlow')
    with tf.device('/GPU:0' if gpu else '/CPU:0'):
        inputs = tf.keras.Input((32,), dtype='int32')
        embed = tf.keras.layers.Embedding(vocab, 64)(inputs)
        attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(embed, embed, use_causal_mask=True)
        output = tf.keras.layers.Dense(vocab)(attention)
        model = tf.keras.Model(inputs, output)
        optimizer = tf.keras.optimizers.Adam(.001)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                loss = loss_fn(y, model(x, training=True))
            optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
            return loss
        def step(index):
            xs, ys = token_batch(tokens, case['batch_size'], 32, index * case['batch_size'] * 32)
            return train_step(tf.constant(xs), tf.constant(ys))
        for i in range(case['warmup_steps']):
            loss = step(i)
        loss.numpy()
        emit_progress(f'Warmup complete; measuring {case["steps"]} language-model steps')
        start = time.perf_counter()
        for i in range(case['steps']):
            loss = step(i + case['warmup_steps'])
        loss_value = float(loss.numpy())
        elapsed = time.perf_counter() - start
        emit_progress(f'Timed language-model training complete in {elapsed:.2f} s; validating')
        validation_tokens = json.loads((Path(case['dataset_path']) / 'valid.json').read_text())
        valid_x, valid_y = token_batch(validation_tokens, case['batch_size'], 32, 0)
        valid_loss = float(loss_fn(tf.constant(valid_y), model(tf.constant(valid_x), training=False)).numpy())
    count = case['batch_size'] * 32 * case['steps']
    return {'status': 'completed', 'metrics': {'elapsed': {'value': elapsed, 'unit': 's'}, 'throughput': {'value': count / elapsed, 'unit': 'tokens/s'}, 'training_loss': {'value': loss_value, 'unit': 'nats/token'}, 'validation_batch_loss': {'value': valid_loss, 'unit': 'nats/token'}, 'validation_batch_perplexity': {'value': perplexity(valid_loss), 'unit': 'ratio'}}, 'work_completed': count}


def train_torch_matmul(case: dict) -> dict:
    """Measure resident PyTorch square matrix products.

    Args:
        case: Fixed step count and selected device.

    Returns:
        Elapsed seconds and conventional GFLOP/s.
    """
    import torch
    gpu = case['device'] != 'cpu'
    if gpu and not torch.cuda.is_available():
        raise RuntimeError('Selected CUDA device is unavailable to PyTorch')
    device = torch.device('cuda:0' if gpu else 'cpu')
    torch.manual_seed(case['seed'])
    x = torch.randn(1024, 1024, device=device)
    y = torch.randn(1024, 1024, device=device)
    for _ in range(case['warmup_steps']):
        result = x @ y
    if gpu:
        torch.cuda.synchronize()
    emit_progress(f'Warmup complete; measuring {case["steps"]} matrix products')
    start = time.perf_counter()
    for _ in range(case['steps']):
        result = x @ y
    if gpu:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    emit_progress(f'Timed matrix products complete in {elapsed:.2f} s')
    work = matmul_flops(1024, case['steps'])
    return {'status': 'completed', 'metrics': {'elapsed': {'value': elapsed, 'unit': 's'}, 'throughput': {'value': work / elapsed / 1e9, 'unit': 'GFLOP/s'}}, 'work_completed': work}


def train_tensorflow_matmul(case: dict) -> dict:
    """Measure resident TensorFlow square matrix products.

    Args:
        case: Fixed step count and selected device.

    Returns:
        Elapsed seconds and conventional GFLOP/s.
    """
    import tensorflow as tf
    gpu = case['device'] != 'cpu'
    if gpu and not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('Selected CUDA device is unavailable to TensorFlow')
    tf.random.set_seed(case['seed'])
    with tf.device('/GPU:0' if gpu else '/CPU:0'):
        x = tf.random.normal((1024, 1024))
        y = tf.random.normal((1024, 1024))
        for _ in range(case['warmup_steps']):
            result = tf.matmul(x, y)
        result.numpy()
        emit_progress(f'Warmup complete; measuring {case["steps"]} matrix products')
        start = time.perf_counter()
        for _ in range(case['steps']):
            result = tf.matmul(x, y)
        result.numpy()
        elapsed = time.perf_counter() - start
        emit_progress(f'Timed matrix products complete in {elapsed:.2f} s')
    work = matmul_flops(1024, case['steps'])
    return {'status': 'completed', 'metrics': {'elapsed': {'value': elapsed, 'unit': 's'}, 'throughput': {'value': work / elapsed / 1e9, 'unit': 'GFLOP/s'}}, 'work_completed': work}


def train_tensorflow_distributed(case: dict) -> dict:
    """Train a CNN with TensorFlow mirrored data parallelism.

    Args:
        case: Frozen multi-GPU workload descriptor.

    Returns:
        Aggregate throughput and elapsed time.
    """
    import tensorflow as tf
    devices = tf.config.list_logical_devices('GPU')
    count = len(case['device_uuids'])
    if len(devices) != count:
        raise RuntimeError('TensorFlow does not see all selected GPUs')
    tf.random.set_seed(case['seed'])
    strategy = tf.distribute.MirroredStrategy(devices=[device.name for device in devices])
    batch = case['batch_size']
    with strategy.scope():
        model = tf.keras.Sequential([tf.keras.layers.Input((32, 32, 3)), tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'), tf.keras.layers.MaxPool2D(), tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'), tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(10)])
        optimizer = tf.keras.optimizers.Adam(.001)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    x = strategy.experimental_distribute_values_from_function(lambda context: tf.random.stateless_normal((batch, 32, 32, 3), seed=(case['seed'], context.replica_id_in_sync_group)))
    y = strategy.experimental_distribute_values_from_function(lambda context: tf.random.stateless_uniform((batch,), seed=(case['seed'] + 1, context.replica_id_in_sync_group), maxval=10, dtype=tf.int32))
    @tf.function
    def step():
        def replica_step(inputs, labels):
            with tf.GradientTape() as tape:
                losses = loss_fn(labels, model(inputs, training=True))
                loss = tf.nn.compute_average_loss(losses, global_batch_size=batch * count)
            optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
            return loss
        return strategy.reduce(tf.distribute.ReduceOp.SUM, strategy.run(replica_step, args=(x, y)), axis=None)
    for _ in range(case['warmup_steps']):
        loss = step()
    loss.numpy()
    start = time.perf_counter()
    for _ in range(case['steps']):
        loss = step()
    loss.numpy()
    elapsed = time.perf_counter() - start
    work = batch * count * case['steps']
    return {'status': 'completed', 'metrics': {'elapsed': {'value': elapsed, 'unit': 's'}, 'throughput': {'value': work / elapsed, 'unit': 'samples/s'}}, 'work_completed': work}


def train_torch_checkpoint(case: dict) -> dict:
    """Measure PyTorch model and optimizer checkpoint save and load.

    Args:
        case: Device, storage root and durability configuration.

    Returns:
        Checkpoint save/load timings and file size.
    """
    import os
    import tempfile
    from pathlib import Path
    import torch
    from bdbench.storage import claim_storage, check_budget, owned_bytes, check_total_budget
    root = Path(case['storage_root'])
    owned = claim_storage(root)
    check_budget(root, owned_bytes(owned), 128 * 1024 * 1024)
    check_total_budget([Path(item) for item in case.get('storage_roots', [str(root)])], 128 * 1024 * 1024)
    if case['device'] != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('Selected CUDA device is unavailable to PyTorch')
    device = torch.device('cpu' if case['device'] == 'cpu' else 'cuda:0')
    torch.manual_seed(case['seed'])
    model = torch.nn.Sequential(torch.nn.Linear(1024, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 10)).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    model(torch.randn(8, 1024, device=device)).sum().backward()
    optimizer.step()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory(prefix='checkpoint-', dir=owned) as tmp:
        path = Path(tmp) / 'model.pt'
        start = time.perf_counter()
        with path.open('wb') as stream:
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, stream)
            if case['durable']:
                stream.flush()
                os.fsync(stream.fileno())
        save_time = time.perf_counter() - start
        size = path.stat().st_size
        start = time.perf_counter()
        saved = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        if device.type == 'cuda':
            torch.cuda.synchronize()
        load_time = time.perf_counter() - start
    return {'status': 'completed', 'metrics': {'save_time': {'value': save_time, 'unit': 's'}, 'load_time': {'value': load_time, 'unit': 's'}, 'checkpoint_size': {'value': size, 'unit': 'bytes'}}, 'work_completed': 1}


def train_tensorflow_checkpoint(case: dict) -> dict:
    """Measure TensorFlow model and optimizer checkpoint save and load.

    Args:
        case: Device, storage root and durability configuration.

    Returns:
        Checkpoint save/load timings and aggregate file size.
    """
    import os
    import tempfile
    from pathlib import Path
    import tensorflow as tf
    from bdbench.storage import claim_storage, check_budget, owned_bytes, check_total_budget
    root = Path(case['storage_root'])
    owned = claim_storage(root)
    check_budget(root, owned_bytes(owned), 128 * 1024 * 1024)
    check_total_budget([Path(item) for item in case.get('storage_roots', [str(root)])], 128 * 1024 * 1024)
    gpu = case['device'] != 'cpu'
    if gpu and not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('Selected CUDA device is unavailable to TensorFlow')
    tf.random.set_seed(case['seed'])
    with tf.device('/GPU:0' if gpu else '/CPU:0'):
        model = tf.keras.Sequential([tf.keras.layers.Input((1024,)), tf.keras.layers.Dense(1024, activation='relu'), tf.keras.layers.Dense(10)])
        optimizer = tf.keras.optimizers.Adam()
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(model(tf.random.normal((8, 1024))))
        optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        with tempfile.TemporaryDirectory(prefix='checkpoint-', dir=owned) as tmp:
            prefix = str(Path(tmp) / 'model')
            start = time.perf_counter()
            name = checkpoint.save(prefix)
            if case['durable']:
                for path in Path(tmp).iterdir():
                    with path.open('rb') as stream:
                        os.fsync(stream.fileno())
            save_time = time.perf_counter() - start
            size = sum(path.stat().st_size for path in Path(tmp).iterdir())
            start = time.perf_counter()
            checkpoint.restore(name).assert_existing_objects_matched()
            load_time = time.perf_counter() - start
    return {'status': 'completed', 'metrics': {'save_time': {'value': save_time, 'unit': 's'}, 'load_time': {'value': load_time, 'unit': 's'}, 'checkpoint_size': {'value': size, 'unit': 'bytes'}}, 'work_completed': 1}


def train_sklearn(case: dict) -> dict:
    """Fit a classical model on generated data and score held-out samples.

    Args:
        case: Frozen estimator and dataset parameters.

    Returns:
        Fit, prediction and accuracy metrics.
    """
    emit_progress('Importing scikit-learn and generating synthetic samples')
    from sklearn.datasets import make_classification
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    x, y = make_classification(n_samples=10000, n_features=64, n_informative=20, random_state=case['seed'])
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=.2, random_state=case['seed'])
    models = {'logistic': lambda: LogisticRegression(max_iter=200, random_state=case['seed']), 'hist_gradient': lambda: HistGradientBoostingClassifier(random_state=case['seed']), 'forest': lambda: RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=case['seed'])}
    model = models[case['model']]()
    emit_progress(f'Fitting {case["model"]} on {len(train_y)} samples')
    start = time.perf_counter()
    model.fit(train_x, train_y)
    fit = time.perf_counter() - start
    emit_progress(f'Fit complete in {fit:.2f} s; scoring held-out samples')
    start = time.perf_counter()
    score = model.score(test_x, test_y)
    predict = time.perf_counter() - start
    return {'status': 'completed', 'metrics': {'elapsed': {'value': fit, 'unit': 's'}, 'prediction_time': {'value': predict, 'unit': 's'}, 'accuracy': {'value': float(score), 'unit': 'fraction'}}, 'work_completed': len(train_y)}


def train_filesystem(case: dict) -> dict:
    """Measure bounded operations in a benchmark-owned storage directory.

    Args:
        case: Storage root, workload type and size.

    Returns:
        Elapsed time and operation throughput.
    """
    import os
    from pathlib import Path
    import tempfile
    from bdbench.storage import claim_storage, check_budget, owned_bytes, check_total_budget
    emit_progress(f'Checking storage budget for {case["model"]}')
    root = Path(case['storage_root'])
    owned = claim_storage(root)
    size = case['size_bytes']
    check_budget(root, owned_bytes(owned), size * 2)
    check_total_budget([Path(item) for item in case.get('storage_roots', [str(root)])], size * 2)
    with tempfile.TemporaryDirectory(prefix='trial-', dir=owned) as temporary:
        emit_progress(f'Running {case["model"]} file operations in benchmark-owned directory')
        directory = Path(temporary)
        data = b'\x91' * min(size, 1048576)
        path = directory / 'data.bin'
        if case['model'] == 'small_files':
            count = max(1, min(1024, size // 4096))
            start = time.perf_counter()
            for index in range(count):
                (directory / str(index)).write_bytes(data[:4096])
            for index in range(count):
                (directory / str(index)).read_bytes()
            elapsed = time.perf_counter() - start
            work = count * 2
            unit = 'operations/s'
        elif case['model'] == 'metadata_remove':
            count = max(1, min(1024, size // 4096))
            for index in range(count):
                (directory / str(index)).write_bytes(data[:64])
            start = time.perf_counter()
            for index in range(count):
                (directory / str(index)).stat()
            for index in range(count):
                (directory / str(index)).unlink()
            elapsed = time.perf_counter() - start
            work = count * 2
            unit = 'operations/s'
        elif case['model'] == 'scalar_logs':
            count = max(1, min(10000, size // 64))
            start = time.perf_counter()
            with path.open('w', encoding='utf-8') as stream:
                for index in range(count):
                    stream.write(json.dumps({'step': index, 'loss': 1.0 / (index + 1)}) + '\n')
                    if index % 100 == 0:
                        stream.flush()
            elapsed = time.perf_counter() - start
            work = count
            unit = 'operations/s'
        elif case['model'] == 'random_read':
            import random
            with path.open('wb') as stream:
                remaining = size
                while remaining:
                    block = data[:min(remaining, len(data))]
                    stream.write(block)
                    remaining -= len(block)
            rng = random.Random(case['seed'])
            count = min(10000, max(1, size // 4096))
            start = time.perf_counter()
            with path.open('rb') as stream:
                for _ in range(count):
                    stream.seek(rng.randrange(max(1, size // 4096)) * 4096)
                    stream.read(4096)
            elapsed = time.perf_counter() - start
            work = count
            unit = 'operations/s'
        elif case['model'] in {'sequential', 'checkpoint'}:
            start = time.perf_counter()
            with path.open('wb') as stream:
                remaining = size
                while remaining:
                    block = data[:min(remaining, len(data))]
                    stream.write(block)
                    remaining -= len(block)
                if case['model'] == 'checkpoint':
                    stream.flush()
                    os.fsync(stream.fileno())
            with path.open('rb') as stream:
                while stream.read(1048576):
                    pass
            elapsed = time.perf_counter() - start
            work = size * 2
            unit = 'bytes/s'
        else:
            raise ValueError('Unknown filesystem workload')
    emit_progress(f'File operations complete in {elapsed:.2f} s; temporary files removed')
    return {'status': 'completed', 'metrics': {'elapsed': {'value': elapsed, 'unit': 's'}, 'throughput': {'value': work / elapsed, 'unit': unit}}, 'work_completed': work}


def train_case(case: dict) -> dict:
    """Dispatch a case to its framework.

    Args:
        case: Frozen case descriptor.

    Returns:
        Worker result.

    Raises:
        ValueError: If the framework is not supported.
    """
    emit_progress(f'Dispatching {case["framework"]}/{case["model"]} on {case["device"]}')
    if case.get('diagnostic') == 'framework_profiler' and not case.get('profile_output'):
        raise ValueError('A session-owned profiler output directory is required')
    if case.get('dataset') == 'cifar10_tfrecord' and case['framework'] != 'tensorflow':
        raise ValueError('TFRecord case requires TensorFlow')
    if case['framework'] == 'pytorch':
        result = train_torch(case)
    elif case['framework'] == 'tensorflow':
        result = train_tensorflow(case)
    elif case['framework'] == 'sklearn':
        result = train_sklearn(case)
    elif case['framework'] == 'filesystem':
        result = train_filesystem(case)
    else:
        raise ValueError('Unsupported framework')
    emit_progress(f'Worker case complete: {result["status"]}')
    return result


if __name__ == '__main__':
    case = json.load(sys.stdin)
    result = train_case(case)
    if case['framework'] == 'pytorch':
        import torch
        result['runtime'] = {'framework': 'pytorch', 'version': torch.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()}
    elif case['framework'] == 'tensorflow':
        import tensorflow as tf
        result['runtime'] = {'framework': 'tensorflow', 'version': tf.__version__, 'build': tf.sysconfig.get_build_info()}
    print(json.dumps(result, allow_nan=False))
