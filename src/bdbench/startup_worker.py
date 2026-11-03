"""Isolated import, initialization and first-step measurements."""

import json
import sys
import time
from bdbench.progress import emit_progress


def measure(case: dict) -> dict:
    """Measure framework import, setup, and the first training step.

    Args:
        case: Framework and selected device.

    Returns:
        Startup timing metrics and framework runtime metadata.
    """
    emit_progress(f'Importing {case["framework"]} for startup timing')
    start = time.perf_counter()
    gpu = case['device'] != 'cpu'
    if case['framework'] == 'pytorch':
        import torch
        import_elapsed = time.perf_counter() - start
        emit_progress(f'Framework import complete in {import_elapsed:.2f} s; initializing PyTorch model')
        init_start = time.perf_counter()
        if gpu and not torch.cuda.is_available():
            raise RuntimeError('Selected CUDA device is unavailable to PyTorch')
        device = torch.device('cuda:0' if gpu else 'cpu')
        model = torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.ReLU(), torch.nn.Linear(512, 10)).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        x = torch.randn(32, 256, device=device)
        y = torch.randint(0, 10, (32,), device=device)
        if gpu:
            torch.cuda.synchronize()
        init_elapsed = time.perf_counter() - init_start
        emit_progress(f'Model initialized in {init_elapsed:.2f} s; measuring first step')
        step_start = time.perf_counter()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        if gpu:
            torch.cuda.synchronize()
        step_elapsed = time.perf_counter() - step_start
        emit_progress(f'First step complete in {step_elapsed:.2f} s')
        runtime = {'framework':'pytorch','version':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version()}
    elif case['framework'] == 'tensorflow':
        import tensorflow as tf
        import_elapsed = time.perf_counter() - start
        emit_progress(f'Framework import complete in {import_elapsed:.2f} s; initializing TensorFlow model')
        init_start = time.perf_counter()
        if gpu and not tf.config.list_physical_devices('GPU'):
            raise RuntimeError('Selected CUDA device is unavailable to TensorFlow')
        with tf.device('/GPU:0' if gpu else '/CPU:0'):
            model = tf.keras.Sequential([tf.keras.layers.Input((256,)), tf.keras.layers.Dense(512, activation='relu'), tf.keras.layers.Dense(10)])
            optimizer = tf.keras.optimizers.Adam()
            x = tf.random.normal((32,256))
            y = tf.random.uniform((32,),maxval=10,dtype=tf.int32)
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            @tf.function
            def step():
                with tf.GradientTape() as tape:
                    value = loss_fn(y, model(x, training=True))
                optimizer.apply_gradients(zip(tape.gradient(value, model.trainable_variables), model.trainable_variables))
                return value
            init_elapsed = time.perf_counter() - init_start
            emit_progress(f'Model initialized in {init_elapsed:.2f} s; measuring first step')
            step_start = time.perf_counter()
            step().numpy()
            step_elapsed = time.perf_counter() - step_start
            emit_progress(f'First step complete in {step_elapsed:.2f} s')
        runtime = {'framework':'tensorflow','version':tf.__version__,'build':tf.sysconfig.get_build_info()}
    else:
        raise ValueError('Unsupported startup framework')
    return {'status':'completed','runtime':runtime,'metrics':{'framework_import_time':{'value':import_elapsed,'unit':'s'},'framework_initialization_time':{'value':init_elapsed,'unit':'s'},'first_training_step_time':{'value':step_elapsed,'unit':'s'}},'work_completed':32}


if __name__ == '__main__':
    print(json.dumps(measure(json.load(sys.stdin)),allow_nan=False))
