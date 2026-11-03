"""Separate compilation and in-process compiled-work reuse diagnostics."""

import json
import sys
import time
from bdbench.progress import emit_progress


def measure(case: dict) -> dict:
    """Measure the first and subsequent compiled training steps.

    Args:
        case: Framework and explicitly selected device.

    Returns:
        First-compile and in-process reuse latencies with runtime metadata.
    """
    gpu = case['device'] != 'cpu'
    emit_progress(f'Importing {case["framework"]} and constructing compiled model')
    if case['framework'] == 'pytorch':
        import torch
        if gpu and not torch.cuda.is_available():
            raise RuntimeError('Selected CUDA device is unavailable to PyTorch')
        device = torch.device('cuda:0' if gpu else 'cpu')
        torch.manual_seed(case['seed'])
        model = torch.compile(torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.ReLU(), torch.nn.Linear(512, 10)).to(device))
        optimizer = torch.optim.Adam(model.parameters())
        x = torch.randn(32, 256, device=device)
        y = torch.randint(0, 10, (32,), device=device)
        def step():
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            if gpu:
                torch.cuda.synchronize()
        emit_progress('Measuring first compiled step')
        start = time.perf_counter()
        step()
        first = time.perf_counter() - start
        emit_progress(f'First compiled step complete in {first:.2f} s; measuring reuse')
        start = time.perf_counter()
        step()
        second = time.perf_counter() - start
        emit_progress(f'Reused step complete in {second:.2f} s')
        runtime = {'framework':'pytorch','version':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version()}
    elif case['framework'] == 'tensorflow':
        import tensorflow as tf
        if gpu and not tf.config.list_physical_devices('GPU'):
            raise RuntimeError('Selected CUDA device is unavailable to TensorFlow')
        tf.random.set_seed(case['seed'])
        with tf.device('/GPU:0' if gpu else '/CPU:0'):
            model = tf.keras.Sequential([tf.keras.layers.Input((256,)), tf.keras.layers.Dense(512, activation='relu'), tf.keras.layers.Dense(10)])
            optimizer = tf.keras.optimizers.Adam()
            x = tf.random.normal((32,256))
            y = tf.random.uniform((32,),maxval=10,dtype=tf.int32)
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            @tf.function(jit_compile=True)
            def step():
                with tf.GradientTape() as tape:
                    loss = loss_fn(y, model(x, training=True))
                optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
                return loss
            emit_progress('Measuring first compiled step')
            start = time.perf_counter()
            step().numpy()
            first = time.perf_counter() - start
            emit_progress(f'First compiled step complete in {first:.2f} s; measuring reuse')
            start = time.perf_counter()
            step().numpy()
            second = time.perf_counter() - start
            emit_progress(f'Reused step complete in {second:.2f} s')
        runtime = {'framework':'tensorflow','version':tf.__version__,'build':tf.sysconfig.get_build_info()}
    else:
        raise ValueError('Unsupported compiler framework')
    return {'status':'completed','runtime':runtime,'metrics':{'first_compiled_step_time':{'value':first,'unit':'s'},'reused_compiled_step_time':{'value':second,'unit':'s'}},'work_completed':64}


if __name__ == '__main__':
    print(json.dumps(measure(json.load(sys.stdin)),allow_nan=False))
