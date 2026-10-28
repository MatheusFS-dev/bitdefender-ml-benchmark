"""Isolated multi-GPU PyTorch distributed training worker."""

import base64
import json
import os
import sys
import time


def distributed_throughput(world_size: int, batch_size: int, steps: int, elapsed: float) -> float:
    """Calculate aggregate data-parallel sample throughput.

    Args:
        world_size: Number of participating ranks.
        batch_size: Samples per rank per step.
        steps: Measured training steps per rank.
        elapsed: Maximum elapsed time across ranks in seconds.

    Returns:
        Aggregate samples per second.

    Raises:
        ValueError: If inputs are nonpositive.
    """
    if min(world_size, batch_size, steps, elapsed) <= 0:
        raise ValueError('Distributed throughput inputs must be positive')
    return world_size * batch_size * steps / elapsed


def main(encoded: str) -> None:
    """Run distributed CNN training and print the rank-zero result.

    Args:
        encoded: URL-safe base64-encoded case descriptor.
    """
    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.nn.parallel import DistributedDataParallel
    case = json.loads(base64.urlsafe_b64decode(encoded))
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl')
    try:
        torch.manual_seed(case['seed'] + rank)
        device = torch.device('cuda', rank)
        model = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10)).to(device)
        model = DistributedDataParallel(model, device_ids=[rank])
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        x = torch.randn(case['batch_size'], 3, 32, 32, device=device)
        y = torch.randint(0, 10, (case['batch_size'],), device=device)
        def step():
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
        for _ in range(case['warmup_steps']):
            step()
        torch.cuda.synchronize()
        dist.barrier()
        start = time.perf_counter()
        for _ in range(case['steps']):
            step()
        torch.cuda.synchronize()
        dist.barrier()
        elapsed = torch.tensor(time.perf_counter() - start, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        if rank == 0:
            seconds = float(elapsed.cpu())
            count = world_size * case['batch_size'] * case['steps']
            print(json.dumps({'status': 'completed', 'runtime': {'framework': 'pytorch', 'version': torch.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()}, 'metrics': {'elapsed': {'value': seconds, 'unit': 's'}, 'throughput': {'value': distributed_throughput(world_size, case['batch_size'], case['steps'], seconds), 'unit': 'samples/s'}}, 'work_completed': count}))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main(sys.argv[1])
