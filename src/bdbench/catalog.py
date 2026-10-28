"""Hardware-neutral benchmark case catalog."""

from bdbench.serialization import canonical_hash


def cpu_thread_levels(physical: int, logical: int, effective: int) -> list[int]:
    """Choose a bounded, deduplicated CPU thread sweep.

    Args:
        physical: Physical CPU core count.
        logical: Logical CPU count.
        effective: CPU count allowed by affinity and resource limits.

    Returns:
        Sorted thread counts bounded by the effective capacity.
    """
    if min(physical, logical, effective) < 1:
        raise ValueError('CPU counts must be positive')
    return sorted({min(effective, n) for n in (1, (physical + 1) // 2, physical, logical)})


def build_cases(settings: dict) -> list[dict]:
    """Construct a deterministic workload list for selected devices.

    Args:
        settings: Profile, explicit GPU UUIDs, and CPU selection.

    Returns:
        Cases with stable identifiers.
    """
    profile = settings.get('profile', 'standard')
    repetitions = settings.get('repetitions') or {'smoke': 1, 'standard': 3, 'complete': 5, 'extended': 5}[profile]
    steps = {'smoke': 10, 'standard': 200, 'complete': 1000, 'extended': 1000}[profile]
    devices = (['cpu'] if settings.get('cpu', True) else []) + list(settings.get('gpu_uuids', []))
    cases = []
    for device in devices:
        for framework in ('pytorch', 'tensorflow'):
            startup = {'framework': framework, 'model': 'startup', 'device': device, 'diagnostic': 'startup', 'repetitions': repetitions, 'seed': settings.get('seed', 1234)}
            startup['case_id'] = canonical_hash(startup)[:16]
            cases.append(startup)
            if profile == 'extended':
                compile_case = {'framework': framework, 'model': 'compile', 'device': device, 'diagnostic': 'compile', 'timeout_s': 900, 'repetitions': repetitions, 'seed': settings.get('seed', 1234)}
                compile_case['case_id'] = canonical_hash(compile_case)[:16]
                cases.append(compile_case)
            for model in ('mlp', 'cnn', 'transformer', 'matmul'):
                case = {'framework': framework, 'model': model, 'device': device, 'batch_size': 32, 'steps': steps, 'warmup_steps': 5, 'repetitions': repetitions, 'seed': settings.get('seed', 1234), 'precision': 'fp32', 'dataset': 'generated'}
                case['case_id'] = canonical_hash(case)[:16]
                cases.append(case)
            cifar = settings.get('datasets', {}).get('cifar10')
            if cifar:
                case = {'framework': framework, 'model': 'cnn', 'device': device, 'batch_size': 32, 'steps': steps, 'warmup_steps': 5, 'repetitions': repetitions, 'seed': settings.get('seed', 1234), 'precision': 'fp32', 'dataset': 'cifar10', 'dataset_path': cifar['path'], 'dataset_sha256': cifar['sha256']}
                case['case_id'] = canonical_hash(case)[:16]
                cases.append(case)
                if profile in {'complete', 'extended'}:
                    quality = dict(case, batch_size=50, steps=1000, fixed_epoch=True, quality_target=0.6, timeout_s=14400)
                    quality.pop('case_id')
                    quality['case_id'] = canonical_hash(quality)[:16]
                    cases.append(quality)
                    image_files = dict(case, dataset='cifar10_jpeg')
                    image_files.pop('case_id')
                    image_files['case_id'] = canonical_hash(image_files)[:16]
                    cases.append(image_files)
                    for layout in ('cifar10_mmap', 'cifar10_shards'):
                        layout_case = dict(case, dataset=layout)
                        layout_case.pop('case_id')
                        layout_case['case_id'] = canonical_hash(layout_case)[:16]
                        cases.append(layout_case)
                    if framework == 'tensorflow':
                        record_case = dict(case, dataset='cifar10_tfrecord')
                        record_case.pop('case_id')
                        record_case['case_id'] = canonical_hash(record_case)[:16]
                        cases.append(record_case)
                if framework == 'pytorch':
                    resnet = dict(case, model='resnet18')
                    resnet.pop('case_id')
                    resnet['case_id'] = canonical_hash(resnet)[:16]
                    cases.append(resnet)
            wiki = settings.get('datasets', {}).get('wikitext2')
            if wiki:
                case = {'framework': framework, 'model': 'transformer_lm', 'device': device, 'batch_size': 16, 'steps': steps, 'warmup_steps': 5, 'repetitions': repetitions, 'seed': settings.get('seed', 1234), 'precision': 'fp32', 'dataset': 'wikitext2', 'dataset_path': wiki['path'], 'dataset_sha256': wiki['sha256']}
                case['case_id'] = canonical_hash(case)[:16]
                cases.append(case)
    if settings.get('cpu', True):
        for model in ('logistic', 'hist_gradient', 'forest'):
            case = {'framework': 'sklearn', 'model': model, 'device': 'cpu', 'repetitions': repetitions, 'seed': settings.get('seed', 1234), 'dataset': 'generated'}
            case['case_id'] = canonical_hash(case)[:16]
            cases.append(case)
    for root in settings.get('storage_roots', []):
        for model in ('sequential', 'random_read', 'small_files', 'metadata_remove', 'scalar_logs', 'checkpoint'):
            case = {'framework': 'filesystem', 'model': model, 'device': 'cpu', 'storage_root': root, 'storage_roots': settings.get('storage_roots', []), 'size_bytes': 1048576 if profile == 'smoke' else 67108864, 'repetitions': repetitions, 'seed': settings.get('seed', 1234)}
            case['case_id'] = canonical_hash(case)[:16]
            cases.append(case)
        for device in devices:
            for framework in ('pytorch', 'tensorflow'):
                for durable in (False, True):
                    case = {'framework': framework, 'model': 'model_checkpoint', 'device': device, 'storage_root': root, 'storage_roots': settings.get('storage_roots', []), 'durable': durable, 'repetitions': repetitions, 'seed': settings.get('seed', 1234)}
                    case['case_id'] = canonical_hash(case)[:16]
                    cases.append(case)
            if profile in {'complete', 'extended'}:
                for framework in ('pytorch', 'tensorflow'):
                    case = {'framework': framework, 'model': 'cnn', 'device': device, 'batch_size': 32, 'steps': steps, 'warmup_steps': 5, 'repetitions': repetitions, 'seed': settings.get('seed', 1234), 'precision': 'fp32', 'dataset': 'generated', 'storage_root': root, 'storage_roots': settings.get('storage_roots', []), 'checkpoint_interval': 50}
                    case['case_id'] = canonical_hash(case)[:16]
                    cases.append(case)
    if profile in {'complete', 'extended'}:
        base_cases = [case for case in cases if case.get('dataset') == 'generated' and case.get('model') == 'cnn' and not case.get('checkpoint_interval') and case['framework'] in {'pytorch', 'tensorflow'}]
        for base in base_cases:
            diagnostic = dict(base, diagnostic='step_latency')
            diagnostic.pop('case_id')
            diagnostic['case_id'] = canonical_hash(diagnostic)[:16]
            cases.append(diagnostic)
            if profile == 'extended' and settings.get('framework_profile'):
                trace_case = dict(base, diagnostic='framework_profiler', steps=30)
                trace_case.pop('case_id')
                trace_case['case_id'] = canonical_hash(trace_case)[:16]
                cases.append(trace_case)
            if profile == 'extended' and settings.get('perf') and base['device'] == 'cpu':
                perf_case = dict(base, diagnostic='perf_stat')
                perf_case.pop('case_id')
                perf_case['case_id'] = canonical_hash(perf_case)[:16]
                cases.append(perf_case)
            alternate = dict(base, batch_size=16)
            alternate.pop('case_id')
            alternate['case_id'] = canonical_hash(alternate)[:16]
            cases.append(alternate)
            if base['device'] != 'cpu':
                mixed = dict(base, precision='mixed_float16')
                mixed.pop('case_id')
                mixed['case_id'] = canonical_hash(mixed)[:16]
                cases.append(mixed)
            if base['device'] == 'cpu':
                physical = settings.get('physical_cores', 1)
                logical = settings.get('logical_cores', physical)
                effective = settings.get('effective_cpus', logical)
                for count in cpu_thread_levels(physical, logical, effective):
                    varied = dict(base, cpu_threads=count)
                    varied.pop('case_id')
                    varied['case_id'] = canonical_hash(varied)[:16]
                    cases.append(varied)
        input_cases = [case for case in cases if case.get('dataset') in {'cifar10', 'cifar10_jpeg'} and case.get('model') == 'cnn' and not case.get('fixed_epoch')]
        for base in input_cases:
            for count in sorted({min(settings.get('effective_cpus', 1), count) for count in (0, 2, 4, 8)}):
                varied = dict(base, input_workers=count)
                varied.pop('case_id')
                varied['case_id'] = canonical_hash(varied)[:16]
                cases.append(varied)
            if base['framework'] == 'pytorch' and base['device'] != 'cpu':
                varied = dict(base, pin_memory=True, input_workers=min(settings.get('effective_cpus', 1), 4))
                varied.pop('case_id')
                varied['case_id'] = canonical_hash(varied)[:16]
                cases.append(varied)
    if profile == 'extended' and len(settings.get('gpu_uuids', [])) >= 2:
        selected = list(settings['gpu_uuids'])
        for framework in ('pytorch', 'tensorflow'):
            case = {'framework': framework, 'model': 'cnn', 'device': selected[0], 'device_uuids': selected, 'distributed': True, 'batch_size': 32, 'steps': steps, 'warmup_steps': 5, 'repetitions': repetitions, 'seed': settings.get('seed', 1234), 'precision': 'fp32', 'dataset': 'generated'}
            case['case_id'] = canonical_hash(case)[:16]
            cases.append(case)
    return cases
