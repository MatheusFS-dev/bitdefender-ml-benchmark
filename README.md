# Bitdefender ML benchmark

A Linux benchmark suite for comparing machine-learning workloads with and without Bitdefender using the same frozen workload manifest. It supports CPU, explicitly selected NVIDIA GPUs, isolated PyTorch and TensorFlow environments, scikit-learn, input pipelines, checkpoints and filesystem controls. Reports separate measured performance from protection evidence and comparison uncertainty.

## Prerequisites

Ubuntu or another compatible Linux distribution, Python 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/), and for GPU cases a compatible NVIDIA driver. Framework dependencies are installed into separate uv environments. Availability of CUDA and framework compatibility have not been tested on this machine.

## Configuration and preparation

Edit the desired `configs/*.toml`. Set `storage.roots` to existing directories you explicitly authorize for the bounded storage cases. The suite uses a marked `.bdbench-data` subdirectory in each selected root. `gpu_uuids` is an explicit list of NVIDIA UUIDs. Leave it empty for CPU-only testing. `./bench.sh doctor` shows detected GPU UUIDs and available Bitdefender process evidence.

Run `./setup.sh` to synchronize the environments, inspect the machine, and prepare a profile. Preparation runs short training probes to calibrate work and, when a storage root is selected, downloads and prepares CIFAR-10 and WikiText-2. Alternatively:

```bash
uv sync --locked --project .
uv sync --locked --project environments/pytorch
uv sync --locked --project environments/tensorflow
./bench.sh doctor
./bench.sh prepare --config configs/smoke.toml --output prepared/smoke.json
```

Use the same prepared manifest for both protection conditions. Change Bitdefender policy only through your authorized administrative process. The scripts do not modify security settings. Avoid other GPU and CPU workloads while running.

## Sessions and reports

Run `./run.sh` and enter the profile, condition label and a unique session name. Alternatively:

```bash
./bench.sh run --manifest prepared/smoke.json --condition protected --session protected-01
./bench.sh run --manifest prepared/smoke.json --condition baseline --session baseline-01
./bench.sh compare --baseline sessions/baseline-01 --protected sessions/protected-01 --output reports/comparison
```

Use `./compare.sh` for interactive comparison, or repeat `--baseline` and `--protected` for multiple independent sessions. Resume an interrupted session with the original manifest and `--resume`. Reports are generated even when worker cases fail.

| Artifact | Contents |
| --- | --- |
| `prepared/*.json` | Frozen case settings and manifest identity |
| `sessions/<name>/manifest.json` | Copy of the frozen manifest |
| `sessions/<name>/environment.json` | Hardware/software identity, operator condition and protection evidence |
| `sessions/<name>/results.jsonl` | Trial results and failures |
| `sessions/<name>/telemetry.jsonl` | Timestamped host, worker, Bitdefender, selected GPU and available RAPL samples |
| `sessions/<name>/report/` | Offline HTML, Markdown, CSV and PNG/PDF resource figures |
| Comparison directory | HTML, CSV and JSON descriptive comparisons |

See [measurement definitions](docs/measurements.md). A single protected session cannot quantify Bitdefender overhead. The read-only product-status command reports the antimalware module when available, but does not establish the full on-access policy or absence of protection.

## Workloads and limits

Profiles are `smoke`, `standard`, `complete` and `extended`. Complete adds targeted batch-size, CPU-thread, precision and input-worker sweeps. Extended adds compiler, optional profiler/perf diagnostics and multi-GPU cases when at least two GPU UUIDs are selected. `setup.sh` also accepts `all` to prepare separate manifests sharing the selected dataset directory.

Generated MLP, CNN, transformer and matrix workloads provide compute controls. Selecting storage enables CIFAR-10 CNN/ResNet-18, WikiText-2 language modeling, JPEG/mmap/NumPy-shard/TFRecord input cases as applicable, model-and-optimizer checkpoints, periodic checkpoint stalls, scalar logs and filesystem operations. Fixed-epoch CIFAR cases report full-test accuracy. Other quality measurements use explicitly identified held-out batches.

Set `perf = true` or `framework_profile = true` in `configs/extended.toml` for separate diagnostic cases. Traces are stored under `sessions/<name>/diagnostics/`. Instrumented results remain separate from primary throughput.

Use repeated independent baseline/protected sessions in alternating order to reduce time-order confounding. Condition labels do not verify policy or exclusions. Preparation and integrity checks can warm caches, so results are not verified cold-cache measurements. Resource samples span worker setup, warmup and execution, while worker elapsed timing covers the stated operation. Timeline figures show the first selected GPU, with all selected GPU samples retained in JSONL.

Implementation was checked with mocked workers and synthetic fixtures only. Framework execution, GPU compatibility, training quality and energy accuracy remain unvalidated on this machine. **The benchmark was not run.**
