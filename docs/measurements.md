# Measurement definitions

Workers use monotonic timing and frozen work from the prepared manifest. Training workers separate warmup from measured work. Startup, compilation, storage and checkpoint cases have operation-specific boundaries. GPU execution is synchronized at the timed boundaries. Public CIFAR-10 and WikiText-2 are prepared only when a storage root is selected. Data hashes are checked before a session.

| Output | Measurement, units and limits |
| --- | --- |
| `elapsed` | Seconds measured by the worker's monotonic performance clock over the fixed training, fit, or filesystem operation. Training time excludes dataset setup and validation. |
| `throughput` | Completed samples, tokens, bytes or operations divided by elapsed seconds, with the unit stored in each result. Generated-input cases use resident data. |
| `step_latency_p50`, `step_latency_p90`, `step_latency_p95`, `step_latency_p99`, `step_latency_max` | Seconds per synchronized step in a separate diagnostic case. Tail percentiles require enough observations. Diagnostic overhead changes throughput and must not be pooled with primary cases. |
| `checkpoint_stall_time`, `checkpoint_count` | Sum of wall-clock seconds spent writing periodic model-and-optimizer checkpoints and number written, in dedicated training cases. They are buffered writes unless stated otherwise. |
| `dataset_load_time` | Seconds to construct the PyTorch CIFAR-10 dataset or read and construct TensorFlow CIFAR-10 input data. Framework measurements are not directly equivalent. |
| `peak_allocated_memory`, `peak_reserved_memory` | PyTorch CUDA allocator peak bytes after reset at the timed boundary. These are not total GPU memory use. |
| `training_loss`, `validation_batch_loss` | Mean token cross-entropy in nats per token for the final training batch or one fixed held-out batch. |
| `validation_batch_perplexity` | Exponential of the held-out batch loss. This is a one-batch estimate, not full-dataset perplexity. Unrepresentable values are missing. |
| `validation_batch_accuracy` | Fraction of correct CIFAR-10 predictions on one fixed held-out batch. It is not full-test accuracy. |
| `validation_accuracy`, `training_elapsed_at_validation_target` | Full CIFAR-10 test accuracy at the end of the fixed-epoch case and training seconds at that checkpoint when accuracy reaches the preselected target. Missing when target is not reached. This is a checkpoint-bound upper estimate, not an observed first-crossing time. |
| `accuracy` | scikit-learn classifier score on the fixed held-out generated subset, as a fraction. |
| `prediction_time` | Seconds to score the scikit-learn held-out subset. |
| `save_time`, `load_time`, `checkpoint_size` | Monotonic time in seconds and total serialized bytes for model and optimizer checkpoints. Buffered and fsynced variants are distinct cases. These paths have not been exercised against the installed frameworks. |
| Filesystem throughput | Sequential read/write, random reads, small-file operations, or a flushed and fsynced file write/read in a selected owned directory. Bytes/s and operations/s are distinct. The fsynced file is a filesystem control, not model checkpoint serialization. |
| Host telemetry | psutil CPU utilization since its previous sample, per-core utilization, available RAM and used swap in bytes, and OS cumulative disk counters. Host-wide disk counters are not process attribution. |
| Bitdefender process counters | Visible process CPU user/system seconds, RSS bytes, I/O counters and creation timestamps from procfs/psutil. Process visibility and permissions can limit coverage, and kernel-side scanning is excluded. |
| RAPL telemetry | Linux powercap per-domain energy counters and ranges in microjoules. `rapl_energy_<domain>` sums observed counter deltas in joules, assuming at most one wrap between samples. It is not whole-PC or process energy. |
| NVML extras | Per-GPU memory utilization percent, memory clock MHz, PCIe TX/RX in NVML-reported KB/s, throttle-reason bitmask and power limit W where supported. NVML PCIe rates reflect its short sampling interval, not total transferred bytes. Unsupported fields retain a reason. |
| GPU telemetry | `nvidia-smi` utilization percent, memory used in MiB, temperature in °C, board power in W, and SM clock in MHz, sampled approximately once per second. These are GPU-wide values, not GPU occupancy. Unavailable queries include a reason. |
| `gpu_board_energy_sampled` | Trapezoidal integration of GPU board watts between the first and last samples of a trial, in joules. It includes sampled worker setup and warmup, omits unsampled edges, and does not measure whole-PC energy. |
| Comparison | Within a matching manifest and recorded environment, median protected/baseline trial values. Throughput loss is `100 × (1 − protected / baseline)`. Other relative changes use `100 × (protected / baseline − 1)`. Seeded 95% bootstrap intervals use independent sessions when multiple sessions are supplied. With one session per condition they use trials and do not account for session-level confounding. An interval requires at least two independent units per condition. |

Bitdefender process names are read from procfs before and after trials. The documented read-only `bduitool get ps` command supplies an allowlisted product version, signatures version and reported antimalware status when accessible. Bitdefender describes this status as referring to on-access scanning in its [Linux troubleshooting guide](https://www.bitdefender.com/business/support/en/77211-343058-on-access-scanning-in-bitdefender-endpoint-security-tools-for-linux.html). A reported on state does not establish that every benchmark path is covered or that exclusions are absent. A change in the observed state or component set marks a trial as contaminated. Product-status and process evidence may be incomplete and do not independently verify the full on-access policy. A condition label is an operator claim, not proof of protection state. A protected-only session cannot establish Bitdefender overhead.

Training quality, profiles, hardware compatibility, energy accuracy and GPU execution have not been validated by an actual benchmark on this machine. No results should be inferred from implementation tests.

Individual JPEG, memory-mapped array, uncompressed NumPy-shard and TensorFlow-record cases use different input representations. Dataset preparation and integrity checking can warm filesystem and security caches before the run. The suite does not claim verified cold-cache results.

## Timing and resource scope

Every scalar result stores its unit, source, scope, availability and missing-value reason. Session records link trial and attempt identifiers to UTC anchors and monotonic timestamps. Telemetry records include actual sample intervals and missed slots. Sampling means are arithmetic means of available observations, not time-weighted means. Peaks are sampled maxima and can miss short spikes. Missing measurements are not zeros.

Host disk summaries subtract consecutive cumulative read/write byte, operation and millisecond counters. Negative reset deltas are excluded. These counters cover the host and may include unrelated processes or stacked block devices. Process RSS sums can double-count shared pages. Raw process records retain PSS where readable, page faults, context switches, CPU and I/O counters.

Startup diagnostics separate framework import, device initialization, first training step and parent-observed process lifetime. Compilation diagnostics separate the first compiled step from a second in-process cached step. They do not establish cross-process compiler-cache reuse. Optional perf counters are process-command measurements with tool-specific inheritance and permission limits. Framework traces expose operation stages without adding their overlapping durations into an invented total.

GPU and RAPL energy integrations span sampled worker-lifecycle intervals. Per-work energy divides those joules by completed worker work, so setup/warmup energy can contribute. Per-GPU board energy and per-domain RAPL energy remain separate to avoid double counting. Multi-GPU scaling efficiency divides aggregate throughput by the sum of matching single-device throughput baselines, when available.

## Artifacts and comparison eligibility

Results preserve failed, skipped and contaminated trials. Comparison excludes ineligible results and requires matching frozen manifests and recorded hardware/software identities. Worker runtime metadata is checked before aggregation. Repeated sessions contribute one median per session to bootstrap uncertainty. Single-session trial intervals do not estimate between-session uncertainty. No global score combines unlike metrics.
