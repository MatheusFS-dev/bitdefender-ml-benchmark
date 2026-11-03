"""Offline figures for telemetry."""

from pathlib import Path


def extract_series(rows: list[dict], source: str, metric: str) -> tuple[list[float], list]:
    """Extract a telemetry series and preserve missing samples as gaps.

    Args:
        rows: Timestamped telemetry samples.
        source: Telemetry source such as host or gpu.
        metric: Metric field in the source value.

    Returns:
        Elapsed seconds and the corresponding sample values.
    """
    if not rows:
        return [], []
    start = rows[0]['monotonic_ns']
    times, values = [], []
    for row in rows:
        times.append((row['monotonic_ns'] - start) / 1e9)
        source_value = row.get(source) or {}
        values.append((source_value.get('value') or {}).get(metric))
    return times, values


def write_telemetry_figures(rows: list[dict], output: Path) -> list[Path]:
    """Render supported telemetry series to PNG and PDF.

    Args:
        rows: Parsed telemetry records.
        output: Directory in which to save figures.

    Returns:
        Created figure paths.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not rows:
        return []
    plt.rcParams.update({'font.family': 'serif', 'figure.facecolor': 'white', 'axes.spines.top': False, 'axes.spines.right': False})
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for source, metric, label in [('host', 'cpu_percent', 'CPU utilization (%)'), ('gpu', 'utilization_percent', 'GPU utilization (%)'), ('gpu', 'power_w', 'GPU board power (W)')]:
        times, values = extract_series(rows, source, metric)
        if all(value is None for value in values):
            continue
        fig, axis = plt.subplots(figsize=(7, 3))
        axis.plot(times, values, color='#47677a', linewidth=1)
        axis.set_xlabel('Elapsed time (s)')
        axis.set_ylabel(label)
        fig.tight_layout()
        for extension in ('png', 'pdf'):
            path = output / f'{source}_{metric}.{extension}'
            fig.savefig(path, dpi=180, bbox_inches='tight')
            paths.append(path)
        plt.close(fig)
    return paths
def plot_comparison(rows: list[dict], output: Path) -> None:
    """Plot relative differences for measured performance metrics.

    Args:
        rows: Comparable metric summaries.
        output: PNG destination.
    """
    import matplotlib.pyplot as plt
    selected = [row for row in rows if row.get('change_percent') is not None and any(word in row['metric'] for word in ('throughput', 'elapsed', 'load_time', 'save_time', 'dataset_load', 'gpu_utilization', 'gpu_power', 'bitdefender_cpu', 'small_files'))]
    selected = selected[:40]
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})
    fig, ax = plt.subplots(figsize=(10, max(3, .28 * len(selected) + 1)))
    labels = [f"{row['case_id']} {row['metric']}" for row in selected]
    values = [row['change_percent'] for row in selected]
    ax.barh(labels, values, color='#527b91')
    ax.axvline(0, color='#333333', linewidth=.7)
    ax.set_xlabel('Diferença relativa (%)')
    ax.grid(axis='x', linestyle='--', linewidth=.4, alpha=.35)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches='tight')
    plt.close(fig)
