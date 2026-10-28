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
