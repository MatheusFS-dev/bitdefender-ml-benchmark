"""Offline session reports for completed and partial result files."""

import csv
import html
import json
from pathlib import Path

from bdbench.statistics import summarize


def write_session_report(session_dir: Path, output_dir: Path) -> list[Path]:
    """Render a descriptive report from an existing session.

    Args:
        session_dir: Directory containing manifest, environment and results.
        output_dir: Destination for report files.

    Returns:
        Paths to the generated HTML, Markdown and CSV files.

    Raises:
        ValueError: If a record is malformed.
    """
    manifest = json.loads((session_dir / 'manifest.json').read_text())
    environment = json.loads((session_dir / 'environment.json').read_text())
    results_path = session_dir / 'results.jsonl'
    results = []
    incomplete_note = ''
    if results_path.exists():
        data = results_path.read_text()
        lines = data.splitlines()
        if data and not data.endswith('\n'):
            lines = lines[:-1]
            incomplete_note = 'An incomplete final result record was omitted from this report.'
        for number, line in enumerate(lines, 1):
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid result on line {number}') from exc
    groups = {}
    for record in results:
        for name, metric in record.get('metrics', {}).items():
            if isinstance(metric, dict):
                key = (str(record.get('case_id', 'unknown')), str(name), str(metric.get('unit', '')))
                groups.setdefault(key, []).append(metric.get('value') if record.get('status') == 'completed' else None)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'summary.csv'
    rows = []
    for (case, name, unit), values in sorted(groups.items()):
        stats = summarize(values)
        rows.append({'case_id': case, 'metric': name, 'unit': unit, **stats})
    figures = []
    plot_note = ''
    telemetry_path = session_dir / 'telemetry.jsonl'
    if telemetry_path.exists():
        from bdbench.plots import write_telemetry_figures
        raw_telemetry = telemetry_path.read_text()
        telemetry_lines = raw_telemetry.splitlines()
        if raw_telemetry and not raw_telemetry.endswith('\n'):
            telemetry_lines = telemetry_lines[:-1]
            plot_note = 'An incomplete final telemetry record was omitted.'
        telemetry = [json.loads(line) for line in telemetry_lines if line.strip()]
        try:
            figures = write_telemetry_figures(telemetry, output_dir / 'figures')
        except ImportError as exc:
            plot_note += f' Figures unavailable: {exc}'
    fields = ['case_id', 'metric', 'unit', 'count', 'mean', 'median', 'std', 'cv', 'p50', 'p90', 'p95', 'p99', 'max']
    with csv_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    state = environment.get('protection', {}).get('observed_state', 'unknown')
    environment_text = json.dumps({key: environment.get(key) for key in ('condition', 'hardware', 'software', 'protection')}, indent=2, ensure_ascii=False)
    planned_trials = {f"{case['case_id']}-{index}" for case in manifest.get('cases', []) if 'case_id' in case for index in range(case.get('repetitions', 0))}
    completed_trials = {record.get('trial_id') for record in results if record.get('status') == 'completed'}
    coverage = f'Planned trials: {len(planned_trials)}. Unfinished trials: {len(planned_trials - completed_trials)}.'
    note = 'A single session cannot establish Bitdefender overhead. Protection evidence may be incomplete. ' + incomplete_note
    case_descriptions = {str(case.get('case_id')): ' / '.join(str(case.get(key, 'unknown')) for key in ('framework', 'model', 'device', 'dataset')) + (f" / diagnostic: {case['diagnostic']}" if case.get('diagnostic') else '') for case in manifest.get('cases', [])}
    lines = ['# Benchmark session report', '', note, '', f"Manifest: {manifest.get('manifest_id', 'unknown')}", f'Protection state: {state}', f'Trials recorded: {len(results)}', coverage, '', '## Environment', '', '```json', environment_text, '```', '', '| Case | Metric | Unit | Count | Median |', '| --- | --- | --- | ---: | ---: |']
    for row in rows:
        clean = lambda value: str(value).replace('|', '\\|').replace('\n', ' ')
        label = case_descriptions.get(row['case_id'], row['case_id'])
        lines.append(f"| {clean(label)} | {clean(row['metric'])} | {clean(row['unit'])} | {row['count']} | {clean(row['median'])} |")
    issues = [f"{record.get('case_id', 'unknown')}: status={record.get('status', 'unknown')}, error={record.get('error', '')}, contamination={record.get('contamination', [])}, warnings={record.get('warnings', [])}" for record in results if record.get('status') != 'completed' or record.get('contamination') or record.get('warnings')]
    scaling = scaling_rows(manifest.get('cases', []), results)
    if scaling:
        lines.extend(['', '## Multi-GPU scaling', '', 'Efficiency is aggregate throughput divided by the sum of matching individual GPU rates.'] + [f"- {item['framework']} {item['devices']}: {item['efficiency']}" for item in scaling])
    if issues:
        lines.extend(['', '## Failures and contamination', ''] + [f'- {issue}' for issue in issues])
    md_path = output_dir / 'summary.md'
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    table = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in [case_descriptions.get(row['case_id'], row['case_id']), row['metric'], row['unit'], row['count'], row['median']]) + '</tr>' for row in rows)
    html_path = output_dir / 'report.html'
    import base64
    embedded = ''.join('<img alt="Telemetry figure" src="data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode('ascii') + '">' for path in figures if path.suffix == '.png')
    issue_html = ''.join('<li>' + html.escape(issue) + '</li>' for issue in issues)
    scaling_html = ''.join('<li>' + html.escape(f"{item['framework']} {item['devices']}: efficiency={item['efficiency']}") + '</li>' for item in scaling)
    html_path.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Benchmark session</title><h1>Benchmark session report</h1><p>' + html.escape(note) + '</p><p>' + html.escape(coverage) + '</p><p>Protection state: ' + html.escape(str(state)) + '</p><p>' + html.escape(plot_note) + '</p><h2>Environment</h2><pre>' + html.escape(environment_text) + '</pre><table><thead><tr><th>Case</th><th>Metric</th><th>Unit</th><th>Count</th><th>Median</th></tr></thead><tbody>' + table + '</tbody></table><h2>Multi-GPU scaling</h2><p>Aggregate throughput divided by the sum of matching single-device rates.</p><ul>' + scaling_html + '</ul><h2>Failures and contamination</h2><ul>' + issue_html + '</ul>' + embedded + '</html>', encoding='utf-8')
    return [html_path, md_path, csv_path, *figures]


def write_comparison_report(rows: list[dict], output_dir: Path, exclusions: list[dict] | None = None) -> list[Path]:
    """Write descriptive comparison artifacts without network access.

    Args:
        rows: Compatible case comparison rows.
        output_dir: Directory in which to write artifacts.
        exclusions: Planned cases lacking comparable trials.

    Returns:
        Paths of HTML, CSV and JSON artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    note = 'Protection policy status is not verified by process evidence alone. Differences are descriptive and may be confounded. Trial bootstrap intervals do not account for session-level confounding; multiple sessions use session-level resampling.'
    json_path = output_dir / 'comparison.json'
    exclusions = exclusions or []
    json_path.write_text(json.dumps({'note': note, 'rows': rows, 'exclusions': exclusions}, indent=2, allow_nan=False), encoding='utf-8')
    fields = ['case_id', 'metric', 'unit', 'baseline', 'protected', 'change_percent', 'ci95_low', 'ci95_high', 'baseline_count', 'protected_count', 'bootstrap_unit', 'baseline_sessions', 'protected_sessions', 'interpretation', 'verified_attribution']
    csv_path = output_dir / 'comparison.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    headers = ''.join(f'<th>{html.escape(field)}</th>' for field in fields)
    body = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(row.get(field, "")))}</td>' for field in fields) + '</tr>' for row in rows)
    html_path = output_dir / 'comparison.html'
    excluded = ''.join('<li>' + html.escape(str(item['case_id'])) + ': ' + html.escape(str(item['reason'])) + '</li>' for item in exclusions)
    html_path.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Benchmark comparison</title><h1>Benchmark comparison</h1><p>' + html.escape(note) + '</p><table><thead><tr>' + headers + '</tr></thead><tbody>' + body + '</tbody></table><h2>Excluded cases</h2><ul>' + excluded + '</ul></html>', encoding='utf-8')
    return [html_path, csv_path, json_path]


def scaling_rows(cases: list[dict], results: list[dict]) -> list[dict]:
    """Calculate multi-GPU scaling against matching individual GPU cases.

    Args:
        cases: Frozen case descriptors.
        results: Trial results for one session.

    Returns:
        Scaling rows with per-device baselines and efficiency ratios.
    """
    from statistics import median
    from bdbench.statistics import scaling_efficiency
    rates = {}
    for result in results:
        if result.get('status') != 'completed' or result.get('contamination'):
            continue
        metric = result.get('metrics', {}).get('throughput', {})
        value = metric.get('value')
        if metric.get('unit') == 'samples/s' and isinstance(value, (int, float)) and not isinstance(value, bool):
            rates.setdefault(result['case_id'], []).append(value)
    output = []
    for distributed in [case for case in cases if case.get('distributed')]:
        singles = []
        for uuid in distributed['device_uuids']:
            matching = [case for case in cases if not case.get('distributed') and not case.get('diagnostic') and case.get('device') == uuid and all(case.get(key) == distributed.get(key) for key in ('framework', 'model', 'dataset', 'precision', 'batch_size')) and not case.get('cpu_threads') and 'input_workers' not in case]
            values = rates.get(matching[0]['case_id'], []) if len(matching) == 1 else []
            singles.append(median(values) if values else None)
        values = rates.get(distributed['case_id'], [])
        multi = median(values) if values else None
        output.append({'case_id': distributed['case_id'], 'framework': distributed['framework'], 'devices': distributed['device_uuids'], 'distributed_samples_per_s': multi, 'single_samples_per_s': singles, 'efficiency': scaling_efficiency(multi, singles)})
    return output
