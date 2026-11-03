#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
condition=${1:?Expected protected or baseline}
[[ "$condition" == protected || "$condition" == baseline ]] || { echo 'Invalid condition' >&2; exit 2; }
command -v uv >/dev/null || { echo 'Missing uv. Install uv before running this script.' >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo 'Missing nvidia-smi or NVIDIA driver.' >&2; exit 1; }
mapfile -t gpu_rows < <(nvidia-smi --query-gpu=uuid,name --format=csv,noheader,nounits)
gpu_uuid=''
for row in "${gpu_rows[@]}"; do
  if [[ "$row" == *'RTX 4070' ]]; then
    [[ -z "$gpu_uuid" ]] || { echo 'More than one RTX 4070 found. Select one explicitly in a custom run.' >&2; exit 1; }
    gpu_uuid=${row%%,*}
  fi
done
[[ -n "$gpu_uuid" ]] || { echo 'RTX 4070 was not found.' >&2; exit 1; }
status=unknown
if [[ -x /opt/bitdefender-security-tools/bin/bduitool ]]; then
  status=$(/opt/bitdefender-security-tools/bin/bduitool get ps 2>/dev/null | awk -F: 'tolower($1) ~ /antimalware status/ {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print tolower($2); exit}')
fi
if [[ "$condition" == protected && "$status" != on ]]; then
  echo 'Bitdefender on-access status is not reported as on. Check bduitool get ps and permissions.' >&2
  exit 1
fi
if [[ "$condition" == baseline && ( "$status" == on || -d /opt/bitdefender-security-tools ) ]]; then
  echo 'Bitdefender is present or reports on. The baseline PC must be without Bitdefender.' >&2
  exit 1
fi
storage_root=${BDBENCH_STORAGE_ROOT:-$project_dir/benchmark-data}
mkdir -p -- "$storage_root"
machine=$(hostname -s | tr -c 'A-Za-z0-9_-' '_')
session=${BDBENCH_SESSION:-${machine}-${condition}-$(date -u +%Y%m%dT%H%M%SZ)}
[[ "$session" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid BDBENCH_SESSION' >&2; exit 1; }
manifest="$project_dir/prepared/$session.json"
[[ ! -e "$manifest" && ! -e "$project_dir/sessions/$session" ]] || { echo "Session already exists: $session" >&2; exit 1; }
uv sync --locked --project "$project_dir"
uv sync --locked --project "$project_dir/environments/pytorch"
uv sync --locked --project "$project_dir/environments/tensorflow"
"$project_dir/bench.sh" prepare --no-calibrate --config "$project_dir/configs/standard.toml" --gpu "$gpu_uuid" --storage-root "$storage_root" --output "$manifest"
"$project_dir/bench.sh" run --manifest "$manifest" --condition "$condition" --session "$session"
echo "Results: $project_dir/sessions/$session"
echo 'Copy the whole sessions/<name> directory to one PC, then run:'
echo './bench.sh compare --baseline /path/to/baseline-session --protected /path/to/protected-session --output reports/comparison'
