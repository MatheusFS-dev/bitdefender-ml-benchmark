#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
condition=${1:?Expected protected or baseline}
[[ "$condition" == protected || "$condition" == baseline ]] || { echo 'Invalid condition' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'Missing nvidia-smi or NVIDIA driver.' >&2; exit 1; }
if ! gpu_output=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader,nounits); then
  echo 'Could not list NVIDIA GPUs with nvidia-smi.' >&2
  exit 1
fi
[[ -n "$gpu_output" ]] || { echo 'No NVIDIA GPUs detected.' >&2; exit 1; }
mapfile -t gpu_rows <<< "$gpu_output"
gpu_uuids=()
gpu_names=()
for row in "${gpu_rows[@]}"; do
  [[ "$row" == *,* ]] || { echo "Invalid nvidia-smi GPU row: $row" >&2; exit 1; }
  gpu_uuids+=("${row%%,*}")
  gpu_names+=("${row#*, }")
done
echo 'Available NVIDIA GPUs:'
for index in "${!gpu_uuids[@]}"; do
  printf '  %d) %s\n' "$((index + 1))" "${gpu_names[$index]}"
done
if ! read -r -p 'Choose a GPU by number: ' choice; then
  echo 'GPU choice required.' >&2
  exit 1
fi
if [[ ! "$choice" =~ ^[1-9][0-9]*$ ]] || (( choice > ${#gpu_uuids[@]} )); then
  echo 'Invalid GPU choice.' >&2
  exit 1
fi
gpu_index=$((choice - 1))
gpu_uuid=${gpu_uuids[$gpu_index]}
gpu_name=${gpu_names[$gpu_index]}
command -v uv >/dev/null || { echo 'Missing uv. Install uv before running this script.' >&2; exit 1; }
status=unknown
if [[ -x /opt/bitdefender-security-tools/bin/bduitool ]]; then
  if status_output=$(/opt/bitdefender-security-tools/bin/bduitool get ps 2>&1); then
    :
  elif command -v sudo >/dev/null && status_output=$(sudo -n /opt/bitdefender-security-tools/bin/bduitool get ps 2>&1); then
    :
  else
    echo 'Cannot read Bitdefender status. Run sudo -v, then rerun this script while the sudo credential is cached.' >&2
    echo "bduitool: $status_output" >&2
    exit 1
  fi
  status=$(printf '%s\n' "$status_output" | awk -F: 'tolower($1) ~ /antimalware status/ {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print tolower($2); exit}')
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
echo "[1/5] GPU selected: $gpu_name ($gpu_uuid); condition: $condition; session: $session"
echo '[2/5] Synchronizing controller dependencies.'
uv sync --locked --project "$project_dir"
echo '[3/5] Synchronizing PyTorch dependencies.'
uv sync --locked --project "$project_dir/environments/pytorch"
echo '[4/5] Synchronizing TensorFlow dependencies.'
uv sync --locked --project "$project_dir/environments/tensorflow"
echo '[5/5] Preparing datasets and manifest, then running benchmark trials.'
"$project_dir/bench.sh" prepare --no-calibrate --config "$project_dir/configs/standard.toml" --gpu "$gpu_uuid" --storage-root "$storage_root" --output "$manifest"
echo "[benchmark] Manifest ready: $manifest"
"$project_dir/bench.sh" run --manifest "$manifest" --condition "$condition" --session "$session"
echo "Results: $project_dir/sessions/$session"
echo 'Copy the whole sessions/<name> directory to one PC, then run:'
echo './bench.sh compare --baseline /path/to/baseline-session --protected /path/to/protected-session --output reports/comparison'
