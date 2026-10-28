#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
read -r -p 'Baseline session names (space-separated): ' -a baseline
read -r -p 'Protected session names (space-separated): ' -a protected
read -r -p 'Output directory: ' output
args=(compare)
for session in "${baseline[@]}"; do
  [[ "$session" =~ ^[A-Za-z0-9_-]+$ ]] || { printf '%s\n' 'Invalid session name' >&2; exit 1; }
  args+=(--baseline "$project_dir/sessions/$session")
done
for session in "${protected[@]}"; do
  [[ "$session" =~ ^[A-Za-z0-9_-]+$ ]] || { printf '%s\n' 'Invalid session name' >&2; exit 1; }
  args+=(--protected "$project_dir/sessions/$session")
done
args+=(--output "$output")
exec "$project_dir/bench.sh" "${args[@]}"
