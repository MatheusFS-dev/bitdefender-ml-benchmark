#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
command -v uv >/dev/null 2>&1 || { printf '%s\n' 'Install uv: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 1; }
uv sync --locked --project "$project_dir"
"$project_dir/bench.sh" doctor
uv sync --locked --project "$project_dir/environments/pytorch"
uv sync --locked --project "$project_dir/environments/tensorflow"
printf '%s\n' 'Profiles: smoke, standard, complete, extended, all. Configure GPU UUIDs in the chosen TOML.'
read -r -p 'Profile [smoke]: ' profile
profile=${profile:-smoke}
case "$profile" in smoke|standard|complete|extended|all) ;; *) printf '%s\n' 'Invalid profile' >&2; exit 1;; esac
printf '%s\n' 'Enter GPU UUIDs from the doctor output separated by spaces, or leave empty to use configured GPUs.'
read -r -p 'GPU UUIDs: ' -a gpus
printf '%s\n' 'Enter an existing storage directory to include storage and CIFAR-10 cases, or leave empty to omit them.'
read -r -p 'Storage directory: ' storage
profiles=("$profile")
if [[ "$profile" == all ]]; then profiles=(smoke standard complete extended); fi
for selected_profile in "${profiles[@]}"; do
args=(prepare --config "$project_dir/configs/$selected_profile.toml" --output "$project_dir/prepared/$selected_profile.json")
for gpu in "${gpus[@]}"; do args+=(--gpu "$gpu"); done
if [[ -n "$storage" ]]; then args+=(--storage-root "$storage"); fi
"$project_dir/bench.sh" "${args[@]}"
done
