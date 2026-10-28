#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
printf '%s\n' 'Profiles: smoke, standard, complete, extended. Use a manifest prepared by setup.sh.'
read -r -p 'Profile [smoke]: ' profile
profile=${profile:-smoke}
case "$profile" in smoke|standard|complete|extended) ;; *) printf '%s\n' 'Invalid profile' >&2; exit 1;; esac
read -r -p 'Protection condition label [protected]: ' condition
condition=${condition:-protected}
read -r -p 'Session name: ' session
[[ "$session" =~ ^[A-Za-z0-9_-]+$ ]] || { printf '%s\n' 'Invalid session name' >&2; exit 1; }
args=(run --manifest "$project_dir/prepared/$profile.json" --condition "$condition" --session "$session")
if [[ -f "$project_dir/sessions/$session/manifest.json" ]]; then
  read -r -p 'Session exists. Resume unfinished trials? [y/N]: ' answer
  [[ "$answer" == y || "$answer" == Y ]] || exit 1
  args+=(--resume)
fi
exec "$project_dir/bench.sh" "${args[@]}"
