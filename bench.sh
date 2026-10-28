#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if ! command -v uv >/dev/null 2>&1; then
  printf '%s\n' 'uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/.' >&2
  exit 1
fi
exec uv run --locked --project "$project_dir" bdbench "$@"
