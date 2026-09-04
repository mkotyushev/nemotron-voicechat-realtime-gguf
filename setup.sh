#!/usr/bin/env bash
# Create the small host-side environment used to download and convert weights.
set -euo pipefail

cd "$(dirname "$0")"
UV=${UV:-uv}

command -v "$UV" >/dev/null 2>&1 || {
    echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
}

if [ ! -x .venv/bin/python ]; then
    "$UV" venv --python 3.12 .venv
fi

VIRTUAL_ENV="$PWD/.venv" "$UV" pip install -r requirements-convert.txt

echo "converter environment ready: $PWD/.venv"
