#!/usr/bin/env bash
# End-to-end smoke test — proves the whole pipeline wires up before the H100 run.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="src:${PYTHONPATH:-}"
echo "== unit tests =="
python -m pytest tests -q || python tests/test_token_extension.py
echo "== end-to-end smoke =="
python scripts/00_smoke.py --config configs/smoke.yaml "$@"
