#!/usr/bin/env bash
# Separate Linux/Pascal environment: never let uv sync replace this CUDA build.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo 'This setup targets Linux x86_64 and GTX 10xx/Pascal GPUs.' >&2
    exit 1
fi
capt_pascal_env="${CAPT_PASCAL_ENV:-$PWD/.venv-pascal}"
if [[ ! -x "$capt_pascal_env/bin/python" ]]; then
    uv venv --python 3.12 "$capt_pascal_env"
fi
# Keep base/test dependencies locked, then replace only the optional torch stack.
UV_PROJECT_ENVIRONMENT="$capt_pascal_env" uv sync --locked --no-extra mass
uv pip install --python "$capt_pascal_env/bin/python" \
    'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124
"$capt_pascal_env/bin/python" -c 'import torch; print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda)'
echo "Ready: $capt_pascal_env/bin/capt12 (invoke directly, without uv run/sync)."
