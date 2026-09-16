#!/usr/bin/env bash
# CPU/disk setup only. Does not start a model or stop existing workloads.
set -euo pipefail
[[ $(uname -m) == aarch64 ]] || { echo 'Expected ARM64 DGX Spark' >&2;exit 2; }
bash "$(dirname "$0")/install-python-headers.sh"
runtime="$HOME/.local/share/spark-gemma-lab/venv"
python3 -m venv "$runtime"
"$runtime/bin/python" -m pip install --upgrade pip
# Official 0.29.0 default wheels target CUDA 13, including ARM64.
# Binary-only fails clearly if a dependency needs a build; no surprise source compilation.
"$runtime/bin/python" -m pip install --only-binary=:all: 'vllm==0.29.0'
"$runtime/bin/python" "$(dirname "$0")/verify_runtime.py"
"$runtime/bin/python" -m pip freeze > "$HOME/.local/share/spark-gemma-lab/runtime-packages.txt"
"$runtime/bin/python" -c 'import vllm, torch; print("vLLM",vllm.__version__,"PyTorch",torch.__version__,"CUDA build",torch.version.cuda)'
echo 'User-owned runtime installed. No GPU inference started.'
