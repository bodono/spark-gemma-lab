#!/usr/bin/env bash
# Run on a Spark. Does not stop or alter existing workloads.
set -euo pipefail
role=${1:?Usage: serve-spark.sh diffusion|autoregressive}
case "$role" in
 diffusion) model=RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic; revision=3b3dae4697494da5a290e9c0461954449e76c4f5; served=diffusiongemma-fp8; extra=(--diffusion-config '{"canvas_length":256}' --override-generation-config '{"max_new_tokens":null}') ;;
 autoregressive) model=RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic; revision=ed35d7abe5d940da41b4ff06eb482feb0be8cb44; served=gemma4-fp8; extra=() ;;
 *) echo 'Unknown role' >&2; exit 2 ;;
esac
if [[ $(uname -m) != aarch64 ]]; then echo 'This launcher is pinned to ARM64 DGX Spark.' >&2; exit 2; fi
# Suspended jobs retain memory but do not contend for GPU compute.
# Refuse active compute processes unless an intentional contended run is requested.
if [[ ${ALLOW_BUSY:-0} != 1 ]]; then
 compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
 while IFS= read -r pid; do
  pid=${pid// /}
  [[ -z "$pid" ]] && continue
  [[ $pid =~ ^[0-9]+$ ]] || { echo 'Cannot identify a GPU process; inspect nvidia-smi before starting.' >&2; exit 3; }
  state=$(ps -p "$pid" -o stat= 2>/dev/null || true)
  state=${state// /}
  case "$state" in T*|t*|'') ;; *) echo "GPU process $pid is active ($state). Pause it first, or deliberately set ALLOW_BUSY=1. No jobs were stopped." >&2;exit 3;;esac
 done <<< "$compute_pids"
fi
capacity=${MAX_NUM_SEQS:-4}
[[ $capacity =~ ^[0-9]+$ ]] && ((capacity>=1 && capacity<=128)) || { echo 'MAX_NUM_SEQS must be 1–128' >&2; exit 2; }
docker_command=(docker)
if ! docker info >/dev/null 2>&1; then docker_command=(sudo docker); fi
image='vllm/vllm-openai:v0.29.0@sha256:18372a7224938643461b846fb64c5c9d3d6e9727e82caf2dc3043e620c9d4d7a'
"${docker_command[@]}" run -d --name "spark-lab-$role" --gpus all --ipc=host \
 -p 127.0.0.1:8000:8000 -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
 -e VLLM_USE_V2_MODEL_RUNNER=1 "$image" "$model" --revision "$revision" \
 --served-model-name "$served" --host 0.0.0.0 --port 8000 \
 --dtype bfloat16 --kv-cache-dtype auto --max-model-len 8192 \
 --max-num-seqs "$capacity" --max-num-batched-tokens 8192 --gpu-memory-utilization 0.75 \
 --language-model-only --attention-backend TRITON_ATTN --reasoning-parser gemma4 \
 --default-chat-template-kwargs '{"enable_thinking":false}' \
 --no-enable-prefix-caching "${extra[@]}"
echo "Launched spark-lab-$role. Follow startup with: sudo docker logs -f spark-lab-$role"
echo 'Check the logs for FP8/compressed-tensors kernel selection before recording hardware results.'
