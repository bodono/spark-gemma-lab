#!/usr/bin/env bash
# Run on a Spark. Does not stop or alter existing workloads.
set -euo pipefail
role=${1:?Usage: serve-spark-user.sh diffusion|autoregressive}
canvas=${CANVAS_LENGTH:-256}
case "$canvas" in 64|128|256|512) ;; *) echo 'CANVAS_LENGTH must be 64, 128, 256 or 512' >&2; exit 2;; esac
steps=${MAX_DENOISING_STEPS:-48}
[[ $steps =~ ^[0-9]+$ ]] && ((steps>=1 && steps<=256)) || { echo "MAX_DENOISING_STEPS must be 1–256" >&2; exit 2; }
case "$role" in
 diffusion) model=RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic; revision=3b3dae4697494da5a290e9c0461954449e76c4f5; served=diffusiongemma-fp8; extra=(--diffusion-config "{\"canvas_length\":$canvas,\"max_denoising_steps\":$steps}" --override-generation-config '{"max_new_tokens":null}') ;;
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
attention=${ATTENTION_BACKEND:-TRITON_ATTN}
case "$attention" in TRITON_ATTN|FLASHINFER) ;; *) echo 'Supported experiment attention backends: TRITON_ATTN or FLASHINFER' >&2;exit 2;; esac
if [[ $role == diffusion && $attention != TRITON_ATTN ]]; then echo 'Diffusion needs dynamic per-request causal masks; keep TRITON_ATTN on GB10.' >&2;exit 2;fi
if [[ ${EXTEND_DIFFUSION_GRAPHS:-0} == 1 && $role == diffusion ]]; then
 extra+=(--compilation-config '{"cudagraph_capture_sizes":[1,2,4,256,512,768,1024],"max_cudagraph_capture_size":1024}')
fi
if [[ ${USE_TUNED_MOE:-0} == 1 ]]; then
 export VLLM_TUNED_CONFIG_FOLDER="$HOME/.local/share/spark-gemma-lab/moe-configs"
 [[ -f "$VLLM_TUNED_CONFIG_FOLDER/E=128,N=704,device_name=NVIDIA_GB10,dtype=fp8_w8a8.json" ]] || { echo 'Tuned MoE config missing' >&2;exit 2; }
fi
if [[ ${ENABLE_TORCH_PROFILER:-0} == 1 ]]; then
 trace_dir="$HOME/.local/share/spark-gemma-lab/traces/$role"
 mkdir -p "$trace_dir"
 extra+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$trace_dir\",\"ignore_frontend\":true,\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"torch_profiler_with_memory\":false,\"torch_profiler_with_flops\":false,\"torch_profiler_use_gzip\":true,\"torch_profiler_dump_cuda_time_total\":true,\"capture_torch_profiler\":false,\"detailed_trace_annotation\":false,\"delay_iterations\":0,\"max_iterations\":20,\"warmup_iterations\":0,\"wait_iterations\":0}")
fi
capacity=${MAX_NUM_SEQS:-4}
[[ $capacity =~ ^[0-9]+$ ]] && ((capacity>=1 && capacity<=128)) || { echo 'MAX_NUM_SEQS must be 1–128' >&2; exit 2; }
runtime="$HOME/.local/share/spark-gemma-lab/venv"
[[ -x "$runtime/bin/vllm" ]] || { echo 'Run install-spark-user.sh first' >&2; exit 2; }
if [[ $role == diffusion ]]; then
 patch_dir="$(dirname "$0")"
 preview_helper=$("$runtime/bin/python" -c 'from importlib.util import find_spec; from pathlib import Path; print(Path(find_spec("vllm").origin).with_name("spark_lab_diffusion_preview.py"))')
 if [[ -f "$preview_helper" ]]; then
  "$runtime/bin/python" "$patch_dir/patch_diffusion_preview.py" --revert --output-dir "$HOME/.local/share/spark-gemma-lab/preview-patch-evidence"
 fi
 "$runtime/bin/python" "$patch_dir/patch_diffusion_controls.py" --apply --output-dir "$HOME/.local/share/spark-gemma-lab/patch-evidence"
 "$runtime/bin/python" "$patch_dir/patch_diffusion_metrics.py" --apply --mode live
 "$runtime/bin/python" "$patch_dir/patch_diffusion_preview.py" --apply --output-dir "$HOME/.local/share/spark-gemma-lab/preview-patch-evidence"
extra+=(--per-request-spec-decode-metrics detailed)
fi
export VLLM_USE_V2_MODEL_RUNNER=1
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$runtime/bin:$CUDA_HOME/bin:$PATH"
command -v ninja >/dev/null || { echo 'Ninja is required for FlashInfer JIT builds' >&2; exit 2; }
devroot="$HOME/.local/share/spark-gemma-lab/python-dev"
[[ -f "$devroot/usr/include/python3.12/Python.h" ]] || { echo 'Run install-python-headers.sh first' >&2; exit 2; }
export CPATH="$devroot/usr/include/python3.12:$devroot/usr/include${CPATH:+:$CPATH}"
exec "$runtime/bin/vllm" serve "$model" --revision "$revision" \
 --served-model-name "$served" --host 127.0.0.1 --port 8000 \
 --dtype bfloat16 --kv-cache-dtype auto --max-model-len 8192 \
 --max-num-seqs "$capacity" --max-num-batched-tokens 8192 --gpu-memory-utilization 0.75 \
 --language-model-only --attention-backend "$attention" --reasoning-parser gemma4 \
 --default-chat-template-kwargs '{"enable_thinking":false}' \
 --no-enable-prefix-caching "${extra[@]}"
