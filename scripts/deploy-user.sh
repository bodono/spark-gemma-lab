#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
role=${1:?Usage: scripts/deploy-user.sh diffusion|autoregressive prepare|serve}
mode=${2:?Choose prepare or serve}
case "$role" in diffusion) target=Spark_1;; autoregressive) target=Spark_2;; *) echo 'Unknown role' >&2;exit 2;;esac
case "$mode" in prepare|serve) ;; *) echo 'Choose prepare or serve' >&2;exit 2;;esac
canvas=${CANVAS_LENGTH:-256}
case "$canvas" in 8|16|32|64|128|256|512) ;; *) exit 2;; esac
capacity=${MAX_NUM_SEQS:-4};busy=${ALLOW_BUSY:-0}
graphs=${EXTEND_DIFFUSION_GRAPHS:-0};tuned=${USE_TUNED_MOE:-0};attention=${ATTENTION_BACKEND:-TRITON_ATTN};steps=${MAX_DENOISING_STEPS:-48};profiler=${ENABLE_TORCH_PROFILER:-0}
[[ $steps =~ ^[0-9]+$ ]] || exit 2
[[ $graphs =~ ^[01]$ && $tuned =~ ^[01]$ && $profiler =~ ^[01]$ ]] || { echo "Invalid tuning switches" >&2;exit 2; }
case "$attention" in TRITON_ATTN|FLASHINFER) ;; *) exit 2;;esac
[[ $capacity =~ ^[0-9]+$ && $busy =~ ^[01]$ ]] || { echo 'Invalid launch environment' >&2;exit 2; }
ssh "$target" 'mkdir -p .local/share/spark-gemma-lab'
scp scripts/install-spark-user.sh scripts/install-python-headers.sh scripts/serve-spark-user.sh scripts/canvas-runtime.py scripts/verify_runtime.py scripts/patch_diffusion_controls.py scripts/patch_diffusion_metrics.py scripts/patch_diffusion_preview.py scripts/patch_diffusion_probabilities.py "$target:.local/share/spark-gemma-lab/"
if [[ $mode == prepare ]]; then
 ssh "$target" 'bash .local/share/spark-gemma-lab/install-spark-user.sh'
 ssh "$target" 'chmod go-w .local/share/spark-gemma-lab .local/share/spark-gemma-lab/venv/bin/vllm'
else
 echo 'Serving in this terminal. Keep it open; Ctrl-C stops this model server.'
 ssh "$target" 'chmod go-w .local/share/spark-gemma-lab .local/share/spark-gemma-lab/venv/bin/vllm'
 ssh -t "$target" "CANVAS_LENGTH=$canvas MAX_NUM_SEQS=$capacity ALLOW_BUSY=$busy EXTEND_DIFFUSION_GRAPHS=$graphs USE_TUNED_MOE=$tuned ATTENTION_BACKEND=$attention MAX_DENOISING_STEPS=$steps ENABLE_TORCH_PROFILER=$profiler bash .local/share/spark-gemma-lab/serve-spark-user.sh $role"
fi
