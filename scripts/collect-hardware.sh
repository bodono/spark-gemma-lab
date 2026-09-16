#!/usr/bin/env bash
set -euo pipefail
output=${1:-results/hardware}
mkdir -p "$output"
for target in Spark_1 Spark_2; do
 ssh "$target" 'date --iso-8601=seconds; hostname; uname -a; nvidia-smi -q; free -h; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits); do ps -p "$pid" -o pid=,stat=,comm=; done' > "$output/$target.txt" 2>&1 || true
 case "$target" in Spark_1) role=diffusion;; Spark_2) role=autoregressive;; esac
 ssh "$target" "cat .local/share/spark-gemma-lab/logs/$role.log" > "$output/$target-startup.log" 2>&1 || true
done
echo "System metadata and native serving logs saved to $output. For Docker deployments, also capture the container logs and image digest."
