#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
role=${1:?Usage: scripts/deploy.sh diffusion|autoregressive}
case "$role" in diffusion) target=Spark_1;; autoregressive) target=Spark_2;; *) echo 'Unknown role' >&2;exit 2;;esac
capacity=${MAX_NUM_SEQS:-4};busy=${ALLOW_BUSY:-0}
[[ $capacity =~ ^[0-9]+$ && $busy =~ ^[01]$ ]] || { echo 'Invalid launch environment' >&2;exit 2; }
ssh "$target" 'mkdir -p .local/share/spark-gemma-lab'
scp scripts/serve-spark.sh "$target:.local/share/spark-gemma-lab/serve-spark.sh"
# Interactive SSH allows sudo to prompt on the Spark; no password is stored.
ssh -t "$target" "MAX_NUM_SEQS=$capacity ALLOW_BUSY=$busy bash .local/share/spark-gemma-lab/serve-spark.sh $role"
