#!/usr/bin/env bash
set -euo pipefail
pids=()
cleanup(){ for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 127.0.0.1:18001:127.0.0.1:8000 Spark_1 &
pids+=("$!")
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 127.0.0.1:18002:127.0.0.1:8000 Spark_2 &
pids+=("$!")
echo 'SSH tunnels: Spark_1 → localhost:18001, Spark_2 → localhost:18002. Keep this terminal open.'
wait
