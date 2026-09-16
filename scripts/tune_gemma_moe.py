#!/usr/bin/env python3
"""Bounded vLLM 0.29.0 Gemma4 FP8 MoE tile search for DGX Spark / GB10.

This is a synthetic expert-layer microbenchmark, NOT a model throughput test.
It measures the same Triton GEMMs, activation, quantization and reduction used by
TritonExperts, through vLLM's functional fused_experts_impl wrapper. Routing is
precomputed and weights are synthetic. Repeated CUDA graph replay measures hot
kernel time; actual model routing and weight/cache traffic still need an A/B run.

Do not run concurrently with model serving or another GPU benchmark. The script
does not stop processes, set clocks, install packages or activate its output.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

E, N, K, TOPK = 128, 704, 2816, 8


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--allow-gpu-run", action="store_true",
                   help="Required acknowledgement that this GPU is reserved for tuning")
    p.add_argument("--tokens", default="1,2,4,256,512,1024")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--replays", type=int, default=40)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--confirm-rounds", type=int, default=5)
    p.add_argument("--min-improvement", type=float, default=0.05)
    p.add_argument("--max-token-grid", type=int, default=8192)
    p.add_argument("--seed", type=int, default=20260916)
    args = p.parse_args()
    args.tokens = sorted(set(int(s) for s in args.tokens.split(",")))
    if not args.allow_gpu_run:
        p.error("No GPU operations performed. Reserve an idle GPU, then pass --allow-gpu-run.")
    if min(args.tokens) < 1 or max(args.tokens) > args.max_token_grid:
        p.error("tokens must be positive and no greater than max-token-grid")
    if args.replays < 5 or min(args.rounds, args.confirm_rounds) < 3:
        p.error("Require at least 5 replays and 3 timing rounds")
    return args


def candidates(default, m):
    # Ten alternatives at most: do not exhaustively cross all tile dimensions.
    if m <= 4:
        tuples = [
            (16, 32, 128, 4, 3), (16, 64, 64, 4, 3),
            (16, 64, 128, 4, 2), (16, 64, 128, 4, 3),
            (16, 64, 128, 4, 5), (16, 64, 256, 4, 2),
            (16, 64, 256, 4, 3), (16, 128, 128, 4, 3),
            (16, 128, 256, 4, 2), (16, 32, 256, 4, 3),
        ]
    else:
        tuples = [
            (16, 64, 128, 4, 3), (32, 64, 128, 4, 3),
            (32, 64, 256, 4, 2), (32, 128, 128, 4, 3),
            (64, 64, 128, 4, 3), (64, 128, 64, 4, 3),
            (64, 128, 128, 4, 3), (64, 128, 256, 8, 2),
            (128, 64, 128, 8, 3), (128, 128, 128, 8, 3),
        ]
    result = [dict(default)]
    for bm, bn, bk, nw, ns in tuples:
        config = {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk,
                  "GROUP_SIZE_M": 1, "SPLIT_K": 1, "num_warps": nw, "num_stages": ns}
        if config not in result:
            result.append(config)
    return result


def main():
    args = parse_args()
    import torch
    import torch.nn.functional as F
    import triton
    import vllm.model_executor.layers.fused_moe as moe_package
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts_impl, get_config_file_name, get_default_config,
    )
    from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input

    version = importlib.metadata.version("vllm")
    if version != "0.29.0":
        raise RuntimeError(f"Inspected and written for vLLM 0.29.0, found {version}")
    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError("This candidate search is scoped to GB10 / SM121")
    if "GB10" not in torch.cuda.get_device_name():
        raise RuntimeError("Expected NVIDIA GB10")
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)

    report = {
        "created_unix": time.time(), "vllm": version, "torch": torch.__version__,
        "triton": triton.__version__, "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "shape": {"experts": E, "expert_intermediate": N, "hidden": K,
                  "topk": TOPK, "w1": [E, 2 * N, K], "w2": [E, K, N]},
        "quantization": "FP8 E4M3 per-channel weights; dynamic per-token activations",
        "activation": "gelu_tanh", "synthetic_weights": True,
        "metric": "CUDA events around CUDA graph replay, ms per complete expert layer",
        "limitations": ["Hot repeated layer, not full-model throughput",
                        "Precomputed synthetic top-k routing, not router timing",
                        "No attention, dense shared MLP, communication or scheduler",
                        "Require full-model A/B before enabling a candidate"],
        "options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "results": [],
    }

    def save():
        tmp = args.output / "report.tmp.json"
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(args.output / "report.json")

    def make_weights(rows, cols):
        # Generate expert by expert to avoid a several-GB FP32 temporary.
        w = torch.empty((E, rows, cols), device="cuda", dtype=torch.float8_e4m3fn)
        scale = torch.empty((E, rows, 1), device="cuda", dtype=torch.float32)
        for e in range(E):
            source = torch.randn((rows, cols), device="cuda", dtype=torch.float32) / math.sqrt(cols)
            s = source.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448.0
            w[e].copy_((source / s).clamp(-448, 448).to(torch.float8_e4m3fn))
            scale[e].copy_(s)
        return w, scale

    w1, s1 = make_weights(2 * N, K)
    w2, s2 = make_weights(K, N)
    torch.cuda.synchronize()

    def make_case(m, seed):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn((m, K), device="cuda", dtype=torch.bfloat16, generator=gen)
        scores = torch.randn((m, E), device="cuda", dtype=torch.float32, generator=gen)
        values, ids = scores.topk(TOPK, dim=-1)
        weights = values.softmax(dim=-1).contiguous()
        return x, weights, ids.to(torch.int32).contiguous()

    def run(config, case):
        # vLLM's override_config context manager in 0.29 lacks a finally clause.
        # Explicit restoration keeps a rejected compile from leaking a config.
        previous = moe_package.get_config()
        moe_package._config = config
        try:
            x, weights, ids = case
            return fused_experts_impl(
                x, w1, w2, weights, ids, activation="gelu_tanh",
                use_fp8_w8a8=True, per_channel_quant=True,
                global_num_experts=E, w1_scale=s1, w2_scale=s2,
            )
        finally:
            moe_package._config = previous

    def errors(actual, expected):
        a, b = actual.float(), expected.float()
        if not torch.isfinite(a).all().item():
            return {"finite": False, "relative_rms": None, "peak_relative_error": None}
        diff = a - b
        return {"finite": True,
                "relative_rms": (diff.norm() / b.norm().clamp_min(1e-12)).item(),
                "peak_relative_error": (diff.abs().max() / b.abs().max().clamp_min(1e-12)).item()}

    def independent_reference(case):
        # Small (<=4-token) FP32 torch-matmul reference using the exact vLLM
        # FP8 input quantizer, BF16 GEMM outputs and FP8 intermediate quantizer.
        # vLLM's gated activation computes float before casting back to BF16.
        x, weights, ids = case
        qx, ax = moe_kernel_quantize_input(x, None, torch.float8_e4m3fn, True)
        output = torch.zeros_like(x, dtype=torch.float32)
        for token in range(x.shape[0]):
            for route in range(TOPK):
                e = ids[token, route].item()
                h = ((qx[token].float() @ w1[e].float().T) * ax[token].float()
                     * s1[e, :, 0]).to(torch.bfloat16).float()
                act = (F.gelu(h[:N], approximate="tanh") * h[N:]).to(torch.bfloat16)[None, :]
                qa, aa = moe_kernel_quantize_input(act, None, torch.float8_e4m3fn, True)
                y = ((qa[0].float() @ w2[e].float().T) * aa.flatten()[0]
                     * s2[e, :, 0] * weights[token, route]).to(torch.bfloat16)
                output[token] += y.float()
        return output.to(torch.bfloat16)

    def graph_for(config, case):
        # Compile and initialize libraries outside capture, on a side stream.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                output = run(config, case)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = run(config, case)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        return graph, output

    def timing(graph, rounds):
        samples = []
        for _ in range(rounds):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(args.replays):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end) / args.replays)
        return samples

    with torch.inference_mode():
        for m in args.tokens:
            default = get_default_config(m, E, N, K, TOPK, "fp8_w8a8", None)
            cases = [make_case(m, args.seed + m + i * 101) for i in range(3)]
            baseline = [run(default, case).clone() for case in cases]
            tiny_case = tuple(t[:min(m, 4)].contiguous() for t in cases[0])
            reference_error = errors(run(default, tiny_case), independent_reference(tiny_case))
            if (not reference_error["finite"] or reference_error["relative_rms"] > 0.03
                    or reference_error["peak_relative_error"] > 0.10):
                report["fatal_reference_error"] = {"m": m, **reference_error}
                save()
                raise RuntimeError(f"Default kernel failed independent reference: {reference_error}")

            condition = {"tokens": m, "default_config": default,
                         "independent_reference_error": reference_error, "candidates": []}
            report["results"].append(condition)
            configs = candidates(default, m)
            random.Random(args.seed + m).shuffle(configs)
            for config in configs:
                result = {"config": config, "is_default": config == default}
                condition["candidates"].append(result)
                try:
                    checks = [errors(run(config, case), ref) for case, ref in zip(cases, baseline)]
                    result["correctness"] = checks
                    if any(not c["finite"] or c["relative_rms"] > 0.01
                           or c["peak_relative_error"] > 0.05 for c in checks):
                        result["status"] = "rejected_correctness"
                    else:
                        graph, out = graph_for(config, cases[0])
                        samples = timing(graph, args.rounds)
                        result.update(status="ok", samples_ms=samples,
                                      median_ms=statistics.median(samples))
                        del graph, out
                        gc.collect()
                except Exception as exc:
                    result.update(status="error", error=f"{type(exc).__name__}: {exc}")
                    # A CUDA device fault cannot safely continue; compile/resource
                    # failures can. synchronize surfaces any sticky device error.
                    torch.cuda.synchronize()
                print(json.dumps({"tokens": m, **result}), flush=True)
                save()

            good = [c for c in condition["candidates"] if c["status"] == "ok"]
            base_result = next((c for c in good if c["is_default"]), None)
            if not base_result:
                raise RuntimeError(f"Default configuration could not be timed at M={m}")
            best = min(good, key=lambda c: c["median_ms"])
            # Alternate the default and candidate so warm-up/clock drift cannot
            # turn one favorable sweep sample into an accepted configuration.
            graphs = [graph_for(cfg, cases[0]) for cfg in (default, best["config"])]
            confirm = [[], []]
            for round_id in range(args.confirm_rounds):
                for idx in ([0, 1] if round_id % 2 == 0 else [1, 0]):
                    confirm[idx].extend(timing(graphs[idx][0], 1))
            default_ms, candidate_ms = [statistics.median(s) for s in confirm]
            gain = 1.0 - candidate_ms / default_ms
            accepted = best["config"] != default and gain >= args.min_improvement
            condition.update(
                confirmation={"default_samples_ms": confirm[0], "candidate_samples_ms": confirm[1],
                              "default_median_ms": default_ms, "candidate_median_ms": candidate_ms,
                              "candidate_config": best["config"], "fraction_faster": gain},
                selected_config=best["config"] if accepted else default,
                accepted_candidate=accepted,
            )
            print(json.dumps({"tokens": m, "confirmation": condition["confirmation"],
                              "accepted": accepted}), flush=True)
            del graphs, cases, baseline
            gc.collect()
            save()

    # vLLM selects the NEAREST token-count entry. Fill every count in the serving
    # range with its default, replacing only measured counts. This deliberately
    # avoids silently applying an untested tile to neighboring token counts.
    export = {str(m): get_default_config(m, E, N, K, TOPK, "fp8_w8a8", None)
              for m in range(1, args.max_token_grid + 1)}
    for result in report["results"]:
        export[str(result["tokens"])] = result["selected_config"]
    export["triton_version"] = triton.__version__
    config_path = args.output / "configs" / get_config_file_name(E, N, "fp8_w8a8")
    config_path.parent.mkdir(exist_ok=True)
    config_path.write_text(json.dumps(export, separators=(",", ":")) + "\n")
    report["candidate_config_file"] = str(config_path)
    report["completed_unix"] = time.time()
    save()
    print(f"Candidate file (not activated): {config_path}", flush=True)
    print("Before activation, compare whole-model latency/output under identical conditions.", flush=True)


if __name__ == "__main__":
    main()
