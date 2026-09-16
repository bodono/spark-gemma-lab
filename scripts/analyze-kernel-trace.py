#!/usr/bin/env python3
"""Summarize vLLM/PyTorch Chrome traces without double-counting CPU as GPU time.

Name/shape attribution is heuristic and explicitly preserves unknown roles.
Use CPU launch correlations to improve attribution when graphs permit it.
Only complete CUDA kernel events enter the kernel-duration totals.
Explicit Triton autotuner work stays in those totals as a separate category.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import re


def intervals_union_us(intervals):
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    start, end = intervals[0]
    total = 0.0
    for a, b in intervals[1:]:
        if a > end:
            total += end - start
            start, end = a, b
        else:
            end = max(end, b)
    return total + end - start


def arg_value(event, *names):
    args = event.get("args", {})
    for key, value in args.items():
        if key.lower() in names:
            return value
    return None


def key(value):
    return None if value is None else str(value)


def cpu_parent_map(events):
    """Build thread-local enclosing CPU-op/annotation chains, plus launch IDs."""
    threads = defaultdict(list)
    external = {}
    correlations = {}
    for i, event in enumerate(events):
        category = event.get("cat", "").lower()
        if any(x in category for x in ("cpu_op", "user_annotation", "cuda_runtime", "cuda_driver")):
            threads[(event.get("pid"), event.get("tid"))].append(i)
            ext = key(arg_value(event, "external id", "external_id"))
            if ext is not None and ("cpu_op" in category or "user_annotation" in category):
                external.setdefault(ext, i)
            corr = key(arg_value(event, "correlation", "correlation id"))
            if corr is not None and ("cuda_runtime" in category or "cuda_driver" in category):
                correlations[corr] = i
    parents = {}
    for indices in threads.values():
        indices.sort(key=lambda i: (events[i]["ts"], -events[i]["dur"]))
        stack = []
        for i in indices:
            current = events[i]
            start, end = current["ts"], current["ts"] + current["dur"]
            while stack and events[stack[-1]]["ts"] + events[stack[-1]]["dur"] < end:
                stack.pop()
            if stack and events[stack[-1]]["ts"] <= start:
                parents[i] = stack[-1]
            stack.append(i)
    return parents, external, correlations


def shape_semantics(chain, vocab, hidden):
    for event in chain:
        dims = arg_value(event, "input dims", "input shapes", "input_dims")
        if isinstance(dims, str):
            try:
                dims = json.loads(dims)
            except (ValueError, TypeError):
                continue
        if not isinstance(dims, list):
            continue
        tensors = [d for d in dims if isinstance(d, list) and len(d) >= 2]
        if len(tensors) < 2:
            continue
        left, right = tensors[0], tensors[1]
        if left[-1] == vocab and right[-2:] == [vocab, hidden]:
            return "self_conditioning_soft_embedding", "CPU matmul input shapes"
        if left[-1] == hidden and right[-2:] == [hidden, vocab]:
            return "output_vocabulary_projection", "CPU matmul input shapes"
    return None


def classify(event, chain, vocab, hidden):
    name = event.get("name", "").lower()
    scopes = " | ".join(e.get("name", "").lower() for e in chain)
    both = name + " | " + scopes
    # Benchmark launches are real GPU work, but must not inflate a model
    # component's apparent steady-state cost. Keep them visible in the total.
    if any("cachingautotuner" in e.get("name", "").lower() for e in chain):
        return "triton_autotuning", "explicit CachingAutotuner CPU ancestry"
    if "spark_lab.self_conditioning" in scopes:
        return "self_conditioning_mlp", "explicit user annotation"
    if re.search(r"fused_moe|moe_align|moe_sum|moe_router|fused_experts|tritonexperts", both):
        return "moe", "kernel/operator name"
    if re.search(r"attention|flash_attn|flash_fwd|fmha|paged_attn|unified_attn", both):
        return "attention", "kernel/operator name"
    shaped = shape_semantics(chain, vocab, hidden)
    if shaped:
        return shaped
    if "spark_lab.diffusion_sampler" in scopes:
        return "diffusion_sampler_including_soft_embedding", "explicit user annotation"
    # Generated Triton helper names may contain a neighboring cutlass_scaled_mm
    # even when their own work is normalization or quantization, not a GEMM.
    # Also avoid treating the model name 'Gemma' as the operation name 'GEMM'.
    if re.search(r"gemm(?!a)|gemv|matmul|cutlass|cublas|aten::mm|aten::bmm|aten::addmm", both):
        return "other_gemm_related_or_fused", "GEMM-related name/context; exact operation and model role unresolved"
    if re.search(r"softmax|gumbel|entropy|sort|topk|top_k|topp|top_p|sampl|multinomial|argmax|random|randint|rand_like", both):
        return "sampling_or_probability_ops", "kernel/operator name; may have other roles"
    if re.search(r"quant|fp8_scale|scaled_fp8", both):
        return "quantization", "kernel/operator name"
    if re.search(r"rms_norm|rmsnorm|layer_norm|layernorm|gelu|silu|rotary|rope", both):
        return "normalization_activation_position", "kernel/operator name"
    return "other_unattributed", "no reliable attribution"


def analyze(path, vocab, hidden, top):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as source:
        trace = json.load(source)
    raw = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
    events = [e for e in raw if e.get("ph") == "X" and isinstance(e.get("ts"), (int, float))
              and isinstance(e.get("dur"), (int, float)) and e["dur"] >= 0]
    parents, external, correlations = cpu_parent_map(events)
    categories = defaultdict(lambda: {"count": 0, "duration_us": 0.0})
    kernels = defaultdict(lambda: {"count": 0, "duration_us": 0.0, "min_us": float("inf"), "max_us": 0.0})
    intervals = []
    correlated = 0
    transfers = []
    for e in events:
        cat = e.get("cat", "").lower()
        if "gpu_memcpy" in cat or "gpu_memset" in cat:
            transfers.append((e["ts"], e["ts"] + e["dur"]))
        if cat not in ("kernel", "gpu_kernel", "cuda_kernel"):
            continue
        start = correlations.get(key(arg_value(e, "correlation", "correlation id")))
        if start is None:
            start = external.get(key(arg_value(e, "external id", "external_id")))
        chain = []
        seen = set()
        while start is not None and start not in seen:
            seen.add(start)
            chain.append(events[start])
            start = parents.get(start)
        if chain:
            correlated += 1
        category, evidence = classify(e, chain, vocab, hidden)
        name = e.get("name", "<unnamed>")
        record = kernels[(category, name, evidence)]
        record["count"] += 1
        record["duration_us"] += e["dur"]
        record["min_us"] = min(record["min_us"], e["dur"])
        record["max_us"] = max(record["max_us"], e["dur"])
        categories[category]["count"] += 1
        categories[category]["duration_us"] += e["dur"]
        intervals.append((e["ts"], e["ts"] + e["dur"]))
    total_us = sum(r["duration_us"] for r in categories.values())
    rows = []
    for (category, name, evidence), data in kernels.items():
        rows.append({"category": category, "name": name, "evidence": evidence,
                     "count": data["count"], "total_ms": data["duration_us"] / 1000,
                     "mean_us": data["duration_us"] / data["count"],
                     "min_us": data["min_us"], "max_us": data["max_us"],
                     "percent_kernel_time": 100 * data["duration_us"] / total_us if total_us else 0})
    rows.sort(key=lambda row: -row["total_ms"])
    category_rows = [{"category": name, "count": d["count"], "total_ms": d["duration_us"] / 1000,
                      "percent_kernel_time": 100 * d["duration_us"] / total_us if total_us else 0}
                     for name, d in categories.items()]
    category_rows.sort(key=lambda row: -row["total_ms"])
    span = max(b for _, b in intervals) - min(a for a, _ in intervals) if intervals else 0
    union = intervals_union_us(intervals)
    autotuning = categories.get("triton_autotuning", {"count": 0, "duration_us": 0.0})
    summary = {
        "trace": str(path), "vllm_version": trace.get("vllm_version") if isinstance(trace, dict) else None,
        "kernel_events": len(intervals), "kernels_with_cpu_launch_context": correlated,
        "summed_kernel_ms": total_us / 1000, "kernel_busy_union_ms": union / 1000,
        "triton_autotuning_kernel_events": autotuning["count"],
        "triton_autotuning_kernel_ms": autotuning["duration_us"] / 1000,
        "non_autotuning_summed_kernel_ms": (total_us - autotuning["duration_us"]) / 1000,
        "first_to_last_kernel_span_ms": span / 1000,
        "kernel_busy_fraction_in_span": union / span if span else None,
        "memory_transfer_union_ms": intervals_union_us(transfers) / 1000,
        "categories": category_rows, "top_kernels": rows[:top],
        "event_categories": dict(Counter(e.get("cat", "") for e in events)),
        "limitations": [
            "Kernel-duration sums can exceed elapsed time with overlapping streams; busy union is reported separately.",
            "CPU op durations are not included in GPU totals. Host/launch gaps are not attributed to a model component.",
            "Names and CPU input shapes provide heuristic attribution. GEMM-related/fused names can include normalization, quantization or other helpers; their exact model roles stay unresolved.",
            "Kernels with explicit CachingAutotuner CPU ancestry have a separate category and remain in the kernel-time denominator. This detects observed autotuning, not every possible compilation or warmup cost.",
            "CUDA graph replay may obscure CPU parentage and tensor shapes; this does not make an unclassified kernel sampler work.",
            "Profiler adds overhead: use separate unprofiled requests for throughput comparisons.",
            "vLLM context/generation labels depend on emitted tokens, so they do not isolate diffusion prompt-only prefill.",
        ],
    }
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--vocab-size", type=int, default=262144)
    parser.add_argument("--hidden-size", type=int, default=2816)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, path in enumerate(args.traces):
        summary, rows = analyze(path, args.vocab_size, args.hidden_size, args.top)
        summaries.append(summary)
        with (args.output / f"kernels-{index}.csv").open("w", newline="") as target:
            fields = ["category", "name", "evidence", "count", "total_ms", "mean_us", "min_us", "max_us", "percent_kernel_time"]
            writer = csv.DictWriter(target, fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"{path.name}: {summary['kernel_events']} kernels, summed {summary['summed_kernel_ms']:.3f} ms, busy union {summary['kernel_busy_union_ms']:.3f} ms")
        for row in summary["categories"]:
            print(f"  {row['category']}: {row['percent_kernel_time']:.2f}% ({row['total_ms']:.3f} ms)")
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
