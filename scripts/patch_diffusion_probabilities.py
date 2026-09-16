#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fix committed DiffusionGemma logprob ownership in pinned vLLM 0.29.0.

The baseline must already contain Spark Lab controls, metrics and previews.
Default is check only; apply/revert only while the owned server is stopped.
No torch/vLLM imports or GPU operations. --vllm-root permits CPU fixture checks.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import tempfile

VERSION = "0.29.0"
PATCH = "SPARK_LAB_DIFFUSION_PROBABILITIES_V1"
RELATIVE = "model_executor/models/diffusion_gemma.py"
BASE_SHA256 = "b379ce87b919491e4a0c4d92df19bcf9dc2f4c49558529dc7da694fa2a06452e"
BACKUP_SUFFIX = ".spark-lab-diffusion-probabilities-v1.bak"

BEFORE = '''        if max_num_logprobs >= 0 and is_committing.any() and self._pending_logprobs:
            parts_ids, parts_lp, parts_ranks = [], [], []
            cu_gen: list[int] = []
            flat_offset = 0
            for i in range(num_reqs):
                cu_gen.append(flat_offset)
                slot = int(slots_np[i])
                if is_decode_np[i] and slot in self._pending_logprobs:
'''

AFTER = '''        if max_num_logprobs >= 0 and is_committing.any() and self._pending_logprobs:
            # SPARK_LAB_DIFFUSION_PROBABILITIES_V1: retain scores for a request
            # that just converged while another request is committing. Only
            # the slots that were committing at entry emit tokens this step.
            # This host copy runs only when logprobs are explicitly requested.
            committing_slots = set(decode_slots[is_committing].tolist())
            parts_ids, parts_lp, parts_ranks = [], [], []
            cu_gen: list[int] = []
            flat_offset = 0
            for i in range(num_reqs):
                cu_gen.append(flat_offset)
                slot = int(slots_np[i])
                if slot in committing_slots and slot in self._pending_logprobs:
'''


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def render(original: bytes) -> bytes:
    if digest(original) != BASE_SHA256:
        raise RuntimeError(f"Refusing uninspected baseline for {RELATIVE}: {digest(original)}")
    source = original.decode()
    if source.count(BEFORE) != 1:
        raise RuntimeError("Expected exactly one committed-logprob source anchor")
    target = source.replace(BEFORE, AFTER, 1)
    ast.parse(target, filename=RELATIVE)
    return target.encode()


def atomic_write(path: Path, data: bytes) -> None:
    mode = path.stat().st_mode & 0o777
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def apply(root: Path, mode: str, output_dir: Path) -> dict:
    if mode not in ("check", "apply", "revert"):
        raise ValueError("Unknown patch mode")
    path = root / RELATIVE
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    current = path.read_bytes()
    if digest(current) == BASE_SHA256:
        original = current
    else:
        source = current.decode()
        if source.count(AFTER) != 1:
            raise RuntimeError(f"Refusing unknown source for {RELATIVE}: {digest(current)}")
        original = source.replace(AFTER, BEFORE, 1).encode()
    patched = render(original)
    if current not in (original, patched):
        raise RuntimeError(f"Refusing unrelated modifications to {RELATIVE}")
    if backup.exists() and backup.read_bytes() != original:
        raise RuntimeError(f"Backup does not match inspected baseline: {backup}")

    target = original if mode == "revert" else patched
    output_dir.mkdir(parents=True, exist_ok=True)
    patch = "".join(difflib.unified_diff(
        original.decode().splitlines(keepends=True),
        patched.decode().splitlines(keepends=True),
        "a/" + RELATIVE, "b/" + RELATIVE,
    ))
    (output_dir / "probabilities.patch").write_text(patch)
    state = "patched" if current == patched else "original"
    manifest = {
        "patch": PATCH,
        "vllm_version": VERSION,
        "mode": mode,
        "root": str(root),
        "relative_path": RELATIVE,
        "original_sha256": digest(original),
        "patched_sha256": digest(patched),
        "initial_state": state,
        "status": "already_selected" if current == target else "pending",
    }
    if mode != "check":
        if mode == "apply" and not backup.exists():
            with backup.open("xb") as stream:
                stream.write(original)
            backup.chmod(path.stat().st_mode & 0o777)
        if current != target:
            atomic_write(path, target)
            manifest["status"] = "applied" if mode == "apply" else "reverted"
    (output_dir / "probabilities-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--apply", action="store_true")
    action.add_argument("--revert", action="store_true")
    parser.add_argument("--vllm-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.vllm_root is None:
        version = importlib.metadata.version("vllm")
        if version != VERSION:
            raise SystemExit(f"Expected vLLM {VERSION}, found {version}")
        args.vllm_root = Path(importlib.util.find_spec("vllm").origin).parent
    mode = "apply" if args.apply else "revert" if args.revert else "check"
    print(json.dumps(apply(args.vllm_root.resolve(), mode, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
