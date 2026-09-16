#!/usr/bin/env python3
"""Render/apply/revert an exact-source guarded vLLM 0.29.0 request-control patch.

No imports of torch/vLLM, no GPU operations, no process management. Applying or
reverting affects only the next server process: restart separately after stopping
the running server. --render-from is a CPU-only fixture/render mode.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import stat

VERSION = "0.29.0"
BACKUP_SUFFIX = ".spark-lab-request-controls-v1.bak"
BASE_HASHES = {
    "model_executor/models/diffusion_gemma.py": "5e780e8a9fd964ca5f5275bbefafc2ce56a6e455b06fddabf00f9b3a8ea63801",
    "sampling_params.py": "2aba9ebd1c3921d601dcf94ac6a1f726e5172fe991e313cd5a69d3eaa02013fa",
}

VALIDATION = '''        # SPARK_LAB_REQUEST_CONTROLS_V1: validate before engine admission.
        controls = self.extra_args or {}
        max_key = "spark_lab_diffusion_max_steps"
        force_key = "spark_lab_diffusion_force_steps"
        if max_key in controls:
            maximum = controls[max_key]
            if type(maximum) is not int or not 1 <= maximum <= 48:
                raise VLLMValidationError(
                    "spark_lab_diffusion_max_steps must be an integer in [1, 48].",
                    parameter=max_key, value=maximum,
                )
        if force_key in controls:
            force = controls[force_key]
            if type(force) not in (int, bool) or force not in (0, 1):
                raise VLLMValidationError(
                    "spark_lab_diffusion_force_steps must be 0 or 1.",
                    parameter=force_key, value=force,
                )
            if force and max_key not in controls:
                raise VLLMValidationError(
                    "Forced diffusion steps require an explicit max-step value.",
                    parameter=max_key, value=None,
                )
'''


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def replace_once(source, old, new):
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one matching source anchor, got {count}: {old[:90]!r}")
    return source.replace(old, new, 1)


def transform(relative, source):
    if sha(source) != BASE_HASHES[relative]:
        raise RuntimeError(f"Refusing uninspected source hash for {relative}: {sha(source)}")
    if relative == "sampling_params.py":
        source = replace_once(source,
                              "    def _verify_args(self) -> None:\n        _verify_num_sequences(self.n, \"n\")\n",
                              "    def _verify_args(self) -> None:\n" + VALIDATION
                              + "        _verify_num_sequences(self.n, \"n\")\n")
    else:
        source = replace_once(source,
            "    tp_group_name: str,\n) -> torch.Tensor:\n    \"\"\"Compiled decode step:",
            "    tp_group_name: str,\n"
            "    # SPARK_LAB_REQUEST_CONTROLS_V1: None preserves native scalar path.\n"
            "    request_max_denoising_steps: torch.Tensor | None = None,\n"
            "    request_force_denoising_steps: torch.Tensor | None = None,\n"
            ") -> torch.Tensor:\n    \"\"\"Compiled decode step:")
        source = replace_once(source,
            "    steps_f = step_tensor[decode_slots].float()\n"
            "    remaining = (max_denoising_steps - steps_f).clamp(min=1.0)\n"
            "    temp = t_min + (t_max - t_min) * (remaining / max_denoising_steps)\n",
            "    steps_f = step_tensor[decode_slots].float()\n"
            "    if request_max_denoising_steps is None:\n"
            "        # Preserve the native arithmetic for default adaptive requests.\n"
            "        remaining = (max_denoising_steps - steps_f).clamp(min=1.0)\n"
            "        temp = t_min + (t_max - t_min) * (remaining / max_denoising_steps)\n"
            "    else:\n"
            "        # An explicit budget also sets this request's temperature horizon.\n"
            "        request_limits = request_max_denoising_steps[decode_slots].float()\n"
            "        remaining = (request_limits - steps_f).clamp(min=1.0)\n"
            "        temp = t_min + (t_max - t_min) * (remaining / request_limits)\n")
        source = replace_once(source,
            "    converged = (stable & confident_tensor[decode_slots] & (new_hist_len >= ST)) | (\n"
            "        step_after >= max_denoising_steps\n"
            "    )\n",
            "    if request_max_denoising_steps is None:\n"
            "        converged = (stable & confident_tensor[decode_slots] & (new_hist_len >= ST)) | (\n"
            "            step_after >= max_denoising_steps\n"
            "        )\n"
            "    else:\n"
            "        # Only explicit force=1 suppresses the stability/confidence exit.\n"
            "        early_converged = (stable & confident_tensor[decode_slots]\n"
            "                           & (new_hist_len >= ST))\n"
            "        early_converged = early_converged & ~request_force_denoising_steps[decode_slots]\n"
            "        converged = early_converged | (step_after >= request_limits)\n")
        source = replace_once(source,
            "        self._num_logits = UvaBackedTensor(max_num_reqs, dtype=torch.int32)\n",
            "        self._num_logits = UvaBackedTensor(max_num_reqs, dtype=torch.int32)\n"
            "\n"
            "        # SPARK_LAB_REQUEST_CONTROLS_V1: persistent slot tensors.\n"
            "        # CPU flags select the native path without inspecting GPU state.\n"
            "        self._request_max_denoising_steps = torch.full(\n"
            "            (max_num_reqs,), diffusion_states.max_denoising_steps,\n"
            "            dtype=torch.int32, device=device,\n"
            "        )\n"
            "        self._request_force_denoising_steps = torch.zeros(\n"
            "            max_num_reqs, dtype=torch.bool, device=device,\n"
            "        )\n"
            "        self._request_has_controls = [False] * max_num_reqs\n")
        source = replace_once(source,
            "    def add_request(self, req_idx: int, prompt_len: int, sampling_params: Any) -> None:\n"
            "        if use_penalty(sampling_params):\n",
            "    def add_request(self, req_idx: int, prompt_len: int, sampling_params: Any) -> None:\n"
            "        # Validated by SamplingParams._verify_args before engine admission.\n"
            "        # Every add (including resumed/preempted requests) overwrites the\n"
            "        # slot so forced settings cannot leak to a later request.\n"
            "        controls = sampling_params.extra_args or {}\n"
            "        maximum = controls.get(\n"
            "            \"spark_lab_diffusion_max_steps\",\n"
            "            self.diffusion_states.max_denoising_steps,\n"
            "        )\n"
            "        force = bool(controls.get(\"spark_lab_diffusion_force_steps\", 0))\n"
            "        self._request_max_denoising_steps[req_idx].fill_(maximum)\n"
            "        self._request_force_denoising_steps[req_idx].fill_(force)\n"
            "        self._request_has_controls[req_idx] = (\n"
            "            maximum != self.diffusion_states.max_denoising_steps or force\n"
            "        )\n"
            "        if use_penalty(sampling_params):\n")
        source = replace_once(source,
            "            tile_slots = decode_slots[tile]\n\n"
            "            scaled = _compiled_sample_step(\n",
            "            tile_slots = decode_slots[tile]\n"
            "            # Small CPU-only lookup, one flag per request in this tile.\n"
            "            use_request_controls = any(\n"
            "                self._request_has_controls[int(slot)]\n"
            "                for slot in decode_slots_np[start_req:end_req]\n"
            "            )\n\n"
            "            scaled = _compiled_sample_step(\n")
        source = replace_once(source,
            "                tp_group_name=self.tp_group_name,\n"
            "            )\n",
            "                tp_group_name=self.tp_group_name,\n"
            "                request_max_denoising_steps=(\n"
            "                    self._request_max_denoising_steps if use_request_controls else None\n"
            "                ),\n"
            "                request_force_denoising_steps=(\n"
            "                    self._request_force_denoising_steps if use_request_controls else None\n"
            "                ),\n"
            "            )\n")
    ast.parse(source)
    return source


def atomic_write(path, data, mode):
    temp = path.with_name(path.name + ".spark-lab-request-controls.tmp")
    if temp.exists():
        raise RuntimeError(f"Refusing to overwrite leftover temporary file: {temp}")
    try:
        temp.write_text(data)
        temp.chmod(mode)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--apply", action="store_true")
    action.add_argument("--revert", action="store_true")
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--render-from", type=Path,
                        help="Directory containing request_controls_base.py and request_controls_sampling_base.py; never applies")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.render_from and (args.apply or args.revert or args.package_root):
        parser.error("--render-from cannot be combined with mutation or package-root")
    if args.render_from:
        package_root = None
    else:
        distribution = importlib.metadata.distribution("vllm")
        if distribution.version != VERSION:
            raise RuntimeError(f"Expected vLLM {VERSION}, found {distribution.version}")
        package_root = args.package_root or Path(distribution.locate_file("vllm"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"patch": "SPARK_LAB_REQUEST_CONTROLS_V1", "vllm_version": VERSION,
                "operation": "revert" if args.revert else "apply" if args.apply else "render", "files": []}
    plans = []
    for relative in BASE_HASHES:
        path = package_root / relative if package_root else None
        backup = Path(str(path) + BACKUP_SUFFIX) if path else None
        if args.render_from:
            filename = "request_controls_sampling_base.py" if relative == "sampling_params.py" else "request_controls_base.py"
            original = (args.render_from / filename).read_text()
            current = original
        else:
            current = path.read_text()
            if sha(current) == BASE_HASHES[relative]:
                original = current
            elif backup.exists():
                original = backup.read_text()
            else:
                raise RuntimeError(f"Unexpected source and no verified backup: {path}")
        patched = transform(relative, original)
        if sha(current) not in (sha(original), sha(patched)):
            raise RuntimeError(f"Refusing to overwrite unrelated modifications: {path}")
        if backup and backup.exists() and backup.read_text() != original:
            raise RuntimeError(f"Backup does not match inspected original: {backup}")
        target = original if args.revert else patched
        manifest["files"].append({"relative_path": relative, "original_sha256": sha(original),
                                  "patched_sha256": sha(patched),
                                  "status": "already_selected" if current == target else "pending"})
        plans.append((path, backup, original, patched, current, target))
    patch = "".join("".join(difflib.unified_diff(original.splitlines(True), patched.splitlines(True),
                                                fromfile="a/" + relative, tofile="b/" + relative))
                    for relative, (_, _, original, patched, _, _) in zip(BASE_HASHES, plans))
    (args.output_dir / "request_controls.patch").write_text(patch)
    # All files and backups pass preflight before any installed source is changed.
    if args.apply or args.revert:
        for record, (path, backup, original, patched, current, target) in zip(manifest["files"], plans):
            mode = stat.S_IMODE(path.stat().st_mode)
            if args.apply and not backup.exists():
                with backup.open("x") as stream:
                    stream.write(original)
                backup.chmod(mode)
            if current != target:
                atomic_write(path, target, mode)
                record["status"] = "reverted" if args.revert else "applied"
    (args.output_dir / "request_controls_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
