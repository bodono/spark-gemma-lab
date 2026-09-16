#!/usr/bin/env python3
"""Audited vLLM 0.29.0 diffusion step telemetry. CPU/file operations only.

Default is a dry-run. Stop vLLM before --apply/--restore, then restart it.
No imports of vLLM or torch; no network access, model loading, or GPU calls.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import stat
import tempfile

VERSION = "0.29.0"
PATCH_ID = "spark-diffusion-metrics-v1"
BACKUP_DIR = ".spark-diffusion-metrics-v1-backup"
ORIGINAL_HASHES = {
    "config/vllm.py": "788430b9211a5b0ccda0be565ad2a0f27e9069370f35d2b1f6a333a6ca135186",
    "v1/metrics/stats.py": "e8c3cbcfb880e7a2a496a1e499d4e4efccf70ba1eef56f4a1a6ee44cc0bf49bc",
    "v1/core/sched/scheduler.py": "abca7134821e2fb5cc8572df5c4a0b570ecf702646254327bc786fc452074983",
    "entrypoints/generate/base/protocol.py": "81b1b5ca411652ddc317d974fa61cf1e3d43b37a4b6cf9dbcf8995caf1c9de5e",
    "v1/engine/output_processor.py": "80a01067f4b3b351239506a3a1754dafdfc83cd162352d95545ba8ebed44c447",
    "outputs.py": "346d1f9204a441867efc3af7c5a99d8a70ef0f69fe607dfa8a4a2557a82ca425",
    "entrypoints/openai/completion/serving.py": "75d05ddddf303734a078a3e42fcb173fef58ea56da09bc0b5c7525cb9bbbf527",
    "entrypoints/openai/chat_completion/serving.py": "ea1f76074a9587c8054f54d30a6ba748b6a5fc4d90dd82c801504505c1da1f92",
}
FINAL_FILES = tuple(list(ORIGINAL_HASHES)[:4]) + ("outputs.py",)
LIVE_FILES = tuple(ORIGINAL_HASHES)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"{label}: expected one exact source anchor, got {text.count(old)}")
    return text.replace(old, new, 1)


def transform(relative: str, source: bytes, mode: str) -> bytes:
    if digest(source) != ORIGINAL_HASHES[relative]:
        raise ValueError(f"{relative}: unsupported original source hash")
    text = source.decode("utf-8")

    def edit(old: str, new: str) -> None:
        nonlocal text
        text = replace_once(text, old, new, relative)

    if relative == "config/vllm.py":
        edit(
            '            self.observability_config.per_request_spec_decode_metrics != "none"\n'
            '            and self.speculative_config is None\n',
            '            self.observability_config.per_request_spec_decode_metrics != "none"\n'
            '            and self.speculative_config is None\n'
            '            and (self.model_config is None or not self.model_config.is_diffusion)\n',
        )
        edit(
            '                "to be enabled (via --speculative-config)."\n',
            '                "or a diffusion model."\n',
        )
    elif relative == "v1/metrics/stats.py":
        edit(
            '    per_step_drafted: list[int] = field(default_factory=list)\n',
            '    per_step_drafted: list[int] = field(default_factory=list)\n'
            '    # Spark lab telemetry: nonzero only on detached diffusion snapshots.\n'
            '    diffusion_trace_version: int = 0\n'
            '    diffusion_num_preemptions: int = 0\n',
        )
        edit(
            '    def to_dict(self) -> dict[str, Any]:\n'
            '        """Payload matching ``SpeculativeDecodingMetrics`` for the response.\n',
            '    def diffusion_snapshot(self, num_preemptions: int):\n'
            '        """Detach mutable lists before transport while generation continues."""\n'
            '        return RequestSpecDecodeMetrics(\n'
            '            num_spec_tokens=self.num_spec_tokens,\n'
            '            histogram=list(self.histogram),\n'
            '            num_draft_tokens=self.num_draft_tokens,\n'
            '            per_step_accepted=list(self.per_step_accepted),\n'
            '            per_step_drafted=list(self.per_step_drafted),\n'
            '            diffusion_trace_version=1,\n'
            '            diffusion_num_preemptions=num_preemptions,\n'
            '        )\n\n'
            '    def to_dict(self) -> dict[str, Any]:\n'
            '        """Payload matching ``SpeculativeDecodingMetrics`` for the response.\n',
        )
        edit(
            '        if self.per_step_accepted:\n'
            '            result["per_step_accepted"] = self.per_step_accepted\n',
            '        if self.diffusion_trace_version:\n'
            '            result["diffusion_trace_version"] = self.diffusion_trace_version\n'
            '            result["diffusion_num_preemptions"] = self.diffusion_num_preemptions\n'
            '        if self.per_step_accepted:\n'
            '            result["per_step_accepted"] = self.per_step_accepted\n',
        )
    elif relative == "v1/core/sched/scheduler.py":
        old = (
            '                        spec_decode_metrics=(\n'
            '                            request.spec_decode_metrics\n'
            '                            if finish_reason is not None\n'
            '                            else None\n'
            '                        ),\n'
        )
        emit_guard = "True" if mode == "live" else "finish_reason is not None"
        edit(old,
            '                        spec_decode_metrics=(\n'
            '                            request.spec_decode_metrics.diffusion_snapshot(\n'
            '                                request.num_preemptions\n'
            '                            )\n'
            '                            if (\n'
            '                                self.vllm_config.model_config.is_diffusion\n'
            '                                and request.spec_decode_metrics is not None\n'
            f'                                and ({emit_guard})\n'
            '                            )\n'
            '                            else request.spec_decode_metrics\n'
            '                            if finish_reason is not None\n'
            '                            else None\n'
            '                        ),\n'
        )
    elif relative == "entrypoints/generate/base/protocol.py":
        edit(
            '    per_step_drafted: list[int] | None = None\n',
            '    per_step_drafted: list[int] | None = None\n'
            '    diffusion_trace_version: int | None = None\n'
            '    diffusion_num_preemptions: int | None = None\n',
        )
    elif relative == "v1/engine/output_processor.py":
        edit(
            '            spec_decode_metrics=self.spec_decode_metrics if finished else None,\n',
            '            spec_decode_metrics=(\n'
            '                self.spec_decode_metrics\n'
            '                if finished or (\n'
            '                    self.spec_decode_metrics is not None\n'
            '                    and self.spec_decode_metrics.diffusion_trace_version == 1\n'
            '                )\n'
            '                else None\n'
            '            ),\n',
        )
    elif relative == "outputs.py":
        edit(
            '                        completion.stop_reason = next_completion.stop_reason\n',
            '                        completion.stop_reason = next_completion.stop_reason\n'
            '                        # Keep the newest cumulative trace when chunks coalesce.\n'
            '                        if next_completion.spec_decode_metrics is not None:\n'
            '                            completion.spec_decode_metrics = (\n'
            '                                next_completion.spec_decode_metrics\n'
            '                            )\n',
        )
    elif relative == "entrypoints/openai/completion/serving.py":
        edit(
            '                    response_json = chunk.model_dump_json(exclude_unset=True)\n',
            '                    if (self.model_config.is_diffusion and num_prompts == 1\n'
            '                            and (request.n or 1) == 1):\n'
            '                        diffusion_stats = build_spec_decoding_metrics(res)\n'
            '                        if diffusion_stats is not None:\n'
            '                            chunk.metrics = PerRequestMetrics(\n'
            '                                speculative_decoding=diffusion_stats\n'
            '                            )\n'
            '                    response_json = chunk.model_dump_json(exclude_unset=True)\n',
        )
    elif relative == "entrypoints/openai/chat_completion/serving.py":
        edit(
            '                    data = chunk.model_dump_json(exclude_unset=True)\n'
            '                    yield f"data: {data}\\n\\n"\n\n'
            '            # once the final token is handled, if stream_options.include_usage\n',
            '                    if self.model_config.is_diffusion and (request.n or 1) == 1:\n'
            '                        diffusion_stats = build_spec_decoding_metrics(res)\n'
            '                        if diffusion_stats is not None:\n'
            '                            chunk.metrics = PerRequestMetrics(\n'
            '                                speculative_decoding=diffusion_stats\n'
            '                            )\n'
            '                    data = chunk.model_dump_json(exclude_unset=True)\n'
            '                    yield f"data: {data}\\n\\n"\n\n'
            '            # once the final token is handled, if stream_options.include_usage\n',
        )
    else:
        raise ValueError(f"Unsupported file: {relative}")
    ast.parse(text, filename=relative)
    return text.encode("utf-8")


def atomic_write(path: Path, data: bytes, permissions: int) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".spark-metrics-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, permissions)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def patch(root: Path, version: str, mode: str = "live", apply: bool = False,
          restore: bool = False) -> dict:
    if version != VERSION:
        raise ValueError(f"Requires vLLM {VERSION}, found {version}")
    if mode not in ("live", "final"):
        raise ValueError("mode must be live or final")
    root = root.resolve()
    backup = root / BACKUP_DIR
    manifest_path = backup / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest:
        if manifest.get("patch_id") != PATCH_ID or manifest.get("version") != VERSION:
            raise ValueError("Unknown backup manifest; refusing to modify files")
        if not restore and manifest.get("mode") != mode:
            raise ValueError("Restore before switching between live and final modes")
        mode = manifest["mode"]
    elif restore:
        return {"status": "not-patched", "patch_id": PATCH_ID}

    selected = LIVE_FILES if mode == "live" else FINAL_FILES
    planned = {}
    for relative in selected:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Not a regular source file: {path}")
        current = path.read_bytes()
        saved = backup / relative
        original = saved.read_bytes() if saved.is_file() else current
        if digest(original) != ORIGINAL_HASHES[relative]:
            raise ValueError(f"{relative}: original hash mismatch; refusing partial patch")
        modified = transform(relative, original, mode)
        if current not in (original, modified):
            raise ValueError(f"{relative}: modified by another edit; refusing to overwrite")
        planned[relative] = (current, original, modified, stat.S_IMODE(path.stat().st_mode))

    changed = [relative for relative, (current, original, modified, _) in planned.items()
               if current != (original if restore else modified)]
    result = {"patch_id": PATCH_ID, "version": version, "mode": mode,
              "status": "would-restore" if restore else "would-apply", "files": changed}
    if not changed:
        result["status"] = "restored" if restore else "already-applied"
    if not apply:
        return result

    if not manifest and changed:
        backup.mkdir(mode=0o700, exist_ok=True)
        for relative, (_, original, modified, _) in planned.items():
            target = backup / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != original:
                raise ValueError(f"Conflicting backup: {relative}")
            target.write_bytes(original)
        manifest = {"patch_id": PATCH_ID, "version": VERSION, "mode": mode,
                    "files": {relative: {"original": digest(original), "patched": digest(modified)}
                              for relative, (_, original, modified, _) in planned.items()}}
        atomic_write(manifest_path, (json.dumps(manifest, indent=2) + "\n").encode(), 0o600)

    written = []
    try:
        for relative in changed:
            current, original, modified, permissions = planned[relative]
            # Detect edits after preflight before replacing that file.
            if (root / relative).read_bytes() != current:
                raise ValueError(f"{relative}: changed after preflight")
            atomic_write(root / relative, original if restore else modified, permissions)
            written.append(relative)
    except BaseException:
        for relative in reversed(written):
            current, _, _, permissions = planned[relative]
            atomic_write(root / relative, current, permissions)
        raise
    if restore and manifest_path.exists():
        os.replace(manifest_path, backup / "manifest.restored.json")
    result["status"] = "restored" if restore else "applied" if changed else "already-applied"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("live", "final"), default="live")
    parser.add_argument("--apply", action="store_true", help="write source changes after preflight")
    parser.add_argument("--restore", action="store_true", help="restore originals (requires --apply)")
    args = parser.parse_args()
    distribution = importlib.metadata.distribution("vllm")
    root = Path(distribution.locate_file("vllm"))
    print(json.dumps(patch(root, distribution.version, args.mode, args.apply, args.restore), indent=2))


if __name__ == "__main__":
    main()
