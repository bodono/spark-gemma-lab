"""CPU regression tests for native committed diffusion logprob ownership."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


patch = load("probability_patch", "scripts/patch_diffusion_probabilities.py")
preview = load("preview_patch", "scripts/patch_diffusion_preview.py")
BASE = preview.unpack(preview.PAYLOAD[patch.RELATIVE]["after"])


class Array:
    """Tiny fake GPU array; rejects unneeded host copies via copy tracking."""
    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(self.values),)
        self.host_copies = 0

    def any(self):
        return any(self.values)

    def __getitem__(self, index):
        if isinstance(index, Array):
            return Array(value for value, keep in zip(self.values, index.values) if keep)
        return self.values[index]

    def tolist(self):
        self.host_copies += 1
        return self.values.copy()


@dataclass
class Scores:
    logprob_token_ids: Array
    logprobs: Array
    selected_token_ranks: Array
    cu_num_generated_tokens: list | None = None


def scores(ids):
    return Scores(Array(ids), Array([-value / 1000 for value in ids]), Array([1] * len(ids)))


def native_commit_block(source):
    """Execute the real patched sampler's complete reassembly branch on CPU."""
    tree = ast.parse(source)
    blocks = [node for node in ast.walk(tree) if isinstance(node, ast.If)
              and ast.unparse(node.test) ==
              "max_num_logprobs >= 0 and is_committing.any() and self._pending_logprobs"]
    assert len(blocks) == 1
    block = ast.unparse(blocks[0])
    code = "def run(self, max_num_logprobs, is_committing, decode_slots, slots_np, is_decode_np):\n"
    code += "    num_reqs = len(slots_np)\n    logprobs_tensors = None\n"
    code += "\n".join("    " + line for line in block.splitlines())
    code += "\n    return logprobs_tensors\n"
    namespace = {
        "LogprobsTensors": Scores,
        "torch": SimpleNamespace(cat=lambda parts: Array(v for p in parts for v in p.values)),
    }
    exec(compile(code, "<native diffusion logprob reassembly>", "exec"), namespace)
    return namespace["run"]


class NativeLogprobOwnershipTests(unittest.TestCase):
    def test_newly_converged_request_retains_scores_until_its_own_commit(self):
        # Slot 42 is prefilling, slot 7 commits, slot 11 just converged.
        # These are slot IDs, deliberately different from request positions.
        run = native_commit_block(patch.render(BASE))
        owner = SimpleNamespace(_pending_logprobs={7: scores([70, 71]), 11: scores([110, 111, 112])})
        result = run(owner, 0, Array([True, False]), Array([7, 11]), [42, 7, 11], [False, True, True])
        self.assertEqual(result.logprob_token_ids.values, [70, 71])
        self.assertEqual(result.cu_num_generated_tokens, [0, 0, 2])
        self.assertEqual(set(owner._pending_logprobs), {11})
        # A changed scheduler order does not change probability ownership.
        result = run(owner, 0, Array([True]), Array([11]), [11, 42], [True, False])
        self.assertEqual(result.logprob_token_ids.values, [110, 111, 112])
        self.assertEqual(result.logprobs.values, [-0.110, -0.111, -0.112])
        self.assertEqual(result.cu_num_generated_tokens, [0, 3])
        self.assertFalse(owner._pending_logprobs)

    def test_regression_fixture_exposes_original_early_pop(self):
        run = native_commit_block(BASE)
        owner = SimpleNamespace(_pending_logprobs={7: scores([70]), 11: scores([110])})
        result = run(owner, 0, Array([True, False]), Array([7, 11]), [7, 11], [True, True])
        # Native unpatched code discards slot 11's scores before it emits tokens.
        self.assertEqual(result.logprob_token_ids.values, [70, 110])
        self.assertFalse(owner._pending_logprobs)

    def test_two_committing_slots_keep_request_order_and_zero_prefill_offset(self):
        run = native_commit_block(patch.render(BASE))
        owner = SimpleNamespace(_pending_logprobs={7: scores([70]), 11: scores([110, 111])})
        result = run(owner, 0, Array([True, True]), Array([11, 7]), [11, 42, 7], [True, False, True])
        self.assertEqual(result.logprob_token_ids.values, [110, 111, 70])
        self.assertEqual(result.cu_num_generated_tokens, [0, 2, 2])
        self.assertFalse(owner._pending_logprobs)

    def test_disabled_logprobs_do_not_inspect_gpu_state(self):
        class ForbiddenGPU:
            def any(self):
                raise AssertionError("Profiling must not inspect probability GPU state")

            def __getitem__(self, index):
                raise AssertionError("Profiling must not copy probability GPU state")

        run = native_commit_block(patch.render(BASE))
        owner = SimpleNamespace(_pending_logprobs={7: scores([70])})
        self.assertIsNone(run(owner, -1, ForbiddenGPU(), ForbiddenGPU(), [7], [True]))
        self.assertEqual(set(owner._pending_logprobs), {7})

    def test_no_commit_retains_pending_scores(self):
        run = native_commit_block(patch.render(BASE))
        owner = SimpleNamespace(_pending_logprobs={11: scores([110])})
        self.assertIsNone(run(owner, 0, Array([False]), Array([11]), [11], [True]))
        self.assertEqual(set(owner._pending_logprobs), {11})


class PatchLifecycleTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory) / "vllm"
        path = root / patch.RELATIVE
        path.parent.mkdir(parents=True)
        path.write_bytes(BASE)
        path.chmod(0o640)
        return root, path, Path(directory) / "evidence"

    def test_check_apply_reapply_revert_and_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            root, path, evidence = self.fixture(directory)
            backup = path.with_name(path.name + patch.BACKUP_SUFFIX)
            self.assertEqual(patch.apply(root, "check", evidence)["status"], "pending")
            self.assertEqual(path.read_bytes(), BASE)
            self.assertFalse(backup.exists())
            self.assertEqual(patch.apply(root, "apply", evidence)["status"], "applied")
            self.assertEqual(backup.read_bytes(), BASE)
            self.assertEqual(path.stat().st_mode & 0o777, 0o640)
            self.assertEqual(patch.apply(root, "apply", evidence)["status"], "already_selected")
            self.assertEqual(patch.apply(root, "revert", evidence)["status"], "reverted")
            self.assertEqual(path.read_bytes(), BASE)
            self.assertEqual(patch.apply(root, "revert", evidence)["status"], "already_selected")
            self.assertTrue((evidence / "probabilities.patch").exists())
            self.assertTrue((evidence / "probabilities-manifest.json").exists())

    def test_unrelated_source_and_corrupt_backup_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root, path, evidence = self.fixture(directory)
            unrelated = BASE + b"\n# unrelated change\n"
            path.write_bytes(unrelated)
            with self.assertRaises(RuntimeError):
                patch.apply(root, "apply", evidence)
            self.assertEqual(path.read_bytes(), unrelated)
            path.write_bytes(BASE)
            patch.apply(root, "apply", evidence)
            path.with_name(path.name + patch.BACKUP_SUFFIX).write_bytes(b"bad backup")
            with self.assertRaises(RuntimeError):
                patch.apply(root, "revert", evidence)
            self.assertEqual(path.read_bytes(), patch.render(BASE))

    def test_preview_stack_unwinds_in_reverse_order_and_reapplies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "vllm"
            for relative, entry in preview.PAYLOAD.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(preview.unpack(entry["after"]))
            evidence = Path(directory) / "evidence"
            patch.apply(root, "apply", evidence)
            # Existing preview refuses a newer overlay until it is unwound.
            with self.assertRaises(RuntimeError):
                preview.apply(root, "revert", evidence)
            patch.apply(root, "revert", evidence)
            preview.apply(root, "revert", evidence)
            preview.apply(root, "apply", evidence)
            patch.apply(root, "apply", evidence)
            self.assertEqual((root / patch.RELATIVE).read_bytes(), patch.render(BASE))


if __name__ == "__main__":
    unittest.main()
