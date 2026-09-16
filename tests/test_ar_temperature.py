"""CPU checks for the pinned native AR temperature validation patch."""
import ast
import importlib.util
import math
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


patch = load("ar_temperature_patch", "scripts/patch_ar_temperature.py")
controls = load("request_controls_patch", "scripts/patch_diffusion_controls.py")
preview = load("preview_patch", "scripts/patch_diffusion_preview.py")
# Reuse the already-shipped source payload, undoing only request controls.
BASE = preview.unpack(preview.PAYLOAD[patch.RELATIVE]["before"])
assert BASE.count(controls.VALIDATION.encode()) == 1
BASE = BASE.replace(controls.VALIDATION.encode(), b"", 1)
assert patch.digest(BASE) == patch.BASE_SHA256


class ValidationError(ValueError):
    def __init__(self, message, **kwargs):
        super().__init__(message)


def temperature_validator(source):
    """Execute the real native method's temperature guards without a GPU."""
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SamplingParams")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_verify_args")
    guards = [node for node in method.body if isinstance(node, ast.If)
              and "self.temperature" in ast.unparse(node.test)]
    code = "def check(self):\n" + "\n".join(
        "    " + line for node in guards for line in ast.unparse(node).splitlines()
    )
    namespace = {"math": math, "VLLMValidationError": ValidationError}
    exec(compile(code, "<native AR temperature guards>", "exec"), namespace)
    return namespace["check"]


class ARTemperatureTests(unittest.TestCase):
    def test_native_guards_accept_100_without_relaxing_invalid_inputs(self):
        old = temperature_validator(BASE)
        new = temperature_validator(patch.render(BASE))
        with self.assertRaisesRegex(ValidationError, r"\[0, 2\]"):
            old(SimpleNamespace(temperature=100))
        for value in [0, 0.1, 2, 3, 100, 1000]:
            new(SimpleNamespace(temperature=value))
        for value in [-1, math.nan, math.inf, -math.inf]:
            with self.assertRaises(ValidationError):
                new(SimpleNamespace(temperature=value))

    def test_patch_preserves_all_other_validation_and_sampling_code(self):
        rendered = patch.render(BASE)
        self.assertEqual(rendered.replace(patch.AFTER.encode(), patch.BEFORE.encode()), BASE)

    def test_check_apply_idempotence_revert_and_backup_integrity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / patch.RELATIVE
            target.write_bytes(BASE)
            target.chmod(0o640)
            evidence = root / "evidence"
            patch.apply(root, "check", evidence)
            self.assertEqual(target.read_bytes(), BASE)
            patch.apply(root, "apply", evidence)
            self.assertEqual(target.read_bytes(), patch.render(BASE))
            self.assertEqual(target.stat().st_mode & 0o777, 0o640)
            self.assertEqual(patch.apply(root, "apply", evidence)["status"], "already_selected")
            patch.apply(root, "revert", evidence)
            self.assertEqual(target.read_bytes(), BASE)
            backup = target.with_name(target.name + patch.BACKUP_SUFFIX)
            backup.write_bytes(b"unrelated backup")
            with self.assertRaisesRegex(RuntimeError, "Backup does not match"):
                patch.apply(root, "apply", evidence)
            self.assertEqual(target.read_bytes(), BASE)

    def test_unknown_source_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / patch.RELATIVE
            altered = BASE + b"\n# unknown modification\n"
            target.write_bytes(altered)
            with self.assertRaisesRegex(RuntimeError, "Refusing"):
                patch.apply(root, "apply", root / "evidence")
            self.assertEqual(target.read_bytes(), altered)


if __name__ == "__main__":
    unittest.main()
