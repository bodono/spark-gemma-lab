# Native vLLM patches

These readable diffs document the vLLM 0.29.0 changes used by the native launcher:

1. Per-request adaptive/fixed denoising controls.
2. Request-specific block metrics and preemption metadata.
3. Opt-in intermediate token previews with asynchronous device-to-host transfer.

Use the guarded `scripts/patch_diffusion_*.py` tools through
`scripts/serve-spark-user.sh`; do not apply these diffs blindly. The tools verify
source hashes and refuse unknown versions. The preview layer depends on the
control/metrics layers and is reverted first when preparing another startup.

No model weights are included. The vLLM-derived source is Apache-2.0; see
[third-party notices](../THIRD_PARTY_NOTICES.md).
