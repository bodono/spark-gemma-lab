# Third-party notices

Original Spark / Gemma Lab application code is provided under the MIT license in `LICENSE`.

## vLLM

The runtime patch scripts and readable diffs include source derived from
[vLLM v0.29.0](https://github.com/vllm-project/vllm/tree/v0.29.0).
In particular, `scripts/patch_diffusion_preview.py` embeds compressed original
and modified vLLM files so it can verify exact source hashes and apply or revert
changes. Upstream copyright and SPDX notices are preserved in those payloads.
The vLLM-derived source, including modifications for per-request controls,
block metrics and intermediate diffusion previews, remains under Apache-2.0;
it is not relicensed by the root MIT license. See `LICENSES/Apache-2.0.txt`.
The readable changes are in `patches/`.

## UI components and package dependencies

The UI uses shadcn-derived components and open-source packages. Their respective
copyrights and licenses continue to apply. The shadcn component license is in
`LICENSES/shadcn-MIT.txt`. Installed package notices are distributed with each
package; exact versions and registry integrity hashes are in `package-lock.json`.

## Models and data

No model weights or PG19 texts are distributed in this repository. Model
checkpoints, their tokenizers, and datasets remain subject to their upstream
terms; the application license does not grant rights to those separate assets.
