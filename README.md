# Spark / Gemma Lab

Run DiffusionGemma and autoregressive Gemma 4 side by side on two NVIDIA DGX Sparks. Send the same editable prompt to both models, watch real output arrive, inspect actual diffusion steps, and collect reproducible latency/throughput measurements.

- One FP8 model per Spark; no MTP or draft model.
- Batch size defaults to 1 and is configurable as concurrent requests.
- Adaptive diffusion demos with actual steps reported for each committed block.
- Optional live intermediate predictions, clearly separated from committed output.
- Committed-token probabilities for both models: translucent red → orange → green highlights with exact values on hover.
- Configurable prompt, input/output lengths and diffusion canvas size.
- Profiling sweeps, PG19 preparation, JSON/JSONL evidence, CSV and plots.

```text
Browser :3000 → local Node bridge :8787
                    ├─ SSH :18001 → Spark_1 :8000 → DiffusionGemma
                    └─ SSH :18002 → Spark_2 :8000 → Gemma 4 AR
```

The UI, bridge and model servers bind to loopback. Inference runs on your Sparks; cloud hosting is unnecessary. Model weights, datasets and private benchmark runs are not included.

## Try the interface without GPUs

Requires Node.js **22.13 or newer**, npm and a Unix-like shell.

```bash
git clone https://github.com/bodono/spark-gemma-lab.git
cd spark-gemma-lab
npm ci
SPARK_LAB_MOCK=1 npm run lab
```

Open [http://127.0.0.1:3000](http://127.0.0.1:3000). Mock mode is visibly marked **synthetic** and cannot provide model performance measurements. Stop it with Ctrl-C before starting the real setup below.

## Set up two DGX Sparks

The native installer targets **Linux ARM64, Python 3.12.3 and CUDA 13.0 at `/usr/local/cuda-13.0`**, with a compatible NVIDIA driver. It also requires `python3-venv`, a C compiler, `curl`, `dpkg-deb`, `tar` and SSH. Allow sufficient disk space for the CUDA environment, two model downloads and compilation caches. The controller requires Linux process descriptors (`pidfd`).

Configure SSH aliases, replacing the placeholders with your hosts and users:

```sshconfig
Host Spark_1
    HostName YOUR_FIRST_SPARK_IP
    User YOUR_SSH_USER

Host Spark_2
    HostName YOUR_SECOND_SPARK_IP
    User YOUR_SSH_USER
```

Connect interactively once to verify each host key and configure SSH key authentication. The app's runtime controller uses noninteractive SSH:

```bash
ssh -o BatchMode=yes Spark_1 true
ssh -o BatchMode=yes Spark_2 true
bash scripts/deploy-user.sh diffusion prepare
bash scripts/deploy-user.sh autoregressive prepare
```

Preparation creates a user-owned environment at `~/.local/share/spark-gemma-lab/venv`, installs **vLLM 0.29.0**, and checks its resolved dependencies. It extracts checksum-verified Python development headers without sudo or changes to system Python. The CUDA 13 stack was developed with PyTorch 2.13.0. The verifier permits one specifically checked cuSPARSELt 0.8.1 ARM64 wheel-tag defect; unrelated dependency errors fail.

Downloads use Hugging Face access configured on each Spark. If authentication is needed, authenticate with `~/.local/share/spark-gemma-lab/venv/bin/hf auth login` on that machine. Local `HF_TOKEN` values are not automatically forwarded over SSH. Follow the checkpoints' model access and license terms.

With both GPUs available, start the managed Diffusion server:

```bash
ssh Spark_1 'python3 .local/share/spark-gemma-lab/canvas-runtime.py ensure 256'
```

This waits for readiness and leaves the owned server running in the background. First startup downloads weights and compiles kernels. Logs are under `~/.local/share/spark-gemma-lab/logs/` on Spark_1. In separate terminals, run:

```bash
bash scripts/deploy-user.sh autoregressive serve
```

```bash
bash scripts/tunnels.sh
```

```bash
npm run lab
```

Keep the AR serving and tunnel terminals open. The launchers refuse active competing GPU processes rather than stopping them. Suspended jobs can still occupy memory. The canvas controller verifies the owned Diffusion process and its idle state before restarting it; it does not manage other workloads.

The defaults in [config/models.json](config/models.json) match these aliases, tunnel ports and served model names. Initial capacity is four sequences per server, with an 8,192-token context and prefix caching disabled. Start profiling at concurrency 1, 2 and 4. Higher client concurrency can introduce queueing rather than a larger GPU batch. Keep configuration metadata consistent with your actual servers.

For private endpoint overrides, copy `config/models.example.json` to `config/models.local.json` and start the app with `SPARK_LAB_CONFIG=config/models.local.json npm run lab`. The local override is ignored by Git.

For a production UI build, use `npm run build` followed by `npm start`. Both commands run locally; no deployment account is needed.

## Models and runtime patches

| Model | Pinned Hugging Face checkpoint | Revision |
|---|---|---|
| DiffusionGemma | `RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic` | `3b3dae4697494da5a290e9c0461954449e76c4f5` |
| Gemma 4 AR | `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic` | `ed35d7abe5d940da41b4ff06eb482feb0be8cb44` |

These are instruction-tuned, non-MTP checkpoints. Quantized layers use FP8 W8A8 compressed tensors; remaining layers and the KV cache use BF16. The native launcher uses Triton attention and MoE, FP8 linear kernels, Model Runner V2, compilation and CUDA graphs. It is an experimental baseline, not a claim of peak hardware performance.

The [native launcher](scripts/serve-spark-user.sh) applies reviewed local patches for [request controls](scripts/patch_diffusion_controls.py), [per-block metrics](scripts/patch_diffusion_metrics.py), [live previews](scripts/patch_diffusion_preview.py) and [batched diffusion probability attribution](scripts/patch_diffusion_probabilities.py). They require vLLM 0.29.0 and matching source hashes, keep backups, and refuse unknown source changes. They are not upstream vLLM features. Updating vLLM requires reviewing and updating the patches.

The supplied Docker alternative is uninstrumented and does not provide the managed canvas/preview/counting workflow described here. Use the native setup for the full application.

## Run a demo

Edit the prompt, batch size and output limit, then select **Run comparison**. For multiple requests, select the request index to inspect the corresponding pair. Output appears at the cadence received from each server; diffusion can commit an entire block at once.

The demo uses adaptive convergence with a **maximum of 48 denoising steps per block**. Actual counts are shown as each block commits; the configured maximum is never substituted for an observed count. Temperature and seed controls apply only to AR. DiffusionGemma uses its own temperature schedule, and its current sampler does not use a request-specific random seed.

**AR temperature** accepts any finite, nonnegative value, including 3 or 100; 0 uses greedy decoding. The [AR temperature patch](scripts/patch_ar_temperature.py) removes vLLM 0.29.0's upper validation limit of 2 on the dedicated AR runtime, preserving finite/nonnegative checks and sampling behavior. The AR launcher applies it with version/source-hash checks and a backup. Restart AR with the updated launcher after installing it; an unpatched server still rejects values above 2. The same values work through the profiling CLI. Diffusion's sampler is unchanged.

Canvas size defaults to **256 tokens**, with 8, 16, 32, 64, 128 and 512 available in both demo and profiling views. Changing it reloads the owned Diffusion server and warms the new runtime before measurement; allow a few minutes. Nondefault sizes are experimental. AR is unaffected. The output limit and canvas size are separate settings.

Live previews display provisional token predictions and may add latency. They never count as completion tokens or first visible committed output. Turn previews off for cleaner timing comparisons. Profiling and all warmups always disable preview capture.

**Token probabilities** is enabled by default for AR demos. Each sampled token gets a translucent box on a continuous probability scale: red at 0%, orange at 50%, green at 100%. Hover to inspect the exact probability and log probability. These are `exp(logprob)` from vLLM's raw model distribution, before temperature/top-k/top-p; they describe the likelihood of the selected token, not whether the answer is correct. The bridge requests `logprobs: true, top_logprobs: 0` only for demo AR requests. Turn the checkbox off to omit collection. Profiling and all warmups always omit it, and demo timings include any collection overhead.

Token boundaries need not match words. Output text and spacing stay unchanged; missing scores or unverified text alignment stay neutral. If several token fragments decode into one Unicode character, its hover readout lists the contributing probabilities without inventing a single character probability. Selected-token records and their verified text spans are included in the exported run.

Diffusion's **Final token probabilities** toggle is also enabled by default for demos. Committed blocks receive the same red/orange/green scale; the provisional live preview keeps its purple animation. These scores come from the final converging denoising pass, after the diffusion temperature schedule and top-k/top-p filtering, conditioned on the current canvas and prefix. They are distinct from AR's raw next-token distribution and are often close to 100% after convergence. Hover labels identify the source; the display never stretches the color scale to manufacture contrast. Collection uses vLLM's existing selected-token logprobs and adds no extra model forward pass. Profiling and all warmups disable both models' probability collection.

The native probability patch fixes a vLLM 0.29.0 batching issue: a committing request could otherwise drain another request's newly converged probability buffer before that request commits. It releases only buffers belonging to requests committing in the current step. The added work stays inside the opt-in logprobs branch. Restart the managed Diffusion server with the updated launcher after installing this patch; the local bridge/UI update alone does not update a running model process.

## Profile and plot

The optional analysis tools require Python 3.12 or newer. Create a local environment and prepare a PG19 test excerpt bank:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-analysis.lock.txt
.venv/bin/python scripts/prepare_pg19.py \
  --count 96 --input-tokens 512 --output data/pg19-512.jsonl
```

The script downloads test books, verifies their inventory/checksums, and records source and tokenizer provenance. No dataset is bundled. The UI can load this prepared bank or a JSONL file with a nonempty `prompt` field on each line.

Keep the real bridge and tunnels running, then launch a matched sweep:

```bash
npm run profile -- \
  --dataset data/pg19-512.jsonl --batch-sizes 1,2,4 \
  --requests 20 --warmups 1 --tokens 512 --canvas-length 256
```

The default task is chat continuation with natural EOS and adaptive denoising. Both models receive the same source text. There are 20 measured requests per model per concurrency; the count must divide evenly by every chosen concurrency. Warmups use separate prompts and are excluded from summaries. The bank must contain at least `max(concurrency) × warmup waves + measured requests` samples.

Additional controls:

- `--input-tokens 2048`: exactly 2,048 shared **source-text** tokens before instructions/chat formatting. Short sources repeat and trim; long ones trim. Both tokenizers and rendered context budgets are checked before timed waves. For contiguous 2K passages instead, prepare a separate bank with `prepare_pg19.py --input-tokens 2048`.
- `--denoising-mode fixed --denoising-steps 16`: exactly 16 passes per block, with early convergence disabled. This also changes the sampler schedule and can substantially reduce output quality; it is a speed/quality experiment.
- `--canvas-length 128`: request a native 128-token canvas, including a managed reload when necessary.
- `--workload raw --output-mode fixed`: raw completion with EOS ignored. Forced length can produce repetitive or special-token tails; it does not guarantee useful text.

Input, formatting and output must fit the configured context. Natural stopping permits different response lengths; reported rates use actual server counts. Each run saves settings, requests, outputs, token IDs, timing, errors and denoising metadata under `results/`. These files can contain your prompts and model output and are ignored by Git.

Plot a saved run, replacing `RUN_ID` with the ID printed by the CLI:

```bash
.venv/bin/python scripts/plot_results.py \
  results/RUN_ID.json --output results/plots/RUN_ID
```

The plotter exports throughput/latency plots, CSV summaries and actual denoising-step distributions. Synthetic data require an explicit `--allow-synthetic` flag and are watermarked.

### Sweep progress
Starting a sweep brings a progress panel into view. It shows model preparation, input preparation, warmups and measured requests, with elapsed time and separate warmup/measurement counters. Preparation is indeterminate; the measured percentage counts finished requests across both models, including failed requests, which are marked separately. Profiling continues if its browser tab refreshes or disconnects; the page reconnects to `/api/activity` and loads the saved results when the sweep finishes. Use **Stop sweep** to cancel. Progress reporting does not enable diffusion previews during profiling.

## Interpret the measurements

- **First visible output:** time to the first nonempty committed content.
- **Complete-answer latency:** request dispatch through the terminal response.
- **Per-user throughput:** server-reported completion tokens divided by full request latency.
- **After-first-block rate:** tokens after the first reported block divided by the time between first and last token-count arrivals. This excludes prefill **and generation of the first whole block**; it is unavailable for one-block answers. The UI reports the pooled rate, with mean/median also saved.
- **Aggregate served throughput:** total tokens divided by measured per-endpoint wave makespans.

These are client-observed serving measurements, including transport and scheduling, not isolated GPU decode timings. Concurrent prefills can affect later generation. A faster response is not automatically an equal-quality response, and equal output limits do not ensure equal output lengths. Batch size means concurrent HTTP requests; actual GPU batch occupancy is not measured. Unavailable or invalid telemetry is labeled explicitly.

## Development

```bash
npm test
npm run check
npm run build
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Tests use CPU fixtures and simulated streams; they do not require model weights or GPUs. Real hardware performance must be measured on your own deployment. Use `SPARK_LAB_MOCK=1 npm run lab` for interface development.

## License

Original project code is released under the [MIT License](LICENSE). Embedded vLLM source and runtime diffs retain Apache-2.0; see [third-party notices](THIRD_PARTY_NOTICES.md). Model weights, datasets and dependencies retain their respective licenses.

Low-level denoising validation and kernel-trace capture require an existing saved
Diffusion chat run, for example:

```bash
BASELINE_RUN=results/RUN_ID.json node scripts/validate-denoising.mjs
```

These diagnostic scripts validate their input before contacting the runtime and
record its hash and request provenance. They are separate from ordinary profiling.
