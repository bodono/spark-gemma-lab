// Run only after unprofiled benchmarks, while the model is otherwise idle.
// CANVAS_LENGTH verifies an already active server; this script never restarts it.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { collectCompletion } from '../server/stream.mjs';

const config = JSON.parse(await readFile('config/models.json', 'utf8'));
const model = config.models.find((m) => m.id === 'diffusion');
if (!model) throw Error('Diffusion model is not configured');
// Explicit input avoids depending on the author's unbundled local result files.
// This uses a prior request as a diagnostic prompt, not its timings as benchmark data.
const baselinePath = process.env.BASELINE_RUN;
if (!baselinePath) throw Error('Set BASELINE_RUN to a saved run JSON containing a Diffusion chat request (for example BASELINE_RUN=results/my-run.json).');
let baselineText, baseline;
try {
  baselineText = await readFile(baselinePath, 'utf8');
  baseline = JSON.parse(baselineText);
} catch (error) {
  throw Error('BASELINE_RUN must name a readable saved-run JSON file.', { cause: error });
}
const candidates = Array.isArray(baseline.results) ? baseline.results : [];
const sample = candidates.find(r => r.model === 'diffusion' && !r.warmup && r.batch_size === 1 && Array.isArray(r.request_payload?.messages))
  ?? candidates.find(r => r.model === 'diffusion' && Array.isArray(r.request_payload?.messages));
if (!sample || !sample.request_payload.messages.length || sample.request_payload.messages.some(m =>
  !m || typeof m.role !== 'string' || typeof m.content !== 'string'))
  throw Error('BASELINE_RUN needs a saved Diffusion chat request with nonempty text messages; raw-completion runs are not supported by this diagnostic.');
const inputProvenance = {
  kind: 'saved_run_request',
  run_id: typeof baseline.id === 'string' ? baseline.id : null,
  run_sha256: createHash('sha256').update(baselineText).digest('hex'),
  request_id: sample.request_id ?? null,
  prompt_hash: sample.prompt_hash ?? null,
  source_run_synthetic: baseline.synthetic === true,
  note: 'Only the saved request supplies the diagnostic input. Timings and output are measured anew; no source-run benchmark results are reused.',
};
const sourcePayload = {
  ...sample.request_payload,
  model: model.model,
  stream: true,
  return_token_ids: true,
  stream_options: { include_usage: true, continuous_usage_stats: true },
};

const canvasLength = Number(process.env.CANVAS_LENGTH ?? 256);
if (!Number.isInteger(canvasLength) || canvasLength < 1 || canvasLength > 32768)
  throw Error('CANVAS_LENGTH must be a positive whole number no larger than 32768');
const bridge = (process.env.SPARK_LAB_BRIDGE || 'http://127.0.0.1:8787').replace(/\/$/, '');
const statusResponse = await fetch(bridge + '/api/runtime', { signal: AbortSignal.timeout(30000) });
if (!statusResponse.ok) throw Error(`Runtime attestation HTTP ${statusResponse.status}`);
const attestation = (await statusResponse.json()).diffusion;
if (attestation?.ready !== true || attestation?.verified !== true || attestation.canvas_length !== canvasLength)
  throw Error(`Canvas ${canvasLength} is not verified active; change it through the app first. This diagnostic does not restart servers.`);

const payload = {
  ...sourcePayload,
  ignore_eos: true,
  vllm_xargs: { spark_lab_diffusion_max_steps: 48, spark_lab_diffusion_force_steps: 0,
    spark_lab_diffusion_preview: 0 },
};
const headers = { 'Content-Type': 'application/json' };
if (process.env[model.apiKeyEnv]) headers.Authorization = `Bearer ${process.env[model.apiKeyEnv]}`;
const base = model.baseUrl.replace(/\/$/, '').replace(/\/v1$/, '');
const tokenResponse = await fetch(base + '/tokenize', {
  method: 'POST', headers, body: JSON.stringify({ model: model.model, messages: payload.messages,
    chat_template_kwargs: payload.chat_template_kwargs, add_generation_prompt: true, add_special_tokens: false }),
  signal: AbortSignal.timeout(30000),
});
if (!tokenResponse.ok) throw Error(`Context preflight HTTP ${tokenResponse.status}`);
const input = await tokenResponse.json();
const capacity = Math.min(config.runtime?.max_model_len ?? Infinity, input.max_model_len ?? Infinity);
if (!Number.isInteger(input.count) || !Number.isFinite(capacity)) throw Error('Missing context preflight counts');
const availableBlocks = Math.floor((capacity - input.count) / canvasLength);
if (availableBlocks < 2) throw Error('Tracing after the first block requires room for at least two complete canvases');
payload.max_tokens = Math.min(Math.max(2048, 4 * canvasLength), availableBlocks * canvasLength);

const label = process.argv[2] ?? 'adaptive48';
if (!/^[a-z0-9-]+$/.test(label)) throw Error('Use a lowercase diagnostic label');
const directory = `runtime/canvas/${canvasLength}/${new Date().toISOString().replace(/[:.]/g, '-')}/trace-${label}`;
const call = async (path) => {
  const response = await fetch(base + path, { method: 'POST', headers, signal: AbortSignal.timeout(180000) });
  if (!response.ok) throw Error(`${path}: ${response.status} ${await response.text()}`);
  return response;
};
await mkdir(directory, { recursive: true });
let result, profilePromise, profileError, profileRequestedMs, profileEnabledMs;
const start = performance.now();
try {
  const response = await fetch(model.baseUrl + '/chat/completions', {
    method: 'POST', headers, body: JSON.stringify(payload), signal: AbortSignal.timeout(180000),
  });
  if (!response.ok) throw Error(await response.text());
  result = await collectCompletion(response.body, {
    start, stripEmptyGemmaChannel: true,
    denoisingOptions: { mode: 'adaptive', max_steps: 48, canvas_length: canvasLength, max_tokens: payload.max_tokens },
    emit(chunk) {
      if (chunk.text && !profilePromise) {
        profileRequestedMs = performance.now() - start;
        profilePromise = call('/start_profile')
          .then(() => { profileEnabledMs = performance.now() - start; })
          .catch((error) => { profileError = error.message; });
      }
    },
  });
} finally {
  if (profilePromise) { await profilePromise; await call('/stop_profile'); }
}
const drafts = result.engine_metrics?.speculative_decoding?.per_step_drafted;
const canvasVerified = Array.isArray(drafts) && drafts.length > 0 && drafts.every((n) => n === canvasLength);
await writeFile(directory + '/request.json', JSON.stringify({
  diagnostic: true, input_provenance: inputProvenance, profiling_active: true, not_for_throughput_comparison: true,
  output_stopping_note: 'Forced length keeps tracing alive after the first canvas; output can contain special-token tails and is not a quality result.',
  requested_canvas_length: canvasLength, actual_canvas_length: attestation.canvas_length,
  observed_drafted_canvas_lengths: Array.isArray(drafts) ? [...new Set(drafts)] : null,
  canvas_length_verified: Boolean(canvasVerified), runtime_attestation: attestation,
  profile_requested_after_first_visible_block_ms: profileRequestedMs,
  profile_enabled_ms: profileEnabledMs, profile_error: profileError,
  created_at: new Date().toISOString(), configuration: config, request_payload: payload, result,
}, null, 2));
if (!canvasVerified || result.denoising?.status !== 'available') throw Error('Trace canvas attribution failed; inspect saved evidence');
if (profileError || !profilePromise) throw Error(profileError || 'No visible block triggered tracing');
console.log(`Profiled diagnostic complete: ${result.completion_tokens} tokens; canvas ${canvasLength}. Evidence: ${directory}/request.json. Verify CUDA events; exclude these timings from throughput results.`);
