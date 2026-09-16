// Real GPU integration check. Reserve Spark_1 and keep the demo idle while running.
// Includes compilation/warmup; these timings are not benchmark results.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
import { collectCompletion } from '../server/stream.mjs';
const config = JSON.parse(await readFile('config/models.json', 'utf8'));
const model = config.models.find(m => m.id === 'diffusion');
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
assert(Number.isInteger(canvasLength) && canvasLength > 0 && canvasLength <= 32768,
  'CANVAS_LENGTH must be a positive whole number no larger than 32768');
const bridge = (process.env.SPARK_LAB_BRIDGE || 'http://127.0.0.1:8787').replace(/\/$/, '');
const statusResponse = await fetch(bridge + '/api/runtime', { signal: AbortSignal.timeout(30000) });
assert(statusResponse.ok, `Runtime attestation HTTP ${statusResponse.status}`);
const attestation = (await statusResponse.json()).diffusion;
assert(attestation?.ready === true && attestation?.verified === true && attestation.canvas_length === canvasLength,
  `Canvas ${canvasLength} is not verified active; change it through the app first. This diagnostic does not restart servers.`);
const headers = { 'Content-Type': 'application/json' };
if (process.env[model.apiKeyEnv]) headers.Authorization = `Bearer ${process.env[model.apiKeyEnv]}`;
const payload = sourcePayload;
const evidence = { created_at: new Date().toISOString(), diagnostic: true, includes_compilation: true,
  configuration: config, input_provenance: inputProvenance, requested_canvas_length: canvasLength, actual_canvas_length: attestation.canvas_length,
  runtime_attestation: attestation, output_stopping_note: 'Forced two-canvas functional checks; special-token tails are not a quality or throughput result.', tests: [] };
const directory = `runtime/canvas/${canvasLength}/${new Date().toISOString().replace(/[:.]/g, '-')}`;
await mkdir(directory, { recursive: true });
async function request(mode, maxSteps, label) {
  const sent = { ...payload, max_tokens: 2 * canvasLength, ignore_eos: true,
    vllm_xargs: { spark_lab_diffusion_max_steps: maxSteps, spark_lab_diffusion_force_steps: mode === 'fixed' ? 1 : 0,
      spark_lab_diffusion_preview: 0 } };
  const tokenResponse = await fetch(model.baseUrl.replace(/\/$/, '').replace(/\/v1$/, '') + '/tokenize', {
    method: 'POST', headers, body: JSON.stringify({ model: model.model, messages: sent.messages,
      chat_template_kwargs: sent.chat_template_kwargs, add_generation_prompt: true, add_special_tokens: false }),
    signal: AbortSignal.timeout(30000),
  });
  assert(tokenResponse.ok, `Context preflight HTTP ${tokenResponse.status}`);
  const input = await tokenResponse.json();
  const capacity = Math.min(config.runtime?.max_model_len ?? Infinity, input.max_model_len ?? Infinity);
  assert(Number.isInteger(input.count) && input.count + sent.max_tokens <= capacity,
    `Two-canvas diagnostic needs ${sent.max_tokens} output tokens plus ${input.count} prompt tokens within ${capacity}`);
  const snapshots = [];
  const start = performance.now();
  const response = await fetch(model.baseUrl + '/chat/completions', {
    method: 'POST', headers, body: JSON.stringify(sent), signal: AbortSignal.timeout(600000),
  });
  if (!response.ok) throw Error(await response.text());
  const result = await collectCompletion(response.body, {
    start, stripEmptyGemmaChannel: true,
    denoisingOptions: { mode, max_steps: maxSteps, canvas_length: canvasLength, max_tokens: sent.max_tokens },
    emit(chunk) { if (chunk.denoising) snapshots.push({ at_ms: chunk.at_ms, denoising: chunk.denoising }); },
  });
  const drafts = result.engine_metrics?.speculative_decoding?.per_step_drafted;
  evidence.tests.push({ label, request_payload: sent, snapshots, result,
    requested_canvas_length: canvasLength, actual_canvas_length: attestation.canvas_length,
    observed_drafted_canvas_lengths: Array.isArray(drafts) ? [...new Set(drafts)] : null });
  assert.equal(result.denoising.status, 'available', JSON.stringify(result.denoising));
  assert(Array.isArray(drafts) && drafts.length && drafts.every(n => n === canvasLength), 'Native canvas drafts differ from requested/attested canvas');
  assert.equal(result.completion_tokens, 2 * canvasLength, 'Expected exactly two full functional-test canvases');
  assert.equal(result.denoising.blocks.length, 2, 'Unexpected canvas count');
  assert.equal(result.output_token_ids.length, result.completion_tokens);
  assert.equal(result.denoising.blocks.reduce((n, b) => n + b.emitted_tokens, 0), result.completion_tokens);
  assert.ok(snapshots.some(s => s.denoising.blocks.length === 1 && !s.denoising.final), 'first block must carry live metadata');
  if (mode === 'fixed') assert.ok(result.denoising.blocks.every(b => b.denoising_steps === maxSteps));
  console.log(label + ': ' + result.denoising.blocks.map(b => b.denoising_steps).join(', ') + ' steps; ' + result.completion_tokens + ' tokens');
}
try {
  await request('adaptive', 48, 'adaptive warmup');
  await request('fixed', 16, 'fixed16 warmup');
  await Promise.all([request('adaptive', 48, 'mixed batch adaptive'), request('fixed', 17, 'mixed batch fixed17')]);
  await request('adaptive', 48, 'adaptive after fixed slot reuse');
  const response = await fetch(model.baseUrl + '/chat/completions', {
    method: 'POST', headers,
    body: JSON.stringify({ ...payload, vllm_xargs: { spark_lab_diffusion_max_steps: 0, spark_lab_diffusion_force_steps: 1 } }),
  });
  assert.equal(response.status, 400, 'bad controls must fail before GPU admission');
  evidence.invalid_request = { status: response.status, response: await response.json() };
  evidence.status = 'passed';
} catch (error) {
  evidence.status = 'failed';
  evidence.error = error.message;
  process.exitCode = 1;
  console.error(error);
} finally {
  await writeFile(directory + '/denoising-validation.json', JSON.stringify(evidence, null, 2));
  console.log('Evidence: ' + directory + '/denoising-validation.json');
}
