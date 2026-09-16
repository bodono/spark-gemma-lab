// Proposed hardware check. Run only while the model servers are idle:
// node scripts/validate-preview-cancel.mjs --run
// Exercises cancellation and subsequent request isolation; physical slot IDs
// are not exposed by the API, so this does not prove a particular slot number.
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { collectCompletion, parseSSE } from '../server/stream.mjs';

assert(process.argv.includes('--run'), 'Pass --run to execute real GPU validation.');
const config = JSON.parse(await readFile(new URL('../config/models.json', import.meta.url), 'utf8'));
const model = config.models.find((m) => m.id === 'diffusion');
assert(model, 'Diffusion model is not configured');
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
const endpoint = model.baseUrl.replace(/\/$/, '') + '/chat/completions';
const evidence = { created_at: new Date().toISOString(), runtime: config.runtime,
  requested_canvas_length: canvasLength, actual_canvas_length: attestation.canvas_length, runtime_attestation: attestation,
  scope: 'Cancellation followed by preview-off/on request isolation; API does not expose physical GPU slot IDs.' };
const ids = new Set();
const shortPrompt = 'Explain how rainbows form in two short paragraphs, using no more than 120 words.';
const payload = (preview, cancelled = false) => ({
  model: model.model,
  messages: [{ role: 'user', content: cancelled
    ? 'Write a detailed explanation of how computers work, in at least 1500 words with many practical examples.'
    : shortPrompt }],
  max_tokens: cancelled ? Math.max(2048, 2 * canvasLength) : canvasLength,
  chat_template_kwargs: { enable_thinking: false },
  vllm_xargs: { spark_lab_diffusion_max_steps: 48, spark_lab_diffusion_force_steps: 0,
    spark_lab_diffusion_preview: preview ? 1 : 0 },
  stream: true, return_token_ids: true,
  stream_options: { include_usage: true, continuous_usage_stats: true },
});
async function checkContext(body) {
  const response = await fetch(model.baseUrl.replace(/\/$/, '').replace(/\/v1$/, '') + '/tokenize', {
    method: 'POST', headers, body: JSON.stringify({ model: model.model, messages: body.messages,
      chat_template_kwargs: body.chat_template_kwargs, add_generation_prompt: true, add_special_tokens: false }),
    signal: AbortSignal.timeout(30000),
  });
  assert(response.ok, `Context preflight HTTP ${response.status}`);
  const info = await response.json();
  const capacity = Math.min(config.runtime?.max_model_len ?? Infinity, info.max_model_len ?? Infinity);
  assert(Number.isInteger(info.count) && info.count + body.max_tokens <= capacity,
    `Diagnostic output ${body.max_tokens} plus prompt ${info.count} exceeds context ${capacity}`);
}
async function open(body, signal) {
  const response = await fetch(endpoint, { method: 'POST', headers,
    body: JSON.stringify(body), signal });
  assert(response.ok, `Runtime HTTP ${response.status}: ${response.ok ? '' : await response.text()}`);
  assert(response.body, 'Missing response body');
  return response;
}
function identity(frame, current) {
  assert.equal(typeof frame.id, 'string', 'Missing SSE request identity');
  if (current) assert.equal(frame.id, current, 'Request identity changed mid-stream');
  if (frame.diffusion_preview) assert.equal(frame.diffusion_preview.request_id, frame.id,
    'Preview belongs to another runtime request');
  return frame.id;
}

// Disconnect during an actual denoising block, before receiving a commit.
const controller = new AbortController();
const cancelledPayload = payload(true, true);
await checkContext(cancelledPayload);
const started = performance.now();
const response = await open(cancelledPayload, AbortSignal.any([controller.signal, AbortSignal.timeout(90000)]));
const cancelled = { request_id: null, received_committed_ids: 0, previews: [], cancelled_at_ms: null };
try {
  for await (const raw of parseSSE(response.body)) {
    assert.notEqual(raw, '[DONE]', 'Request finished before the cancellation target');
    const frame = JSON.parse(raw);
    assert(!frame.error, JSON.stringify(frame.error));
    cancelled.request_id = identity(frame, cancelled.request_id);
    cancelled.received_committed_ids += frame.choices?.[0]?.token_ids?.length ?? 0;
    if (!frame.diffusion_preview) continue;
    const p = frame.diffusion_preview;
    assert.notEqual(p.attribution_valid, false, 'Preview attribution unavailable');
    assert.equal(p.block_index, 1, 'Cancellation target moved beyond block 1');
    assert.equal(p.token_ids.length, canvasLength, 'Cancelled preview does not use the verified canvas');
    cancelled.previews.push(p);
    if (p.denoising_step >= 3) {
      assert.equal(cancelled.received_committed_ids, 0, 'A commit arrived before cancellation');
      cancelled.cancelled_at_ms = performance.now() - started;
      controller.abort();
      break;
    }
  }
} finally {
  controller.abort();
}
assert(cancelled.cancelled_at_ms != null, 'No cancellable prediction received');
ids.add(cancelled.request_id);
evidence.cancelled = cancelled;
console.log('Disconnected during block 1 after an observed denoising step.');

async function complete(preview) {
  const body = payload(preview);
  await checkContext(body);
  const started = performance.now(), frames = [], observed = [];
  const response = await open(body, AbortSignal.timeout(120000));
  const [completionBody, identityBody] = response.body.tee();
  const audit = (async () => {
    let requestId = null;
    for await (const raw of parseSSE(identityBody)) {
      if (raw === '[DONE]') break;
      const frame = JSON.parse(raw);
      assert(!frame.error, JSON.stringify(frame.error));
      requestId = identity(frame, requestId);
      if (frame.diffusion_preview) observed.push(frame.diffusion_preview);
    }
    assert(requestId && !ids.has(requestId), 'An earlier request identity was reused');
    ids.add(requestId);
    return requestId;
  })();
  const [result, requestId] = await Promise.all([
    collectCompletion(completionBody, { start: started, stripEmptyGemmaChannel: true,
      previewEnabled: true, emitPreview: (p) => frames.push(p),
      denoisingOptions: { mode: 'adaptive', max_steps: 48, canvas_length: canvasLength, max_tokens: body.max_tokens } }),
    audit,
  ]);
  assert.equal(result.denoising.status, 'available', result.denoising.reason);
  const drafts = result.engine_metrics?.speculative_decoding?.per_step_drafted;
  assert(Array.isArray(drafts) && drafts.length && drafts.every(n => n === canvasLength),
    'Native canvas drafts differ from requested/attested canvas');
  assert(Array.isArray(result.output_token_ids));
  assert.equal(result.output_token_ids.length, result.completion_tokens);
  assert.equal(result.chunks.map((c) => c.text).join(''), result.text);
  assert(result.finish_reason && result.ttft_ms >= 0 && result.elapsed_ms >= result.ttft_ms);
  assert.equal(result.denoising.blocks.length, 1, 'Expected one completed canvas');
  if (!preview) {
    assert.equal(observed.length, 0, 'Preview-off request leaked a raw runtime snapshot');
    assert.equal(frames.length, 0, 'Preview-off request emitted a bridge snapshot');
  } else {
    assert(frames.length > 0 && !frames.some((p) => p.unavailable));
    assert.equal(frames[0].block_index, 1, 'Block index did not reset');
    assert.equal(frames[0].denoising_step, 1, 'Denoising step did not reset');
    assert(frames.every((p, i) => p.block_index === 1 && p.denoising_step === i + 1));
    assert.equal(frames.length, result.denoising.blocks[0].denoising_steps);
    assert.deepEqual(frames.at(-1).token_ids.slice(0, result.completion_tokens), result.output_token_ids);
  }
  return { request_id: requestId, preview_enabled: preview, request_payload: body, ...result,
    requested_canvas_length: canvasLength, actual_canvas_length: attestation.canvas_length,
    observed_drafted_canvas_lengths: [...new Set(drafts)], previews: frames };
}

// Sequential requests let the runtime release/reuse its ordinary request slots.
// Detect leaked opt-in state, old request IDs, old block numbers, and stale tokens.
evidence.after_cancel_off = await complete(false);
evidence.after_cancel_on = await complete(true);
const outputDir = new URL(`../runtime/canvas/${canvasLength}/${new Date().toISOString().replace(/[:.]/g, '-')}/`, import.meta.url);
await mkdir(outputDir, { recursive: true });
await writeFile(new URL('cancel-reuse-evidence.json', outputDir), JSON.stringify(evidence, null, 2) + '\n');
console.log(`PASS: cancellation followed by preview-off/on requests; identities, resets, and committed IDs verified. Evidence: ${outputDir.pathname}`);
