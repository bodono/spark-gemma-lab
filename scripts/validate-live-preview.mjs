// Real hardware validation; model servers must be idle. Does not restart them.
// Demo telemetry only: benchmark-mode rejection is checked through the bridge.
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { collectCompletion, parseSSE } from '../server/stream.mjs';
const config = JSON.parse(await readFile(new URL('../config/models.json', import.meta.url), 'utf8'));
const model = config.models.find(m => m.id === 'diffusion');
const canvasLength = Number(process.env.CANVAS_LENGTH ?? 256);
if (!Number.isInteger(canvasLength) || canvasLength < 1 || canvasLength > 32768)
  throw Error('CANVAS_LENGTH must be a positive whole number no larger than 32768');
const bridge = (process.env.SPARK_LAB_BRIDGE || 'http://127.0.0.1:8787').replace(/\/$/, '');
const statusResponse = await fetch(bridge + '/api/runtime', { signal: AbortSignal.timeout(30000) });
if (!statusResponse.ok) throw Error(`Runtime attestation HTTP ${statusResponse.status}`);
const attestation = (await statusResponse.json()).diffusion;
if (attestation?.ready !== true || attestation?.verified !== true || attestation.canvas_length !== canvasLength)
  throw Error(`Selected canvas ${canvasLength} is not verified active; change it through the app first. Diagnostics do not restart servers.`);
const outputDir = new URL(`../runtime/canvas/${canvasLength}/${new Date().toISOString().replace(/[:.]/g, '-')}/`, import.meta.url);
await mkdir(outputDir, { recursive: true });
const assert = (value, message) => { if (!value) throw Error(message); };
const headers = { 'Content-Type': 'application/json' };
if (process.env[model.apiKeyEnv]) headers.Authorization = `Bearer ${process.env[model.apiKeyEnv]}`;
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
    `Diagnostic needs ${body.max_tokens} output tokens plus its prompt within context ${capacity}`);
}
function observedCanvas(result) {
  const counts = result.engine_metrics?.speculative_decoding?.per_step_drafted;
  assert(Array.isArray(counts) && counts.length && counts.every(n => n === canvasLength),
    'Observed native canvas drafts differ from requested/attested canvas');
  return { requested_canvas_length: canvasLength, actual_canvas_length: attestation.canvas_length,
    observed_drafted_canvas_lengths: [...new Set(counts)] };
}
const shortPrompt = 'Explain how a diffusion language model generates text, using an analogy that a curious engineer would understand. Then compare it with autoregressive generation. Use three short paragraphs, totalling no more than 150 words.';
const longPrompt = 'Explain how diffusion language models work and compare them with autoregressive generation. Give a detailed explanation of at least 400 words, with practical examples.';
async function direct(preview, { prompt = shortPrompt, maxTokens = 2 * canvasLength, forceLength = false } = {}) {
  const frames = [];
  const body = {
    model: model.model, messages: [{ role: 'user', content: prompt }],
    max_tokens: maxTokens, chat_template_kwargs: { enable_thinking: false },
    ignore_eos: forceLength,
    vllm_xargs: { spark_lab_diffusion_max_steps: 48, spark_lab_diffusion_force_steps: 0,
      spark_lab_diffusion_preview: preview ? 1 : 0 },
    stream: true, return_token_ids: true,
    stream_options: { include_usage: true, continuous_usage_stats: true },
  };
  await checkContext(body);
  const started = performance.now();
  const response = await fetch(model.baseUrl + '/chat/completions', {
    method: 'POST', headers, body: JSON.stringify(body),
    signal: AbortSignal.timeout(180000),
  });
  assert(response.ok, `Runtime HTTP ${response.status}: ${response.ok ? '' : await response.text()}`);
  const result = await collectCompletion(response.body, {
    start: started, stripEmptyGemmaChannel: true, previewEnabled: true,
    denoisingOptions: { mode: 'adaptive', max_steps: 48, canvas_length: canvasLength, max_tokens: maxTokens },
    emitPreview: p => frames.push({ ...p, at_ms: performance.now() - started }),
  });
  assert(result.denoising.status === 'available', 'Invalid real step attribution');
  assert(result.output_token_ids.length === result.completion_tokens, 'Token usage mismatch');
  const canvas = observedCanvas(result);
  assert(preview ? frames.length > 0 : frames.length === 0, 'Preview flag was not respected');
  if (preview) {
    assert(!frames.some(p => p.unavailable), 'Preview unavailable');
    assert(frames[0].at_ms < result.ttft_ms, 'No prediction before committed text');
    let offset = 0;
    for (const block of result.denoising.blocks) {
      const seen = frames.filter(p => p.block_index === block.block_index);
      assert(seen.length === block.denoising_steps, `Lost/repeated steps for block${block.block_index}`);
      assert(seen.every((p, i) => p.denoising_step === i + 1), 'Misordered step identities');
      assert(JSON.stringify(seen.at(-1).token_ids.slice(0, block.emitted_tokens)) ===
        JSON.stringify(result.output_token_ids.slice(offset, offset + block.emitted_tokens)),
      `Final prediction differs from committed IDs in block${block.block_index}`);
      offset += block.emitted_tokens;
    }
  }
  return { preview_enabled: preview, payload: body, ...result, ...canvas, previews: frames };
}
const evidence = { created_at: new Date().toISOString(), runtime: config.runtime,
  requested_canvas_length: canvasLength, actual_canvas_length: attestation.canvas_length, runtime_attestation: attestation,
  note: 'Functional validation and an alternating adaptive demo overhead pilot. Warmups excluded. Variable steps and output length prevent a clean raw-latency causal estimate.' };
// Two-block equality proves previews are actual model guesses, not UI animation.
// Force this functional check only; special-token tails are not a quality or throughput result.
evidence.multi_block = await direct(true, { prompt: longPrompt, maxTokens: 2 * canvasLength, forceLength: true });
assert(evidence.multi_block.denoising.blocks.length >= 2, 'Expected at least two blocks');
console.log('Two-block snapshots match committed tokens, ordered actual steps, and precede TTFT.');
evidence.mixed = await Promise.all([direct(true), direct(false)]);
console.log('Concurrent preview-on/off requests remain isolated.');
// Warm both paths before recording latency. Never use these as benchmark points.
evidence.warmups = [await direct(false), await direct(true)];
evidence.latency_pilot = [];
for (let pair = 0; pair < 4; pair++) {
  for (const preview of pair % 2 ? [true, false] : [false, true]) {
    const result = await direct(preview);
    evidence.latency_pilot.push(result);
    console.log(JSON.stringify({ pair, preview, seconds: result.elapsed_ms / 1000,
      tokens: result.completion_tokens, steps: result.denoising.mean_steps,
      preview_frames: result.previews.length }));
  }
}
// Try to enable preview on a profile; bridge must explicitly turn it off for
// both the warmup and measured request. This deliberately uses the real API.
const response = await fetch(bridge + '/api/run', {
  method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ kind: 'profile', prompt: shortPrompt, max_tokens: canvasLength, canvas_length: canvasLength,
    batch_size: 1, batch_sizes: [1], repeats: 1, warmups: 1, seed: 42,
    temperature: 0, denoising_mode: 'adaptive', diffusion_preview: true }),
});
assert(response.ok, `Bridge HTTP ${response.status}: ${response.ok ? '' : await response.text()}`);
let profile, previews = 0;
for await (const raw of parseSSE(response.body)) {
  const event = JSON.parse(raw);
  if (event.type === 'preview') previews++;
  if (event.type === 'error') throw Error(event.message);
  if (event.type === 'complete') profile = event.run;
}
assert(profile?.status === 'complete', 'Profile did not complete');
assert(profile.settings.canvas_length === canvasLength && profile.configuration.runtime.canvas_length === canvasLength,
  'Bridge profile did not preserve requested and verified canvas');
assert(profile.settings.diffusion_preview === false && previews === 0, 'Profile enabled previews');
for (const result of [...profile.warmup_results, ...profile.results]) {
  if (result.model === 'diffusion') {
    assert(result.request_payload.vllm_xargs.spark_lab_diffusion_preview === 0, 'Profile did not force runtime off');
    assert(!result.diffusion_preview, 'Profile collected preview telemetry');
  }
}
evidence.profile_gate = { id: profile.id, warmups: profile.warmup_results.length,
  measured: profile.results.length, preview_events: previews, settings: profile.settings };
await writeFile(new URL('hardware-validation.json', outputDir), JSON.stringify(evidence, null, 2) + '\n');
console.log(`PASS: profiling and its warmups explicitly disabled snapshots; evidence saved to ${outputDir.pathname}`);
