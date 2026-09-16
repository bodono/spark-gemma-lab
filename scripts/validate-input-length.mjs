// Real hardware functional checks. Run while the lab is idle; not benchmark points.
import assert from 'node:assert/strict';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { parseSSE } from '../server/stream.mjs';
const dataset = (
  await readFile(new URL('../data/pg19-512.jsonl', import.meta.url), 'utf8')
)
  .trim()
  .split('\n')
  .slice(0, 4)
  .map(JSON.parse);
const base = {
  kind: 'profile',
  prompt: 'Explain how a diffusion language model works.',
  batch_size: 1,
  batch_sizes: [1],
  requests_per_condition: 1,
  repeats: 1,
  warmups: 1,
  max_tokens: 64,
  temperature: 0,
  seed: 42,
  diffusion_preview: true,
  denoising_mode: 'adaptive',
};
async function request(extra) {
  const response = await fetch('http://127.0.0.1:8787/api/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...base, ...extra }),
    signal: AbortSignal.timeout(180000),
  });
  assert.equal(
    response.status,
    200,
    response.ok ? '' : (await response.text()).slice(0, 200),
  );
  let run,
    error,
    previews = 0;
  for await (const raw of parseSSE(response.body)) {
    const event = JSON.parse(raw);
    if (event.type === 'error') error = event.message;
    if (event.type === 'preview') previews++;
    if (event.type === 'complete') run = event.run;
  }
  return { run, error, previews };
}
const checks = [];
for (const setting of [
  {
    input_tokens: 128,
    dataset,
    dataset_name: 'PG19 validation subset',
    batch_sizes: [1, 2],
    requests_per_condition: 2,
  },
  { input_tokens: 2048, dataset, dataset_name: 'PG19 validation subset' },
  { input_tokens: 512, workload: 'raw', warmups: 0 },
]) {
  const { run, error, previews } = await request(setting);
  assert(!error, error);
  assert.equal(run.status, 'complete');
  assert.equal(run.settings.diffusion_preview, false);
  assert.equal(previews, 0);
  for (const r of [...run.results, ...run.warmup_results]) {
    assert.equal(r.input_length_ok, true, r.error);
    assert.equal(
      r.input_preparation.actual_source_tokens,
      setting.input_tokens,
    );
    assert.equal(r.prompt_token_ids.length, r.prompt_tokens);
    if (r.model === 'diffusion')
      assert.equal(r.request_payload.vllm_xargs.spark_lab_diffusion_preview, 0);
  }
  const groups = run.summaries.map((s) =>
    run.results
      .filter((r) => r.model === s.model && r.batch_size === s.batch_size)
      .map((r) => r.effective_prompt_hash)
      .sort(),
  );
  groups.forEach((group) => assert.deepEqual(group, groups[0]));
  checks.push({
    run_id: run.id,
    input_tokens: setting.input_tokens,
    workload: setting.workload || 'continuation',
    policies: [...new Set(run.input_preparation.map((p) => p.policy))],
    warmups: run.warmup_results.length,
    measured: run.results.length,
    preview_events: previews,
    totals: run.summaries.map((s) => ({
      model: s.model,
      batch_size: s.batch_size,
      min: s.min_prompt_tokens,
      max: s.max_prompt_tokens,
    })),
  });
  console.log(JSON.stringify(checks.at(-1)));
}
// The source+output fits, but the AR chat template pushes total context over.
const overflow = await request({
  prompt: 'hello world',
  input_tokens: 2,
  max_tokens: 8180,
  warmups: 0,
});
assert(
  !overflow.run && /exceeds the 8192-token context/.test(overflow.error),
  overflow.error,
);
const evidence = {
  created_at: new Date().toISOString(),
  purpose: 'Functional validation; excluded from benchmark plots',
  checks,
  rendered_context_error: overflow.error,
};
const path = new URL('../runtime/input-length/', import.meta.url);
await mkdir(path, { recursive: true });
await writeFile(
  new URL('hardware-validation.json', path),
  JSON.stringify(evidence, null, 2) + '\n',
);
console.log(
  'PASS: source targets, rendered counts/IDs, warmups, matched cohorts, profiling preview gate, and SSE context error.',
);
