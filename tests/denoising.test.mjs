import test from 'node:test';
import assert from 'node:assert/strict';
import { decodeDenoisingTrace } from '../server/denoising.mjs';
import { collectCompletion } from '../server/stream.mjs';
import { validate, runExperiment } from '../server/runner.mjs';
const metrics = (steps, extra = {}) => {
  const accepted = steps.flatMap((count) => [...Array(count).fill(0), 256]);
  return {
    speculative_decoding: {
      per_step_accepted: accepted,
      per_step_drafted: accepted.map(() => 256),
      diffusion_trace_version: 1,
      diffusion_num_preemptions: 0,
      ...extra,
    },
  };
};
test('actual per-block counts exclude commits and align a trimmed final canvas', () => {
  const d = decodeDenoisingTrace(metrics([3, 16, 5]), 600, { final: true });
  assert.equal(d.status, 'available');
  assert.deepEqual(
    d.blocks.map((b) => b.denoising_steps),
    [3, 16, 5],
  );
  assert.deepEqual(
    d.blocks.map((b) => b.emitted_tokens),
    [256, 256, 88],
  );
  assert.equal(d.blocks[2].canvas_tokens, 256);
  assert.equal(d.mean_steps, 8);
});
test('missing, malformed and preempted traces never produce invented counts', () => {
  assert.equal(decodeDenoisingTrace(null, 512).status, 'unavailable');
  assert.equal(
    decodeDenoisingTrace(
      metrics([3], { diffusion_trace_version: undefined }),
      256,
    ).status,
    'unavailable',
  );
  const preempted = decodeDenoisingTrace(
    metrics([3], { diffusion_num_preemptions: 1 }),
    256,
  );
  assert.equal(preempted.status, 'unavailable');
  assert.deepEqual(preempted.blocks, []);
  for (const extra of [
    { per_step_drafted: [256] },
    { per_step_accepted: [0, 0, 0, 257] },
    { per_step_drafted: [256, 256, -1, 256] },
    { per_step_accepted: [0, 0, 256, 0] },
  ])
    assert.equal(
      decodeDenoisingTrace(metrics([3], extra), 256, { final: true }).status,
      'invalid',
    );
  assert.equal(decodeDenoisingTrace(metrics([49]), 256).status, 'invalid');
  assert.equal(decodeDenoisingTrace(metrics([3]), 512).status, 'invalid');
  assert.equal(decodeDenoisingTrace(metrics([3, 4]), 256).status, 'invalid');
  assert.equal(
    decodeDenoisingTrace(metrics([3, 4]), 257).blocks[1].emitted_tokens,
    1,
  );
});
test('fixed-step traces must match the requested count exactly', () => {
  assert.equal(
    decodeDenoisingTrace(metrics([16, 16]), 512, {
      mode: 'fixed',
      max_steps: 16,
    }).status,
    'available',
  );
  assert.equal(
    decodeDenoisingTrace(metrics([3, 16]), 512, {
      mode: 'fixed',
      max_steps: 16,
    }).status,
    'invalid',
  );
});
test('streaming cumulative snapshots update metadata without duplicating text or blocks', async () => {
  const frame = (text, tokens, steps) => ({
    choices: [{ delta: { content: text } }],
    usage: { completion_tokens: tokens, prompt_tokens: 20 },
    metrics: metrics(steps),
  });
  const frames = [
    frame('First. ', 256, [7]),
    frame('Second.', 310, [7, 19]),
    {
      choices: [],
      usage: { completion_tokens: 310, prompt_tokens: 20 },
      metrics: metrics([7, 19]),
    },
  ];
  const bytes = new TextEncoder().encode(
    frames.map((f) => `data: ${JSON.stringify(f)}\n\n`).join('') +
      'data: [DONE]\n\n',
  );
  const chunks = [];
  let clock = 0;
  const r = await collectCompletion(
    new ReadableStream({
      start(c) {
        c.enqueue(bytes);
        c.close();
      },
    }),
    {
      start: 0,
      now: () => ++clock * 100,
      denoisingOptions: { max_steps: 48 },
      emit: (chunk) => chunks.push(chunk),
    },
  );
  assert.equal(r.text, 'First. Second.');
  assert.equal(chunks[0].denoising.blocks.length, 1);
  assert.equal(chunks[1].denoising.blocks.length, 2);
  assert.equal(r.denoising.blocks.length, 2);
  assert.equal(r.denoising.mean_steps, 13);
  assert.equal(r.denoising.blocks[1].emitted_tokens, 54);
  assert.equal(r.post_first_block_tps, 54 / 0.1);
  assert.equal(r.denoising.final, true);
});
test('demo always uses adaptive48 and profiling validates fixed-step budget', () => {
  const input = {
    kind: 'demo',
    prompt: 'hello',
    batch_size: 1,
    max_tokens: 512,
    seed: 42,
    temperature: 0,
  };
  assert.equal(
    validate({ ...input, denoising_mode: 'fixed', denoising_steps: 16 })
      .denoising_mode,
    'adaptive',
  );
  assert.equal(validate(input).denoising_steps, 48);
  const profile = {
    ...input,
    kind: 'profile',
    batch_sizes: [1],
    repeats: 1,
    warmups: 0,
    denoising_mode: 'fixed',
  };
  assert.equal(validate(profile).denoising_steps, 16);
  assert.throws(
    () => validate({ ...profile, denoising_steps: 49 }),
    /denoising_steps/,
  );
  assert.throws(
    () => validate({ ...profile, denoising_steps: 0 }),
    /denoising_steps/,
  );
});
test('fixed-step runs send namespaced controls and reject silent server fallback', async () => {
  for (const suppliedTrace of [metrics([16, 16]), metrics([3, 16]), null]) {
    let sent;
    const run = await runExperiment(
      validate({
        kind: 'profile',
        prompt: 'hello',
        batch_size: 1,
        batch_sizes: [1],
        max_tokens: 512,
        seed: 42,
        temperature: 0,
        repeats: 1,
        warmups: 0,
        denoising_mode: 'fixed',
        denoising_steps: 16,
      }),
      { models: [{ id: 'diffusion', model: 'd', baseUrl: 'http://d/v1' }] },
      {
        signal: new AbortController().signal,
        emit() {},
        async fetcher(url, options) {
          sent = JSON.parse(options.body);
          const frame = {
            choices: [
              { delta: { content: 'answer' }, finish_reason: 'length' },
            ],
            usage: { completion_tokens: 512 },
            metrics: suppliedTrace,
          };
          return new Response(
            new ReadableStream({
              start(c) {
                c.enqueue(
                  new TextEncoder().encode(
                    `data: ${JSON.stringify(frame)}\n\ndata: [DONE]\n\n`,
                  ),
                );
                c.close();
              },
            }),
          );
        },
      },
    );
    assert.deepEqual(sent.vllm_xargs, {
      spark_lab_diffusion_max_steps: 16,
      spark_lab_diffusion_force_steps: 1,
      spark_lab_diffusion_preview: 0,
    });
    assert.equal(
      run.status,
      suppliedTrace?.speculative_decoding.per_step_accepted.length === 34
        ? 'complete'
        : 'partial',
    );
    if (run.status === 'partial') {
      assert.equal(run.summaries[0].valid, false);
      assert.equal(run.summaries[0].aggregate_tps, null);
      assert.equal(run.results[0].text, 'answer');
    }
  }
});
