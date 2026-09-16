import test from 'node:test';
import assert from 'node:assert/strict';
import { collectCompletion } from '../server/stream.mjs';
import { validate, runExperiment } from '../server/runner.mjs';

const input = {
  kind: 'demo',
  prompt: 'hello',
  batch_size: 1,
  max_tokens: 512,
  seed: 42,
  temperature: 0,
};
const prediction = (block = 1, step = 1, extra = {}) => ({
  version: 1,
  block_index: block,
  denoising_step: step,
  token_ids: [10, 20],
  text: 'Provisional draft',
  final: false,
  ...extra,
});
const metrics = {
  speculative_decoding: {
    diffusion_trace_version: 1,
    diffusion_num_preemptions: 0,
    per_step_accepted: [0, 256],
    per_step_drafted: [256, 256],
  },
};
const content = (text, count, extra = {}) => ({
  choices: [{ delta: { content: text }, token_ids: [count] }],
  usage: { completion_tokens: count, prompt_tokens: 12 },
  ...extra,
});
function stream(frames) {
  return new Response(
    frames
      .map(
        (frame) =>
          `data: ${typeof frame === 'string' ? frame : JSON.stringify(frame)}\n\n`,
      )
      .join(''),
  ).body;
}
function clock() {
  let t = 0;
  return () => (t += 100);
}

test('preview settings are opt-in, strictly boolean and always off for profiling', () => {
  assert.equal(validate(input).diffusion_preview, false);
  assert.equal(
    validate({ ...input, diffusion_preview: true }).diffusion_preview,
    true,
  );
  assert.throws(
    () => validate({ ...input, diffusion_preview: 'true' }),
    /boolean/,
  );
  assert.equal(
    validate({
      ...input,
      kind: 'profile',
      batch_sizes: [1],
      repeats: 1,
      warmups: 1,
      diffusion_preview: true,
    }).diffusion_preview,
    false,
  );
});

test('preview snapshots never alter committed text, counts, chunks or timing bookkeeping', async () => {
  const committed = [content('First ', 256), content('second.', 512), '[DONE]'];
  const baseline = await collectCompletion(stream(committed), {
    start: 0,
    now: clock(),
  });
  const snapshots = [],
    emitted = [];
  const result = await collectCompletion(
    stream([
      { choices: [], diffusion_preview: prediction() },
      committed[0],
      { choices: [], diffusion_preview: prediction(2) },
      committed[1],
      committed[2],
    ]),
    {
      start: 0,
      now: clock(),
      previewEnabled: true,
      emitPreview: (frame) => snapshots.push(frame),
      emit: (chunk) => emitted.push(chunk),
    },
  );
  const { diffusion_preview, ...actual } = result;
  assert.deepEqual(actual, baseline);
  assert.equal(diffusion_preview.received_frames, 2);
  assert.equal(snapshots.length, 2);
  assert.deepEqual(
    emitted,
    baseline.chunks.map((chunk) => ({ ...chunk, ttft_ms: 100 })),
  );
  assert.equal(result.output_token_ids.length, 2);
  assert.ok(!JSON.stringify(result.chunks).includes('Provisional'));
});

test('mixed committed frames preserve real output once and reject stale preview metadata', async () => {
  const previews = [];
  const result = await collectCompletion(
    stream([
      content('Final.', 256, { metrics, diffusion_preview: prediction() }),
      '[DONE]',
    ]),
    {
      start: 0,
      now: clock(),
      previewEnabled: true,
      emitPreview: (p) => previews.push(p),
      denoisingOptions: { max_steps: 48 },
    },
  );
  assert.equal(result.text, 'Final.');
  assert.equal(result.completion_tokens, 256);
  assert.equal(result.denoising.blocks[0].denoising_steps, 1);
  assert.equal(previews.length, 0);
});

test('preview hides only a verified initial empty Gemma channel and preserves token IDs', async () => {
  const exact = [100, 45518, 107, 101, 9259];
  const cases = [
    { ids: exact, block: 1, text: 'thought\nHello', expected: 'Hello' },
    { ids: [45518, 107, 9259], block: 1, text: 'thought\nHello', expected: 'thought\nHello' },
    { ids: [100, 45518, 107, 9259], block: 1, text: 'thought\nHello', expected: 'thought\nHello' },
    { ids: exact, block: 2, text: 'thought\nHello', expected: 'thought\nHello' },
    { ids: exact, block: 1, text: 'Different text', expected: 'Different text' },
  ];
  for (const c of cases) {
    const received = [];
    await collectCompletion(stream([
      { diffusion_preview: prediction(c.block, 1, { token_ids: c.ids, text: c.text }) },
      content('Final', 1), '[DONE]',
    ]), { start: 0, now: clock(), previewEnabled: true,
      stripEmptyGemmaChannel: true, emitPreview: p => received.push(p) });
    assert.equal(received[0].text, c.expected);
    assert.deepEqual(received[0].token_ids, c.ids);
  }
});

test('malformed, duplicate and reordered snapshots are dropped without damaging output', async () => {
  const frames = [
    prediction(1, 2),
    prediction(1, 2),
    prediction(1, 1),
    prediction(1, 3, { version: 2 }),
    prediction(1, 3, { token_ids: [-1] }),
    prediction(1, 49),
    prediction(0, 3),
    prediction(1, 3, { token_ids: Array(257).fill(0) }),
    prediction(1, 3, { text: {} }),
    prediction(1, 3),
  ];
  const received = [];
  const result = await collectCompletion(
    stream([
      ...frames.map((diffusion_preview) => ({
        choices: [],
        diffusion_preview,
      })),
      content('Real output.', 10),
      '[DONE]',
    ]),
    {
      start: 0,
      now: clock(),
      previewEnabled: true,
      emitPreview: (p) => received.push(p),
    },
  );
  assert.deepEqual(
    received.map((p) => p.denoising_step),
    [2, 3],
  );
  assert.equal(result.diffusion_preview.dropped_frames, 8);
  assert.equal(result.text, 'Real output.');
});

test('preemption invalidates the live preview permanently, not committed output', async () => {
  const previews = [];
  const result = await collectCompletion(
    stream([
      { diffusion_preview: prediction() },
      {
        diffusion_preview: {
          version: 1,
          attribution_valid: false,
          reason: 'request_preempted',
        },
      },
      { diffusion_preview: prediction(1, 2) },
      content('Real.', 10),
      '[DONE]',
    ]),
    {
      start: 0,
      now: clock(),
      previewEnabled: true,
      emitPreview: (p) => previews.push(p),
    },
  );
  assert.equal(previews.length, 2);
  assert.deepEqual(previews[1], {
    unavailable: true,
    reason: 'request_preempted',
  });
  assert.equal(result.diffusion_preview.unavailable, 'request_preempted');
  assert.equal(result.text, 'Real.');
});

test('preview-only streams still contain no completed model output', async () => {
  await assert.rejects(
    () =>
      collectCompletion(
        stream([{ diffusion_preview: prediction() }, '[DONE]']),
        { start: 0, previewEnabled: true },
      ),
    /no generated text/,
  );
});

test('runner enforces profile gate for every warmup and measured request even when validation is bypassed', async () => {
  for (const workload of ['continuation', 'raw'])
    for (const mode of ['adaptive', 'fixed']) {
      const settings = validate({
        ...input,
        kind: 'profile',
        batch_sizes: [1],
        repeats: 1,
        warmups: 1,
        dataset: [{ prompt: 'warmup' }, { prompt: 'measure' }],
        workload,
        denoising_mode: mode,
        denoising_steps: 1,
      });
      // Adversarial direct caller bypasses validate's forced-false normalization.
      settings.diffusion_preview = true;
      const sent = [],
        events = [];
      const run = await runExperiment(
        settings,
        {
          models: [
            { id: 'diffusion', model: 'd', baseUrl: 'http://d/v1' },
            { id: 'autoregressive', model: 'a', baseUrl: 'http://a/v1' },
          ],
        },
        {
          signal: new AbortController().signal,
          emit: (e) => events.push(e),
          fetcher: async (url, options) => {
            const payload = JSON.parse(options.body);
            sent.push({ url, payload });
            return new Response(
              stream([
                { choices: [], diffusion_preview: prediction() },
                content('Final.', 256, { metrics }),
                '[DONE]',
              ]),
            );
          },
        },
      );
      assert.equal(run.status, 'complete');
      assert.equal(run.settings.diffusion_preview, false);
      assert.equal(sent.length, 4);
      for (const { payload, url } of sent) {
        if (payload.model === 'd')
          assert.equal(payload.vllm_xargs.spark_lab_diffusion_preview, 0);
        else assert.equal(payload.vllm_xargs, undefined);
        assert.ok(
          url.endsWith(
            workload === 'raw' ? '/completions' : '/chat/completions',
          ),
        );
      }
      assert.equal(events.filter((e) => e.type === 'preview').length, 0);
      assert.ok(
        [...run.results, ...run.warmup_results].every(
          (r) => !r.diffusion_preview,
        ),
      );
    }
});

test('demo preview on/off flags route only diffusion frames with the matching request identity', async () => {
  for (const enabled of [true, false]) {
    const events = [],
      sent = [];
    const run = await runExperiment(
      validate({ ...input, diffusion_preview: enabled }),
      {
        models: [{ id: 'diffusion', model: 'd', baseUrl: 'http://d/v1' }],
      },
      {
        signal: new AbortController().signal,
        emit: (e) => events.push(e),
        fetcher: async (url, options) => {
          sent.push(JSON.parse(options.body));
          return new Response(
            stream([
              { choices: [], diffusion_preview: prediction() },
              content('Final.', 256, { metrics }),
              '[DONE]',
            ]),
          );
        },
      },
    );
    assert.equal(
      sent[0].vllm_xargs.spark_lab_diffusion_preview,
      enabled ? 1 : 0,
    );
    const previews = events.filter((e) => e.type === 'preview');
    assert.equal(previews.length, enabled ? 1 : 0);
    if (enabled)
      assert.equal(previews[0].request_id, run.results[0].request_id);
    assert.ok(
      events
        .filter((e) => ['start', 'chunk'].includes(e.type))
        .every((e) => e.request_id === run.results[0].request_id),
    );
    assert.equal(run.results[0].text, 'Final.');
  }
});

test('empty-choice preview frames with pending trace steps are not final output', async () => {
  const pendingMetrics = {
    speculative_decoding: {
      ...metrics.speculative_decoding,
      per_step_accepted: [0, 256, 0, 0],
      per_step_drafted: [256, 256, 256, 256],
    },
  };
  const completeMetrics = {
    speculative_decoding: {
      ...metrics.speculative_decoding,
      per_step_accepted: [0, 256, 0, 0, 256],
      per_step_drafted: [256, 256, 256, 256, 256],
    },
  };
  const snapshots = [],
    chunks = [];
  const r = await collectCompletion(
    stream([
      content('Block one. ', 256, { metrics }),
      {
        choices: [],
        usage: { completion_tokens: 256 },
        metrics: pendingMetrics,
        diffusion_preview: prediction(2, 2),
      },
      content('Block two.', 512, { metrics: completeMetrics }),
      '[DONE]',
    ]),
    {
      start: 0,
      now: clock(),
      previewEnabled: true,
      denoisingOptions: { max_steps: 48 },
      emitPreview: (p) => snapshots.push(p),
      emit: (c) => chunks.push(c),
    },
  );
  assert.equal(snapshots.length, 1);
  assert.equal(chunks[1].denoising.status, 'available');
  assert.equal(chunks[1].denoising.final, false);
  assert.equal(r.denoising.status, 'available');
  assert.deepEqual(
    r.denoising.blocks.map((b) => b.denoising_steps),
    [1, 2],
  );
});
