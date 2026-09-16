import test from 'node:test';
import assert from 'node:assert/strict';
import { validate, runExperiment } from '../server/runner.mjs';
import { decodeDenoisingTrace } from '../server/denoising.mjs';
import { collectCompletion } from '../server/stream.mjs';
import { prepareCanvasRuntime, verifyCanvas } from '../server/canvas.mjs';
const base = {
  kind: 'demo',
  prompt: 'hello',
  max_tokens: 1024,
  batch_size: 1,
  seed: 42,
  temperature: 0,
};
const config = {
  runtime: { capacity: 4, canvas_length: 256 },
  models: [
    { id: 'diffusion', model: 'diffusion', baseUrl: 'http://diffusion/v1' },
  ],
};
const trace = (canvas, blocks = 2) => ({
  speculative_decoding: {
    diffusion_trace_version: 1,
    diffusion_num_preemptions: 0,
    per_step_accepted: Array.from({ length: blocks }, () => [0, canvas]).flat(),
    per_step_drafted: Array(blocks * 2).fill(canvas),
  },
});
const sse = (frames) =>
  new Response(
    frames.map((f) => 'data: ' + JSON.stringify(f) + '\n\n').join('') +
      'data: [DONE]\n\n',
  );
const frames = (canvas, blocks = 2) =>
  Array.from({ length: blocks }, (_, i) => ({
    choices: [
      {
        delta: { content: `Block ${i + 1}.` },
        token_ids: Array(canvas).fill(42),
      },
    ],
    usage: { prompt_tokens: 10, completion_tokens: canvas * (i + 1) },
    metrics: trace(canvas, i + 1),
  }));
const verified = (canvas) => ({
  verified: true,
  ready: true,
  canvas_length: canvas,
  pid: 123,
  starttime: 'test',
});
test('both views default to 256 and reject unsupported canvas lengths', () => {
  for (const kind of ['demo', 'profile']) {
    const input = { ...base, kind, batch_sizes: [1], warmups: 0, repeats: 1 };
    assert.equal(validate(input).canvas_length, 256);
    for (const canvas_length of [64, 128, 256, 512])
      assert.equal(
        validate({ ...input, canvas_length }).canvas_length,
        canvas_length,
      );
    for (const canvas_length of [0, 32, 129, 1024, 1.5, '128'])
      assert.throws(
        () => validate({ ...input, canvas_length }),
        /canvas_length/,
      );
  }
});
test('128-token blocks align partial final output; any wrong runtime width is invalid', () => {
  const result = decodeDenoisingTrace(trace(128), 200, {
    canvas_length: 128,
    final: true,
  });
  assert.equal(result.status, 'available');
  assert.equal(result.canvas_length, 128);
  assert.deepEqual(
    result.blocks.map((b) => b.emitted_tokens),
    [128, 72],
  );
  for (const requested of [64, 256, 512])
    assert.equal(
      decodeDenoisingTrace(trace(128), 200, { canvas_length: requested })
        .status,
      'invalid',
    );
});
test('512-token metrics exclude the entire first block; one block has no generation-only rate', async () => {
  for (const blocks of [1, 2]) {
    let time = 0;
    const r = await collectCompletion(sse(frames(512, blocks)).body, {
      start: 0,
      now: () => (time += 100),
      denoisingOptions: { canvas_length: 512, max_tokens: 1024 },
    });
    assert.equal(r.first_block_tokens, 512);
    assert.equal(r.post_first_block_tps, blocks === 1 ? null : 5120);
    assert.equal(r.denoising.status, 'available');
  }
});
test('preview bounds track canvas and output budget', async () => {
  for (const canvas of [64, 128, 512]) {
    const preview = (block, ids) => ({
      diffusion_preview: {
        version: 1,
        final: false,
        block_index: block,
        denoising_step: 1,
        text: 'draft',
        token_ids: Array(ids).fill(42),
      },
    });
    const previews = [];
    const r = await collectCompletion(
      sse([
        preview(1, canvas),
        preview(2, canvas + 1),
        preview(3, canvas),
        ...frames(canvas),
      ]).body,
      {
        start: 0,
        previewEnabled: true,
        emitPreview: (p) => previews.push(p),
        denoisingOptions: { canvas_length: canvas, max_tokens: canvas * 2 },
      },
    );
    assert.equal(previews.length, 1);
    assert.equal(previews[0].token_ids.length, canvas);
    assert.equal(r.diffusion_preview.dropped_frames, 2);
  }
});
test('failed or mismatched runtime preparation prevents all tokenizer/generation requests', async () => {
  for (const prepareRuntime of [
    async () => {
      throw Error('busy');
    },
    async () => verified(256),
  ]) {
    let calls = 0;
    await assert.rejects(
      () =>
        runExperiment(validate({ ...base, canvas_length: 128 }), config, {
          signal: new AbortController().signal,
          emit() {},
          fetcher() {
            calls++;
          },
          prepareRuntime,
        }),
      /busy|canvas length/,
    );
    assert.equal(calls, 0);
  }
  assert.throws(
    () => verifyCanvas({ ...verified(128), verified: false }, 128),
    /verified/,
  );
});
test('new runtime warms before timing with preview disabled and saves actual canvas independently of current config', async () => {
  const settings = validate({ ...base, canvas_length: 128 });
  const options = { signal: new AbortController().signal, emit() {} };
  const sent = [];
  const fetcher = async (_url, opts) => {
    sent.push(JSON.parse(opts.body));
    return sse(frames(128));
  };
  const runtime = await prepareCanvasRuntime(settings, config, {
    ...options,
    fetcher,
    command: async () => verified(128),
  });
  assert.equal(sent[0].vllm_xargs.spark_lab_diffusion_preview, 0);
  assert.equal(runtime.warmup.results[0].denoising.canvas_length, 128);
  const cached = await prepareCanvasRuntime(settings, config, {
    ...options,
    fetcher,
    command: async () => verified(128),
  });
  assert.equal(cached.warmup.status, 'already_warmed');
  assert.equal(sent.length, 1);
  const larger = await prepareCanvasRuntime(
    { ...settings, batch_sizes: [1, 2] },
    config,
    {
      ...options,
      fetcher,
      command: async () => verified(128),
    },
  );
  assert.equal(larger.warmup.status, 'complete');
  assert.equal(larger.warmup.results.length, 2);
  assert.equal(sent.length, 3);

  const run = await runExperiment(settings, config, {
    ...options,
    fetcher,
    prepareRuntime: async () => runtime,
  });
  assert.equal(run.status, 'complete');
  assert.equal(run.configuration.runtime.canvas_length, 128);
  assert.equal(config.runtime.canvas_length, 256);
  assert.equal(run.settings.canvas_length, 128);
  assert.equal(run.summaries[0].diffusion_canvas_length, 128);
  assert.equal(run.results[0].canvas_length_ok, true);
});
test('attestation alone cannot hide a different canvas in the completion trace', async () => {
  const run = await runExperiment(
    validate({ ...base, canvas_length: 512 }),
    config,
    {
      signal: new AbortController().signal,
      emit() {},
      prepareRuntime: async () => verified(512),
      fetcher: async () => sse(frames(256)),
    },
  );
  assert.equal(run.status, 'partial');
  assert.equal(run.results[0].canvas_length_ok, false);
  assert.equal(run.summaries[0].valid, false);
});

test('any failed configured warmup stops profiling before measured requests', async () => {
  for (const failure of ['http', 'short_output']) {
    let calls = 0;
    const run = await runExperiment(
      validate({
        ...base,
        kind: 'profile',
        batch_sizes: [1],
        repeats: 1,
        warmups: 1,
        output_mode: 'fixed',
        max_tokens: 16,
      }),
      {
        models: [{ id: 'autoregressive', model: 'a', baseUrl: 'http://a/v1' }],
      },
      {
        signal: new AbortController().signal,
        emit() {},
        fetcher: async () => {
          calls++;
          return failure === 'http'
            ? new Response('not ready', { status: 500 })
            : sse(frames(8, 1));
        },
      },
    );
    assert.equal(calls, 1);
    assert.equal(run.status, 'partial');
    assert.equal(run.results.length, 0);
    assert.equal(run.summaries.length, 0);
  }
});
