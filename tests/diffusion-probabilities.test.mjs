import test from 'node:test';
import assert from 'node:assert/strict';
import { collectCompletion } from '../server/stream.mjs';
import { validate, runExperiment } from '../server/runner.mjs';

const header = ['<|channel>', 'thought', '\n', '<channel|>'];
const headerIds = [100, 45518, 107, 101];
const record = (token, logprob = -0.5) => ({
  token,
  logprob,
  bytes: [...new TextEncoder().encode(token)],
  top_logprobs: [],
});
const frame = (content, tokens, ids, extra = {}) => ({
  choices: [
    {
      delta: { content },
      token_ids: ids,
      logprobs: { content: tokens.map((token) => record(token)) },
      ...extra,
    },
  ],
});
const body = (frames) =>
  new Response(
    frames
      .map((f) => `data: ${typeof f === 'string' ? f : JSON.stringify(f)}\n\n`)
      .join(''),
  ).body;
function options(extra = {}) {
  let time = 0;
  return {
    start: 0,
    now: () => (time += 100),
    tokenProbabilitiesEnabled: true,
    tokenProbabilitiesKind: 'diffusion',
    stripEmptyGemmaChannel: true,
    ...extra,
  };
}
const input = {
  kind: 'demo',
  prompt: 'hello',
  batch_size: 1,
  max_tokens: 24,
  seed: 42,
  temperature: 0,
};
const config = {
  models: [
    { id: 'diffusion', model: 'd', baseUrl: 'http://fixture-d/v1' },
    { id: 'autoregressive', model: 'a', baseUrl: 'http://fixture-a/v1' },
  ],
};

test('diffusion final distribution aligns committed content and audits verified hidden header/EOS', async () => {
  const text = 'A café ☕';
  const frames = [
    {
      ...frame(
        header.join('') + text,
        [...header, 'A', ' café', ' ☕', '<eos>'],
        [...headerIds, 20, 21, 22, 1],
        { finish_reason: 'stop', stop_reason: null },
      ),
      usage: { completion_tokens: 8, prompt_tokens: 12 },
    },
    '[DONE]',
  ];
  const deltas = [];
  const result = await collectCompletion(
    body(frames),
    options({ emitTokenProbabilities: (d) => deltas.push(structuredClone(d)) }),
  );
  const plain = await collectCompletion(
    body(frames),
    options({ tokenProbabilitiesEnabled: false }),
  );
  const p = result.diffusion_token_probabilities;
  assert.equal(result.text, text);
  assert.equal(result.ar_token_probabilities, undefined);
  assert.equal(p.status, 'available');
  assert.equal(p.logprobs_mode, 'final_denoising');
  assert.equal(p.tokens.length, 8);
  assert.deepEqual(p.omitted_token_indexes, [0, 1, 2, 3, 7]);
  assert.deepEqual(
    p.spans.map((s) => text.slice(s.start, s.end)),
    ['A', ' café', ' ☕'],
  );
  assert.deepEqual(
    p.spans.map((s) => s.token_indexes),
    [[4], [5], [6]],
  );
  for (const token of p.tokens) assert.equal(token.probability, Math.exp(-0.5));
  for (const key of [
    'text',
    'chunks',
    'ttft_ms',
    'elapsed_ms',
    'token_progress',
    'completion_tokens',
    'output_token_ids',
    'post_first_block_tps',
  ])
    assert.deepEqual(result[key], plain[key], key);
  assert.deepEqual(
    deltas.flatMap((d) => d.tokens),
    p.tokens,
  );
  assert.deepEqual(
    deltas.flatMap((d) => d.spans),
    p.spans,
  );
});

test('an empty Gemma header split across frames never colors hidden tokens or shifts visible offsets', async () => {
  const result = await collectCompletion(
    body([
      frame(
        header.slice(0, 2).join(''),
        header.slice(0, 2),
        headerIds.slice(0, 2),
      ),
      frame(
        header.slice(2).join('') + 'Answer',
        [...header.slice(2), 'Answer', '<eos>'],
        [...headerIds.slice(2), 30, 1],
        { finish_reason: 'stop', stop_reason: null },
      ),
      '[DONE]',
    ]),
    options(),
  );
  assert.equal(result.text, 'Answer');
  assert.equal(result.diffusion_token_probabilities.status, 'available');
  assert.deepEqual(result.diffusion_token_probabilities.spans, [
    { start: 0, end: 6, token_indexes: [4] },
  ]);
});

test('unverified hidden special tokens never receive invented diffusion probabilities', async () => {
  for (const frames of [
    [frame(header.join('') + 'A', [...header, 'A'], [9, 8, 7, 6, 20])],
    [
      frame('A', ['A', '<eos>'], [20, 9], {
        finish_reason: 'stop',
        stop_reason: null,
      }),
    ],
    [
      frame('A', ['A', '<eos>'], [20, 1], {
        finish_reason: 'length',
        stop_reason: null,
      }),
    ],
  ]) {
    const result = await collectCompletion(
      body([...frames, '[DONE]']),
      options(),
    );
    assert.equal(result.text, 'A');
    assert.equal(result.diffusion_token_probabilities.status, 'invalid');
  }
  const visible = 'A <eos>';
  const result = await collectCompletion(
    body([
      frame(visible, ['A ', '<eos>'], [20, 1], { finish_reason: 'stop' }),
      '[DONE]',
    ]),
    options(),
  );
  assert.equal(result.text, visible);
  assert.equal(result.diffusion_token_probabilities.status, 'available');
  assert.equal(
    result.diffusion_token_probabilities.spans.at(-1).end,
    visible.length,
  );
});

test('provisional purple preview frames remain separate from final probability metadata', async () => {
  const previews = [];
  const provisional = {
    version: 1,
    block_index: 1,
    denoising_step: 2,
    token_ids: [40, 41],
    text: 'changing guess',
    final: false,
  };
  const result = await collectCompletion(
    body([
      { choices: [], diffusion_preview: provisional },
      frame('Final', ['Final'], [50]),
      '[DONE]',
    ]),
    options({ previewEnabled: true, emitPreview: (p) => previews.push(p) }),
  );
  assert.equal(result.text, 'Final');
  assert.deepEqual(previews, [provisional]);
  assert.equal(result.diffusion_preview.received_frames, 1);
  assert.equal(result.diffusion_token_probabilities.tokens.length, 1);
  assert.equal(result.diffusion_token_probabilities.tokens[0].token, 'Final');
});

test('diffusion probability selection is boolean, independent of AR, and forced off in profiling', () => {
  assert.equal(validate(input).diffusion_token_probabilities, false);
  assert.equal(
    validate({ ...input, diffusion_token_probabilities: true })
      .diffusion_token_probabilities,
    true,
  );
  for (const value of ['true', 1, 0, {}, null])
    assert.throws(
      () => validate({ ...input, diffusion_token_probabilities: value }),
      /boolean/,
    );
  const settings = validate({
    ...input,
    kind: 'profile',
    batch_sizes: [1],
    repeats: 1,
    warmups: 1,
    diffusion_token_probabilities: true,
  });
  assert.equal(settings.diffusion_token_probabilities, false);
});

test('each model opts in independently and diffusion metadata retains request identity', async () => {
  for (const diffusionEnabled of [false, true]) {
    const sent = [],
      events = [];
    const run = await runExperiment(
      validate({
        ...input,
        diffusion_token_probabilities: diffusionEnabled,
        ar_token_probabilities: false,
      }),
      config,
      {
        signal: new AbortController().signal,
        emit: (e) => events.push(e),
        fetcher: async (_, request) => {
          sent.push(JSON.parse(request.body));
          return new Response(
            body([frame('Answer', ['Answer'], [10]), '[DONE]']),
          );
        },
      },
    );
    for (const request of sent) {
      assert.equal(
        request.logprobs,
        request.model === 'd' && diffusionEnabled ? true : undefined,
      );
      assert.equal(
        request.top_logprobs,
        request.model === 'd' && diffusionEnabled ? 0 : undefined,
      );
    }
    const d = run.results.find((r) => r.model === 'diffusion');
    const probabilityEvents = events.filter(
      (e) => e.type === 'token_probabilities',
    );
    assert.equal(
      d.diffusion_token_probabilities?.status,
      diffusionEnabled ? 'available' : undefined,
    );
    assert.equal(d.ar_token_probabilities, undefined);
    assert.equal(
      run.results.find((r) => r.model === 'autoregressive')
        .diffusion_token_probabilities,
      undefined,
    );
    if (diffusionEnabled) assert.ok(probabilityEvents.length > 0);
    for (const event of probabilityEvents) {
      assert.equal(event.model, 'diffusion');
      assert.equal(event.index, d.index);
      assert.equal(event.request_id, d.request_id);
      assert.equal(event.logprobs_mode, 'final_denoising');
    }
  }
});

test('both probability collectors and previews stay off for raw/chat profiling and all warmups', async () => {
  for (const workload of ['raw', 'continuation']) {
    const settings = validate({
      ...input,
      kind: 'profile',
      workload,
      batch_sizes: [1],
      repeats: 1,
      warmups: 1,
      dataset: [{ prompt: 'Warm' }, { prompt: 'Measure' }],
    });
    Object.assign(settings, {
      diffusion_token_probabilities: true,
      ar_token_probabilities: true,
      diffusion_preview: true,
    });
    const sent = [],
      events = [];
    const run = await runExperiment(settings, config, {
      signal: new AbortController().signal,
      emit: (e) => events.push(e),
      fetcher: async (_, request) => {
        sent.push(JSON.parse(request.body));
        return new Response(body([frame('A', ['A'], [10]), '[DONE]']));
      },
    });
    assert.equal(sent.length, 4);
    for (const request of sent) {
      assert.equal(request.logprobs, undefined);
      assert.equal(request.top_logprobs, undefined);
      if (request.model === 'd')
        assert.equal(request.vllm_xargs.spark_lab_diffusion_preview, 0);
    }
    assert.equal(run.settings.diffusion_token_probabilities, false);
    assert.equal(run.settings.ar_token_probabilities, false);
    assert.equal(
      events.filter(
        (e) => e.type === 'token_probabilities' || e.type === 'preview',
      ).length,
      0,
    );
    assert.ok(
      [...run.results, ...run.warmup_results].every(
        (r) => !r.diffusion_token_probabilities && !r.ar_token_probabilities,
      ),
    );
  }
});

test('an interrupted diffusion stream preserves attributed final text without estimating missing metrics', async () => {
  const run = await runExperiment(
    validate({ ...input, diffusion_token_probabilities: true }),
    { models: [config.models[0]] },
    {
      signal: new AbortController().signal,
      emit: () => {},
      fetcher: async () =>
        new Response(
          body([
            frame(
              header.join('') + 'Partial',
              [...header, 'Partial'],
              [...headerIds, 20],
            ),
          ]),
        ),
    },
  );
  const result = run.results[0];
  assert.equal(result.status, 'error');
  assert.equal(result.text, 'Partial');
  assert.equal(result.completion_tokens, null);
  assert.equal(result.tokens_per_second, null);
  assert.equal(result.ar_token_probabilities, undefined);
  assert.equal(result.diffusion_token_probabilities.status, 'partial');
  assert.deepEqual(result.diffusion_token_probabilities.spans, [
    { start: 0, end: 7, token_indexes: [4] },
  ]);
});

test('direct demo callers cannot enable diffusion scores on warmups or through nonboolean values', async () => {
  for (const value of [true, 'true', 1]) {
    const settings = validate(input);
    settings.diffusion_token_probabilities = value;
    settings.warmups = 1;
    const sent = [];
    const run = await runExperiment(
      settings,
      { models: [config.models[0]] },
      {
        signal: new AbortController().signal,
        emit: () => {},
        fetcher: async (_, request) => {
          sent.push(JSON.parse(request.body));
          return new Response(body([frame('A', ['A'], [10]), '[DONE]']));
        },
      },
    );
    assert.equal(sent.length, 2);
    assert.equal(sent[0].logprobs, undefined);
    assert.equal(sent[1].logprobs, value === true ? true : undefined);
    assert.equal(
      run.warmup_results[0].diffusion_token_probabilities,
      undefined,
    );
  }
});
