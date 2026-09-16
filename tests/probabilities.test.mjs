import test from 'node:test';
import assert from 'node:assert/strict';
import { collectCompletion } from '../server/stream.mjs';
import { validate, runExperiment } from '../server/runner.mjs';

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
const token = (text, logprob = -0.5, extra = {}) => ({
  token: text,
  bytes: [...new TextEncoder().encode(text)],
  logprob,
  top_logprobs: [],
  ...extra,
});
const frame = (text, records, choice = {}, extra = {}) => ({
  choices: [
    {
      delta: { content: text },
      ...(records === undefined ? {} : { logprobs: { content: records } }),
      ...choice,
    },
  ],
  ...extra,
});
function stream(frames, split = 7) {
  const bytes = new TextEncoder().encode(
    frames
      .map(
        (f) => `data: ${typeof f === 'string' ? f : JSON.stringify(f)}\r\n\r\n`,
      )
      .join(''),
  );
  return new ReadableStream({
    start(c) {
      for (let i = 0; i < bytes.length; i += split)
        c.enqueue(bytes.slice(i, i + split));
      c.close();
    },
  });
}
function clock() {
  let time = 0;
  return () => (time += 100);
}
async function collect(frames, options = {}) {
  return collectCompletion(stream(frames, 1), {
    start: 0,
    now: clock(),
    tokenProbabilitiesEnabled: true,
    ...options,
  });
}
function assertMetadata(value) {
  assert.equal(value.version, 1);
  assert.equal(value.source, 'choice.logprobs.content');
  assert.equal(value.logprobs_mode, 'raw');
  assert.equal(value.offset_unit, 'utf16');
}
function assertAppendOnly(deltas, final) {
  assert.ok(deltas.length > 0);
  const tokens = [],
    spans = [];
  for (const delta of deltas) {
    assertMetadata(delta);
    assert.equal(typeof delta.final, 'boolean');
    for (const item of delta.tokens) {
      assert.equal(
        item.index,
        tokens.length,
        'token deltas append once in order',
      );
      tokens.push(item);
    }
    for (const span of delta.spans) {
      assert.ok(span.end > span.start);
      assert.ok(span.start >= (spans.at(-1)?.end ?? 0));
      assert.ok(span.token_indexes.every((index) => index < tokens.length));
      spans.push(span);
    }
  }
  assert.equal(deltas.filter((delta) => delta.final).length, 1);
  assert.equal(deltas.at(-1).final, true);
  assert.equal(deltas.at(-1).status, final.status);
  assert.deepEqual(tokens, final.tokens);
  assert.deepEqual(spans, final.spans);
}

test('coalesced native records retain sampled probabilities and UTF-16 spans across split UTF-8 transport', async () => {
  const deltas = [];
  const result = await collect(
    [
      frame(
        'café 🦉!',
        [token('café', -0.2), token(' ', -1), token('🦉', -2), token('!', 0)],
        { token_ids: [10, 11, 12, 13] },
        { usage: { completion_tokens: 4, prompt_tokens: 9 } },
      ),
      '[DONE]',
    ],
    { emitTokenProbabilities: (delta) => deltas.push(structuredClone(delta)) },
  );
  const probabilities = result.ar_token_probabilities;
  assertMetadata(probabilities);
  assert.equal(probabilities.status, 'available');
  assert.equal(result.text, 'café 🦉!');
  assert.deepEqual(
    probabilities.tokens.map((record) => ({
      index: record.index,
      token: record.token,
      bytes: record.bytes,
      token_id: record.token_id,
      logprob: record.logprob,
      probability: record.probability,
    })),
    [
      {
        index: 0,
        token: 'café',
        bytes: [99, 97, 102, 195, 169],
        token_id: 10,
        logprob: -0.2,
        probability: Math.exp(-0.2),
      },
      {
        index: 1,
        token: ' ',
        bytes: [32],
        token_id: 11,
        logprob: -1,
        probability: Math.exp(-1),
      },
      {
        index: 2,
        token: '🦉',
        bytes: [240, 159, 166, 137],
        token_id: 12,
        logprob: -2,
        probability: Math.exp(-2),
      },
      {
        index: 3,
        token: '!',
        bytes: [33],
        token_id: 13,
        logprob: 0,
        probability: 1,
      },
    ],
  );
  assert.deepEqual(probabilities.spans, [
    { start: 0, end: 4, token_indexes: [0] },
    { start: 4, end: 5, token_indexes: [1] },
    { start: 5, end: 7, token_indexes: [2] },
    { start: 7, end: 8, token_indexes: [3] },
  ]);
  assertAppendOnly(deltas, probabilities);
});

test('empty decoded records group with the next visible token and remain separate probability samples', async () => {
  const deltas = [],
    originals = [],
    snapshots = [];
  const result = await collect(
    [
      frame('', [token('', -3)], { token_ids: [40] }),
      frame('é', [token('', -4), token('é', -0.25)], { token_ids: [41, 42] }),
      frame('🦉', [token('', -5), token('🦉', -0.5)], { token_ids: [43, 44] }),
      '[DONE]',
    ],
    {
      emitTokenProbabilities(delta) {
        originals.push(delta);
        snapshots.push(structuredClone(delta));
        deltas.push(structuredClone(delta));
      },
    },
  );
  assert.equal(result.text, 'é🦉');
  assert.equal(result.ar_token_probabilities.status, 'available');
  assert.equal(result.ar_token_probabilities.tokens.length, 5);
  assert.deepEqual(result.ar_token_probabilities.spans, [
    { start: 0, end: 1, token_indexes: [0, 1, 2] },
    { start: 1, end: 3, token_indexes: [3, 4] },
  ]);
  assert.deepEqual(
    originals,
    snapshots,
    'later alignment must not mutate earlier events',
  );
  assertAppendOnly(deltas, result.ar_token_probabilities);
});

test('metadata-only stop records preserve ordinary chunks, counts and clocks', async () => {
  const frames = [
    { choices: [{ delta: { role: 'assistant' } }] },
    frame(
      'A',
      [token('A', -0.3)],
      { token_ids: [7] },
      { usage: { completion_tokens: 1 } },
    ),
    frame('', [token('<eos>', -0.1)], {
      token_ids: [2],
      finish_reason: 'stop',
      stop_reason: 2,
    }),
    { choices: [], usage: { completion_tokens: 1, prompt_tokens: 4 } },
    '[DONE]',
  ];
  const baseline = await collect(frames, { tokenProbabilitiesEnabled: false });
  const deltas = [],
    chunks = [];
  const result = await collect(frames, {
    emit: (chunk) => chunks.push(chunk),
    emitTokenProbabilities: (delta) => deltas.push(delta),
  });
  const { ar_token_probabilities, ...ordinary } = result;
  assert.deepEqual(ordinary, baseline);
  assert.equal(ar_token_probabilities.status, 'available');
  assert.equal(chunks.length, 1);
  assert.equal(result.ttft_ms, 100);
  assert.equal(result.elapsed_ms, 200);
  assertAppendOnly(deltas, ar_token_probabilities);
});

test('probability collection preserves exact output, latency, token progress and chunk events', async () => {
  const frames = [
    { choices: [{ delta: { role: 'assistant' } }] },
    { choices: [{ delta: { reasoning_content: 'reason' } }] },
    frame(
      'First ',
      [token('First', -0.2), token(' ', -0.7)],
      { token_ids: [1, 2] },
      { usage: { completion_tokens: 2 } },
    ),
    frame(
      'café 🦉',
      [token('café', -0.3), token(' ', -0.4), token('🦉', -0.8)],
      { token_ids: [3, 4, 5] },
      { usage: { completion_tokens: 5 } },
    ),
    {
      choices: [{ delta: {}, finish_reason: 'stop' }],
      usage: { completion_tokens: 5, prompt_tokens: 12 },
    },
    '[DONE]',
  ];
  const baselineChunks = [],
    enabledChunks = [],
    disabledEvents = [];
  const baseline = await collect(frames, {
    tokenProbabilitiesEnabled: false,
    emit: (chunk) => baselineChunks.push(chunk),
    emitTokenProbabilities: (delta) => disabledEvents.push(delta),
  });
  const enabled = await collect(frames, {
    emit: (chunk) => enabledChunks.push(chunk),
  });
  const { ar_token_probabilities, ...ordinary } = enabled;
  assert.deepEqual(ordinary, baseline);
  assert.deepEqual(enabledChunks, baselineChunks);
  assert.deepEqual(disabledEvents, []);
  assert.equal(ar_token_probabilities.status, 'available');
  assert.equal(enabled.text, 'First café 🦉');
  assert.equal(enabled.ttft_ms, 200);
  assert.equal(enabled.elapsed_ms, 400);
  assert.deepEqual(enabled.token_progress, [
    { at_ms: 200, completion_tokens: 2 },
    { at_ms: 300, completion_tokens: 5 },
  ]);
});

test('only a proven trailing stop token may be unspanned, and visible EOS text remains exact', async () => {
  for (const visible of [false, true]) {
    const text = visible ? 'A<eos>' : 'A';
    const result = await collect([
      frame(text, [token('A'), token('<eos>', -0.1)], {
        token_ids: [10, 2],
        finish_reason: 'stop',
        stop_reason: 2,
      }),
      '[DONE]',
    ]);
    assert.equal(result.text, text);
    assert.equal(result.ar_token_probabilities.status, 'available');
    assert.equal(result.ar_token_probabilities.tokens.length, 2);
    assert.equal(result.ar_token_probabilities.tokens[1].token_id, 2);
    assert.deepEqual(result.ar_token_probabilities.spans, [
      { start: 0, end: 1, token_indexes: [0] },
      ...(visible ? [{ start: 1, end: 6, token_indexes: [1] }] : []),
    ]);
  }
});

test('unproven, nonterminal and mismatched stop tokens invalidate attribution without altering text', async () => {
  const cases = [
    { token_ids: [10, 2], finish_reason: 'length', stop_reason: 2 },
    { token_ids: [10, 2], finish_reason: 'stop', stop_reason: 3 },
    { token_ids: [10, 2], finish_reason: 'stop', stop_reason: '2' },
    { finish_reason: 'stop', stop_reason: 2 },
    { token_ids: [10], finish_reason: 'stop', stop_reason: 2 },
  ];
  for (const choice of cases) {
    const result = await collect([
      frame('A', [token('A'), token('<eos>')], choice),
      '[DONE]',
    ]);
    assert.equal(result.text, 'A');
    assert.equal(result.ar_token_probabilities.status, 'invalid');
    assert.equal(typeof result.ar_token_probabilities.reason, 'string');
  }
  const interior = await collect([
    frame('AB', [token('A'), token('<eos>'), token('B')], {
      token_ids: [10, 2, 11],
      finish_reason: 'stop',
      stop_reason: 2,
    }),
    '[DONE]',
  ]);
  assert.equal(interior.text, 'AB');
  assert.equal(interior.ar_token_probabilities.status, 'invalid');
});

test('absent token IDs remain optional, while mismatched ID arrays invalidate attribution', async () => {
  for (const ids of [undefined, [10], [10, 11, 12]]) {
    const result = await collect([
      frame('AB', [token('A'), token('B')], ids ? { token_ids: ids } : {}),
      '[DONE]',
    ]);
    assert.equal(
      result.ar_token_probabilities.status,
      ids ? 'invalid' : 'available',
    );
    assert.ok(
      result.ar_token_probabilities.tokens.every(
        (record) => record.token_id == null,
      ),
    );
  }
});

test('missing probabilities are unavailable and mixed attribution gaps never fabricate samples', async () => {
  const unavailable = await collect([frame('Missing metadata'), '[DONE]']);
  assert.equal(unavailable.ar_token_probabilities.status, 'unavailable');
  assert.deepEqual(unavailable.ar_token_probabilities.tokens, []);
  assert.deepEqual(unavailable.ar_token_probabilities.spans, []);
  const partial = await collect([
    frame('A', [token('A')]),
    frame('B'),
    '[DONE]',
  ]);
  assert.equal(partial.text, 'AB');
  assert.equal(partial.ar_token_probabilities.status, 'invalid');
  assert.equal(partial.ar_token_probabilities.tokens.length, 1);
  for (const frames of [
    [frame('A'), frame('A', [token('A')]), '[DONE]'],
    [frame('A', [token('A')]), frame('A'), frame('A', [token('A')]), '[DONE]'],
  ]) {
    const result = await collect(frames);
    assert.equal(result.ar_token_probabilities.status, 'invalid');
    assert.equal(result.text, frames.length === 3 ? 'AA' : 'AAA');
    assert.ok(
      result.ar_token_probabilities.spans.every((span) => span.start !== 1),
      'later scored tokens must not be assigned to the same-looking unscored middle character',
    );
  }
});

test('sentinel logprobs are explicitly unavailable, while ordinary underflow remains a numeric probability', async () => {
  const result = await collect([
    frame('ABCD', [
      token('A', -9999),
      token('B', -10000),
      token('C', -800),
      token('D', 0),
    ]),
    '[DONE]',
  ]);
  assert.equal(result.ar_token_probabilities.status, 'partial');
  assert.deepEqual(
    result.ar_token_probabilities.tokens.map((record) => record.probability),
    [null, null, 0, 1],
  );
  assert.deepEqual(
    result.ar_token_probabilities.tokens.map((record) => record.logprob),
    [-9999, -10000, -800, 0],
  );
});

test('combining marks and ZWJ emoji share one span without normalizing committed text', async () => {
  for (const records of [
    [token('e'), token('\u0301')],
    [token('👩'), token('\u200d'), token('🔬')],
  ]) {
    const deltas = [];
    const text = records.map((record) => record.token).join('');
    const result = await collect(
      [...records.map((record) => frame(record.token, [record])), '[DONE]'],
      {
        emitTokenProbabilities: (delta) => deltas.push(structuredClone(delta)),
      },
    );
    assert.equal(result.text, text);
    assert.equal(result.ar_token_probabilities.status, 'available');
    assert.deepEqual(result.ar_token_probabilities.spans, [
      {
        start: 0,
        end: text.length,
        token_indexes: records.map((record, index) => index),
      },
    ]);
    assertAppendOnly(deltas, result.ar_token_probabilities);
  }
});

test('malformed native metadata and mismatches invalidate coloring, never model completion', async () => {
  const cases = [
    { content: 'bad' },
    { content: [null] },
    { content: [token('A', 'bad')] },
    { content: [token('A', null)] },
    { content: [token('A', 0.1)] },
    { content: [token('A', -1, { token: 12 })] },
    { content: [token('A', -1, { bytes: [-1] })] },
    { content: [token('A', -1, { bytes: [256] })] },
    { content: [token('A', -1, { bytes: [65.5] })] },
    { content: [token('A', -1, { bytes: [66] })] },
    { content: [token('B')] },
  ];
  for (const logprobs of cases) {
    const result = await collect([
      frame('A', undefined, { logprobs }, { usage: { completion_tokens: 1 } }),
      '[DONE]',
    ]);
    assert.equal(result.text, 'A');
    assert.equal(result.completion_tokens, 1);
    assert.equal(result.tokens_per_second, 5);
    assert.equal(
      result.ar_token_probabilities.status,
      'invalid',
      JSON.stringify(logprobs),
    );
    assert.equal(typeof result.ar_token_probabilities.reason, 'string');
  }
});

test('probabilities are opt-in and strictly boolean, and validation always disables profiling', () => {
  assert.equal(validate(input).ar_token_probabilities, false);
  assert.equal(
    validate({ ...input, ar_token_probabilities: false })
      .ar_token_probabilities,
    false,
  );
  assert.equal(
    validate({ ...input, ar_token_probabilities: true }).ar_token_probabilities,
    true,
  );
  for (const value of ['true', 1, 0, {}, [], null])
    assert.throws(
      () => validate({ ...input, ar_token_probabilities: value }),
      /boolean/,
    );
  assert.equal(
    validate({
      ...input,
      kind: 'profile',
      batch_sizes: [1],
      repeats: 1,
      warmups: 1,
      ar_token_probabilities: true,
    }).ar_token_probabilities,
    false,
  );
});

test('demo requests and live events route probabilities only to the opted-in autoregressive request', async () => {
  for (const enabled of [true, false]) {
    const sent = [],
      events = [];
    const run = await runExperiment(
      validate({ ...input, ar_token_probabilities: enabled }),
      {
        models: [
          ...config.models,
          {
            id: 'other',
            model: 'autoregressive',
            baseUrl: 'http://fixture-other/v1',
          },
        ],
      },
      {
        signal: new AbortController().signal,
        emit: (event) => events.push(event),
        fetcher: async (url, options) => {
          sent.push(JSON.parse(options.body));
          return new Response(
            stream([
              frame(
                'Answer',
                [token('Answer')],
                { token_ids: [10] },
                { usage: { completion_tokens: 1 } },
              ),
              '[DONE]',
            ]),
          );
        },
      },
    );
    assert.equal(run.status, 'complete');
    for (const payload of sent) {
      if (enabled && payload.model === 'a') {
        assert.equal(payload.logprobs, true);
        assert.equal(payload.top_logprobs, 0);
      } else {
        assert.equal(Object.hasOwn(payload, 'logprobs'), false);
        assert.equal(Object.hasOwn(payload, 'top_logprobs'), false);
      }
    }
    const probabilityEvents = events.filter(
      (event) => event.type === 'token_probabilities',
    );
    const arResult = run.results.find(
      (result) => result.model === 'autoregressive',
    );
    assert.ok(run.results.every((result) => result.text === 'Answer'));
    assert.equal(
      run.results.find((result) => result.model === 'diffusion')
        .ar_token_probabilities,
      undefined,
    );
    if (enabled) {
      assert.equal(arResult.ar_token_probabilities.status, 'available');
      assert.ok(
        probabilityEvents.every(
          (event) =>
            event.model === 'autoregressive' &&
            event.request_id === arResult.request_id &&
            event.index === arResult.index,
        ),
      );
      assertAppendOnly(probabilityEvents, arResult.ar_token_probabilities);
    } else {
      assert.deepEqual(probabilityEvents, []);
      assert.equal(arResult.ar_token_probabilities, undefined);
    }
  }
});

test('profiling and warmups omit probability flags and events even when validation is bypassed', async () => {
  for (const workload of ['continuation', 'raw']) {
    const settings = validate({
      ...input,
      kind: 'profile',
      batch_sizes: [1],
      repeats: 1,
      warmups: 1,
      workload,
      dataset: [{ prompt: 'Warmup' }, { prompt: 'Measure' }],
    });
    settings.ar_token_probabilities = true;
    const sent = [],
      events = [];
    const run = await runExperiment(settings, config, {
      signal: new AbortController().signal,
      emit: (event) => events.push(event),
      fetcher: async (url, options) => {
        sent.push(JSON.parse(options.body));
        return new Response(
          stream([
            frame(
              'Answer',
              [token('Answer')],
              {},
              { usage: { completion_tokens: 1 } },
            ),
            '[DONE]',
          ]),
        );
      },
    });
    assert.equal(run.status, 'complete');
    assert.equal(run.settings.ar_token_probabilities, false);
    assert.equal(sent.length, 4);
    assert.equal(run.warmup_results.length, 2);
    assert.ok(
      sent.every(
        (payload) =>
          !Object.hasOwn(payload, 'logprobs') &&
          !Object.hasOwn(payload, 'top_logprobs'),
      ),
    );
    assert.equal(
      events.filter((event) => event.type === 'token_probabilities').length,
      0,
    );
    assert.ok(
      [...run.results, ...run.warmup_results].every(
        (result) => !result.ar_token_probabilities,
      ),
    );
  }
});

test('direct demo callers cannot enable probabilities with truthy nonbooleans or on warmups', async () => {
  for (const value of ['true', 1, true]) {
    const settings = validate(input);
    settings.ar_token_probabilities = value;
    settings.warmups = 1;
    const events = [],
      sent = [];
    const run = await runExperiment(
      settings,
      {
        models: [config.models[1]],
      },
      {
        signal: new AbortController().signal,
        emit: (event) => events.push(event),
        fetcher: async (url, options) => {
          sent.push(JSON.parse(options.body));
          return new Response(
            stream([
              frame('A', [token('A')], {}, { usage: { completion_tokens: 1 } }),
              '[DONE]',
            ]),
          );
        },
      },
    );
    assert.equal(sent.length, 2);
    assert.equal(Object.hasOwn(sent[0], 'logprobs'), false);
    assert.equal(Object.hasOwn(sent[0], 'top_logprobs'), false);
    assert.equal(run.warmup_results[0].ar_token_probabilities, undefined);
    const probabilityEvents = events.filter(
      (event) => event.type === 'token_probabilities',
    );
    if (value === true) {
      assert.equal(sent[1].logprobs, true);
      assert.equal(sent[1].top_logprobs, 0);
      assert.ok(probabilityEvents.length > 0);
      assert.ok(
        probabilityEvents.every(
          (event) => event.request_id === run.results[0].request_id,
        ),
      );
    } else {
      assert.equal(Object.hasOwn(sent[1], 'logprobs'), false);
      assert.equal(Object.hasOwn(sent[1], 'top_logprobs'), false);
      assert.deepEqual(probabilityEvents, []);
    }
  }
});

test('a failed stream preserves validated probability samples and partial text without invented throughput', async () => {
  const events = [];
  const run = await runExperiment(
    validate({ ...input, ar_token_probabilities: true }),
    {
      models: [config.models[1]],
    },
    {
      signal: new AbortController().signal,
      emit: (event) => events.push(event),
      fetcher: async () =>
        new Response(
          stream([
            frame(
              'Partial',
              [token('Partial')],
              { token_ids: [10] },
              { usage: { completion_tokens: 1 } },
            ),
          ]),
        ),
    },
  );
  assert.equal(run.status, 'partial');
  const result = run.results[0];
  assert.equal(result.status, 'error');
  assert.equal(result.text, 'Partial');
  assert.equal(result.chunks.length, 1);
  assert.equal(result.completion_tokens, null);
  assert.equal(result.tokens_per_second, null);
  assert.equal(result.ar_token_probabilities.status, 'partial');
  assert.equal(result.ar_token_probabilities.tokens[0].token, 'Partial');
  assert.deepEqual(result.ar_token_probabilities.spans, [
    { start: 0, end: 7, token_indexes: [0] },
  ]);
  assertAppendOnly(
    events.filter((event) => event.type === 'token_probabilities'),
    result.ar_token_probabilities,
  );
});
