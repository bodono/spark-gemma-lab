import test from 'node:test';
import assert from 'node:assert/strict';
import { collectCompletion, parseSSE } from '../server/stream.mjs';
import { summarize, percentile } from '../server/metrics.mjs';
import { validate, runExperiment } from '../server/runner.mjs';
const frame = (content, extra = {}) => ({
  choices: [{ delta: { content }, finish_reason: null }],
  ...extra,
});
function stream(frames, split = 7) {
  const raw = frames
      .map(
        (f) =>
          'data: ' +
          (typeof f === 'string' ? f : JSON.stringify(f)) +
          '\r\n\r\n',
      )
      .join(''),
    bytes = new TextEncoder().encode(raw);
  return new ReadableStream({
    start(c) {
      for (let i = 0; i < bytes.length; i += split)
        c.enqueue(bytes.slice(i, i + split));
      c.close();
    },
  });
}
function clock() {
  let t = 0;
  return () => (t += 100);
}
test('SSE framing preserves split UTF-8, CRLF, multiline data and comments', async () => {
  const frames = [];
  for await (const x of parseSSE(stream([frame('café 🦉'), '[DONE]'], 1)))
    frames.push(x);
  assert.equal(JSON.parse(frames[0]).choices[0].delta.content, 'café 🦉');
  assert.equal(frames[1], '[DONE]');
  const body = new Response(': comment\ndata: {"x":\ndata: 1}\n\n').body;
  for await (const x of parseSSE(body))
    assert.deepEqual(JSON.parse(x), { x: 1 });
});
test('TTFT ignores roles, reasoning and usage; cumulative usage replaces count', async () => {
  const result = await collectCompletion(
    stream([
      { choices: [{ delta: { role: 'assistant' } }] },
      { choices: [{ delta: { reasoning_content: 'thought' } }] },
      frame('hello', { usage: { completion_tokens: 1 } }),
      frame(' world', { usage: { completion_tokens: 2 } }),
      {
        choices: [{ delta: {}, finish_reason: 'stop' }],
        usage: { completion_tokens: 3, prompt_tokens: 10 },
      },
      '[DONE]',
    ]),
    { start: 0, now: clock() },
  );
  assert.equal(result.ttft_ms, 200);
  assert.equal(result.last_content_ms, 300);
  assert.equal(result.completion_tokens, 3);
  assert.equal(result.elapsed_ms, 500);
  assert.equal(result.tokens_per_second, 6);
  assert.equal(result.text, 'hello world');
});
test('one block has finite E2E throughput; missing usage remains unavailable', async () => {
  const a = await collectCompletion(
    stream([
      frame('one block', { usage: { completion_tokens: 256 } }),
      '[DONE]',
    ]),
    { start: 0, now: clock() },
  );
  assert.equal(a.tokens_per_second, 1280);
  assert.equal(a.post_first_block_tps, null);
  const b = await collectCompletion(stream([frame('hello'), '[DONE]']), {
    start: 0,
    now: clock(),
  });
  assert.equal(b.completion_tokens, null);
  assert.equal(b.tokens_per_second, null);
  assert.equal(b.post_first_block_tps, null);
});

test('raw completion prompt IDs are retained from the first choice without counting them as output', async () => {
  const r = await collectCompletion(
    stream([
      {
        choices: [
          { text: 'One', prompt_token_ids: [5, 6, 7], token_ids: [10] },
        ],
        usage: { prompt_tokens: 3, completion_tokens: 1 },
      },
      {
        choices: [
          {
            text: ' two',
            prompt_token_ids: null,
            token_ids: [11],
            finish_reason: 'stop',
          },
        ],
        usage: { prompt_tokens: 3, completion_tokens: 2 },
      },
      '[DONE]',
    ]),
    { start: 0, now: clock() },
  );
  assert.deepEqual(r.prompt_token_ids, [5, 6, 7]);
  assert.deepEqual(r.output_token_ids, [10, 11]);
  assert.equal(r.completion_tokens, 2);
  assert.equal(r.text, 'One two');
});
test('post-first-block generation rate subtracts the entire first diffusion block', async () => {
  const r = await collectCompletion(
    stream([
      frame('block one', { usage: { completion_tokens: 256 } }),
      frame('block two', { usage: { completion_tokens: 512 } }),
      { choices: [], usage: { completion_tokens: 512 } },
      '[DONE]',
    ]),
    { start: 0, now: clock() },
  );
  assert.equal(r.first_block_tokens, 256);
  assert.equal(r.token_progress.length, 2);
  assert.equal(r.post_first_block_tps, 256 / 0.1);
  assert.notEqual(r.post_first_block_tps, 511 / 0.1);
});
test('Gemma empty channel is decoded across fragments with raw text preserved', async () => {
  const r = await collectCompletion(
    stream([
      frame('<|chan'),
      frame('nel>thought\n<channel|>'),
      frame('Answer', { usage: { completion_tokens: 256 } }),
      '[DONE]',
    ]),
    { start: 0, now: clock(), stripEmptyGemmaChannel: true },
  );
  assert.equal(r.text, 'Answer');
  assert.equal(r.ttft_ms, 300);
  assert.equal(r.completion_tokens, 256);
  assert.equal(
    r.chunks.map((c) => c.raw_text).join(''),
    '<|channel>thought\n<channel|>Answer',
  );
  for (const text of [
    'thought\nAnswer',
    '<|channel>thought\nActual thought<channel|>Answer',
    '<|chan',
  ]) {
    const x = await collectCompletion(stream([frame(text), '[DONE]']), {
      start: 0,
      now: clock(),
      stripEmptyGemmaChannel: true,
    });
    assert.equal(x.text, text);
  }
});
test('malformed, empty and prematurely terminated streams fail', async () => {
  await assert.rejects(
    () => collectCompletion(stream(['not json']), { start: 0 }),
    /Malformed/,
  );
  await assert.rejects(
    () => collectCompletion(stream(['[DONE]']), { start: 0 }),
    /no generated text/,
  );
  await assert.rejects(
    () => collectCompletion(stream([frame('partial')]), { start: 0 }),
    /without a completion marker/,
  );
});
test('frontier uses measured wave durations and suppresses incomplete waves', () => {
  const results = [
    {
      status: 'complete',
      completion_tokens: 100,
      tokens_per_second: 100,
      elapsed_ms: 1000,
      ttft_ms: 100,
    },
    {
      status: 'complete',
      completion_tokens: 100,
      tokens_per_second: 50,
      elapsed_ms: 2000,
      ttft_ms: 100,
    },
  ];
  const a = summarize('a', 2, [{ makespan_ms: 2200, results }]);
  assert.equal(a.mean_user_tps, 75);
  assert.equal(a.aggregate_tps, 200 / 2.2);
  assert.equal(a.p50_latency_ms, 1500);
  assert.equal(a.valid, true);
  for (const bad of [
    { status: 'error' },
    { ...results[0], completion_tokens: null },
    { ...results[0], fixed_length_ok: false },
  ]) {
    const b = summarize('a', 2, [
      { makespan_ms: 2200, results: [results[0], bad] },
    ]);
    assert.equal(b.valid, false);
    assert.equal(b.aggregate_tps, null);
  }
  assert.equal(percentile([], 0.5), null);
});
test('post-first summary pools token intervals and reports textless tails without invalidating timing', () => {
  const results = [1000, 4000, 10000].map((ms, index) => ({
    status: 'complete',
    completion_tokens: 512,
    fixed_length_ok: true,
    tokens_per_second: 512 / ((1000 + ms) / 1000),
    elapsed_ms: 1000 + ms,
    ttft_ms: 1000,
    post_first_block_tps: 256 / (ms / 1000),
    token_progress: [
      { at_ms: 1000, completion_tokens: 256 },
      { at_ms: 1000 + ms, completion_tokens: 512 },
    ],
    text: index === 0 ? 'A' : 'AB',
    chunks: [
      { at_ms: 1000, text: 'A' },
      ...(index === 0 ? [] : [{ at_ms: 1000 + ms, text: 'B' }]),
    ],
  }));
  const summary = summarize('diffusion', 1, [{ makespan_ms: 18000, results }]);
  assert.equal(summary.valid, true);
  assert.equal(summary.mean_post_first_block_tps, (256 + 64 + 25.6) / 3);
  assert.equal(summary.median_post_first_block_tps, 64);
  assert.equal(summary.pooled_post_first_block_tokens, 768);
  assert.equal(summary.pooled_post_first_block_ms, 15000);
  assert.equal(summary.pooled_post_first_block_tps, 768 / 15);
  assert.equal(summary.pooled_post_first_block_requests, 3);
  assert.equal(summary.post_first_visibility_checked_requests, 3);
  assert.equal(summary.requests_without_post_first_visible_text, 1);
  assert.equal(summary.output_characters_min, 1);
  assert.equal(summary.output_characters_median, 2);
  assert.equal(summary.output_characters_max, 2);
  const invalid = summarize('diffusion', 1, [
    { makespan_ms: 18000, results: [...results, { status: 'error' }] },
  ]);
  assert.equal(invalid.pooled_post_first_block_tps, null);
  assert.equal(invalid.median_post_first_block_tps, null);
});
const settings = {
  kind: 'demo',
  prompt: 'hello',
  batch_size: 2,
  max_tokens: 24,
  temperature: 0,
  seed: 42,
};
const config = {
  models: [
    { id: 'a', model: 'a', baseUrl: 'http://a/v1' },
    { id: 'b', model: 'b', baseUrl: 'http://b/v1' },
  ],
};
test('validation rejects pathological settings and short corpus', () => {
  assert.throws(() => validate({ ...settings, batch_size: 0 }));
  assert.throws(() => validate({ ...settings, max_tokens: NaN }));
  assert.throws(
    () =>
      validate({
        ...settings,
        kind: 'profile',
        batch_sizes: [1, 2],
        repeats: 3,
        warmups: 1,
        dataset: [{ prompt: 'only one' }],
      }),
    /Dataset needs/,
  );
});
test('both endpoints dispatch same prompts concurrently with a real stream fixture', async () => {
  const sent = [];
  let release;
  const gate = new Promise((r) => (release = r));
  const fetcher = async (url, options) => {
    sent.push({ url, body: JSON.parse(options.body) });
    if (sent.length === 4) release();
    await gate;
    return new Response(
      stream([frame('answer', { usage: { completion_tokens: 24 } }), '[DONE]']),
    );
  };
  const r = await runExperiment(validate(settings), config, {
    signal: new AbortController().signal,
    emit: () => {},
    fetcher,
  });
  assert.equal(r.status, 'complete');
  assert.equal(sent.length, 4);
  assert.equal(new Set(sent.map((x) => x.body.messages[0].content)).size, 1);
  assert.equal(r.results.length, 4);
  assert.equal(r.synthetic, false);
  assert.equal(r.summaries.length, 2);
});
test('diffusion omits unsupported sampler overrides in chat and raw profiling requests', async () => {
  for (const kind of ['demo', 'profile']) {
    const sent = [];
    const r = await runExperiment(
      validate({
        ...settings,
        kind,
        batch_size: 1,
        batch_sizes: [1],
        repeats: 1,
        warmups: 0,
        workload: 'raw',
        output_mode: 'fixed',
        dataset: kind === 'profile' ? [{ prompt: 'book' }] : [],
      }),
      {
        models: [
          { id: 'diffusion', model: 'd', baseUrl: 'http://d/v1' },
          { id: 'autoregressive', model: 'a', baseUrl: 'http://a/v1' },
        ],
      },
      {
        signal: new AbortController().signal,
        emit: () => {},
        fetcher: async (url, options) => {
          sent.push({ url, body: JSON.parse(options.body) });
          return new Response(
            stream([
              frame('answer', { usage: { completion_tokens: 24 } }),
              '[DONE]',
            ]),
          );
        },
      },
    );
    const d = sent.find((x) => x.body.model === 'd').body;
    const a = sent.find((x) => x.body.model === 'a').body;
    assert.equal(Object.hasOwn(d, 'temperature'), false);
    assert.equal(Object.hasOwn(d, 'seed'), false);
    assert.equal(a.temperature, 0);
    assert.equal(a.seed, 42);
    assert.equal(d.max_tokens, a.max_tokens);
    assert.equal(d.ignore_eos, kind === 'profile' ? true : undefined);
    assert.deepEqual(
      r.results.find((x) => x.model === 'diffusion').request_payload,
      d,
    );
  }
});
test('HTTP errors are retained without fabricated rates', async () => {
  const r = await runExperiment(validate(settings), config, {
    signal: new AbortController().signal,
    emit: () => {},
    fetcher: async () => new Response('unavailable', { status: 503 }),
  });
  assert.equal(r.status, 'partial');
  assert.equal(r.results[0].status, 'error');
  assert.equal(r.summaries[0].aggregate_tps, null);
});
test('warmups excluded, matched raw corpus and fixed output length validated', async () => {
  const samples = Array.from({ length: 4 }, (_, i) => ({
      prompt: 'book ' + i,
      book_id: i,
    })),
    sent = [];
  const r = await runExperiment(
    validate({
      ...settings,
      kind: 'profile',
      batch_sizes: [1, 2],
      warmups: 1,
      repeats: 1,
      workload: 'raw',
      output_mode: 'fixed',
      dataset: samples,
    }),
    config,
    {
      signal: new AbortController().signal,
      emit: () => {},
      fetcher: async (url, options) => {
        sent.push({ url, body: JSON.parse(options.body) });
        return new Response(
          stream([
            frame('answer', { usage: { completion_tokens: 24 } }),
            '[DONE]',
          ]),
        );
      },
    },
  );
  assert.equal(r.results.length, 6);
  assert.equal(r.warmup_results.length, 6);
  assert.ok(
    sent.every((x) => x.url.endsWith('/completions') && x.body.ignore_eos),
  );
  assert.ok(r.summaries.every((s) => s.valid));
  assert.equal(
    r.summaries.find((s) => s.model === 'a' && s.batch_size === 2).requests,
    2,
  );
});
test('fixed request count holds the measured corpus constant across concurrency', async () => {
  const input = {
    ...settings,
    kind: 'profile',
    batch_sizes: [1, 2],
    warmups: 1,
    repeats: 1,
    requests_per_condition: 4,
    workload: 'raw',
    output_mode: 'fixed',
    dataset: Array.from({ length: 6 }, (_, i) => ({ prompt: 'book ' + i })),
  };
  assert.throws(
    () => validate({ ...input, requests_per_condition: 3 }),
    /divisible/,
  );
  assert.throws(
    () => validate({ ...input, dataset: input.dataset.slice(0, 5) }),
    /at least 6/,
  );
  const r = await runExperiment(validate(input), config, {
    signal: new AbortController().signal,
    emit: () => {},
    fetcher: async () =>
      new Response(
        stream([
          frame('answer', { usage: { completion_tokens: 24 } }),
          '[DONE]',
        ]),
      ),
  });
  for (const model of ['a', 'b']) {
    for (const size of [1, 2]) {
      const rows = r.results.filter(
        (x) => x.model === model && x.batch_size === size,
      );
      assert.deepEqual(
        rows.map((x) => x.request_payload.prompt),
        ['book 2', 'book 3', 'book 4', 'book 5'],
      );
      const summary = r.summaries.find(
        (x) => x.model === model && x.batch_size === size,
      );
      assert.equal(summary.requests, 4);
      assert.equal(summary.wave_count, 4 / size);
    }
  }
});
test('default corpus profiling sends matched chat continuations and respects natural stopping', async () => {
  const input = {
    ...settings,
    kind: 'profile',
    batch_sizes: [1],
    warmups: 0,
    repeats: 1,
    requests_per_condition: 1,
    dataset: [{ prompt: 'The sailor watched the moon rise over the water.' }],
  };
  const sent = [];
  const r = await runExperiment(validate(input), config, {
    signal: new AbortController().signal,
    emit: () => {},
    fetcher: async (url, options) => {
      sent.push({ url, body: JSON.parse(options.body) });
      return new Response(
        stream([
          frame('A short continuation.', { usage: { completion_tokens: 7 } }),
          '[DONE]',
        ]),
      );
    },
  });
  assert.equal(r.settings.workload, 'continuation');
  assert.equal(r.settings.output_mode, 'natural');
  assert.ok(sent.every((x) => x.url.endsWith('/chat/completions')));
  assert.deepEqual(sent[0].body.messages, sent[1].body.messages);
  assert.ok(sent[0].body.messages[0].content.includes(input.dataset[0].prompt));
  assert.notEqual(sent[0].body.messages[0].content, input.dataset[0].prompt);
  assert.ok(sent.every((x) => x.body.ignore_eos === false));
  assert.ok(sent.every((x) => x.body.return_token_ids === true));
  assert.equal(
    r.results[0].effective_prompt_hash,
    r.results[1].effective_prompt_hash,
  );
  assert.ok(r.results.every((x) => x.fixed_length_ok === null));
  assert.ok(r.summaries.every((x) => x.valid));
  assert.equal(r.status, 'complete');
  assert.throws(() => validate({ ...input, workload: 'unknown' }), /workload/);
  assert.throws(
    () => validate({ ...input, output_mode: 'unknown' }),
    /output_mode/,
  );
});
test('cancellation aborts requests and persists cancelled results', async () => {
  const c = new AbortController();
  const r = await runExperiment(validate(settings), config, {
    signal: c.signal,
    emit: (e) => {
      if (e.type === 'start') c.abort();
    },
    fetcher: async () => {
      throw Error('should not fetch');
    },
  });
  assert.equal(r.status, 'cancelled');
  assert.ok(r.results.every((x) => x.status === 'cancelled'));
  assert.ok(r.summaries.every((x) => !x.valid));
});

test('partial output survives a failed stream without token estimates', async () => {
  const r = await runExperiment(
    validate({ ...settings, batch_size: 1 }),
    config,
    {
      signal: new AbortController().signal,
      emit: () => {},
      fetcher: async () => new Response(stream([frame('partial answer')])),
    },
  );
  assert.equal(r.status, 'partial');
  assert.ok(
    r.results.every(
      (result) =>
        result.text === 'partial answer' &&
        result.chunks.length === 1 &&
        result.tokens_per_second === null,
    ),
  );
});

test('a live demo uses the edited prompt even after loading a profiling corpus', async () => {
  const sent = [];
  await runExperiment(
    validate({
      ...settings,
      prompt: 'the edited prompt',
      dataset: [{ prompt: 'old corpus prompt' }],
    }),
    config,
    {
      signal: new AbortController().signal,
      emit: () => {},
      fetcher: async (url, options) => {
        sent.push(JSON.parse(options.body));
        return new Response(
          stream([
            frame('answer', { usage: { completion_tokens: 24 } }),
            '[DONE]',
          ]),
        );
      },
    },
  );
  assert.ok(
    sent.every(
      (request) => request.messages[0].content === 'the edited prompt',
    ),
  );
});
