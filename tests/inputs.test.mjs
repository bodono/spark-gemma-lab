import test from 'node:test';
import assert from 'node:assert/strict';
import { validate, runExperiment } from '../server/runner.mjs';
import { prepareInputs, verifyInputLength } from '../server/inputs.mjs';

const config = {
  runtime: { max_model_len: 8192, canvas_length: 256 },
  models: ['diffusion', 'autoregressive'].map((id) => ({
    id,
    model: id,
    baseUrl: `http://${id}/v1`,
  })),
};
const settings = (extra = {}) =>
  validate({
    kind: 'profile',
    prompt: 'Hello world',
    max_tokens: 10,
    batch_size: 1,
    batch_sizes: [1],
    repeats: 1,
    warmups: 0,
    seed: 42,
    temperature: 0,
    input_tokens: 5,
    ...extra,
  });
const encode = (text) => [...Buffer.from(text, 'utf8')];
function fixture({
  capacity = 8192,
  badUsage = false,
  drift = false,
  onTokenize,
} = {}) {
  const calls = [];
  const render = (body) =>
    body.messages
      ? [
          300,
          ...(body.model === 'autoregressive' ? [301, 302, 303, 304] : []),
          ...encode(body.messages[0].content),
          305,
        ]
      : encode(body.prompt);
  const fetcher = async (url, options) => {
    options.signal?.throwIfAborted();
    const body = JSON.parse(options.body);
    const route = new URL(url).pathname;
    calls.push({ route, body });
    if (route === '/tokenize') {
      assert.equal(body.add_special_tokens, false);
      onTokenize?.();
      const tokens = render(body);
      if (drift && body.model === 'autoregressive') tokens[0]++;
      return Response.json({
        tokens,
        count: tokens.length,
        max_model_len: capacity,
      });
    }
    if (route === '/detokenize')
      return Response.json({
        prompt: Buffer.from(body.tokens).toString('utf8'),
      });
    assert.match(route, /^\/v1\/(chat\/)?completions$/);
    const tokens = render(body);
    return new Response(
      'data: ' +
        JSON.stringify({
          ...(body.messages ? { prompt_token_ids: tokens } : {}),
          choices: [
            {
              delta: { content: 'Answer' },
              token_ids: [42, 43],
              ...(!body.messages ? { prompt_token_ids: tokens } : {}),
              finish_reason: 'stop',
            },
          ],
          usage: {
            prompt_tokens: tokens.length - (badUsage ? 1 : 0),
            completion_tokens: 2,
          },
        }) +
        '\n\ndata: [DONE]\n\n',
    );
  };
  return { calls, fetcher };
}
const options = (fetcher) => ({
  fetcher,
  signal: new AbortController().signal,
  emit: () => {},
});

test('input target is optional, integer-only, and ignored by the demo', async () => {
  assert.equal(settings({ input_tokens: undefined }).input_tokens, null);
  for (const input_tokens of [0, 1.2, -1, '512', 32769])
    assert.throws(() => settings({ input_tokens }), /input_tokens/);
  const demo = settings({ kind: 'demo', input_tokens: -1 });
  const f = fixture();
  await runExperiment(demo, config, options(f.fetcher));
  assert(f.calls.every((c) => c.route.includes('/completions')));
  const original = fixture();
  await runExperiment(
    settings({ input_tokens: null }),
    config,
    options(original.fetcher),
  );
  assert(original.calls.every((c) => c.route.includes('/completions')));
  assert(
    original.calls.every((c) => c.body.messages[0].content === 'Hello world'),
  );
});

test('trimmed/repeated sources are exact and retain distinct rendered model counts', async () => {
  for (const [target, policy] of [
    [5, 'trim'],
    [11, 'unchanged'],
    [30, 'repeat_then_trim'],
  ]) {
    const f = fixture();
    const prepared = await prepareInputs(
      settings({ input_tokens: target }),
      config,
      options(f.fetcher),
    );
    const p = prepared.get('Hello world');
    assert.equal(encode(p.source).length, target);
    assert.equal(p.audit.policy, policy);
    assert.equal(p.audit.original_source_tokens, 11);
    assert.equal(p.byModel.get('diffusion').count, target + 2);
    assert.equal(p.byModel.get('autoregressive').count, target + 6);
    if (target > 11) assert(p.source.includes('\n\n'));
  }
});

test('all input preparation precedes generation and matched prompts persist across batch sizes', async () => {
  const f = fixture();
  const dataset = ['warm one', 'warm two', 'measure one', 'measure two'].map(
    (prompt) => ({ prompt, input_tokens: 99 }),
  );
  const run = await runExperiment(
    settings({
      dataset,
      batch_sizes: [1, 2],
      warmups: 1,
      requests_per_condition: 2,
    }),
    config,
    options(f.fetcher),
  );
  assert.equal(run.status, 'complete');
  const firstGeneration = f.calls.findIndex((c) =>
    c.route.includes('/completions'),
  );
  assert(firstGeneration > 0);
  assert(
    f.calls
      .slice(firstGeneration)
      .every((c) => c.route.includes('/completions')),
  );
  assert.equal(run.input_preparation.length, 4);
  assert(run.results.every((r) => r.input_length_ok));
  assert(run.results.every((r) => r.sample_metadata.input_tokens === 99));
  for (const r of [...run.results, ...run.warmup_results]) {
    assert.equal(r.input_preparation.actual_source_tokens, 5);
    if (r.model === 'diffusion')
      assert.equal(r.request_payload.vllm_xargs.spark_lab_diffusion_preview, 0);
  }
  const sets = run.summaries.map((s) =>
    run.results
      .filter((r) => r.model === s.model && r.batch_size === s.batch_size)
      .map((r) => r.effective_prompt_hash)
      .sort(),
  );
  for (const set of sets) assert.deepEqual(set, sets[0]);
  assert(
    run.summaries.every((s) => s.min_prompt_tokens === s.max_prompt_tokens),
  );
});

test('Unicode boundaries and equal-count tokenizer drift fail before generation', async () => {
  const f = fixture();
  await assert.rejects(
    prepareInputs(
      settings({ prompt: '🌈x', input_tokens: 3 }),
      config,
      options(f.fetcher),
    ),
    /losslessly/,
  );
  assert(f.calls.every((c) => !c.route.includes('/completions')));
  const different = fixture({ drift: true });
  await assert.rejects(
    prepareInputs(settings(), config, options(different.fetcher)),
    /losslessly/,
  );
});

test('rendered context limits and cancellation are enforced before generation', async () => {
  const f = fixture({ capacity: 18 });
  await assert.rejects(
    prepareInputs(settings({ input_tokens: 5 }), config, options(f.fetcher)),
    /exceeds the 18-token context/,
  );
  assert(f.calls.every((c) => !c.route.includes('/completions')));
  const controller = new AbortController();
  const cancelled = fixture({ onTokenize: () => controller.abort() });
  await assert.rejects(
    prepareInputs(settings(), config, {
      ...options(cancelled.fetcher),
      signal: controller.signal,
    }),
    /abort/i,
  );
});

test('raw profiling sizes current prompts without adding chat formatting', async () => {
  const f = fixture();
  const run = await runExperiment(
    settings({ workload: 'raw' }),
    config,
    options(f.fetcher),
  );
  assert.equal(run.status, 'complete');
  assert(run.results.every((r) => r.prompt_tokens === 5 && r.input_length_ok));
  assert(
    f.calls
      .filter((c) => c.route.includes('/completions'))
      .every((c) => c.route === '/v1/completions'),
  );
});

test('observed input mismatch invalidates measured points and warmup mismatch stops the sweep', async () => {
  const f = fixture({ badUsage: true });
  const measured = await runExperiment(settings(), config, options(f.fetcher));
  assert.equal(measured.status, 'partial');
  assert(
    measured.results.every(
      (r) => r.status === 'error' && r.input_length_ok === false,
    ),
  );
  assert(measured.summaries.every((s) => !s.valid && s.aggregate_tps === null));
  const warm = fixture({ badUsage: true });
  const stopped = await runExperiment(
    settings({ warmups: 1, repeats: 2 }),
    config,
    options(warm.fetcher),
  );
  assert.equal(stopped.status, 'partial');
  assert.equal(stopped.results.length, 0);
  assert.equal(stopped.warmup_results.length, 2);
  assert.equal(
    verifyInputLength(
      { prompt_tokens: 2, prompt_token_ids: [1, 3] },
      { count: 2, tokens: [1, 2] },
    ),
    false,
  );
});
