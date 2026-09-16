import test from 'node:test';
import assert from 'node:assert/strict';
import { createProgress } from '../server/progress.mjs';
import { validate, runExperiment } from '../server/runner.mjs';
const settings = (extra = {}) =>
  validate({
    kind: 'profile',
    prompt: 'Private fixture prompt',
    max_tokens: 24,
    batch_size: 1,
    batch_sizes: [1, 2],
    repeats: 2,
    warmups: 1,
    temperature: 0,
    seed: 42,
    ...extra,
  });
const config = {
  runtime: { canvas_length: 256 },
  models: ['diffusion', 'autoregressive'].map((id) => ({
    id,
    model: id,
    baseUrl: 'http://fixture/v1',
  })),
};
const terminal = (request_id, extra = {}) => ({
  request_id,
  status: 'complete',
  warmup: false,
  ...extra,
});

test('progress totals include every model/condition and keep warmups separate', () => {
  for (const [input, expected, warmups] of [
    [{ requests_per_condition: 20, batch_sizes: [1, 2, 4] }, 120, 14],
    [{ repeats: 3, batch_sizes: [1, 2, 4], warmups: 2 }, 42, 28],
    [{ kind: 'demo', batch_size: 3 }, 6, 0],
  ]) {
    const events = [];
    const p = createProgress(settings(input), 2, (e) => events.push(e));
    assert.equal(p.snapshot.total_requests, expected);
    assert.equal(p.snapshot.warmup_total_requests, warmups);
    assert.equal(p.snapshot.stage, 'preparing');
    assert.equal(p.snapshot.completed_requests, 0);
    assert.equal(events.length, 1);
    assert(!JSON.stringify(events).includes('Private fixture'));
    assert(Number.isFinite(Date.parse(p.snapshot.started_at)));
    assert(Number.isFinite(Date.parse(p.snapshot.updated_at)));
  }
});
test('terminal requests count once; unsuccessful warmups never inflate measured completion', () => {
  const p = createProgress(settings(), 2);
  p.phase({
    stage: 'warmup',
    message: 'Warmup',
    batch_size: 2,
    wave: 1,
    waves: 1,
  });
  p.request(terminal('warm-ok', { warmup: true }));
  p.request(
    terminal('warm-error', {
      warmup: true,
      status: 'error',
      text: 'private output',
    }),
  );
  p.request(terminal('warm-error', { warmup: true, status: 'error' }));
  assert.equal(p.snapshot.warmup_completed_requests, 2);
  assert.equal(p.snapshot.completed_requests, 0);
  assert.equal(p.snapshot.failed_requests, 1);
  p.phase({
    stage: 'measuring',
    message: 'Measuring',
    batch_size: 1,
    wave: 1,
    waves: 2,
  });
  p.request(terminal('measured-invalid', { fixed_length_ok: false }));
  p.request(terminal('measured-cancelled', { status: 'cancelled' }));
  p.request(terminal('measured-ok'));
  assert.equal(p.snapshot.completed_requests, 3);
  assert.equal(p.snapshot.warmup_completed_requests, 2);
  assert.equal(p.snapshot.failed_requests, 3);
  p.finish({ id: 'saved-run', status: 'partial' });
  assert.equal(p.snapshot.stage, 'error');
  assert.equal(p.snapshot.run_status, 'partial');
  assert.equal(p.snapshot.run_id, 'saved-run');
  assert.equal(p.snapshot.batch_size, undefined);
  assert(!JSON.stringify(p.snapshot).includes('private output'));
});
test('runtime logs and exception details do not enter snapshots', () => {
  const p = createProgress(settings(), 2);
  p.phase({ type: 'phase', message: 'private runtime output' });
  assert(!JSON.stringify(p.snapshot).includes('private runtime output'));
  p.finish({ id: 'not-persisted', status: 'complete' });
  p.fail();
  assert.equal(p.snapshot.stage, 'error');
  assert.equal(p.snapshot.run_id, undefined);
});
test('runner emits monotonic counts across all warmups and measured waves', async () => {
  const events = [];
  const run = await runExperiment(
    settings({ requests_per_condition: 4 }),
    config,
    {
      signal: new AbortController().signal,
      emit: (e) => events.push(e),
      synthetic: true,
    },
  );
  const snapshots = events
    .filter((e) => e.type === 'progress')
    .map((e) => e.progress);
  assert.equal(run.status, 'complete');
  assert(snapshots.some((p) => p.stage === 'warmup'));
  assert(snapshots.some((p) => p.stage === 'measuring'));
  for (let i = 1; i < snapshots.length; i++) {
    assert(
      snapshots[i].completed_requests >= snapshots[i - 1].completed_requests,
    );
    assert(
      snapshots[i].warmup_completed_requests >=
        snapshots[i - 1].warmup_completed_requests,
    );
  }
  const final = snapshots.at(-1);
  assert.equal(final.completed_requests, 16);
  assert.equal(final.total_requests, 16);
  assert.equal(final.warmup_completed_requests, 6);
  assert.equal(final.warmup_total_requests, 6);
  assert.equal(final.failed_requests, 0);
  assert.equal(final.stage, 'complete');
  assert.equal(final.run_id, run.id);
});
test('runtime preparation is observable and cancellation leaves a truthful snapshot', async () => {
  const controller = new AbortController(),
    events = [];
  let notifyPrepared;
  const prepared = new Promise((resolve) => {
    notifyPrepared = resolve;
  });
  const execution = runExperiment(settings(), config, {
    signal: controller.signal,
    emit: (e) => events.push(e),
    fetcher: () => {
      throw Error('No inference should be requested');
    },
    prepareRuntime: async () => {
      notifyPrepared();
      await new Promise((resolve, reject) =>
        controller.signal.addEventListener(
          'abort',
          () => reject(Error('Cancelled')),
          { once: true },
        ),
      );
    },
  });
  await prepared;
  const initial = events.find((e) => e.type === 'progress').progress;
  assert.equal(initial.stage, 'preparing');
  assert.equal(initial.completed_requests, 0);
  controller.abort();
  await assert.rejects(execution, /Cancelled/);
  const final = events.filter((e) => e.type === 'progress').at(-1).progress;
  assert.equal(final.stage, 'cancelled');
  assert.equal(final.run_status, 'cancelled');
  assert.equal(final.completed_requests, 0);
  assert.equal(final.run_id, undefined);
});
test('input preparation and HTTP failures produce honest counters without error-text leakage', async () => {
  const events = [];
  await assert.rejects(
    runExperiment(settings({ input_tokens: 512 }), config, {
      signal: new AbortController().signal,
      emit: (e) => events.push(e),
      synthetic: true,
    }),
    /real tokenizer endpoints/,
  );
  assert(
    events.some((e) => e.type === 'progress' && e.progress.stage === 'inputs'),
  );
  assert.equal(events.at(-1).progress.stage, 'error');
  assert.equal(events.at(-1).progress.completed_requests, 0);
  const measured = [];
  const run = await runExperiment(
    settings({ batch_sizes: [1], repeats: 1, warmups: 0 }),
    config,
    {
      signal: new AbortController().signal,
      emit: (e) => measured.push(e),
      fetcher: async () =>
        new Response('private error details', { status: 500 }),
    },
  );
  assert.equal(run.status, 'partial');
  const final = measured.filter((e) => e.type === 'progress').at(-1).progress;
  assert.equal(final.completed_requests, 2);
  assert.equal(final.total_requests, 2);
  assert.equal(final.failed_requests, 2);
  assert.equal(final.warmup_completed_requests, 0);
  assert.equal(final.stage, 'error');
  assert(!JSON.stringify(final).includes('private error details'));
});
