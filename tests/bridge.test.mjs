import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdir, mkdtemp, readFile, rm } from 'node:fs/promises';
import path from 'node:path';
import { once } from 'node:events';
import http from 'node:http';
import { parseSSE } from '../server/stream.mjs';
const payload = {
  kind: 'profile',
  prompt: 'fixture',
  batch_size: 1,
  batch_sizes: [1, 2],
  max_tokens: 24,
  temperature: 0,
  seed: 42,
  repeats: 2,
  warmups: 1,
};

test('bridge API: host/origin protection, streaming, export, locking and cancellation', async () => {
  const work = path.resolve('../../work');
  await mkdir(work, { recursive: true });
  const results = await mkdtemp(path.join(work, 'bridge-test-'));
  const server = spawn(process.execPath, ['server/index.mjs'], {
    env: {
      ...process.env,
      SPARK_LAB_MOCK: '1',
      SPARK_LAB_PORT: '18787',
      SPARK_LAB_RESULTS: results,
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  try {
    await Promise.race([
      once(server.stdout, 'data'),
      new Promise((_, reject) =>
        setTimeout(() => reject(Error('Bridge startup timeout')), 5000).unref(),
      ),
    ]);
    const base = 'http://127.0.0.1:18787';
    const hostileHostStatus = await new Promise((resolve, reject) => {
      http
        .get(
          base + '/api/config',
          { headers: { Host: 'evil.example:18787' } },
          (r) => {
            r.resume();
            resolve(r.statusCode);
          },
        )
        .on('error', reject);
    });
    assert.equal(hostileHostStatus, 403);
    assert.equal(
      (
        await fetch(base + '/api/config', {
          headers: { Origin: 'https://evil.example' },
        })
      ).status,
      403,
    );
    assert.equal((await fetch(base + '/api/config')).status, 200);
    assert.deepEqual(
      await fetch(base + '/api/activity').then((r) => r.json()),
      { active: false, progress: null },
    );
    const response = await fetch(base + '/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    assert.equal(response.status, 200);
    const activity = await fetch(base + '/api/activity').then((r) => r.json());
    assert.equal(activity.active, true);
    assert.equal(activity.progress.kind, 'profile');
    assert.equal(activity.progress.total_requests, 12);
    assert.equal(activity.progress.warmup_total_requests, 6);
    assert(
      ['preparing', 'warmup', 'measuring'].includes(activity.progress.stage),
    );
    assert(!JSON.stringify(activity.progress).includes('fixture'));
    const blocked = await fetch(base + '/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    assert.equal(blocked.status, 409);
    let run;
    const progressEvents = [];
    for await (const s of parseSSE(response.body)) {
      const e = JSON.parse(s);
      if (e.type === 'complete') run = e.run;
      if (e.type === 'progress') progressEvents.push(e.progress);
    }
    assert.ok(run);
    assert.equal(run.synthetic, true);
    assert.equal(run.results.length, 12);
    assert.equal(run.warmup_results.length, 6);
    assert.equal(run.summaries.length, 4);
    const finished = await fetch(base + '/api/activity').then((r) => r.json());
    assert.equal(finished.active, false);
    assert.equal(finished.progress.stage, 'complete');
    assert.equal(finished.progress.run_id, run.id);
    assert.equal(finished.progress.completed_requests, 12);
    assert.equal(finished.progress.warmup_completed_requests, 6);
    assert.equal(finished.progress.failed_requests, 0);
    assert(progressEvents.some((p) => p.stage === 'preparing'));
    assert(progressEvents.some((p) => p.stage === 'warmup'));
    assert(progressEvents.some((p) => p.stage === 'measuring'));
    assert.equal(progressEvents.at(-1).stage, 'complete');
    const saved = await fetch(base + '/api/runs/' + run.id).then((r) =>
      r.json(),
    );
    assert.equal(saved.id, run.id);
    assert.equal(
      (await readFile(path.join(results, run.id + '.jsonl'), 'utf8'))
        .trim()
        .split('\n').length,
      12,
    );
    const unicodePrompt = 'Compare 🧪 experiments in a café';
    const unicodeBody = Buffer.from(
      JSON.stringify({ ...payload, kind: 'demo', prompt: unicodePrompt }),
    );
    const splitAt = unicodeBody.indexOf(Buffer.from('🧪')) + 1;
    const fragmented = await new Promise((resolve, reject) => {
      const request = http.request(
        base + '/api/run',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
        },
        (response) => {
          const chunks = [];
          response.on('data', (chunk) => chunks.push(chunk));
          response.on('end', () =>
            resolve(Buffer.concat(chunks).toString('utf8')),
          );
          response.on('error', reject);
        },
      );
      request.on('error', reject);
      request.write(unicodeBody.subarray(0, splitAt));
      setTimeout(() => request.end(unicodeBody.subarray(splitAt)), 20);
    });
    const unicodeRun = fragmented
      .split('\n')
      .filter((line) => line.startsWith('data: '))
      .map((line) => JSON.parse(line.slice(6)))
      .find((event) => event.type === 'complete').run;
    assert.equal(unicodeRun.settings.prompt, unicodePrompt);
    assert.ok(
      unicodeRun.results.every(
        (result) =>
          result.request_payload.messages[0].content === unicodePrompt,
      ),
    );
    const activityNow = () =>
      fetch(base + '/api/activity').then((r) => r.json());
    const waitForIdle = async () => {
      for (let attempt = 0; attempt < 100; attempt++) {
        const current = await activityNow();
        if (!current.active) return current;
        await new Promise((resolve) => setTimeout(resolve, 20));
      }
      throw Error('Bridge did not settle');
    };
    const cancel = (started_at) =>
      fetch(base + '/api/cancel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ started_at }),
      });
    assert.equal(
      (await cancel(finished.progress.started_at)).status,
      409,
      'no active run is benign',
    );

    // Refreshing the owner closes its SSE connection, but profiling continues.
    const owner = new AbortController();
    const disconnected = await fetch(base + '/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...payload,
        batch_sizes: [1],
        warmups: 0,
        repeats: 4,
      }),
      signal: owner.signal,
    });
    assert.equal(disconnected.status, 200);
    const disconnectedStart = await activityNow();
    owner.abort();
    await new Promise((resolve) => setTimeout(resolve, 20));
    assert.equal(
      (await activityNow()).active,
      true,
      'profile must survive owner disconnect',
    );
    const recovered = await waitForIdle();
    assert.equal(
      recovered.progress.started_at,
      disconnectedStart.progress.started_at,
    );
    assert.equal(recovered.progress.stage, 'complete');
    assert.equal(recovered.progress.completed_requests, 8);
    const recoveredRun = await fetch(
      base + '/api/runs/' + recovered.progress.run_id,
    ).then((r) => r.json());
    assert.equal(recoveredRun.status, 'complete');
    assert.equal(recoveredRun.results.length, 8);

    const cancelled = await fetch(base + '/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...payload, repeats: 100 }),
    });
    const current = await activityNow();
    assert.notEqual(current.progress.started_at, recovered.progress.started_at);
    assert.equal(
      (await cancel(recovered.progress.started_at)).status,
      409,
      'a stale tab cannot cancel the next run',
    );
    assert.equal((await activityNow()).active, true);
    assert.equal(
      (await activityNow()).progress.started_at,
      current.progress.started_at,
    );
    assert.equal(
      (
        await fetch(base + '/api/cancel', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Origin: 'https://evil.example',
          },
          body: JSON.stringify({ started_at: current.progress.started_at }),
        })
      ).status,
      403,
    );
    assert.equal(
      (
        await fetch(base + '/api/cancel', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: '{}',
        })
      ).status,
      400,
    );
    assert.equal((await cancel(current.progress.started_at)).status, 202);
    let abortedRun;
    for await (const raw of parseSSE(cancelled.body)) {
      const event = JSON.parse(raw);
      if (event.type === 'complete') abortedRun = event.run;
    }
    assert.equal(
      abortedRun.status,
      'cancelled',
      'explicit cancellation reaches the owning SSE connection',
    );
    const cancelledActivity = await waitForIdle();
    assert.equal(cancelledActivity.progress.stage, 'cancelled');
    assert.equal(cancelledActivity.progress.run_id, abortedRun.id);
    assert.equal(
      cancelledActivity.progress.completed_requests,
      abortedRun.results.length,
    );
    assert.equal(
      cancelledActivity.progress.warmup_completed_requests,
      abortedRun.warmup_results.length,
    );
    assert(
      cancelledActivity.progress.completed_requests <
        cancelledActivity.progress.total_requests,
    );
    assert.equal((await cancel(current.progress.started_at)).status, 409);

    // Demos continue to cancel when their owning browser disconnects.
    const demoOwner = new AbortController();
    await fetch(base + '/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...payload, kind: 'demo' }),
      signal: demoOwner.signal,
    });
    demoOwner.abort();
    const demoCancelled = await waitForIdle();
    assert.equal(demoCancelled.progress.kind, 'demo');
    assert.equal(demoCancelled.progress.stage, 'cancelled');
    const demoRun = await fetch(
      base + '/api/runs/' + demoCancelled.progress.run_id,
    ).then((r) => r.json());
    assert.equal(demoRun.status, 'cancelled');
    const invalid = await fetch(base + '/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...payload, max_tokens: 0 }),
    });
    assert.equal(invalid.status, 400);
    const preparationFailure = await fetch(base + '/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...payload, input_tokens: 512 }),
    });
    assert.equal(preparationFailure.status, 200);
    const errors = [];
    for await (const raw of parseSSE(preparationFailure.body))
      errors.push(JSON.parse(raw));
    const failure = errors.find((e) => e.type === 'error');
    assert.match(failure.message, /real tokenizer endpoints/);
    assert(
      errors.some(
        (e) => e.type === 'progress' && e.progress.stage === 'inputs',
      ),
    );
    const failedActivity = await fetch(base + '/api/activity').then((r) =>
      r.json(),
    );
    assert.equal(failedActivity.active, false);
    assert.equal(failedActivity.progress.stage, 'error');
    assert.equal(failedActivity.progress.completed_requests, 0);
    assert.equal(failedActivity.progress.run_id, undefined);
    assert.equal((await fetch(base + '/api/health')).status, 200);
  } finally {
    server.kill('SIGTERM');
    await once(server, 'exit').catch(() => {});
    await rm(results, { recursive: true, force: true });
  }
});
