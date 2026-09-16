import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdir, mkdtemp, readFile, readdir, rm } from 'node:fs/promises';
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
    const response = await fetch(base + '/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    assert.equal(response.status, 200);
    const blocked = await fetch(base + '/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    assert.equal(blocked.status, 409);
    let run;
    for await (const s of parseSSE(response.body)) {
      const e = JSON.parse(s);
      if (e.type === 'complete') run = e.run;
    }
    assert.ok(run);
    assert.equal(run.synthetic, true);
    assert.equal(run.results.length, 12);
    assert.equal(run.warmup_results.length, 6);
    assert.equal(run.summaries.length, 4);
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
    const controller = new AbortController();
    const cancelled = await fetch(base + '/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...payload, repeats: 100 }),
      signal: controller.signal,
    });
    await cancelled.body.getReader().read();
    controller.abort();
    let abortedRun;
    for (let attempt = 0; attempt < 50 && !abortedRun; attempt++) {
      await new Promise((r) => setTimeout(r, 20));
      for (const f of (await readdir(results)).filter(
        (f) => f.endsWith('.json') && f !== run.id + '.json',
      )) {
        const x = JSON.parse(await readFile(path.join(results, f), 'utf8'));
        if (x.status === 'cancelled') abortedRun = x;
      }
    }
    assert.ok(abortedRun, 'client cancellation is saved');
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
    assert.equal(errors.length, 1);
    assert.equal(errors[0].type, 'error');
    assert.match(errors[0].message, /real tokenizer endpoints/);
    assert.equal((await fetch(base + '/api/health')).status, 200);
  } finally {
    server.kill('SIGTERM');
    await once(server, 'exit').catch(() => {});
    await rm(results, { recursive: true, force: true });
  }
});
