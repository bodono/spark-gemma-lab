import http from 'node:http';
import { readFile, mkdir, writeFile, readdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { validate, runExperiment } from './runner.mjs';
import { canvasCommand, prepareCanvasRuntime } from './canvas.mjs';
const root = fileURLToPath(new URL('../', import.meta.url)),
  resultsDir = process.env.SPARK_LAB_RESULTS || path.join(root, 'results');
const configPath =
  process.env.SPARK_LAB_CONFIG || path.join(root, 'config/models.json');
const synthetic = process.env.SPARK_LAB_MOCK === '1';
const port = Number(process.env.SPARK_LAB_PORT || 8787);
let active = null;
await mkdir(resultsDir, { recursive: true });
const getConfig = async () => JSON.parse(await readFile(configPath, 'utf8'));
const json = (res, status, body) => {
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(body));
};
const server = http.createServer(async (req, res) => {
  if (![`127.0.0.1:${port}`, `localhost:${port}`].includes(req.headers.host)) {
    json(res, 403, { error: 'Invalid Host header' });
    return;
  }
  const origin = req.headers.origin;
  if (origin && !/^http:\/\/(localhost|127\.0\.0\.1):\d+$/.test(origin)) {
    json(res, 403, { error: 'Only local workspace origins are allowed' });
    return;
  }
  if (origin) {
    res.setHeader('Access-Control-Allow-Origin', origin);
    res.setHeader('Vary', 'Origin');
  }
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }
  try {
    const url = new URL(req.url, 'http://127.0.0.1');
    if (req.method === 'GET' && url.pathname === '/api/config') {
      json(res, 200, { ...(await getConfig()), synthetic });
      return;
    }
    if (req.method === 'GET' && url.pathname === '/api/datasets/pg19') {
      try {
        const raw = await readFile(
          path.join(root, 'data/pg19-512.jsonl'),
          'utf8',
        );
        json(
          res,
          200,
          raw
            .split('\n')
            .filter(Boolean)
            .map((line) => JSON.parse(line)),
        );
      } catch {
        json(res, 404, {
          error:
            'Prepare data/pg19-512.jsonl first with scripts/prepare_pg19.py',
        });
      }
      return;
    }
    if (req.method === 'GET' && url.pathname === '/api/runtime') {
      json(res, 200, {
        diffusion: synthetic
          ? {
              synthetic: true,
              ready: true,
              verified: false,
              canvas_length: 256,
            }
          : await canvasCommand(await getConfig(), 'status'),
      });
      return;
    }
    if (req.method === 'GET' && url.pathname === '/api/activity') {
      json(res, 200, { active: Boolean(active) });
      return;
    }
    if (req.method === 'GET' && url.pathname === '/api/health') {
      const c = await getConfig();
      const checks = await Promise.all(
        c.models.map(async (m) => {
          if (synthetic) return [m.id, { ok: true, synthetic: true }];
          try {
            const key = process.env[m.apiKeyEnv];
            const r = await fetch(m.baseUrl.replace(/\/$/, '') + '/models', {
              headers: key ? { Authorization: `Bearer ${key}` } : {},
              signal: AbortSignal.timeout(5000),
            });
            if (!r.ok) throw Error(`HTTP ${r.status}`);
            const b = await r.json();
            if (!b.data?.some((x) => x.id === m.model))
              throw Error(`Server does not advertise ${m.model}`);
            return [m.id, { ok: true }];
          } catch (e) {
            return [m.id, { ok: false, error: e.message }];
          }
        }),
      );
      json(res, 200, Object.fromEntries(checks));
      return;
    }
    if (req.method === 'GET' && url.pathname === '/api/runs') {
      const names = (await readdir(resultsDir)).filter((f) =>
        f.endsWith('.json'),
      );
      const runs = await Promise.all(
        names.map(async (f) => {
          try {
            const r = JSON.parse(
              await readFile(path.join(resultsDir, f), 'utf8'),
            );
            return {
              id: r.id,
              created_at: r.created_at,
              kind: r.kind,
              status: r.status,
              synthetic: r.synthetic,
              settings: {
                denoising_mode: r.settings?.denoising_mode,
                denoising_steps: r.settings?.denoising_steps,
                diffusion_preview: r.settings?.diffusion_preview,
                input_tokens: r.settings?.input_tokens,
                canvas_length: r.settings?.canvas_length,
              },
              configuration: {
                runtime: {
                  canvas_length: r.configuration?.runtime?.canvas_length,
                  max_denoising_steps:
                    r.configuration?.runtime?.max_denoising_steps,
                },
              },
            };
          } catch {
            return null;
          }
        }),
      );
      json(
        res,
        200,
        runs
          .filter(Boolean)
          .sort((a, b) => b.created_at.localeCompare(a.created_at))
          .slice(0, 100),
      );
      return;
    }
    if (
      req.method === 'GET' &&
      /^\/api\/runs\/[a-f0-9-]{36}$/.test(url.pathname)
    ) {
      const id = url.pathname.split('/').pop();
      try {
        const data = await readFile(
          path.join(resultsDir, id + '.json'),
          'utf8',
        );
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(data);
      } catch {
        json(res, 404, { error: 'Run not found' });
      }
      return;
    }
    if (req.method === 'POST' && url.pathname === '/api/run') {
      if (active) {
        json(res, 409, {
          error:
            'An experiment is already running. Wait or cancel it before starting another.',
        });
        return;
      }
      if (!req.headers['content-type']?.startsWith('application/json')) {
        json(res, 415, { error: 'Use application/json' });
        return;
      }
      active = new AbortController();
      const controller = active;
      let started = false;
      res.on('close', () => {
        if (!res.writableEnded) controller.abort();
      });
      try {
        const bodyChunks = [];
        let bytes = 0;
        for await (const chunk of req) {
          bytes += chunk.length;
          if (bytes > 20 * 1024 * 1024) throw Error('Payload exceeds 20 MB');
          bodyChunks.push(chunk);
        }
        const raw = Buffer.concat(bodyChunks).toString('utf8');
        const settings = validate(JSON.parse(raw));
        const config = await getConfig();
        if (
          settings.input_tokens != null &&
          config.runtime?.max_model_len &&
          settings.input_tokens + settings.max_tokens >
            config.runtime.max_model_len
        )
          throw Error(
            `Input text plus output must fit the ${config.runtime.max_model_len}-token context; chat formatting also needs room.`,
          );
        if (
          config.runtime?.max_model_len &&
          settings.max_tokens >= config.runtime.max_model_len
        )
          throw Error(
            `Output limit must be below the configured ${config.runtime.max_model_len}-token context budget, leaving room for the prompt.`,
          );
        controller.signal.throwIfAborted();
        res.writeHead(200, {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          Connection: 'keep-alive',
          'X-Accel-Buffering': 'no',
        });
        started = true;
        res.flushHeaders();
        const emit = (e) => {
          // Slow browsers may miss provisional frames; never let them build an
          // unbounded telemetry backlog ahead of the authoritative completion.
          if (
            e.type === 'preview' &&
            !e.preview.unavailable &&
            res.writableLength > 65536
          )
            return;
          if (!res.destroyed) res.write('data: ' + JSON.stringify(e) + '\n\n');
        };
        const heartbeat = setInterval(() => {
          if (!res.destroyed) res.write(': keepalive\n\n');
        }, 15000);
        try {
          const run = await runExperiment(settings, config, {
            signal: controller.signal,
            emit,
            synthetic,
            prepareRuntime: prepareCanvasRuntime,
          });
          await writeFile(
            path.join(resultsDir, run.id + '.json'),
            JSON.stringify(run, null, 2),
          );
          await writeFile(
            path.join(resultsDir, run.id + '.jsonl'),
            run.results
              .map((r) => JSON.stringify({ ...r, run_id: run.id, synthetic }))
              .join('\n') + '\n',
          );
          emit({ type: 'complete', run });
          res.end();
        } finally {
          clearInterval(heartbeat);
        }
      } catch (e) {
        if (!started) json(res, 400, { error: e.message });
        else {
          res.write(
            'data: ' +
              JSON.stringify({ type: 'error', message: e.message }) +
              '\n\n',
          );
          res.end();
        }
      } finally {
        active = null;
      }
      return;
    }
    json(res, 404, { error: 'Not found' });
  } catch (e) {
    if (!res.headersSent) json(res, 500, { error: e.message });
    else res.end();
  }
});
server.listen(port, '127.0.0.1', () =>
  console.log(
    `Spark Lab bridge: http://127.0.0.1:${port}${synthetic ? ' [SYNTHETIC TEST MODE]' : ''}`,
  ),
);
function shutdown() {
  active?.abort();
  server.close();
  server.closeIdleConnections();
}
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
