import { spawn } from 'node:child_process';
import { collectCompletion } from './stream.mjs';

const warmed = new Set();
// The remote helper verifies Linux process identity, idle state and startup args.
// Never interpolate request text into the SSH command.
export function canvasCommand(
  config,
  action,
  canvasLength,
  { signal, emit = () => {} } = {},
) {
  const host = config.runtime?.canvas_controller?.ssh_host;
  if (!host || !/^[A-Za-z0-9_][A-Za-z0-9_.@-]*$/.test(host))
    throw Error(
      'Configure a trusted canvas_controller.ssh_host for the native Diffusion runtime',
    );
  if (!['status', 'ensure'].includes(action))
    throw Error('Invalid canvas action');
  if (action === 'ensure' && ![64, 128, 256, 512].includes(canvasLength))
    throw Error('Unsupported canvas length');
  return new Promise((resolve, reject) => {
    const child = spawn(
      'ssh',
      [
        '-o',
        'BatchMode=yes',
        '-o',
        'ConnectTimeout=8',
        host,
        `python3 .local/share/spark-gemma-lab/canvas-runtime.py ${action}${action === 'ensure' ? ` ${canvasLength}` : ''}`,
      ],
      { signal, stdio: ['ignore', 'pipe', 'pipe'] },
    );
    let stdout = '',
      stderr = '',
      pending = '';
    const timeout = setTimeout(
      () => child.kill('SIGTERM'),
      action === 'status' ? 15000 : 720000,
    );
    child.stdout.on('data', (chunk) => {
      stdout += chunk;
      if (stdout.length > 65536) child.kill('SIGTERM');
    });
    child.stderr.on('data', (chunk) => {
      stderr = (stderr + chunk).slice(-4000);
      pending += chunk;
      const lines = pending.split('\n');
      pending = lines.pop();
      for (const line of lines)
        if (line.trim()) emit({ type: 'phase', message: line.slice(0, 500) });
    });
    child.on('error', (error) => {
      clearTimeout(timeout);
      reject(error);
    });
    child.on('close', (code) => {
      clearTimeout(timeout);
      if (signal?.aborted)
        return reject(
          Error(
            'Cancelled during runtime preparation; a reload already started may finish in the background.',
          ),
        );
      if (code !== 0)
        return reject(
          Error(
            `Diffusion runtime preparation failed: ${stderr.trim() || stdout.trim() || `SSH exited ${code}`}`,
          ),
        );
      try {
        resolve(JSON.parse(stdout));
      } catch {
        reject(Error('Invalid runtime attestation from Spark'));
      }
    });
  });
}

export function verifyCanvas(attestation, requested) {
  if (
    !attestation?.verified ||
    !attestation?.ready ||
    attestation.canvas_length !== requested
  )
    throw Error(
      `Requested canvas ${requested} does not match a ready, verified Diffusion runtime (reported ${attestation?.canvas_length ?? 'unknown'})`,
    );
  return attestation;
}

export async function prepareCanvasRuntime(
  settings,
  config,
  { signal, emit, fetcher = fetch, command = canvasCommand },
) {
  emit({
    type: 'phase',
    message: `Preparing ${settings.canvas_length}-token diffusion canvas · reloads and warmup are excluded from timing`,
  });
  const attestation = verifyCanvas(
    await command(config, 'ensure', settings.canvas_length, { signal, emit }),
    settings.canvas_length,
  );
  const count = Math.min(
    config.runtime?.capacity ?? 4,
    Math.max(...settings.batch_sizes),
  );
  const identity = `${attestation.boot_id ?? ''}:${attestation.pid}:${attestation.starttime}:${attestation.canvas_length}:c=${count}`;
  const result = {
    ...attestation,
    requested_canvas_length: settings.canvas_length,
    verified_at: new Date().toISOString(),
  };
  if (warmed.has(identity))
    return {
      ...result,
      warmup: { status: 'already_warmed', runtime_identity: identity },
    };
  emit({
    type: 'phase',
    message: `Warming ${settings.canvas_length}-token canvas · previews off · excluded from measurements`,
  });
  const model = config.models.find((m) => m.id === 'diffusion');
  const key = process.env[model.apiKeyEnv];
  const settled = await Promise.allSettled(
    Array.from({ length: count }, async () => {
      signal.throwIfAborted();
      const start = performance.now();
      const payload = {
        model: model.model,
        messages: [
          {
            role: 'user',
            content:
              'Explain how a computer processes instructions. Write several detailed paragraphs.',
          },
        ],
        max_tokens: settings.canvas_length * 2,
        ignore_eos: true,
        chat_template_kwargs: { enable_thinking: false },
        stream: true,
        return_token_ids: true,
        stream_options: { include_usage: true, continuous_usage_stats: true },
        vllm_xargs: {
          spark_lab_diffusion_max_steps: 48,
          spark_lab_diffusion_force_steps: 0,
          spark_lab_diffusion_preview: 0,
        },
      };
      const response = await fetcher(
        model.baseUrl.replace(/\/$/, '') + '/chat/completions',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(key ? { Authorization: `Bearer ${key}` } : {}),
          },
          body: JSON.stringify(payload),
          signal: AbortSignal.any([signal, AbortSignal.timeout(240000)]),
        },
      );
      if (!response.ok)
        throw Error(
          `Canvas warmup failed: HTTP ${response.status}: ${(await response.text()).slice(0, 400)}`,
        );
      const measured = await collectCompletion(response.body, {
        start,
        denoisingOptions: {
          canvas_length: settings.canvas_length,
          max_tokens: payload.max_tokens,
          mode: 'adaptive',
          max_steps: 48,
        },
      });
      if (
        measured.denoising?.status !== 'available' ||
        measured.completion_tokens !== payload.max_tokens
      )
        throw Error(
          'Canvas warmup trace does not verify the requested native block size: ' +
            (measured.denoising?.reason ?? 'incorrect output count'),
        );
      return {
        completion_tokens: measured.completion_tokens,
        elapsed_ms: measured.elapsed_ms,
        denoising: measured.denoising,
        preview_enabled: false,
      };
    }),
  );
  const failed = settled.find((item) => item.status === 'rejected');
  if (failed) throw failed.reason;
  const warmups = settled.map((item) => item.value);
  signal.throwIfAborted();
  warmed.add(identity);
  return {
    ...result,
    warmup: {
      status: 'complete',
      runtime_identity: identity,
      results: warmups,
    },
  };
}
