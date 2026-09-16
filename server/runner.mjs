import {
  appendProbabilityDelta,
  interruptedProbabilities,
} from './probabilities.mjs';
import { createProgress } from './progress.mjs';
import { createHash, randomUUID } from 'node:crypto';
import { collectCompletion } from './stream.mjs';
import { summarize } from './metrics.mjs';
import { prepareInputs, promptSpec, verifyInputLength } from './inputs.mjs';
const hash = (s) => createHash('sha256').update(s).digest('hex');
const integer = (v, min, max, name) => {
  if (!Number.isInteger(v) || v < min || v > max)
    throw Error(`${name} must be an integer from ${min} to ${max}`);
  return v;
};
export function validate(input) {
  if (!['demo', 'profile'].includes(input.kind))
    throw Error('kind must be demo or profile');
  if (
    typeof input.prompt !== 'string' ||
    !input.prompt.trim() ||
    input.prompt.length > 100000
  )
    throw Error('Enter a prompt between 1 and 100,000 characters');
  const x = {
    kind: input.kind,
    prompt: input.prompt,
    max_tokens: integer(input.max_tokens, 1, 32768, 'max_tokens'),
    batch_size: integer(input.batch_size, 1, 128, 'batch_size'),
    seed: integer(input.seed, 0, 2147483647, 'seed'),
    temperature: input.temperature,
    canvas_length: input.canvas_length ?? 256,
  };
  if (![8, 16, 32, 64, 128, 256, 512].includes(x.canvas_length))
    throw Error('canvas_length must be 8, 16, 32, 64, 128, 256 or 512 tokens');
  if (
    input.diffusion_preview != null &&
    typeof input.diffusion_preview !== 'boolean'
  )
    throw Error('diffusion_preview must be a boolean');
  if (
    input.ar_token_probabilities !== undefined &&
    typeof input.ar_token_probabilities !== 'boolean'
  )
    throw Error('ar_token_probabilities must be a boolean');
  x.ar_token_probabilities =
    input.kind === 'demo' && input.ar_token_probabilities === true;
  if (
    input.diffusion_token_probabilities !== undefined &&
    typeof input.diffusion_token_probabilities !== 'boolean'
  )
    throw Error('diffusion_token_probabilities must be a boolean');
  x.diffusion_token_probabilities =
    input.kind === 'demo' && input.diffusion_token_probabilities === true;
  // Profiling, including its warmups, must never request intermediate snapshots.
  x.diffusion_preview =
    input.kind === 'demo' && input.diffusion_preview === true;
  x.denoising_mode =
    input.kind === 'demo' ? 'adaptive' : (input.denoising_mode ?? 'adaptive');
  if (!['adaptive', 'fixed'].includes(x.denoising_mode))
    throw Error('denoising_mode must be adaptive or fixed');
  x.denoising_steps =
    x.denoising_mode === 'adaptive'
      ? 48
      : integer(input.denoising_steps ?? 16, 1, 48, 'denoising_steps');
  if (
    typeof x.temperature !== 'number' ||
    !Number.isFinite(x.temperature) ||
    x.temperature < 0
  )
    throw Error('AR temperature must be a finite number greater than or equal to 0');
  x.dataset =
    input.kind === 'profile' && Array.isArray(input.dataset)
      ? input.dataset
      : [];
  if (
    x.dataset.length > 20000 ||
    x.dataset.some(
      (r) =>
        typeof r.prompt !== 'string' ||
        !r.prompt.trim() ||
        r.prompt.length > 100000,
    )
  )
    throw Error('Invalid dataset: expected nonempty prompt strings');
  x.dataset_name =
    input.kind === 'demo'
      ? 'Current prompt'
      : String(input.dataset_name || 'Current prompt').slice(0, 250);
  if (x.kind === 'profile') {
    x.input_tokens =
      input.input_tokens == null
        ? null
        : integer(input.input_tokens, 1, 32768, 'input_tokens');
    x.workload = input.workload ?? 'continuation';
    if (!['continuation', 'raw'].includes(x.workload))
      throw Error('workload must be continuation or raw');
    x.output_mode = input.output_mode ?? 'natural';
    if (!['natural', 'fixed'].includes(x.output_mode))
      throw Error('output_mode must be natural or fixed');
    if (
      !Array.isArray(input.batch_sizes) ||
      !input.batch_sizes.length ||
      input.batch_sizes.length > 12
    )
      throw Error('Choose 1–12 batch sizes');
    x.batch_sizes = [
      ...new Set(
        input.batch_sizes.map((n) => integer(n, 1, 128, 'batch size')),
      ),
    ];
    x.repeats = integer(input.repeats, 1, 100, 'repeats');
    x.warmups = integer(input.warmups, 0, 10, 'warmups');
    if (input.requests_per_condition != null) {
      x.requests_per_condition = integer(
        input.requests_per_condition,
        1,
        10000,
        'requests_per_condition',
      );
      if (x.batch_sizes.some((size) => x.requests_per_condition % size !== 0))
        throw Error(
          'Requests per condition must be divisible by every batch size',
        );
    }
    const requiredSamples =
      Math.max(...x.batch_sizes) * x.warmups +
      (x.requests_per_condition ?? Math.max(...x.batch_sizes) * x.repeats);
    if (x.dataset.length && x.dataset.length < requiredSamples)
      throw Error(
        `Dataset needs at least ${requiredSamples} prompts for disjoint warmups and measured requests`,
      );
  } else {
    x.batch_sizes = [x.batch_size];
    x.repeats = 1;
    x.warmups = 0;
  }
  return x;
}
async function syntheticResponse(signal) {
  const text =
    'This is a synthetic integration-test response. It does not come from a GPU or measure model performance.';
  return new Response(
    new ReadableStream({
      async start(c) {
        try {
          for (const t of [text.slice(0, 35), text.slice(35)]) {
            await new Promise((resolve, reject) => {
              const t = setTimeout(resolve, 30);
              signal.addEventListener(
                'abort',
                () => {
                  clearTimeout(t);
                  reject(Error('Cancelled'));
                },
                { once: true },
              );
            });
            c.enqueue(
              new TextEncoder().encode(
                'data: ' +
                  JSON.stringify({
                    choices: [{ delta: { content: t }, finish_reason: null }],
                  }) +
                  '\n\n',
              ),
            );
          }
          c.enqueue(
            new TextEncoder().encode(
              'data: ' +
                JSON.stringify({
                  choices: [{ delta: {}, finish_reason: 'length' }],
                  usage: { prompt_tokens: 12, completion_tokens: 24 },
                }) +
                '\n\ndata: [DONE]\n\n',
            ),
          );
          c.close();
        } catch (e) {
          c.error(e);
        }
      },
    }),
  );
}
export async function runExperiment(
  settings,
  config,
  {
    signal,
    emit,
    synthetic = false,
    fetcher = fetch,
    prepareRuntime,
    progressTracker,
  },
) {
  settings = {
    ...settings,
    diffusion_preview:
      settings.kind === 'demo' && settings.diffusion_preview === true,
    ar_token_probabilities:
      settings.kind === 'demo' && settings.ar_token_probabilities === true,
    diffusion_token_probabilities:
      settings.kind === 'demo' &&
      settings.diffusion_token_probabilities === true,
  };
  const publish = emit;
  const progress =
    progressTracker ?? createProgress(settings, config.models.length, publish);
  emit = (event) => {
    if (event.type === 'phase') progress.phase(event);
    publish(event);
  };
  try {
    // The production bridge attests the native runtime before tokenizer work or timers.
    const runtimePreparation =
      !synthetic && prepareRuntime
        ? await prepareRuntime(settings, config, { signal, emit, fetcher })
        : null;
    if (runtimePreparation) {
      if (
        !runtimePreparation.verified ||
        !runtimePreparation.ready ||
        runtimePreparation.canvas_length !== settings.canvas_length
      )
        throw Error(
          'Requested canvas length does not match the verified active runtime',
        );
      config = {
        ...config,
        runtime: {
          ...config.runtime,
          canvas_length: runtimePreparation.canvas_length,
        },
      };
    } else if (
      settings.canvas_length !== (config.runtime?.canvas_length ?? 256) &&
      !synthetic
    ) {
      throw Error(
        'Changing canvas length requires verified runtime preparation',
      );
    }
    const run = {
      schema_version: 5,
      ...(runtimePreparation
        ? { runtime_preparation: runtimePreparation }
        : {}),
      id: randomUUID(),
      kind: settings.kind,
      created_at: new Date().toISOString(),
      synthetic,
      status: 'running',
      settings,
      configuration: config,
      results: [],
      summaries: [],
      waves: [],
      warmup_results: [],
      measurement: {
        clock: 'local monotonic performance.now',
        rate: 'server completion tokens / end-to-end seconds',
        aggregate:
          'sum completion tokens / sum measured per-endpoint wave makespans',
        batch_semantics: 'concurrent HTTP requests, not observed GPU batch',
        post_first_block_rate:
          'Tokens after the first reported token block / arrival time from that block to the last token-count increase; excludes prefill AND the first output block, not an isolated kernel decode measurement',
        workload_note:
          'Dataset continuation mode wraps each excerpt in the same chat instruction for both models. Natural output mode respects EOS; fixed mode forces length and may count special-token tails. Token IDs are retained for auditing.',
        input_length_note:
          'Optional input_tokens is the shared source-text length before instructions/chat formatting. Short sources repeat with two newlines; long sources trim. Lossless tokenizer agreement and rendered context budgets are verified before any timed waves. Actual per-model prompt counts/IDs are checked after generation. Repeated excerpts are not longer contiguous corpus passages.',
        canvas_note:
          'Native Diffusion canvas length is verified before timing. Changes restart only the owned Diffusion server and trigger an unmeasured warmup; other sizes are experimental. AR is unaffected.',
        sampler_note:
          'Temperature and seed apply only to AR. Its native temperature validation ceiling is patched to allow finite nonnegative values above 2. Diffusion uses its own temperature schedule and does not use a request-specific random seed.',
        probability_note:
          'Optional token probabilities are demo-only and may add overhead. AR uses raw sampled-token model log probabilities. Diffusion uses final converging denoising-pass probabilities after its configured temperature/top-k/top-p filters, conditioned on the current canvas and prefix; these differ from AR probabilities. Profiling and all warmups omit logprobs. Display spans require exact text attribution; they are not probabilities of correctness.',
        preview_note:
          'Live intermediate predictions are opt-in for demos only. They are not completion tokens or first real output. Demo timings include any collection/transport overhead. Profiling and its warmups explicitly disable GPU snapshot collection.',
        denoising_note:
          'Demo uses adaptive convergence with a 48-step maximum. Fixed-step profiling disables early convergence and changes the schedule budget; it is a speed/quality experiment. Per-block counts come from request-specific scheduler events, exclude commit passes, and are unavailable after preemption.',
        presentation_note:
          'Diffusion chat strips only a verified empty Gemma thought-channel prefix. Original deltas remain in chunks[].raw_text; token counts and timings are not estimated.',
      },
    };
    if (settings.kind === 'profile' && settings.input_tokens != null)
      emit({
        type: 'phase',
        stage: 'inputs',
        message:
          'Preparing and verifying input token lengths; excluded from measurement.',
      });
    const preparedInputs = await prepareInputs(settings, config, {
      signal,
      emit,
      fetcher,
      synthetic,
    });
    if (preparedInputs.size)
      run.input_preparation = [...preparedInputs.values()].map(
        ({ audit }) => audit,
      );
    let waveSequence = 0;
    async function wave(batchSize, repeat, warmup) {
      const waveId = waveSequence++;
      emit({
        type: 'phase',
        stage: warmup ? 'warmup' : 'measuring',
        batch_size: batchSize,
        wave: repeat + 1,
        waves: warmup
          ? settings.warmups
          : settings.requests_per_condition
            ? settings.requests_per_condition / batchSize
            : settings.repeats,
        message: `${warmup ? 'Warmup' : 'Measuring'} · c=${batchSize} · wave ${repeat + 1}/${warmup ? settings.warmups : settings.requests_per_condition ? settings.requests_per_condition / batchSize : settings.repeats}`,
      });
      // Prepare every request before releasing the asynchronous wave to both endpoints.
      const offset = warmup
        ? repeat * batchSize
        : Math.max(...settings.batch_sizes) * settings.warmups +
          repeat * batchSize;
      const prompts = Array.from({ length: batchSize }, (_, i) =>
        settings.dataset.length
          ? settings.dataset[offset + i]
          : { prompt: settings.prompt },
      );
      await Promise.all(
        config.models.map(async (model) => {
          const waveStart = performance.now();
          const results = await Promise.all(
            prompts.map(async (sample, index) => {
              const prepared = preparedInputs.get(sample.prompt);
              const expectedInput = prepared?.byModel.get(model.id);
              const allowPreview =
                settings.kind === 'demo' &&
                settings.diffusion_preview === true &&
                model.id === 'diffusion' &&
                !warmup;
              const allowTokenProbabilities =
                settings.kind === 'demo' &&
                ((settings.ar_token_probabilities === true &&
                  model.id === 'autoregressive') ||
                  (settings.diffusion_token_probabilities === true &&
                    model.id === 'diffusion')) &&
                !warmup;
              const probabilityKind =
                model.id === 'diffusion' ? 'diffusion' : 'ar';
              const probabilityField =
                model.id === 'diffusion'
                  ? 'diffusion_token_probabilities'
                  : 'ar_token_probabilities';
              const requestId = randomUUID(),
                requestStart = performance.now();
              const base = {
                model: model.id,
                index,
                request_id: requestId,
                wave: waveId,
                repeat,
                warmup,
                batch_size: batchSize,
                canvas_length:
                  model.id === 'diffusion' ? settings.canvas_length : null,
                prompt_hash: hash(sample.prompt),
                ...(prepared ? { input_preparation: prepared.audit } : {}),
                sample_metadata: Object.fromEntries(
                  Object.entries(sample).filter(([key]) => key !== 'prompt'),
                ),
                started_at: new Date().toISOString(),
                dispatch_offset_ms: requestStart - waveStart,
              };
              if (!warmup)
                emit({
                  type: 'start',
                  model: model.id,
                  index,
                  request_id: requestId,
                });
              const { raw, prompt, fields } =
                prepared?.spec ?? promptSpec(settings, sample.prompt);
              base.effective_prompt_hash = hash(prompt);
              const payload = {
                model: model.model,
                ...fields,
                max_tokens: settings.max_tokens,
                ...(model.id === 'diffusion'
                  ? {
                      vllm_xargs: {
                        spark_lab_diffusion_max_steps: settings.denoising_steps,
                        spark_lab_diffusion_force_steps:
                          settings.denoising_mode === 'fixed' ? 1 : 0,
                        spark_lab_diffusion_preview: allowPreview ? 1 : 0,
                      },
                    }
                  : { temperature: settings.temperature, seed: settings.seed }),
                stream: true,
                return_token_ids: true,
                ...(allowTokenProbabilities
                  ? { logprobs: true, top_logprobs: 0 }
                  : {}),
                stream_options: {
                  include_usage: true,
                  continuous_usage_stats: true,
                },
                ...(settings.kind === 'profile'
                  ? { ignore_eos: settings.output_mode === 'fixed' }
                  : {}),
              };
              base.request_payload = payload;
              let result,
                probabilitySnapshot,
                probabilityFinal = false;
              const receiveProbabilities = (delta) => {
                probabilityFinal = delta.final === true;
                probabilitySnapshot = appendProbabilityDelta(
                  probabilitySnapshot,
                  delta,
                );
                emit({
                  type: 'token_probabilities',
                  model: model.id,
                  index,
                  request_id: requestId,
                  ...delta,
                });
              };
              const received = [];
              try {
                signal.throwIfAborted();
                const key = process.env[model.apiKeyEnv];
                const response = synthetic
                  ? await syntheticResponse(signal)
                  : await fetcher(
                      model.baseUrl.replace(/\/$/, '') +
                        (raw ? '/completions' : '/chat/completions'),
                      {
                        method: 'POST',
                        headers: {
                          'Content-Type': 'application/json',
                          ...(key ? { Authorization: `Bearer ${key}` } : {}),
                        },
                        body: JSON.stringify(payload),
                        signal: AbortSignal.any([
                          signal,
                          AbortSignal.timeout(1200000),
                        ]),
                      },
                    );
                if (!response.ok)
                  throw Error(
                    `HTTP ${response.status}: ${(await response.text()).slice(0, 800)}`,
                  );
                if (!response.body) throw Error('Missing response stream');
                const measured = await collectCompletion(response.body, {
                  start: requestStart,
                  previewEnabled: allowPreview,
                  tokenProbabilitiesEnabled: allowTokenProbabilities,
                  tokenProbabilitiesKind: probabilityKind,
                  emitTokenProbabilities: receiveProbabilities,
                  emitPreview: (preview) =>
                    emit({
                      type: 'preview',
                      model: model.id,
                      index,
                      request_id: requestId,
                      preview,
                    }),
                  stripEmptyGemmaChannel: model.id === 'diffusion' && !raw,
                  denoisingOptions:
                    model.id === 'diffusion'
                      ? {
                          mode: settings.denoising_mode,
                          max_steps: settings.denoising_steps,
                          canvas_length: settings.canvas_length,
                          max_tokens: settings.max_tokens,
                        }
                      : null,
                  emit: (chunk) => {
                    received.push(chunk);
                    if (!warmup)
                      emit({
                        type: 'chunk',
                        model: model.id,
                        index,
                        request_id: requestId,
                        ...chunk,
                      });
                  },
                });
                const verifiedFixedSteps =
                  model.id !== 'diffusion' ||
                  settings.denoising_mode !== 'fixed' ||
                  synthetic ||
                  measured.denoising?.status === 'available';
                const canvasOk =
                  model.id !== 'diffusion' || synthetic || !runtimePreparation
                    ? null
                    : measured.denoising?.status === 'available' &&
                      measured.denoising?.canvas_length ===
                        settings.canvas_length;
                const inputLengthOk = expectedInput
                  ? verifyInputLength(measured, expectedInput)
                  : null;
                const validationErrors = [];
                if (!verifiedFixedSteps)
                  validationErrors.push(
                    'The server did not verify the requested fixed denoising count: ' +
                      (measured.denoising?.reason ?? 'trace unavailable'),
                  );
                if (canvasOk === false)
                  validationErrors.push(
                    'The completed trace did not verify the active canvas: ' +
                      (measured.denoising?.reason ?? 'trace unavailable'),
                  );
                if (inputLengthOk === false)
                  validationErrors.push(
                    `Input length verification failed: expected ${expectedInput.count} rendered tokens and matching IDs, received ${measured.prompt_tokens ?? 'no token count'}.`,
                  );
                result = {
                  ...base,
                  ...measured,
                  status: validationErrors.length ? 'error' : 'complete',
                  input_length_ok: inputLengthOk,
                  canvas_length_ok: canvasOk,
                  ...(validationErrors.length
                    ? { error: validationErrors.join(' ') }
                    : {}),
                  fixed_length_ok:
                    settings.kind === 'profile' &&
                    settings.output_mode === 'fixed'
                      ? synthetic ||
                        measured.completion_tokens === settings.max_tokens
                      : null,
                };
              } catch (e) {
                if (allowTokenProbabilities && !probabilityFinal) {
                  probabilitySnapshot = interruptedProbabilities(
                    probabilitySnapshot,
                    {
                      kind: probabilityKind,
                    },
                  );
                  receiveProbabilities({
                    ...probabilitySnapshot,
                    tokens: [],
                    spans: [],
                    final: true,
                  });
                }
                result = {
                  ...base,
                  text: received.map((chunk) => chunk.text).join(''),
                  reasoning: received.map((chunk) => chunk.reasoning).join(''),
                  chunks: received,
                  ttft_ms: received.find((chunk) => chunk.text)?.at_ms ?? null,
                  status: signal.aborted ? 'cancelled' : 'error',
                  error: String(e.message || e),
                  elapsed_ms: performance.now() - requestStart,
                  completion_tokens: null,
                  input_length_ok: expectedInput ? false : null,
                  tokens_per_second: null,
                  ...(allowTokenProbabilities
                    ? { [probabilityField]: probabilitySnapshot }
                    : {}),
                  ...(model.id === 'diffusion'
                    ? {
                        denoising: {
                          status: 'unavailable',
                          reason:
                            'The request did not complete; block attribution is unavailable.',
                          blocks: [],
                          mean_steps: null,
                        },
                      }
                    : {}),
                };
              }
              progress.request(result);
              if (!warmup) emit({ type: 'result', result });
              return result;
            }),
          );
          const w = {
            model: model.id,
            batch_size: batchSize,
            repeat,
            warmup,
            wave: waveId,
            makespan_ms: performance.now() - waveStart,
            results,
          };
          run.waves.push(w);
          if (warmup) run.warmup_results.push(...results);
          else run.results.push(...results);
        }),
      );
    }
    conditions: for (const size of settings.batch_sizes) {
      for (let i = 0; i < settings.warmups && !signal.aborted; i++) {
        await wave(size, i, true);
        if (
          run.warmup_results.some(
            (r) =>
              r.status !== 'complete' ||
              r.fixed_length_ok === false ||
              r.input_length_ok === false ||
              r.canvas_length_ok === false,
          )
        ) {
          emit({
            type: 'phase',
            message:
              'Warmup failed or did not pass validation; measured requests were not started.',
          });
          break conditions;
        }
      }
      const measuredWaves = settings.requests_per_condition
        ? settings.requests_per_condition / size
        : settings.repeats;
      for (let i = 0; i < measuredWaves && !signal.aborted; i++)
        await wave(size, i, false);
      for (const model of config.models) {
        const waves = run.waves.filter(
          (w) => !w.warmup && w.model === model.id && w.batch_size === size,
        );
        if (waves.length) {
          const summary = {
            ...summarize(model.id, size, waves),
            diffusion_canvas_length: settings.canvas_length,
          };
          run.summaries.push(summary);
          emit({ type: 'summary', summary });
        }
      }
      if (signal.aborted) break;
    }
    run.status = signal.aborted
      ? 'cancelled'
      : run.warmup_results.some(
            (r) =>
              r.status !== 'complete' ||
              r.fixed_length_ok === false ||
              r.input_length_ok === false ||
              r.canvas_length_ok === false,
          ) ||
          run.results.some(
            (r) => r.status !== 'complete' || r.fixed_length_ok === false,
          )
        ? 'partial'
        : 'complete';
    run.finished_at = new Date().toISOString();
    // The bridge publishes completion after it has persisted the run, making the
    // terminal run_id immediately retrievable by a refreshed client.
    if (!progressTracker) progress.finish(run);
    return run;
  } catch (error) {
    progress.fail(signal.aborted);
    throw error;
  }
}
