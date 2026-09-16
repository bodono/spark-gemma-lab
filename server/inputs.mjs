import { createHash } from 'node:crypto';

const digest = (value) => createHash('sha256').update(value).digest('hex');
const sameIds = (a, b) =>
  Array.isArray(a) &&
  Array.isArray(b) &&
  a.length === b.length &&
  a.every((id, i) => id === b[i]);

// Keep preflight rendering and the actual inference request identical.
export function promptSpec(settings, source) {
  const raw = settings.kind === 'profile' && settings.workload === 'raw';
  const prompt =
    settings.kind === 'profile' && settings.dataset.length && !raw
      ? 'Continue the following book passage in the same narrative style. Write at least 800 words of new prose. Return only the continuation, without commentary or repeating the excerpt.\n\n<passage>\n' +
        source +
        '\n</passage>\n\nContinuation:'
      : source;
  return {
    raw,
    prompt,
    fields: raw
      ? { prompt, add_special_tokens: false }
      : {
          messages: [{ role: 'user', content: prompt }],
          chat_template_kwargs: { enable_thinking: false },
        },
  };
}

export function verifyInputLength(measured, expected) {
  return (
    measured.prompt_tokens === expected.count &&
    sameIds(measured.prompt_token_ids, expected.tokens)
  );
}

// CPU tokenizer API calls only. This completes for every used source before
// warmup/measurement wave timers start; cached text is shared by both models.
export async function prepareInputs(
  settings,
  config,
  { signal, emit, fetcher = fetch, synthetic = false },
) {
  if (settings.kind !== 'profile' || settings.input_tokens == null)
    return new Map();
  if (synthetic)
    throw Error(
      'Exact input sizing requires real tokenizer endpoints; leave input length blank for synthetic tests.',
    );
  const target = settings.input_tokens;
  const largest = Math.max(...settings.batch_sizes);
  const required =
    largest * settings.warmups +
    (settings.requests_per_condition ?? largest * settings.repeats);
  const sources = [
    ...new Set(
      settings.dataset.length
        ? settings.dataset.slice(0, required).map((s) => s.prompt)
        : [settings.prompt],
    ),
  ];
  const prepared = new Map();
  async function post(model, route, payload) {
    signal.throwIfAborted();
    const key = process.env[model.apiKeyEnv];
    const root = model.baseUrl.replace(/\/$/, '').replace(/\/v1$/, '');
    const response = await fetcher(root + route, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(key ? { Authorization: `Bearer ${key}` } : {}),
      },
      body: JSON.stringify({ model: model.model, ...payload }),
      signal: AbortSignal.any([signal, AbortSignal.timeout(30000)]),
    });
    if (!response.ok)
      throw Error(
        `${model.label || model.id} input preparation: HTTP ${response.status}: ${(await response.text()).slice(0, 500)}`,
      );
    return response.json();
  }
  async function tokenize(model, fields) {
    const data = await post(model, '/tokenize', {
      ...fields,
      add_special_tokens: false,
    });
    if (
      !Array.isArray(data.tokens) ||
      data.count !== data.tokens.length ||
      data.tokens.some((id) => !Number.isInteger(id) || id < 0)
    )
      throw Error(`${model.id}: invalid tokenizer response`);
    return data;
  }
  const reference = config.models[0];
  if (!reference) throw Error('No model endpoints configured');
  for (const source of sources) {
    emit({
      type: 'phase',
      message: `Preparing input lengths · ${prepared.size + 1}/${sources.length} sources · ${target} source tokens`,
    });
    const original = await tokenize(reference, { prompt: source });
    if (!original.count) throw Error('Source text has no tokenizer tokens');
    let expanded = source,
      encoded = original,
      repetitions = 1;
    if (original.count < target) {
      // Repeat text with an explicit separator, then tokenize the boundaries.
      repetitions = Math.ceil(target / original.count) + 1;
      for (let attempt = 0; attempt < 4; attempt++) {
        if (repetitions * (source.length + 2) > 1000000)
          throw Error(
            'Repeated source would exceed the input preparation size limit',
          );
        expanded = Array(repetitions).fill(source).join('\n\n');
        encoded = await tokenize(reference, { prompt: expanded });
        if (encoded.count >= target) break;
        repetitions *= 2;
      }
    }
    if (encoded.count < target)
      throw Error('Cannot fill the requested input length from this source');
    const selectedIds = encoded.tokens.slice(0, target);
    const shaped =
      original.count === target
        ? source
        : (await post(reference, '/detokenize', { tokens: selectedIds }))
            .prompt;
    if (typeof shaped !== 'string' || !shaped.trim())
      throw Error(
        'Requested input length produces empty text; choose a longer length',
      );
    const verified = await Promise.all(
      config.models.map((m) => tokenize(m, { prompt: shaped })),
    );
    if (
      verified.some(
        (v) => v.count !== target || !sameIds(v.tokens, selectedIds),
      )
    )
      throw Error(
        `Cannot represent exactly ${target} source tokens losslessly with both tokenizers. The cut may split a Unicode character or retokenize differently; adjust the input length or source text.`,
      );
    const spec = promptSpec(settings, shaped);
    const rendered = await Promise.all(
      config.models.map((m) =>
        tokenize(m, {
          ...spec.fields,
          ...(!spec.raw ? { add_generation_prompt: true } : {}),
        }),
      ),
    );
    const byModel = new Map();
    for (let i = 0; i < config.models.length; i++) {
      const model = config.models[i],
        input = rendered[i];
      const capacity = Math.min(
        config.runtime?.max_model_len ?? Infinity,
        input.max_model_len ?? Infinity,
      );
      if (input.count + settings.max_tokens > capacity)
        throw Error(
          `${model.label || model.id}: ${target} source tokens render to ${input.count} input tokens; plus ${settings.max_tokens} output tokens exceeds the ${capacity}-token context. Reduce input or output length.`,
        );
      byModel.set(model.id, input);
    }
    const policy =
      original.count < target
        ? 'repeat_then_trim'
        : original.count > target
          ? 'trim'
          : 'unchanged';
    prepared.set(source, {
      source: shaped,
      spec,
      byModel,
      audit: {
        requested_source_tokens: target,
        actual_source_tokens: target,
        original_source_tokens: original.count,
        policy,
        expansion_copies: repetitions,
        repeat_separator: repetitions > 1 ? '\n\n' : null,
        original_source_hash: digest(source),
        shaped_source_hash: digest(shaped),
        source_token_ids_hash: digest(JSON.stringify(selectedIds)),
        rendered_inputs: Object.fromEntries(
          config.models.map((m, i) => [
            m.id,
            {
              prompt_tokens: rendered[i].count,
              prompt_token_ids_hash: digest(JSON.stringify(rendered[i].tokens)),
              tokenizer_checkpoint: m.checkpoint ?? m.model,
              tokenizer_revision: m.revision ?? null,
            },
          ]),
        ),
      },
    });
  }
  return prepared;
}
