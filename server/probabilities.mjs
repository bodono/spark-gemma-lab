const descriptor = {
  version: 1,
  source: 'choice.logprobs.content',
  logprobs_mode: 'raw',
  offset_unit: 'utf16',
};
const probabilityDescriptor = (kind) =>
  kind === 'diffusion'
    ? {
        ...descriptor,
        logprobs_mode: 'final_denoising',
        probability_semantics:
          'Final converging denoising-pass probabilities after the configured diffusion temperature, top-k and top-p filters, conditioned on the current canvas and prefix.',
      }
    : descriptor;
const decoder = new TextDecoder('utf-8', { fatal: true });

// vLLM returns UTF-8 bytes of its incrementally decoded token string, not raw
// tokenizer byte fragments. Empty decoded tokens belong with the next visible
// token; their separate conditional probabilities must remain separate.
export function createTokenProbabilities(
  emit = () => {},
  { kind = 'ar', hiddenPrefixTokens = [], hiddenTerminalTokens = [] } = {},
) {
  const metadata = probabilityDescriptor(kind);
  const tokens = [],
    spans = [],
    candidates = [],
    pendingEmpty = [],
    omitted = [],
    stopEligible = new Set();
  const segmenter = new Intl.Segmenter('und', { granularity: 'grapheme' });
  let text = '',
    cursor = 0,
    nextToken = 0,
    invalid = null,
    lastState = null,
    attributionGap = false,
    prefixPending = false,
    prefixStripped = false,
    prefixVerified = false;
  const state = (final) => {
    if (invalid) return { status: 'invalid', reason: invalid };
    if (!tokens.length)
      return {
        status: 'unavailable',
        reason:
          'The model server did not return sampled-token log probabilities.',
      };
    if (
      nextToken < tokens.length ||
      pendingEmpty.length ||
      candidates.length ||
      cursor < text.length
    )
      return {
        status: 'partial',
        reason: final
          ? 'Only the displayed spans with exact token attribution are available.'
          : 'Waiting for matching token metadata and visible text.',
      };
    if (tokens.some((token) => token.probability == null))
      return {
        status: 'partial',
        reason:
          'The server reported an unavailable probability for one or more tokens.',
      };
    return { status: 'available' };
  };
  const publishSpans = (final) => {
    if (!candidates.length) return;
    // A following combining mark or ZWJ may extend the last grapheme. Delay
    // only coloring of the unsettled tail, never authoritative response text.
    const origin = candidates[0].start;
    const boundaries = new Set(
      [...segmenter.segment(text.slice(origin))].map(
        (part) => origin + part.index,
      ),
    );
    boundaries.add(text.length);
    while (candidates.length) {
      const first = candidates[0];
      while (!boundaries.has(first.end) && candidates.length > 1) {
        const next = candidates.splice(1, 1)[0];
        first.end = next.end;
        first.token_indexes.push(...next.token_indexes);
      }
      if (!boundaries.has(first.end) || (!final && first.end === text.length))
        break;
      spans.push(candidates.shift());
    }
  };
  const align = (final = false) => {
    if (invalid) return;
    // The presentation layer must first prove that the complete prefix was
    // removed at absolute response start. Its token IDs AND decoded strings
    // must then match before any probability record can be hidden.
    if (prefixPending) return;
    if (prefixStripped && !prefixVerified) {
      if (!hiddenPrefixTokens.length || nextToken !== 0) {
        invalid =
          'The hidden response prefix could not be attributed to tokens.';
        return;
      }
      for (
        let i = 0;
        i < Math.min(tokens.length, hiddenPrefixTokens.length);
        i++
      ) {
        const expected = hiddenPrefixTokens[i];
        if (
          tokens[i].token_id !== expected.token_id ||
          tokens[i].token !== expected.token
        ) {
          invalid =
            'The hidden response prefix does not match its verified token IDs and text.';
          return;
        }
      }
      if (tokens.length < hiddenPrefixTokens.length) return;
      for (let i = 0; i < hiddenPrefixTokens.length; i++) omitted.push(i);
      nextToken = hiddenPrefixTokens.length;
      prefixVerified = true;
    }
    while (nextToken < tokens.length) {
      const token = tokens[nextToken];
      const remaining = text.slice(cursor);
      // The only hidden nonempty token we can prove is the explicitly named
      // terminal stop token. Do not drop arbitrary special-looking strings.
      if (!remaining && stopEligible.has(token.index)) {
        omitted.push(token.index);
        nextToken++;
        continue;
      }
      if (!token.token) {
        pendingEmpty.push(token.index);
        nextToken++;
        continue;
      }
      if (remaining.startsWith(token.token)) {
        candidates.push({
          start: cursor,
          end: cursor + token.token.length,
          token_indexes: [...pendingEmpty, token.index],
        });
        pendingEmpty.length = 0;
        cursor += token.token.length;
        nextToken++;
        continue;
      }
      if (!final && token.token.startsWith(remaining)) break;
      invalid = 'Token probability text does not match the generated response.';
      break;
    }
    if (!invalid) publishSpans(final);
  };
  const notify = (tokenStart, spanStart, final = false, override = null) => {
    const status = override ?? state(final);
    const signature = JSON.stringify(status);
    if (
      final ||
      tokens.length > tokenStart ||
      spans.length > spanStart ||
      (lastState !== null && signature !== lastState) ||
      invalid
    ) {
      emit({
        ...metadata,
        ...status,
        tokens: tokens.slice(tokenStart),
        spans: spans.slice(spanStart),
        ...(final && omitted.length
          ? { omitted_token_indexes: [...omitted] }
          : {}),
        final,
      });
      lastState = signature;
    }
  };
  return {
    observe(choice, visibleText, presentation = {}) {
      const tokenStart = tokens.length,
        spanStart = spans.length;
      text = visibleText;
      prefixPending = presentation.prefixPending === true;
      prefixStripped = presentation.prefixStripped === true;
      const raw = choice?.logprobs?.content;
      const visibleDelta = choice?.delta?.content ?? choice?.text ?? '';
      if (
        (!Array.isArray(raw) || raw.length === 0) &&
        ((typeof visibleDelta === 'string' && visibleDelta.length > 0) ||
          choice?.token_ids?.length > 0)
      )
        attributionGap = true;
      // Repeated text must never let a later scored token inherit the position
      // of an earlier unscored token. Native logprobs are attached per choice.
      if (
        attributionGap &&
        (tokens.length || (Array.isArray(raw) && raw.length))
      )
        invalid =
          'Sampled-token probability records are missing; exact attribution is unavailable.';

      if (raw != null && !Array.isArray(raw))
        invalid = 'Malformed sampled-token probability metadata.';
      if (Array.isArray(raw)) {
        const ids = choice?.token_ids;
        const idsMatch =
          Array.isArray(ids) &&
          ids.length === raw.length &&
          ids.every((id) => Number.isInteger(id) && id >= 0);
        if (Array.isArray(ids) && !idsMatch)
          invalid = 'Token IDs and sampled-token probabilities do not align.';
        for (let i = 0; i < raw.length; i++) {
          if (tokens.length >= 65536) {
            invalid =
              'Sampled-token probability metadata exceeded its size limit.';
            break;
          }
          const entry = raw[i];
          const validToken =
            typeof entry?.token === 'string' && entry.token.length <= 65536;
          const validBytes =
            Array.isArray(entry?.bytes) &&
            entry.bytes.length <= 65536 &&
            entry.bytes.every(
              (byte) => Number.isInteger(byte) && byte >= 0 && byte <= 255,
            );
          const validLogprob =
            typeof entry?.logprob === 'number' &&
            Number.isFinite(entry.logprob) &&
            entry.logprob <= 0;
          if (!validToken || !validBytes || !validLogprob)
            invalid = 'Malformed sampled-token probability metadata.';
          if (validToken && validBytes) {
            try {
              if (decoder.decode(Uint8Array.from(entry.bytes)) !== entry.token)
                invalid = 'Token text and probability bytes do not match.';
            } catch {
              invalid = 'Token probability bytes are not valid UTF-8.';
            }
          }
          const token = {
            index: tokens.length,
            token: validToken ? entry.token : '',
            bytes: validBytes ? [...entry.bytes] : null,
            ...(idsMatch ? { token_id: ids[i] } : {}),
            logprob: validLogprob ? entry.logprob : null,
            probability:
              validLogprob && entry.logprob > -9999
                ? Math.exp(entry.logprob)
                : null,
          };
          tokens.push(token);
          if (
            i === raw.length - 1 &&
            choice?.finish_reason === 'stop' &&
            idsMatch &&
            ((Number.isInteger(choice?.stop_reason) &&
              token.token_id === choice.stop_reason) ||
              hiddenTerminalTokens.some(
                (expected) =>
                  token.token_id === expected.token_id &&
                  token.token === expected.token,
              ))
          )
            stopEligible.add(token.index);
        }
      }
      align();
      notify(tokenStart, spanStart);
    },
    finish(visibleText, { interrupted = false, presentation } = {}) {
      const tokenStart = tokens.length,
        spanStart = spans.length;
      text = visibleText;
      if (presentation) {
        prefixPending = presentation.prefixPending === true;
        prefixStripped = presentation.prefixStripped === true;
      }
      align(true);
      const status =
        interrupted && !invalid
          ? {
              status: tokens.length ? 'partial' : 'unavailable',
              reason:
                'The request ended before complete probability attribution was available.',
            }
          : state(true);
      notify(tokenStart, spanStart, true, status);
      return {
        ...metadata,
        ...status,
        tokens,
        spans,
        ...(omitted.length ? { omitted_token_indexes: [...omitted] } : {}),
      };
    },
  };
}

// Accumulate append-only streaming metadata, including on interrupted requests.
export function appendProbabilityDelta(previous, delta) {
  const { final, tokens = [], spans = [], ...metadata } = delta;
  const snapshot = {
    ...previous,
    ...metadata,
    tokens: [...(previous?.tokens ?? []), ...tokens],
    spans: [...(previous?.spans ?? []), ...spans],
  };
  if (metadata.reason === undefined) delete snapshot.reason;
  return snapshot;
}
export function interruptedProbabilities(previous, { kind = 'ar' } = {}) {
  return {
    ...probabilityDescriptor(kind),
    ...previous,
    status:
      previous?.status === 'invalid'
        ? 'invalid'
        : previous?.tokens?.length
          ? 'partial'
          : 'unavailable',
    reason:
      previous?.status === 'invalid'
        ? previous.reason
        : 'The request ended before complete probability attribution was available.',
    tokens: previous?.tokens ?? [],
    spans: previous?.spans ?? [],
  };
}
