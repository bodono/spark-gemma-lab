import { createTokenProbabilities } from './probabilities.mjs';
import { decodeDenoisingTrace } from './denoising.mjs';
// SSE is transport framing, not token framing. UTF-8 and CRLF may split anywhere.
export async function* parseSSE(body) {
  const reader = body.getReader(),
    decoder = new TextDecoder();
  let buffer = '',
    data = [];
  const take = (line) => {
    if (line === '') {
      const out = data.length ? data.join('\n') : null;
      data = [];
      return out;
    }
    if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
    return null;
  };
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      let i;
      while ((i = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, i).replace(/\r$/, '');
        buffer = buffer.slice(i + 1);
        const out = take(line);
        if (out !== null) yield out;
      }
      if (done) break;
    }
    if (buffer) {
      const out = take(buffer.replace(/\r$/, ''));
      if (out !== null) yield out;
    }
    if (data.length) yield data.join('\n');
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
export async function collectCompletion(
  body,
  {
    start,
    now = () => performance.now(),
    emit = () => {},
    previewEnabled = false,
    emitPreview = () => {},
    tokenProbabilitiesEnabled = false,
    tokenProbabilitiesKind = 'ar',
    emitTokenProbabilities = () => {},
    stripEmptyGemmaChannel = false,
    denoisingOptions = null,
  },
) {
  // vLLM 0.29's delegated parser bypasses Gemma extraction when thinking is off.
  // Decode only the verified empty channel at absolute start; preserve every raw delta.
  const canvasLength = denoisingOptions?.canvas_length ?? 256;
  const maxBlocks = Math.ceil(
    (denoisingOptions?.max_tokens ?? 32768) / canvasLength,
  );
  const maxSteps = denoisingOptions?.max_steps ?? 48;
  const emptyChannel = '<|channel>thought\n<channel|>';
  let pendingPrefix = '',
    inspectPrefix = stripEmptyGemmaChannel,
    strippedPrefix = false;
  let text = '',
    reasoning = '',
    first = null,
    last = null,
    usage = null,
    finishReason = null,
    terminal = false;
  const probabilities = tokenProbabilitiesEnabled
    ? createTokenProbabilities(emitTokenProbabilities, {
        kind: tokenProbabilitiesKind,
        hiddenPrefixTokens:
          tokenProbabilitiesKind === 'diffusion' && stripEmptyGemmaChannel
            ? [
                { token_id: 100, token: '<|channel>' },
                { token_id: 45518, token: 'thought' },
                { token_id: 107, token: '\n' },
                { token_id: 101, token: '<channel|>' },
              ]
            : [],
        // The pinned Gemma tokenizer's EOS is omitted from visible chat text.
        // vLLM's diffusion path reports finish_reason=stop, stop_reason=null.
        hiddenTerminalTokens:
          tokenProbabilitiesKind === 'diffusion'
            ? [{ token_id: 1, token: '<eos>' }]
            : [],
      })
    : null;
  const chunks = [],
    tokenProgress = [],
    outputTokenIds = [];
  let promptTokenIds = null;
  let engineMetrics = null,
    denoising = null;
  let lastPreview = null,
    previewFrames = 0,
    droppedPreviews = 0,
    previewUnavailable = null;
  try {
    for await (const raw of parseSSE(body)) {
      if (raw === '[DONE]') {
        terminal = true;
        break;
      }
      let frame;
      try {
        frame = JSON.parse(raw);
      } catch {
        throw Error('Malformed SSE JSON from model server');
      }
      if (frame.error)
        throw Error(frame.error.message || JSON.stringify(frame.error));
      if (frame.usage?.completion_tokens != null) usage = frame.usage;
      if (frame.metrics) engineMetrics = frame.metrics;
      const choice = frame.choices?.[0];
      if (Array.isArray(choice?.token_ids))
        outputTokenIds.push(...choice.token_ids);
      if (Array.isArray(frame.prompt_token_ids))
        promptTokenIds = frame.prompt_token_ids;
      // Raw completion streams attach input IDs to the choice once; chat puts
      // them at the top level. Later null fields must not erase the first array.
      else if (
        promptTokenIds == null &&
        Array.isArray(choice?.prompt_token_ids)
      )
        promptTokenIds = choice.prompt_token_ids;
      if (choice?.finish_reason != null) finishReason = choice.finish_reason;
      const rawContent = choice?.delta?.content ?? choice?.text ?? '';
      let content = rawContent;
      const thought =
        choice?.delta?.reasoning_content ?? choice?.delta?.reasoning ?? '';
      if (typeof content !== 'string' || typeof thought !== 'string')
        throw Error('Expected text delta from model server');
      if (inspectPrefix && content) {
        pendingPrefix += content;
        if (
          emptyChannel.startsWith(pendingPrefix) &&
          pendingPrefix.length < emptyChannel.length
        )
          content = '';
        else {
          strippedPrefix = pendingPrefix.startsWith(emptyChannel);
          content = strippedPrefix
            ? pendingPrefix.slice(emptyChannel.length)
            : pendingPrefix;
          pendingPrefix = '';
          inspectPrefix = false;
        }
      }
      const count = frame.usage?.completion_tokens;
      const advances =
        Number.isInteger(count) &&
        count > (tokenProgress.at(-1)?.completion_tokens ?? 0);
      const at = rawContent || thought || advances ? now() - start : null;
      if (denoisingOptions && frame.metrics) {
        denoising = decodeDenoisingTrace(
          engineMetrics,
          usage?.completion_tokens,
          {
            ...denoisingOptions,
            final: finishReason != null,
          },
        );
      }
      // Independent, provisional metadata: never append these IDs/text to the
      // completion, chunks, usage, token progress, or latency timestamps.
      if (previewEnabled && frame.diffusion_preview != null) {
        const p = frame.diffusion_preview;
        if (
          p?.version === 1 &&
          p.attribution_valid === false &&
          !previewUnavailable
        ) {
          previewUnavailable = String(
            p.reason || 'Preview attribution unavailable',
          ).slice(0, 200);
          emitPreview({ unavailable: true, reason: previewUnavailable });
        }
        const valid =
          p &&
          p.version === 1 &&
          p.final === false &&
          Number.isInteger(p.block_index) &&
          p.block_index > 0 &&
          p.block_index <= maxBlocks &&
          Number.isInteger(p.denoising_step) &&
          p.denoising_step > 0 &&
          p.denoising_step <= maxSteps &&
          Array.isArray(p.token_ids) &&
          p.token_ids.length > 0 &&
          p.token_ids.length <= canvasLength &&
          p.token_ids.every(
            (id) => Number.isInteger(id) && id >= 0 && id < 2147483648,
          ) &&
          typeof p.text === 'string' &&
          p.text.length <= 65536;
        const committedBlock =
          denoising?.status === 'available'
            ? Math.max(0, ...denoising.blocks.map((block) => block.block_index))
            : 0;
        if (
          !previewUnavailable &&
          valid &&
          previewFrames < maxBlocks * maxSteps &&
          p.block_index > committedBlock &&
          (!lastPreview ||
            p.block_index > lastPreview.block_index ||
            (p.block_index === lastPreview.block_index &&
              p.denoising_step > lastPreview.denoising_step))
        ) {
          let previewText = p.text;
          if (stripEmptyGemmaChannel && p.block_index === 1) {
            if (previewText.startsWith(emptyChannel))
              previewText = previewText.slice(emptyChannel.length);
            // Runtime previews skip special-token delimiters when decoding.
            // Verified against the pinned Gemma tokenizer: only this complete
            // four-ID prefix proves an empty thought channel. Bare prose such
            // as "thought\n" must remain untouched. Retain original token IDs.
            else if (
              [100, 45518, 107, 101].every((id, i) => p.token_ids[i] === id) &&
              previewText.startsWith('thought\n')
            )
              previewText = previewText.slice('thought\n'.length);
          }
          const preview = {
            version: 1,
            block_index: p.block_index,
            denoising_step: p.denoising_step,
            token_ids: p.token_ids,
            final: false,
            text: previewText,
          };
          lastPreview = {
            block_index: p.block_index,
            denoising_step: p.denoising_step,
          };
          previewFrames++;
          emitPreview(preview);
        } else droppedPreviews++;
      }
      if (advances) tokenProgress.push({ at_ms: at, completion_tokens: count });
      if (rawContent || thought) {
        if (content) {
          first ??= at;
          last = at;
          text += content;
        }
        reasoning += thought;
        chunks.push({
          at_ms: at,
          text: content,
          raw_text: rawContent,
          reasoning: thought,
          ...(denoising ? { denoising } : {}),
        });
        emit({
          text: content,
          raw_text: rawContent,
          reasoning: thought,
          ttft_ms: first,
          at_ms: at,
          ...(denoising ? { denoising } : {}),
        });
      }
      if (denoisingOptions && frame.metrics && !rawContent && !thought) {
        const metadata = {
          text: '',
          raw_text: '',
          reasoning: '',
          at_ms: now() - start,
          denoising,
          metadata_only: true,
        };
        chunks.push(metadata);
        emit(metadata);
      }
      probabilities?.observe(choice, text, {
        prefixPending: inspectPrefix,
        prefixStripped: strippedPrefix,
      });
    }
    if (pendingPrefix) {
      const at = now() - start;
      first ??= at;
      last = at;
      text += pendingPrefix;
      const flush = {
        at_ms: at,
        text: pendingPrefix,
        raw_text: '',
        reasoning: '',
        presentation_flush: true,
      };
      chunks.push(flush);
      emit({ ...flush, ttft_ms: first });
    }
    if (!terminal && !finishReason)
      throw Error('Model stream ended without a completion marker');
    if (!text && !reasoning) throw Error('Model returned no generated text');
  } catch (error) {
    // The authoritative partial text is now final for this failed request.
    // Flush its last stable grapheme's metadata without inventing completion.
    probabilities?.finish(text, { interrupted: true });
    throw error;
  }
  const elapsed = now() - start;
  const n =
    Number.isInteger(usage?.completion_tokens) && usage.completion_tokens >= 0
      ? usage.completion_tokens
      : null;
  const firstBlock = tokenProgress[0],
    lastBlock = tokenProgress.at(-1);
  const postFirstMs =
    tokenProgress.length > 1 ? lastBlock.at_ms - firstBlock.at_ms : null;
  const postFirstTokens =
    n != null && firstBlock ? n - firstBlock.completion_tokens : null;
  return {
    text,
    reasoning,
    ttft_ms: first,
    last_content_ms: last,
    elapsed_ms: elapsed,
    completion_tokens: n,
    prompt_tokens: usage?.prompt_tokens ?? null,
    tokens_per_second: n != null && elapsed > 0 ? n / (elapsed / 1000) : null,
    first_block_tokens: firstBlock?.completion_tokens ?? null,
    post_first_block_tps:
      postFirstMs > 0 &&
      postFirstTokens > 0 &&
      lastBlock.completion_tokens === n
        ? postFirstTokens / (postFirstMs / 1000)
        : null,
    post_first_block_ms: postFirstMs,
    token_progress: tokenProgress,
    output_token_ids: outputTokenIds.length ? outputTokenIds : null,
    prompt_token_ids: promptTokenIds,
    token_count_source: n != null ? 'server_usage' : 'unavailable',
    finish_reason: finishReason,
    ...(denoisingOptions
      ? {
          denoising: decodeDenoisingTrace(engineMetrics, n, {
            ...denoisingOptions,
            final: true,
          }),
          engine_metrics: engineMetrics,
        }
      : {}),
    ...(probabilities
      ? {
          [tokenProbabilitiesKind === 'diffusion'
            ? 'diffusion_token_probabilities'
            : 'ar_token_probabilities']: probabilities.finish(text, {
            presentation: {
              prefixPending: false,
              prefixStripped: strippedPrefix,
            },
          }),
        }
      : {}),
    chunks,
    ...(previewEnabled
      ? {
          diffusion_preview: {
            enabled: true,
            received_frames: previewFrames,
            dropped_frames: droppedPreviews,
            ...(previewUnavailable ? { unavailable: previewUnavailable } : {}),
          },
        }
      : {}),
  };
}
