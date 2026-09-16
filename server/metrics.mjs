export function percentile(values, q) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b),
    p = (sorted.length - 1) * q;
  const lo = Math.floor(p),
    hi = Math.ceil(p);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (p - lo);
}
export function summarize(model, batchSize, waves) {
  const results = waves.flatMap((w) => w.results),
    good = results.filter((r) => r.status === 'complete');
  const valid =
    results.length > 0 &&
    good.length === results.length &&
    good.every(
      (r) =>
        r.completion_tokens != null &&
        r.fixed_length_ok !== false &&
        r.input_length_ok !== false,
    );
  const sum = (arr) => arr.reduce((a, b) => a + b, 0);
  const postFirst = good.filter((r) => Number.isFinite(r.post_first_block_tps));
  // Pool measured token/time intervals instead of averaging reciprocal times.
  // Keep the arithmetic mean for compatibility, but expose the median and the
  // weighted rate so a few very fast (possibly textless) canvases cannot hide
  // the distribution. These still count server tokens, including special tokens.
  const postFirstIntervals = good.flatMap((r) => {
    const progress = r.token_progress;
    if (!Array.isArray(progress) || progress.length < 2) return [];
    const first = progress[0],
      last = progress.at(-1);
    const tokens = r.completion_tokens - first.completion_tokens,
      milliseconds = last.at_ms - first.at_ms;
    return Number.isInteger(r.completion_tokens) &&
      Number.isInteger(first.completion_tokens) &&
      last.completion_tokens === r.completion_tokens &&
      tokens > 0 &&
      Number.isFinite(milliseconds) &&
      milliseconds > 0
      ? [{ tokens, milliseconds }]
      : [];
  });
  const postFirstTokens = sum(postFirstIntervals.map((x) => x.tokens)),
    postFirstMs = sum(postFirstIntervals.map((x) => x.milliseconds));
  const visibilityChecked = good.filter(
    (r) =>
      Array.isArray(r.token_progress) &&
      r.token_progress.length > 1 &&
      Array.isArray(r.chunks),
  );
  const noPostFirstText = visibilityChecked.filter(
    (r) =>
      !r.chunks.some(
        (chunk) =>
          typeof chunk.text === 'string' &&
          chunk.text.length > 0 &&
          chunk.at_ms > r.token_progress[0].at_ms,
      ),
  );
  const outputCharacters = good
    .filter((r) => typeof r.text === 'string')
    .map((r) => Array.from(r.text).length);
  const inputCounts = good.map((r) => r.prompt_tokens).filter(Number.isInteger);
  return {
    model,
    batch_size: batchSize,
    requests: results.length,
    successes: good.length,
    valid,
    min_prompt_tokens: inputCounts.length ? Math.min(...inputCounts) : null,
    max_prompt_tokens: inputCounts.length ? Math.max(...inputCounts) : null,
    mean_prompt_tokens: inputCounts.length
      ? sum(inputCounts) / inputCounts.length
      : null,
    mean_user_tps: valid
      ? sum(good.map((r) => r.tokens_per_second)) / good.length
      : null,
    aggregate_tps: valid
      ? sum(good.map((r) => r.completion_tokens)) /
        (sum(waves.map((w) => w.makespan_ms)) / 1000)
      : null,
    p50_latency_ms: percentile(
      good.map((r) => r.elapsed_ms),
      0.5,
    ),
    p95_latency_ms: percentile(
      good.map((r) => r.elapsed_ms),
      0.95,
    ),
    mean_ttft_ms: good.filter((r) => r.ttft_ms != null).length
      ? sum(good.map((r) => r.ttft_ms || 0)) /
        good.filter((r) => r.ttft_ms != null).length
      : null,
    wave_count: waves.length,
    post_first_block_requests: postFirst.length,
    mean_post_first_block_tps:
      valid && postFirst.length
        ? sum(postFirst.map((r) => r.post_first_block_tps)) / postFirst.length
        : null,
    median_post_first_block_tps:
      valid && postFirst.length
        ? percentile(
            postFirst.map((r) => r.post_first_block_tps),
            0.5,
          )
        : null,
    pooled_post_first_block_requests: postFirstIntervals.length,
    pooled_post_first_block_tokens: postFirstTokens,
    pooled_post_first_block_ms: postFirstMs,
    pooled_post_first_block_tps:
      valid && postFirstIntervals.length
        ? postFirstTokens / (postFirstMs / 1000)
        : null,
    post_first_visibility_checked_requests: visibilityChecked.length,
    requests_without_post_first_visible_text: noPostFirstText.length,
    output_characters_min: outputCharacters.length
      ? Math.min(...outputCharacters)
      : null,
    output_characters_median: percentile(outputCharacters, 0.5),
    output_characters_max: outputCharacters.length
      ? Math.max(...outputCharacters)
      : null,
    failed: results.length - good.length,
  };
}
