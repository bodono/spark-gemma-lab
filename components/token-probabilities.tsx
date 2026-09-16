import type { CSSProperties, ReactNode } from 'react';

export type TokenProbability = {
  index: number;
  token: string;
  bytes: number[] | null;
  token_id?: number;
  logprob: number | null;
  probability: number | null;
};
export type TokenProbabilities = {
  version: 1;
  status: 'available' | 'partial' | 'unavailable' | 'invalid';
  reason?: string;
  source: string;
  logprobs_mode: string;
  offset_unit: 'utf16';
  tokens: TokenProbability[];
  spans: { start: number; end: number; token_indexes: number[] }[];
};

// Red → orange → green. A linear scale uses the model's probability directly.
function probabilityStyle(probability: number): CSSProperties {
  const stops = [
    [213, 53, 48],
    [227, 137, 24],
    [32, 151, 81],
  ];
  const part = probability < 0.5 ? 0 : 1;
  const mix = probability < 0.5 ? probability * 2 : (probability - 0.5) * 2;
  const rgb = stops[part]
    .map((start, i) => Math.round(start + (stops[part + 1][i] - start) * mix))
    .join(', ');
  return {
    backgroundColor: `rgba(${rgb}, 0.14)`,
    boxShadow: `inset 0 0 0 1px rgba(${rgb}, 0.30)`,
  };
}
function validProbability(p: number | null | undefined): p is number {
  return typeof p === 'number' && Number.isFinite(p) && p >= 0 && p <= 1;
}
function describeToken(token: TokenProbability) {
  const p = token.probability;
  const value = validProbability(p)
    ? p > 0 && p < 0.0001
      ? `${(p * 100).toExponential(2)}%`
      : `${(p * 100).toFixed(2)}%`
    : 'unavailable';
  return `${JSON.stringify(token.token) || 'Token'}: ${value}${token.logprob == null ? '' : ` (log p = ${token.logprob.toFixed(4)})`}`;
}
function splitsSurrogate(text: string, offset: number) {
  return (
    offset > 0 &&
    offset < text.length &&
    /[\uD800-\uDBFF]/.test(text[offset - 1]) &&
    /[\uDC00-\uDFFF]/.test(text[offset])
  );
}

export function TokenProbabilityText({
  text,
  data,
  kind = 'autoregressive',
}: {
  text: string;
  data?: TokenProbabilities;
  kind?: 'autoregressive' | 'diffusion';
}) {
  if (!data || data.status === 'invalid' || data.offset_unit !== 'utf16')
    return text;
  const pieces: ReactNode[] = [];
  let end = 0;
  const tokens = new Map(data.tokens.map((token) => [token.index, token]));
  for (const span of data.spans) {
    if (
      !Number.isInteger(span.start) ||
      !Number.isInteger(span.end) ||
      span.start < end ||
      span.end <= span.start ||
      span.end > text.length ||
      splitsSurrogate(text, span.start) ||
      splitsSurrogate(text, span.end)
    )
      return text;
    if (span.start > end) pieces.push(text.slice(end, span.start));
    const records = span.token_indexes.map((index) => tokens.get(index));
    const single = records.length === 1 ? records[0] : undefined;
    const colored = single && validProbability(single.probability);
    const probabilityLabel =
      kind === 'diffusion'
        ? 'Final denoising probability of selected token'
        : 'Raw model probability of selected token';
    const probabilityNote =
      kind === 'diffusion'
        ? 'After the diffusion temperature schedule and token filtering; not a measure of factual accuracy.'
        : 'Before temperature / top-p; not a measure of factual accuracy.';
    const title = records.every((record) => record != null)
      ? `${records.length > 1 ? 'This text combines multiple token fragments; shown without a single color.\n' : ''}${probabilityLabel}${records.length > 1 ? 's' : ''}:\n${records.map((record) => describeToken(record!)).join('\n')}\n${probabilityNote}`
      : 'Token probability unavailable';
    pieces.push(
      <span
        key={`${span.start}:${span.end}`}
        className={
          colored
            ? 'probability-token'
            : 'probability-token probability-neutral'
        }
        style={colored ? probabilityStyle(single.probability!) : undefined}
        title={title}
      >
        {text.slice(span.start, span.end)}
      </span>,
    );
    end = span.end;
  }
  pieces.push(text.slice(end));
  return pieces;
}

export function TokenProbabilityLegend({
  data,
  running,
  kind = 'autoregressive',
}: {
  data?: TokenProbabilities;
  running: boolean;
  kind?: 'autoregressive' | 'diffusion';
}) {
  const missing =
    !data || data.status === 'unavailable' || data.status === 'invalid';
  return (
    <div className="probability-legend">
      <div
        className="probability-scale"
        aria-label="Token probability: red 0%, orange 50%, green 100%"
      >
        <span className="probability-swatch low" /> Low
        <span className="probability-swatch mid" /> Intermediate
        <span className="probability-swatch high" /> High
      </div>
      <span title={data?.reason}>
        {missing
          ? running
            ? 'Waiting for token probabilities…'
            : data
              ? 'Probabilities unavailable for this response; text stays uncolored.'
              : 'Run a comparison to collect token probabilities.'
          : 'Hover a token for its probability. Unscored tokens stay neutral.'}
      </span>
      <small>
        {kind === 'diffusion'
          ? 'Final denoising probability, after the diffusion temperature schedule and token filtering. '
          : 'Raw model likelihood of each selected token, before temperature / top-p. '}
        Not factual confidence.
      </small>
    </div>
  );
}
