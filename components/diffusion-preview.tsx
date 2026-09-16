'use client';

import { memo, useEffect, useMemo, useRef } from 'react';

export type DiffusionPrediction = {
  version: 1;
  block_index: number;
  denoising_step: number;
  token_ids: number[];
  text: string;
  final: false;
};

// Every displayed word comes from a received model prediction. No animation
// timer, fabricated candidates, or delayed release of committed output.
export const DiffusionPreview = memo(function DiffusionPreview({
  prediction,
  continuation,
}: {
  prediction: DiffusionPrediction;
  continuation: boolean;
}) {
  const previous = useRef<string[]>([]);
  const words = useMemo(
    () => prediction.text.split(/(\s+)/u),
    [prediction.text],
  );
  const rendered = useMemo(
    () =>
      words.map((word, index) => ({
        word,
        changed: Boolean(word.trim()) && word !== previous.current[index],
      })),
    [words, prediction.block_index, prediction.denoising_step],
  );
  useEffect(() => {
    previous.current = words;
  }, [words]);
  return (
    <span className="diffusion-preview" aria-hidden="true">
      {continuation ? ' ' : ''}
      {rendered.map(({ word, changed }, index) => (
        <span
          key={`${index}:${changed ? prediction.denoising_step : 'stable'}`}
          className={changed ? 'prediction-word changed' : 'prediction-word'}
        >
          {word}
        </span>
      ))}
    </span>
  );
});
