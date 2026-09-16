// Real hardware checks; changes the owned Diffusion runtime. The default series ends at 256.
// CANVAS_LENGTHS can override the series; a failed check stops immediately for inspection.
import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { parseSSE } from '../server/stream.mjs';
const lengths = (process.env.CANVAS_LENGTHS || '128,512,64,256')
  .split(',')
  .map(Number);
const bridge = process.env.SPARK_LAB_BRIDGE || 'http://127.0.0.1:8787';
const output = new URL('../runtime/canvas/', import.meta.url);
await mkdir(output, { recursive: true });
let evidence = [];
try {
  evidence = JSON.parse(
    await readFile(new URL('hardware-validation.json', output), 'utf8'),
  ).cases;
} catch {}
for (const canvas of lengths) {
  for (const kind of canvas === 128 ? ['demo', 'profile'] : ['demo']) {
    const body = {
      kind,
      canvas_length: canvas,
      prompt:
        'Write a detailed explanation of how computers work, with at least twelve long paragraphs. Cover processors, memory, instructions, operating systems, networking, and programming. Keep going with concrete examples.',
      batch_size: 1,
      batch_sizes: kind === 'profile' ? [1, 2] : [1],
      requests_per_condition: 2,
      repeats: 1,
      warmups: kind === 'profile' ? 1 : 0,
      input_tokens: kind === 'profile' ? 128 : null,
      max_tokens: canvas * 2,
      seed: 42,
      temperature: 0,
      diffusion_preview: true,
      denoising_mode: 'adaptive',
      output_mode: kind === 'profile' ? 'fixed' : 'natural',
      workload: 'continuation',
    };
    const response = await fetch(bridge + '/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!response.ok) throw Error(await response.text());
    let run,
      previewFrames = 0,
      maxPreviewTokens = 0;
    for await (const line of parseSSE(response.body)) {
      const event = JSON.parse(line);
      if (event.type === 'phase') console.log(canvas, kind, event.message);
      if (event.type === 'error') throw Error(event.message);
      if (event.type === 'preview' && !event.preview.unavailable) {
        previewFrames++;
        maxPreviewTokens = Math.max(
          maxPreviewTokens,
          event.preview.token_ids.length,
        );
        assert(event.preview.token_ids.length <= canvas);
      }
      if (event.type === 'complete') run = event.run;
    }
    assert(run, 'missing completion');
    assert.equal(
      run.status,
      'complete',
      JSON.stringify(run.results.map((r) => r.error)),
    );
    assert.equal(run.settings.canvas_length, canvas);
    assert.equal(run.configuration.runtime.canvas_length, canvas);
    assert.equal(run.runtime_preparation.verified, true);
    const diffusion = run.results.filter((r) => r.model === 'diffusion');
    assert(diffusion.every((r) => r.canvas_length_ok === true));
    assert(
      diffusion.every((r) =>
        r.denoising.blocks.every((b) => b.canvas_tokens <= canvas),
      ),
    );
    if (kind === 'profile') {
      assert.equal(previewFrames, 0);
      assert.equal(run.settings.diffusion_preview, false);
      assert(
        [...run.results, ...run.warmup_results]
          .filter((r) => r.model === 'diffusion')
          .every(
            (r) =>
              r.request_payload.vllm_xargs.spark_lab_diffusion_preview === 0,
          ),
      );
      assert(diffusion.every((r) => r.denoising.blocks.length === 2));
    } else {
      assert(previewFrames > 0);
      assert.equal(maxPreviewTokens, canvas);
    }
    const record = {
      canvas_length: canvas,
      kind,
      run_id: run.id,
      preview_frames: previewFrames,
      max_preview_tokens: maxPreviewTokens,
      runtime: run.runtime_preparation,
      results: diffusion.map((r) => ({
        request_id: r.request_id,
        completion_tokens: r.completion_tokens,
        canvas_length_ok: r.canvas_length_ok,
        denoising: r.denoising,
        post_first_block_tps: r.post_first_block_tps,
      })),
    };
    evidence.push(record);
    await writeFile(
      new URL('hardware-validation.json', output),
      JSON.stringify(
        {
          validated_at: new Date().toISOString(),
          purpose:
            'Functional canvas/runtime validation, not a performance comparison',
          cases: evidence,
        },
        null,
        2,
      ) + '\n',
    );
    console.log(
      'PASS',
      canvas,
      kind,
      run.id,
      'previews',
      previewFrames,
      'blocks',
      diffusion.map((r) => r.denoising.blocks.length).join(','),
    );
  }
}
