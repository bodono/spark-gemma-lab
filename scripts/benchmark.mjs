import { parseArgs } from 'node:util';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { parseSSE } from '../server/stream.mjs';
const { values } = parseArgs({
  options: {
    bridge: { type: 'string', default: 'http://127.0.0.1:8787' },
    dataset: { type: 'string' },
    workload: { type: 'string', default: 'continuation' },
    'output-mode': { type: 'string', default: 'natural' },
    'denoising-mode': { type: 'string', default: 'adaptive' },
    'denoising-steps': { type: 'string', default: '16' },
    prompt: {
      type: 'string',
      default:
        'Explain the difference between diffusion and autoregressive language models.',
    },
    'batch-sizes': { type: 'string', default: '1,2,4' },
    repeats: { type: 'string' },
    requests: { type: 'string' },
    warmups: { type: 'string', default: '1' },
    tokens: { type: 'string', default: '512' },
    'input-tokens': { type: 'string' },
    'canvas-length': { type: 'string', default: '256' },
    temperature: { type: 'string', default: '0' },
    seed: { type: 'string', default: '42' },
    output: { type: 'string', default: 'results' },
  },
});
if (values.repeats && values.requests)
  throw Error('Choose --requests or --repeats, not both');
if (!['adaptive', 'fixed'].includes(values['denoising-mode']))
  throw Error('--denoising-mode must be adaptive or fixed');
const fixedSteps = Number(values['denoising-steps']);
if (!Number.isInteger(fixedSteps) || fixedSteps < 1 || fixedSteps > 48)
  throw Error('--denoising-steps must be a whole number from 1 to 48');
const dataset = values.dataset
  ? (await readFile(values.dataset, 'utf8'))
      .split('\n')
      .filter(Boolean)
      .map(JSON.parse)
  : [];
const body = {
  kind: 'profile',
  prompt: values.prompt,
  batch_size: 1,
  batch_sizes: values['batch-sizes'].split(',').map(Number),
  repeats: +(values.repeats || 5),
  requests_per_condition: values.repeats ? undefined : +(values.requests || 20),
  warmups: +values.warmups,
  max_tokens: +values.tokens,
  canvas_length: +values['canvas-length'],
  input_tokens: values['input-tokens'] == null ? null : +values['input-tokens'],
  temperature: +values.temperature,
  seed: +values.seed,
  dataset,
  workload: values.workload,
  output_mode: values['output-mode'],
  denoising_mode: values['denoising-mode'],
  denoising_steps: values['denoising-mode'] === 'fixed' ? fixedSteps : 48,
  dataset_name: values.dataset || 'Current prompt',
};
const controller = new AbortController();
process.on('SIGINT', () => controller.abort());
try {
  console.log(
    body.input_tokens == null
      ? 'Input: original source length.'
      : `Input: ${body.input_tokens} source-text tokens before formatting; short sources repeat, long sources trim. Actual rendered counts are saved.`,
  );
  console.log(
    body.denoising_mode === 'fixed'
      ? `Diffusion: fixed ${body.denoising_steps} steps/block requested. Fixed steps may reduce output quality; inspect generated text.`
      : 'Diffusion: adaptive, maximum 48 steps/block requested.',
  );
  const r = await fetch(values.bridge + '/api/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: controller.signal,
  });
  if (!r.ok) throw Error(await r.text());
  let run;
  const reportedBlocks = new Map();
  for await (const line of parseSSE(r.body)) {
    const e = JSON.parse(line);
    if (e.type === 'phase') console.log(e.message);
    if (e.type === 'start' && e.model === 'diffusion')
      reportedBlocks.delete(e.index);
    if (
      e.type === 'chunk' &&
      e.model === 'diffusion' &&
      e.denoising?.status === 'available'
    ) {
      const previous = reportedBlocks.get(e.index) || new Map();
      for (const block of e.denoising.blocks || []) {
        if (previous.get(block.block_index) !== block.denoising_steps) {
          console.log(
            `Diffusion request ${e.index + 1}: block ${block.block_index}, ${block.denoising_steps} actual steps (${block.emitted_tokens} emitted tokens)`,
          );
          previous.set(block.block_index, block.denoising_steps);
        }
      }
      reportedBlocks.set(e.index, previous);
    }
    if (e.type === 'result' && e.result.model === 'diffusion') {
      const { denoising, index, batch_size } = e.result;
      const label = `Diffusion c=${batch_size} request ${index + 1}`;
      if (denoising?.status === 'available') {
        const blocks = (denoising.blocks || [])
          .map((b) => `block ${b.block_index}=${b.denoising_steps}`)
          .join(', ');
        const mean = Number.isFinite(denoising.mean_steps)
          ? `; mean ${denoising.mean_steps.toFixed(2)} steps/block`
          : '';
        console.log(
          `${label}: actual steps ${blocks || 'no completed blocks reported'}${mean}`,
        );
      } else {
        console.log(
          `${label}: actual steps unavailable${denoising?.status === 'invalid' ? ' (invalid metadata)' : ''} — ${denoising?.reason || 'Server did not report denoising metadata.'}`,
        );
      }
    }
    if (e.type === 'summary') console.log(JSON.stringify(e.summary));
    if (e.type === 'error') throw Error(e.message);
    if (e.type === 'complete') run = e.run;
  }
  if (!run) throw Error('Run ended without saved results');
  await mkdir(values.output, { recursive: true });
  const filename = path.join(values.output, run.id + '.json');
  await writeFile(filename, JSON.stringify(run, null, 2));
  console.log(
    `Saved ${filename} (${run.status}${run.synthetic ? ', SYNTHETIC' : ''})`,
  );
  if (run.status !== 'complete') process.exitCode = 2;
} catch (e) {
  console.error(e.message);
  process.exitCode = 1;
}
