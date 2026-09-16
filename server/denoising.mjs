// vLLM's diffusion path records one zero-token event per denoising pass and
// one positive-token event per canvas commit. Never count the commit as a step.
export function decodeDenoisingTrace(metrics, completionTokens, options = {}) {
  const base = {
    mode: options.mode ?? 'adaptive',
    max_steps: options.max_steps ?? 48,
    canvas_length: options.canvas_length ?? 256,
    source: 'vllm_per_request_scheduler_trace',
    blocks: [],
    mean_steps: null,
  };
  const unavailable = (reason, status = 'unavailable') => ({
    ...base,
    status,
    reason,
  });
  const trace = metrics?.speculative_decoding;
  if (!trace)
    return unavailable('The server has not supplied a per-block trace.');
  if (trace.diffusion_trace_version !== 1)
    return unavailable(
      'This trace lacks verified diffusion/preemption metadata.',
    );
  if (
    !Number.isInteger(trace.diffusion_num_preemptions) ||
    trace.diffusion_num_preemptions < 0
  )
    return unavailable(
      'The trace does not identify request preemptions.',
      'invalid',
    );
  if (trace.diffusion_num_preemptions > 0)
    return unavailable(
      'This request was preempted; discarded denoising work prevents exact per-block attribution.',
    );
  const accepted = trace.per_step_accepted,
    drafted = trace.per_step_drafted;
  if (
    !Array.isArray(accepted) ||
    !Array.isArray(drafted) ||
    accepted.length !== drafted.length
  )
    return unavailable('The denoising trace is incomplete.', 'invalid');
  if (!Number.isInteger(completionTokens) || completionTokens < 0)
    return unavailable(
      'Server output-token counts are required to align blocks.',
    );
  const canvasLength = options.canvas_length ?? 256;
  const blocks = [];
  let steps = 0,
    emitted = 0;
  for (let i = 0; i < accepted.length; i++) {
    const a = accepted[i],
      d = drafted[i];
    if (
      !Number.isInteger(a) ||
      !Number.isInteger(d) ||
      a < 0 ||
      d < 1 ||
      a > d ||
      d !== canvasLength
    )
      return unavailable(
        'Invalid canvas counts in the denoising trace.',
        'invalid',
      );
    if (!a) {
      steps++;
      continue;
    }
    if (steps < 1 || steps > base.max_steps)
      return unavailable(
        'A block has a denoising count outside its configured budget.',
        'invalid',
      );
    if (base.mode === 'fixed' && steps !== base.max_steps)
      return unavailable(
        'The observed denoising count does not match the requested fixed count.',
        'invalid',
      );
    if (emitted >= completionTokens)
      return unavailable(
        'A committed block has no corresponding output tokens.',
        'invalid',
      );
    const tokens = Math.min(a, Math.max(0, completionTokens - emitted));
    blocks.push({
      block_index: blocks.length + 1,
      denoising_steps: steps,
      canvas_tokens: a,
      emitted_tokens: tokens,
    });
    emitted += tokens;
    steps = 0;
  }
  if (emitted !== completionTokens || (options.final && steps))
    return unavailable(
      'The trace does not align with the completed output.',
      'invalid',
    );
  if (!blocks.length)
    return unavailable('Waiting for the first committed block.');
  return {
    ...base,
    status: 'available',
    blocks,
    mean_steps:
      blocks.reduce((sum, block) => sum + block.denoising_steps, 0) /
      blocks.length,
    final: options.final === true,
  };
}
