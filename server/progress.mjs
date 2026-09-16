// Progress is operational telemetry, separate from benchmark timings and output.
// Only counts and fixed phase labels enter this snapshot, never request content.
export function createProgress(settings, modelCount, emit = () => {}) {
  const started = new Date().toISOString();
  let state = {
    kind: settings.kind,
    stage: 'preparing',
    message: 'Preparing model runtime; this work is excluded from measurement.',
    started_at: started,
    updated_at: started,
    completed_requests: 0,
    total_requests:
      modelCount *
      settings.batch_sizes.reduce(
        (sum, size) =>
          sum + (settings.requests_per_condition ?? settings.repeats * size),
        0,
      ),
    warmup_completed_requests: 0,
    warmup_total_requests:
      modelCount *
      settings.warmups *
      settings.batch_sizes.reduce((sum, size) => sum + size, 0),
    failed_requests: 0,
    run_status: 'running',
  };
  const seen = new Set();
  const publish = () => emit({ type: 'progress', progress: { ...state } });
  const update = (fields) => {
    state = { ...state, ...fields, updated_at: new Date().toISOString() };
    publish();
  };
  const clearWave = () => {
    delete state.batch_size;
    delete state.wave;
    delete state.waves;
  };
  publish();
  return {
    get snapshot() {
      return { ...state };
    },
    phase(event) {
      // Unstructured subprocess phase lines may contain runtime logs. Keep them
      // on the existing phase channel; they must not leak into status snapshots.
      if (!['preparing', 'inputs', 'warmup', 'measuring'].includes(event.stage))
        return;
      clearWave();
      const fields = { stage: event.stage, message: event.message };
      for (const key of ['batch_size', 'wave', 'waves'])
        if (Number.isInteger(event[key]) && event[key] > 0)
          fields[key] = event[key];
      update(fields);
    },
    request(result) {
      if (seen.has(result.request_id)) return;
      seen.add(result.request_id);
      const counter = result.warmup
        ? 'warmup_completed_requests'
        : 'completed_requests';
      const failed =
        result.status !== 'complete' ||
        result.fixed_length_ok === false ||
        result.input_length_ok === false ||
        result.canvas_length_ok === false;
      update({
        [counter]: state[counter] + 1,
        failed_requests: state.failed_requests + Number(failed),
      });
    },
    finish(run) {
      clearWave();
      update({
        stage:
          run.status === 'complete'
            ? 'complete'
            : run.status === 'cancelled'
              ? 'cancelled'
              : 'error',
        message:
          run.status === 'complete'
            ? 'Experiment complete.'
            : run.status === 'cancelled'
              ? 'Experiment cancelled.'
              : 'Experiment finished with failed requests or validation errors.',
        run_id: run.id,
        run_status: run.status,
      });
    },
    fail(cancelled = false) {
      clearWave();
      // An exception can prevent persistence, so never advertise a saved run.
      delete state.run_id;
      update({
        stage: cancelled ? 'cancelled' : 'error',
        message: cancelled
          ? 'Experiment cancelled.'
          : 'Experiment failed; see the error details.',
        run_status: cancelled ? 'cancelled' : 'error',
      });
    },
  };
}
