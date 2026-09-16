'use client';

import { CheckCircle2, CircleAlert, LoaderCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';

export type RunProgressState = {
  kind: 'demo' | 'profile';
  stage:
    | 'preparing'
    | 'inputs'
    | 'warmup'
    | 'measuring'
    | 'complete'
    | 'cancelled'
    | 'error';
  message: string;
  started_at: string;
  updated_at: string;
  completed_requests: number;
  total_requests: number;
  warmup_completed_requests: number;
  warmup_total_requests: number;
  failed_requests: number;
  batch_size?: number;
  wave?: number;
  waves?: number;
  run_id?: string;
  run_status?: string;
};

export function newerProgress(
  previous: RunProgressState | null,
  next: RunProgressState,
) {
  if (
    previous &&
    (previous.started_at > next.started_at ||
      (previous.started_at === next.started_at &&
        previous.updated_at > next.updated_at))
  )
    return previous;
  return next;
}

export function RunProgress({
  progress,
  active,
  connected = true,
  onCancel,
  cancelling = false,
}: {
  progress: RunProgressState;
  active: boolean;
  connected?: boolean;
  onCancel?: () => void;
  cancelling?: boolean;
}) {
  const terminal = ['complete', 'cancelled', 'error'].includes(progress.stage);
  const warming = progress.stage === 'warmup';
  const completed = warming
    ? progress.warmup_completed_requests
    : progress.completed_requests;
  const total = warming
    ? progress.warmup_total_requests
    : progress.total_requests;
  const determinate =
    (warming || progress.stage === 'measuring' || terminal) && total > 0;
  const percent = determinate
    ? Math.min(100, Math.round((completed / total) * 100))
    : null;
  const end = active ? Date.now() : Date.parse(progress.updated_at);
  const seconds = Math.max(
    0,
    Math.floor((end - Date.parse(progress.started_at)) / 1000),
  );
  const elapsed =
    seconds < 60
      ? `${seconds}s`
      : `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  const titles = {
    preparing: 'Preparing models',
    inputs: 'Preparing inputs',
    warmup: 'Warming models',
    measuring:
      progress.kind === 'profile'
        ? 'Sweep in progress'
        : 'Comparison in progress',
    complete:
      progress.kind === 'profile' ? 'Sweep complete' : 'Comparison complete',
    cancelled: 'Run cancelled',
    error: 'Run finished with errors',
  };
  const Icon =
    progress.stage === 'complete'
      ? CheckCircle2
      : terminal
        ? CircleAlert
        : LoaderCircle;
  return (
    <section
      className={`run-progress ${terminal ? progress.stage : 'active'}`}
      aria-label="Run progress"
    >
      <div className="run-progress-heading">
        <div role="status" aria-live="polite">
          <Icon
            size={20}
            className={terminal ? '' : 'progress-spinner'}
            aria-hidden="true"
          />
          <strong>{titles[progress.stage]}</strong>
        </div>
        <div className="run-progress-actions">
          <span className="run-progress-time">
            {elapsed} elapsed{percent != null ? ` · ${percent}%` : ''}
          </span>
          {active && progress.kind === 'profile' && onCancel && (
            <Button
              variant="outline"
              size="sm"
              disabled={cancelling || !progress.run_status}
              onClick={onCancel}
            >
              {cancelling ? 'Stopping…' : 'Stop sweep'}
            </Button>
          )}
        </div>
      </div>
      <p className="run-progress-message">{progress.message}</p>
      <Progress
        value={percent}
        aria-label={
          warming ? 'Warmup requests completed' : 'Measured requests completed'
        }
        aria-valuetext={
          determinate
            ? `${completed} of ${total} requests finished`
            : progress.message
        }
        className={
          determinate ? 'run-progress-bar' : 'run-progress-bar indeterminate'
        }
      />
      <div className="run-progress-counts">
        <span>
          Measured requests:{' '}
          <strong>
            {progress.completed_requests} / {progress.total_requests || '—'}
          </strong>
        </span>
        <span>
          Warmup requests:{' '}
          <strong>
            {progress.warmup_completed_requests} /{' '}
            {progress.warmup_total_requests}
          </strong>
        </span>
        {progress.failed_requests > 0 && (
          <span className="run-progress-failures">
            {progress.failed_requests} unsuccessful
          </span>
        )}
      </div>
      <p className="run-progress-note">
        {!connected
          ? 'Progress connection interrupted. Reconnecting…'
          : progress.stage === 'preparing'
            ? 'A canvas change can take a few minutes to reload. Preparation and warmup are excluded from measured timings.'
            : warming
              ? 'Warmup requests are excluded from the benchmark. Measured requests follow automatically.'
              : progress.stage === 'inputs'
                ? 'Matching and checking input tokens before measurement starts.'
                : 'Request counts include both models. Results appear as each batch-size condition finishes.'}
      </p>
    </section>
  );
}
