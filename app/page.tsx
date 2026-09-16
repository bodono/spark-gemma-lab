'use client';
import { useEffect, useRef, useState } from 'react';
import {
  Play,
  Square,
  ArrowUpRight,
  Download,
  Activity,
  Settings2,
  FlaskConical,
  RefreshCw,
  Zap,
  Check,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import {
  TokenProbabilityText,
  TokenProbabilityLegend,
  type TokenProbabilities,
} from '@/components/token-probabilities';
import {
  DiffusionPreview,
  type DiffusionPrediction,
} from '@/components/diffusion-preview';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import {
  RunProgress,
  newerProgress,
  type RunProgressState,
} from '@/components/run-progress';
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from '@/components/ui/table';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LabelList,
} from 'recharts';

type Model = {
  id: string;
  label: string;
  host: string;
  model: string;
  baseUrl: string;
};
type DenoisingMode = 'adaptive' | 'fixed';
type DenoisingSettings = {
  denoising_mode?: DenoisingMode;
  denoising_steps?: number;
  diffusion_preview?: boolean;
  ar_token_probabilities?: boolean;
  diffusion_token_probabilities?: boolean;
  input_tokens?: number | null;
  canvas_length?: number;
};
type Denoising = {
  status: 'available' | 'unavailable' | 'invalid';
  reason?: string;
  mode?: DenoisingMode;
  max_steps?: number;
  blocks?: {
    block_index: number;
    denoising_steps: number;
    canvas_tokens: number;
    emitted_tokens: number;
  }[];
  mean_steps?: number | null;
  source?: string;
};
type Result = {
  model: string;
  index: number;
  batch_size?: number;
  request_id?: string;
  text: string;
  reasoning?: string;
  status: string;
  error?: string;
  ttft_ms?: number | null;
  elapsed_ms?: number;
  completion_tokens?: number | null;
  tokens_per_second?: number | null;
  post_first_block_tps?: number | null;
  first_block_tokens?: number | null;
  finish_reason?: string;
  denoising?: Denoising;
  preview?: DiffusionPrediction;
  previewUnavailable?: string;
  ar_token_probabilities?: TokenProbabilities;
  diffusion_token_probabilities?: TokenProbabilities;
  diffusion_preview?: {
    enabled: boolean;
    received_frames: number;
    dropped_frames: number;
  };
};
type Summary = {
  model: string;
  batch_size: number;
  min_prompt_tokens?: number | null;
  max_prompt_tokens?: number | null;
  mean_user_tps: number | null;
  aggregate_tps: number | null;
  p50_latency_ms: number | null;
  p95_latency_ms: number | null;
  mean_ttft_ms: number | null;
  pooled_post_first_block_tps?: number | null;
  median_post_first_block_tps?: number | null;
  requests_without_post_first_visible_text?: number;
  successes: number;
  requests: number;
  valid: boolean;
  label?: string;
};
type Run = {
  id: string;
  kind: string;
  created_at: string;
  synthetic: boolean;
  results: Result[];
  summaries: Summary[];
  settings?: DenoisingSettings;
  configuration?: {
    runtime?: {
      max_denoising_steps?: number;
      denoising_mode?: DenoisingMode;
      canvas_length?: number;
    };
  };
  [key: string]: unknown;
};
const defaultPrompt =
  'Explain how a diffusion language model generates text, using an analogy that a curious engineer would understand. Then compare it with autoregressive generation. Use three short paragraphs, totalling no more than 150 words.';
const fmt = (n: number | null | undefined, d = 1) =>
  n == null ? '—' : n.toLocaleString(undefined, { maximumFractionDigits: d });
const api = () =>
  `http://${typeof window === 'undefined' ? '127.0.0.1' : window.location.hostname}:8787`;
function denoisingLabel(
  settings?: DenoisingSettings,
  runtime?: Run['configuration'],
) {
  const mode = settings?.denoising_mode ?? runtime?.runtime?.denoising_mode;
  const steps =
    settings?.denoising_steps ?? runtime?.runtime?.max_denoising_steps;
  const canvas = settings?.canvas_length ?? runtime?.runtime?.canvas_length;
  const suffix = canvas == null ? '' : ` · ${canvas}-token canvas`;
  if (mode === 'adaptive')
    return `Adaptive · max ${steps ?? 48} steps/block${suffix}`;
  if (mode === 'fixed')
    return `Fixed · ${steps ?? 'unrecorded'} steps/block requested${suffix}`;
  return steps == null
    ? 'Denoising settings not recorded'
    : `Saved cap: ${steps} steps · mode not recorded${suffix}`;
}
function DenoisingReadout({ result }: { result?: Result }) {
  const data = result?.denoising;
  const waiting = !result || result.status === 'running';
  if (data?.status !== 'available') {
    return (
      <div className="denoising-readout" aria-live="polite">
        <strong>Actual denoising steps</strong>
        <span>
          {!data && waiting
            ? 'Awaiting server metadata.'
            : `${data?.status === 'invalid' ? 'Unavailable (invalid metadata)' : 'Unavailable'} — ${data?.reason || 'This run did not report actual denoising steps.'}`}
        </span>
      </div>
    );
  }
  return (
    <div className="denoising-readout" aria-live="polite">
      <div className="denoising-heading">
        <strong>Actual denoising steps</strong>
        {result?.status === 'complete' &&
          data.mean_steps != null &&
          Number.isFinite(data.mean_steps) && (
            <span>Mean {fmt(data.mean_steps, 2)} steps/block</span>
          )}
      </div>
      <div className="denoising-blocks">
        {data.blocks?.length ? (
          data.blocks.map((block) => (
            <span
              className="denoising-chip"
              key={block.block_index}
              title={`${block.canvas_tokens} canvas tokens · ${block.emitted_tokens} emitted tokens`}
            >
              Block {block.block_index} · <b>{block.denoising_steps} steps</b>
            </span>
          ))
        ) : (
          <span>No completed blocks reported.</span>
        )}
      </div>
      <small title={data.source}>Reported by the inference server</small>
    </div>
  );
}
function download(name: string, data: string, type = 'application/json') {
  const u = URL.createObjectURL(new Blob([data], { type }));
  const a = document.createElement('a');
  a.href = u;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(u), 1000);
}
export default function Home() {
  const [models, setModels] = useState<Model[]>([
    {
      id: 'diffusion',
      label: 'DiffusionGemma',
      host: 'Spark_1',
      model: 'RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic',
      baseUrl: 'http://127.0.0.1:18001/v1',
    },
    {
      id: 'autoregressive',
      label: 'Gemma 4',
      host: 'Spark_2',
      model: 'RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic',
      baseUrl: 'http://127.0.0.1:18002/v1',
    },
  ]);
  const [runSynthetic, setRunSynthetic] = useState(false);
  const [liveDiffusionPreview, setLiveDiffusionPreview] = useState(true);
  const [showTokenProbabilities, setShowTokenProbabilities] = useState(true);
  const [showDiffusionTokenProbabilities, setShowDiffusionTokenProbabilities] =
    useState(true);
  const [activePreview, setActivePreview] = useState(false);
  const [previewRun, setPreviewRun] = useState(0);
  const [contextBudget, setContextBudget] = useState(8192);
  const [profileDenoisingMode, setProfileDenoisingMode] =
    useState<DenoisingMode>('adaptive');
  const [fixedDenoisingSteps, setFixedDenoisingSteps] = useState(16);
  const [inputTokens, setInputTokens] = useState('');
  const [canvasLength, setCanvasLength] = useState(256);
  const [managedDiffusion, setManagedDiffusion] = useState(false);
  const [runProgress, setRunProgress] = useState<RunProgressState | null>(null);
  const [remoteActive, setRemoteActive] = useState(false);
  const [progressConnected, setProgressConnected] = useState(true);
  const [cancelling, setCancelling] = useState(false);
  const restoredRunId = useRef<string | null>(null);
  const reconnectingSince = useRef<string | null>(null);
  const [, tickProgress] = useState(0);
  const recoveredRun = useRef<string | null>(null);
  const progressPanel = useRef<HTMLDivElement | null>(null);
  const [activeDenoising, setActiveDenoising] =
    useState<DenoisingSettings | null>(null);
  const [health, setHealth] = useState<
    Record<string, { ok: boolean; error?: string }>
  >({});
  const [bridge, setBridge] = useState(false);
  const [synthetic, setSynthetic] = useState(false);
  const [prompt, setPrompt] = useState(defaultPrompt),
    [batch, setBatch] = useState(1),
    [tokens, setTokens] = useState(512),
    [temp, setTemp] = useState(0),
    [seed, setSeed] = useState(42);
  const [batches, setBatches] = useState('1, 2, 4'),
    [requestCount, setRequestCount] = useState(20),
    [warmups, setWarmups] = useState(1),
    [dataset, setDataset] = useState<
      { prompt: string; [key: string]: unknown }[]
    >([]),
    [datasetName, setDatasetName] = useState('Current prompt'),
    [workload, setWorkload] = useState('continuation'),
    [outputMode, setOutputMode] = useState('natural');
  const [results, setResults] = useState<Result[]>([]),
    [summaries, setSummaries] = useState<Summary[]>([]),
    [run, setRun] = useState<Run | null>(null),
    [running, setRunning] = useState(false),
    [phase, setPhase] = useState('Ready for a comparison'),
    [error, setError] = useState(''),
    [selected, setSelected] = useState(0),
    [elapsed, setElapsed] = useState(0),
    [history, setHistory] = useState<Run[]>([]),
    [tab, setTab] = useState('demo');
  const busy = running || remoteActive;
  const displayedDenoising = run
    ? denoisingLabel(run.settings, run.configuration)
    : activeDenoising
      ? denoisingLabel(activeDenoising)
      : denoisingLabel({
          denoising_mode: 'adaptive',
          denoising_steps: 48,
          canvas_length: canvasLength,
        });
  const displayedPreview = run?.settings?.diffusion_preview ?? activePreview;
  const ctrl = useRef<AbortController | null>(null),
    start = useRef(0);
  async function connect() {
    try {
      const r = await fetch(api() + '/api/config');
      if (!r.ok) throw Error('Bridge unavailable');
      const c = (await r.json()) as {
        models: Model[];
        synthetic: boolean;
        runtime?: {
          max_model_len?: number;
          max_denoising_steps?: number;
          canvas_controller?: { ssh_host?: string };
        };
      };
      setModels(c.models);
      setManagedDiffusion(Boolean(c.runtime?.canvas_controller?.ssh_host));
      setContextBudget(c.runtime?.max_model_len || 8192);
      setSynthetic(c.synthetic);
      setBridge(true);
      const h = await fetch(api() + '/api/health').then((r) => r.json());
      setHealth(h as Record<string, { ok: boolean; error?: string }>);
      const runs = await fetch(api() + '/api/runs').then((r) => r.json());
      setHistory(runs as Run[]);
    } catch {
      setBridge(false);
    }
  }
  useEffect(() => {
    if (running) return;
    void connect();
    const timer = setInterval(() => void connect(), 15000);
    return () => clearInterval(timer);
  }, [running]);
  useEffect(() => {
    if (!running) return;
    const id = setInterval(
      () => setElapsed((performance.now() - start.current) / 1000),
      50,
    );
    return () => clearInterval(id);
  }, [running]);
  useEffect(() => {
    if (!bridge) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    const controller = new AbortController();
    const poll = async () => {
      try {
        const response = await fetch(api() + '/api/activity', {
          signal: AbortSignal.any([
            controller.signal,
            AbortSignal.timeout(5000),
          ]),
        });
        if (!response.ok) throw Error('Progress unavailable');
        const activity = (await response.json()) as {
          active: boolean;
          progress?: RunProgressState | null;
        };
        if (stopped) return;
        setRemoteActive(activity.active);
        setProgressConnected(true);
        if (activity.progress) {
          const progress = activity.progress;
          if (
            reconnectingSince.current &&
            progress.started_at >= reconnectingSince.current
          ) {
            setError('');
            setPhase(progress.message);
            reconnectingSince.current = null;
          }
          setRunProgress((previous) => newerProgress(previous, progress));
          if (!running && activity.active) {
            if (recoveredRun.current !== progress.started_at)
              setTab(progress.kind === 'profile' ? 'profile' : 'demo');
            recoveredRun.current = progress.started_at;
          }
          if (
            !running &&
            !activity.active &&
            (recoveredRun.current === progress.started_at ||
              (progress.run_id && restoredRunId.current !== progress.run_id))
          ) {
            setPhase(progress.message);
            if (progress.run_id) {
              const saved = await fetch(
                api() + '/api/runs/' + progress.run_id,
                { signal: controller.signal },
              );
              if (!saved.ok) throw Error('Saved run unavailable');
              const finished = (await saved.json()) as Run;
              if (stopped) return;
              setRun(finished);
              setRunSynthetic(finished.synthetic);
              setSummaries(finished.summaries);
              setResults(finished.results);
              setActiveDenoising(finished.settings ?? null);
              restoredRunId.current = progress.run_id;
              if (recoveredRun.current === progress.started_at)
                setTab(finished.kind === 'profile' ? 'profile' : 'demo');
            }
            recoveredRun.current = null;
          }
        }
      } catch {
        if (!stopped) setProgressConnected(false);
      } finally {
        if (!stopped) timer = setTimeout(poll, 1000);
      }
    };
    void poll();
    return () => {
      stopped = true;
      controller.abort();
      clearTimeout(timer);
    };
  }, [bridge, running]);
  useEffect(() => {
    if (!busy) {
      setCancelling(false);
      return;
    }
    const timer = setInterval(() => tickProgress((value) => value + 1), 1000);
    return () => clearInterval(timer);
  }, [busy]);
  useEffect(() => {
    if (running && tab === 'profile')
      progressPanel.current?.scrollIntoView({
        behavior: 'smooth',
        block: 'start',
      });
  }, [running, tab]);
  async function stopRun() {
    if (runProgress?.kind !== 'profile') {
      ctrl.current?.abort();
      return;
    }
    setCancelling(true);
    try {
      const response = await fetch(api() + '/api/cancel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ started_at: runProgress.started_at }),
      });
      if (!response.ok) {
        const body = (await response.json()) as { error?: string };
        throw Error(body.error || 'Unable to cancel the sweep');
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : String(error));
      setCancelling(false);
    }
  }
  async function execute(kind: string) {
    if (!Number.isFinite(temp) || temp < 0) {
      setError('AR temperature must be a finite number greater than or equal to 0');
      return;
    }
    const requestedInput =
      kind === 'profile' && inputTokens !== '' ? Number(inputTokens) : null;
    if (
      requestedInput != null &&
      (!Number.isInteger(requestedInput) ||
        requestedInput < 1 ||
        requestedInput + tokens > contextBudget)
    ) {
      setError(
        `Input length must be a positive whole number; input plus output must fit the ${fmt(contextBudget, 0)}-token context, with room for chat formatting.`,
      );
      return;
    }
    const denoising: DenoisingSettings =
      kind === 'profile' && profileDenoisingMode === 'fixed'
        ? { denoising_mode: 'fixed', denoising_steps: fixedDenoisingSteps }
        : { denoising_mode: 'adaptive', denoising_steps: 48 };
    if (
      !Number.isInteger(denoising.denoising_steps) ||
      denoising.denoising_steps! < 1 ||
      denoising.denoising_steps! > 48
    ) {
      setError('Denoising steps must be a whole number from 1 to 48.');
      return;
    }
    setError('');
    setActiveDenoising({
      ...denoising,
      input_tokens: requestedInput,
      canvas_length: canvasLength,
    });
    setRunSynthetic(synthetic);
    setResults([]);
    setPreviewRun((value) => value + 1);
    const wantPreview = kind === 'demo' && liveDiffusionPreview;
    setActivePreview(wantPreview);
    setSummaries([]);
    setRun(null);
    setSelected(0);
    setRunning(true);
    setCancelling(false);
    const startedAt = new Date().toISOString();
    setRunProgress({
      kind: kind as 'demo' | 'profile',
      stage: 'preparing',
      message: kind === 'profile' ? 'Starting sweep…' : 'Starting comparison…',
      started_at: startedAt,
      updated_at: startedAt,
      completed_requests: 0,
      total_requests: 0,
      warmup_completed_requests: 0,
      warmup_total_requests: 0,
      failed_requests: 0,
    });
    setProgressConnected(true);
    start.current = performance.now();
    setElapsed(0);
    setPhase(
      kind === 'profile' ? 'Preparing sweep' : 'Dispatching to both Sparks',
    );
    const c = new AbortController();
    ctrl.current = c;
    let receivedServerProgress = false;
    let serverReportedError = false;
    try {
      const response = await fetch(api() + '/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: c.signal,
        body: JSON.stringify({
          kind,
          prompt,
          batch_size: batch,
          max_tokens: tokens,
          input_tokens: requestedInput,
          canvas_length: canvasLength,
          temperature: temp,
          seed,
          batch_sizes: batches.split(',').map((x) => Number(x.trim())),
          repeats: 5,
          requests_per_condition: requestCount,
          warmups,
          dataset,
          dataset_name: datasetName,
          workload,
          output_mode: outputMode,
          diffusion_preview: wantPreview,
          ar_token_probabilities: kind === 'demo' && showTokenProbabilities,
          diffusion_token_probabilities:
            kind === 'demo' && showDiffusionTokenProbabilities,
          ...denoising,
        }),
      });
      if (!response.ok) {
        serverReportedError = true;
        throw Error(
          ((await response.json()) as { error?: string }).error ||
            response.statusText,
        );
      }
      if (!response.body) throw Error('No stream returned');
      const reader = response.body.getReader(),
        decoder = new TextDecoder();
      let buffer = '',
        sawComplete = false;
      try {
        while (true) {
          const { value, done } = await reader.read();
          buffer += decoder.decode(value, { stream: !done });
          let ix;
          while ((ix = buffer.indexOf('\n\n')) >= 0) {
            const block = buffer.slice(0, ix);
            buffer = buffer.slice(ix + 2);
            const line = block.split('\n').find((l) => l.startsWith('data: '));
            if (!line) continue;
            const e = JSON.parse(line.slice(6));
            if (e.type === 'phase') setPhase(e.message);
            if (e.type === 'progress') {
              receivedServerProgress = true;
              setRunProgress((previous) => newerProgress(previous, e.progress));
            }
            if (e.type === 'start')
              setResults((p) => [
                ...p.filter(
                  (r) => !(r.model === e.model && r.index === e.index),
                ),
                { ...e, text: '', status: 'running' },
              ]);
            if (e.type === 'chunk')
              setResults((p) =>
                p.map((r) =>
                  r.model === e.model &&
                  r.index === e.index &&
                  r.request_id === e.request_id
                    ? {
                        ...r,
                        text: r.text + (e.text || ''),
                        reasoning: (r.reasoning || '') + (e.reasoning || ''),
                        ttft_ms: e.ttft_ms ?? r.ttft_ms,
                        denoising: e.denoising ?? r.denoising,
                        preview: (
                          e.denoising?.status === 'available'
                            ? e.denoising.blocks?.some(
                                (block: { block_index: number }) =>
                                  block.block_index >=
                                  (r.preview?.block_index ?? Infinity),
                              )
                            : Boolean(e.text)
                        )
                          ? undefined
                          : r.preview,
                      }
                    : r,
                ),
              );
            if (
              e.type === 'token_probabilities' &&
              (e.model === 'autoregressive' || e.model === 'diffusion')
            )
              setResults((previous) =>
                previous.map((result) => {
                  if (
                    result.model !== e.model ||
                    result.index !== e.index ||
                    result.request_id !== e.request_id ||
                    result.status !== 'running'
                  )
                    return result;
                  const field =
                    e.model === 'diffusion'
                      ? 'diffusion_token_probabilities'
                      : 'ar_token_probabilities';
                  return {
                    ...result,
                    [field]: {
                      version: e.version,
                      status: e.status,
                      reason: e.reason,
                      source: e.source,
                      logprobs_mode: e.logprobs_mode,
                      offset_unit: e.offset_unit,
                      tokens: [
                        ...(result[field]?.tokens ?? []),
                        ...(e.tokens ?? []),
                      ],
                      spans: [
                        ...(result[field]?.spans ?? []),
                        ...(e.spans ?? []),
                      ],
                    },
                  };
                }),
              );
            if (e.type === 'result')
              setResults((p) => [
                ...p.filter(
                  (r) =>
                    !(r.model === e.result.model && r.index === e.result.index),
                ),
                e.result,
              ]);
            if (e.type === 'preview' && wantPreview && e.model === 'diffusion')
              setResults((previous) =>
                previous.map((result) => {
                  if (
                    result.model !== e.model ||
                    result.index !== e.index ||
                    result.request_id !== e.request_id ||
                    result.status !== 'running'
                  )
                    return result;
                  if (e.preview.unavailable)
                    return {
                      ...result,
                      preview: undefined,
                      previewUnavailable: e.preview.reason,
                    };
                  if (
                    result.previewUnavailable ||
                    result.denoising?.blocks?.some(
                      (block) => block.block_index >= e.preview.block_index,
                    )
                  )
                    return result;
                  const p = result.preview;
                  if (
                    p &&
                    (p.block_index > e.preview.block_index ||
                      (p.block_index === e.preview.block_index &&
                        p.denoising_step >= e.preview.denoising_step))
                  )
                    return result;
                  return { ...result, preview: e.preview };
                }),
              );
            if (e.type === 'summary') setSummaries((p) => [...p, e.summary]);
            if (e.type === 'complete') {
              sawComplete = true;
              setRunSynthetic(e.run.synthetic);
              setRun(e.run);
              setHistory((p) => [e.run, ...p].slice(0, 30));
              setPhase(
                e.run.status === 'complete'
                  ? 'Run complete'
                  : e.run.status === 'cancelled'
                    ? 'Run cancelled'
                    : 'Run finished with errors',
              );
            }
            if (e.type === 'error') {
              serverReportedError = true;
              throw Error(e.message);
            }
          }
          if (done) break;
        }
        if (!sawComplete)
          throw Error(
            'Connection ended before the run was saved. Results may be incomplete.',
          );
      } finally {
        await reader.cancel().catch(() => {});
        reader.releaseLock();
      }
    } catch (e) {
      const transportFailure = !serverReportedError && !c.signal.aborted;
      const recovering =
        kind === 'profile' && transportFailure && receivedServerProgress;
      const message = c.signal.aborted
        ? 'Run cancelled.'
        : e instanceof Error
          ? e.message
          : String(e);
      setError(recovering ? '' : message);
      setPhase(
        recovering
          ? 'Connection interrupted. Reconnecting to sweep…'
          : 'Stopped',
      );
      if (transportFailure) {
        reconnectingSince.current = startedAt;
        setProgressConnected(false);
      }
      // Server snapshots stay authoritative. A local transport error must never
      // supersede a completed run with a newer client-generated timestamp.
      if (!receivedServerProgress)
        setRunProgress((previous) =>
          previous && !previous.run_status
            ? {
                ...previous,
                stage: c.signal.aborted ? 'cancelled' : 'error',
                message,
                updated_at: new Date().toISOString(),
              }
            : previous,
        );
      if (!recovering)
        setResults((previous) =>
          previous.map((result) =>
            result.status === 'running'
              ? { ...result, status: 'cancelled' }
              : result,
          ),
        );
    } finally {
      setRunning(false);
      ctrl.current = null;
    }
  }
  async function loadPreparedDataset() {
    try {
      const response = await fetch(api() + '/api/datasets/pg19');
      if (!response.ok)
        throw Error(
          'Prepared PG19 bank is missing. Run scripts/prepare_pg19.py first.',
        );
      setDataset(
        (await response.json()) as { prompt: string; [key: string]: unknown }[],
      );
      setDatasetName('PG19 test · 512-token excerpts');
    } catch (e) {
      setError(String(e));
    }
  }
  async function loadDataset(file: File) {
    try {
      const lines = (await file.text())
        .split('\n')
        .filter(Boolean)
        .map((x) => JSON.parse(x));
      if (
        !lines.length ||
        lines.some((x) => typeof x.prompt !== 'string' || !x.prompt.trim())
      )
        throw Error('Use JSONL with one non-empty prompt field per line.');
      setDataset(lines);
      setDatasetName(file.name);
    } catch (e) {
      setError(String(e));
    }
  }
  const ready = bridge && models.every((m) => health[m.id]?.ok);
  const canRun =
    !busy &&
    bridge &&
    models.every(
      (m) => health[m.id]?.ok || (m.id === 'diffusion' && managedDiffusion),
    );
  const left = results.find(
      (r) => r.model === 'diffusion' && r.index === selected,
    ),
    right = results.find(
      (r) => r.model === 'autoregressive' && r.index === selected,
    );
  const ratio =
    left?.status === 'complete' &&
    right?.status === 'complete' &&
    left.elapsed_ms &&
    right.elapsed_ms
      ? right.elapsed_ms / left.elapsed_ms
      : null;
  return (
    <main className="workspace">
      <header className="topbar">
        <div className="brand">
          <span className="logo">
            <Zap size={22} />
          </span>
          <strong>Spark / Gemma Lab</strong>
          <span className="edition">26B · FP8</span>
        </div>
        <div className="connection">
          <span className={'dot ' + (ready ? 'on' : '')} />
          {ready
            ? 'Two endpoints ready'
            : bridge
              ? 'Waiting for model servers'
              : 'Local bridge offline'}
          <Button
            variant="ghost"
            size="icon"
            aria-label="Refresh connection"
            onClick={connect}
          >
            <RefreshCw size={16} />
          </Button>
        </div>
      </header>
      <div className="titleline">
        <div>
          <p className="eyebrow">TWO SPARKS. ONE PROMPT.</p>
          <h1>See the difference.</h1>
          <p className="subtitle">
            Diffusion and autoregression, side by side.
          </p>
        </div>
        <div className="hardware">
          <Activity size={17} />
          <span>
            2 × NVIDIA DGX Spark
            <br />
            <small>One model per GB10 · MTP disabled</small>
          </span>
        </div>
      </div>
      {(synthetic || runSynthetic) && (
        <div className="notice">
          Synthetic test mode — generated fixtures, not model performance. Shown
          values are test fixtures and must not be used as hardware
          measurements.
        </div>
      )}
      {!bridge && (
        <div className="notice">
          Start the local bridge with <code>npm run bridge</code>, then refresh
          the connection.
        </div>
      )}
      <Tabs value={tab} onValueChange={(v) => setTab(String(v))}>
        <div className="tabrow">
          <TabsList variant="line">
            <TabsTrigger value="demo">
              <Zap />
              Live comparison
            </TabsTrigger>
            <TabsTrigger value="profile">
              <FlaskConical />
              Profiling
            </TabsTrigger>
            <TabsTrigger value="setup">
              <Settings2 />
              Connection & runs
            </TabsTrigger>
          </TabsList>
          <span className="mono">
            {running ? `${fmt(elapsed, 2)} s elapsed` : 'LOCAL WORKSPACE'}
          </span>
        </div>
        {(busy || runProgress?.kind === 'profile') && (
          <div ref={progressPanel} className="run-progress-container">
            {runProgress ? (
              <RunProgress
                progress={runProgress}
                active={busy}
                connected={progressConnected}
                onCancel={() => void stopRun()}
                cancelling={cancelling}
              />
            ) : (
              <p className="notice" role="status">
                A run is already active. Waiting for progress…
              </p>
            )}
          </div>
        )}
        <TabsContent value="demo">
          <section className="promptbox">
            <label className="field-title" htmlFor="prompt">
              Your prompt <span>Sent unchanged to both models</span>
            </label>
            <Textarea
              id="prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              disabled={busy}
              className="prompt"
            />
            <div className="controls">
              <label>
                Batch size
                <Input
                  aria-label="Batch size"
                  type="number"
                  min={1}
                  max={128}
                  value={batch}
                  onChange={(e) => setBatch(+e.target.value)}
                  disabled={busy}
                />
              </label>
              <label>
                Output limit
                <Input
                  aria-label="Output token limit"
                  type="number"
                  min={1}
                  max={contextBudget - 1}
                  value={tokens}
                  onChange={(e) => setTokens(+e.target.value)}
                  disabled={busy}
                />
              </label>
              <label>
                Diffusion canvas tokens
                <select
                  aria-label="Diffusion canvas tokens"
                  value={canvasLength}
                  onChange={(e) => setCanvasLength(Number(e.target.value))}
                  disabled={busy}
                >
                  {[8, 16, 32, 64, 128, 256, 512].map((size) => (
                    <option key={size} value={size}>
                      {size}
                      {size === 256 ? ' (default)' : ''}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                AR temperature
                <Input
                  aria-label="Gemma 4 temperature"
                  type="number"
                  min={0}
                  step={0.1}
                  value={temp}
                  onChange={(e) => setTemp(+e.target.value)}
                  disabled={busy}
                />
              </label>
              <label>
                AR seed
                <Input
                  aria-label="Gemma 4 seed"
                  type="number"
                  min={0}
                  value={seed}
                  onChange={(e) => setSeed(+e.target.value)}
                  disabled={busy}
                />
              </label>
              <div className="action">
                {running ? (
                  <Button
                    variant="outline"
                    onClick={() => void stopRun()}
                    disabled={
                      cancelling ||
                      (runProgress?.kind === 'profile' &&
                        !runProgress.run_status)
                    }
                  >
                    <Square />
                    Stop
                  </Button>
                ) : (
                  <Button
                    onClick={() => execute('demo')}
                    disabled={!canRun || !prompt.trim()}
                  >
                    <Play />
                    Run comparison
                  </Button>
                )}
              </div>
            </div>
            <p className="method-note">
              Canvas size is tokens per diffusion block. Changing it reloads and
              warms DiffusionGemma before the comparison (a few minutes). Sizes
              other than 256 are experimental.
            </p>
          </section>
          <div className="runline">
            <span>
              <span className={'dot ' + (running ? 'pulse' : '')} />
              {phase} · {displayedDenoising}
            </span>
            <div>
              {batch > 1 && (
                <label>
                  Showing request{' '}
                  <Input
                    className="request-index"
                    type="number"
                    aria-label="Displayed request"
                    min={1}
                    max={batch}
                    value={selected + 1}
                    onChange={(e) =>
                      setSelected(
                        Math.min(batch - 1, Math.max(0, +e.target.value - 1)),
                      )
                    }
                  />
                </label>
              )}
              {run && (
                <Button
                  variant="ghost"
                  onClick={() =>
                    download(run.id + '.json', JSON.stringify(run, null, 2))
                  }
                >
                  <Download />
                  Export run
                </Button>
              )}
            </div>
          </div>
          <div className="comparison">
            {models.map((m, i) => {
              const r = results.find(
                (v) => v.model === m.id && v.index === selected,
              );
              const confirmedBlocks =
                r?.denoising?.status === 'available'
                  ? r.denoising.blocks
                  : undefined;
              const pendingBlock = confirmedBlocks?.length
                ? Math.max(
                    ...confirmedBlocks.map((block) => block.block_index),
                  ) + 1
                : undefined;
              const emittedTokens =
                confirmedBlocks?.reduce(
                  (sum, block) => sum + block.emitted_tokens,
                  0,
                ) ?? 0;
              const previewRunning =
                m.id === 'diffusion' &&
                activePreview &&
                r?.status === 'running' &&
                !r.error &&
                emittedTokens < tokens;
              const showRefinement = previewRunning && r?.preview;
              return (
                <article
                  className={'model-panel ' + (i ? 'ar' : 'diffusion')}
                  key={m.id}
                >
                  <div className="model-heading">
                    <div>
                      <p className="model-kicker">
                        {i ? 'AUTOREGRESSIVE' : 'DIFFUSION'}
                      </p>
                      <h2>
                        {m.label} <span>26B</span>
                      </h2>
                    </div>
                    <div className="model-heading-controls">
                      <span className="host">{m.host}</span>
                      {m.id === 'diffusion' && (
                        <label
                          className="animation-toggle"
                          title="Stream actual intermediate predictions for live demos. May add a little latency; always disabled for profiling."
                        >
                          <input
                            type="checkbox"
                            checked={liveDiffusionPreview}
                            disabled={busy}
                            onChange={(event) =>
                              setLiveDiffusionPreview(event.target.checked)
                            }
                          />
                          Live diffusion preview
                        </label>
                      )}
                      {m.id === 'autoregressive' && (
                        <label
                          className="animation-toggle probability-toggle"
                          title="Color sampled tokens by their raw model probability. Always disabled for profiling."
                        >
                          <input
                            type="checkbox"
                            checked={showTokenProbabilities}
                            disabled={busy}
                            onChange={(event) =>
                              setShowTokenProbabilities(event.target.checked)
                            }
                          />
                          Token probabilities
                        </label>
                      )}
                      {m.id === 'diffusion' && (
                        <label
                          className="animation-toggle probability-toggle"
                          title="Color committed tokens by their final denoising probabilities. Always disabled for profiling."
                        >
                          <input
                            type="checkbox"
                            checked={showDiffusionTokenProbabilities}
                            disabled={busy}
                            onChange={(event) =>
                              setShowDiffusionTokenProbabilities(
                                event.target.checked,
                              )
                            }
                          />
                          Final token probabilities
                        </label>
                      )}
                    </div>
                  </div>
                  <div className="metrics">
                    <div>
                      <strong>
                        {fmt(
                          r?.elapsed_ms == null ? null : r.elapsed_ms / 1000,
                          2,
                        )}
                        <small> s</small>
                      </strong>
                      <span>Completion time</span>
                    </div>
                    <div>
                      <strong>
                        {fmt(r?.ttft_ms == null ? null : r.ttft_ms / 1000, 2)}
                        <small> s</small>
                      </strong>
                      <span>First committed output</span>
                    </div>
                    <div>
                      <strong>
                        {fmt(r?.tokens_per_second)}
                        <small> tok/s</small>
                      </strong>
                      <span>End-to-end rate</span>
                    </div>
                    <div title="Excludes prefill and the first emitted token block. Requires at least two blocks with server token counts.">
                      <strong>
                        {fmt(r?.post_first_block_tps)}
                        <small> tok/s</small>
                      </strong>
                      <span>After first block</span>
                    </div>
                  </div>
                  <div
                    className="output"
                    aria-label={`${m.label} output`}
                    aria-busy={r?.status === 'running'}
                  >
                    {r?.error ? (
                      <p className="error">{r.error}</p>
                    ) : r?.text || showRefinement ? (
                      <div className="response-text">
                        {m.id === 'autoregressive' && showTokenProbabilities ? (
                          <TokenProbabilityText
                            text={r?.text ?? ''}
                            data={r?.ar_token_probabilities}
                          />
                        ) : m.id === 'diffusion' &&
                          showDiffusionTokenProbabilities ? (
                          <TokenProbabilityText
                            text={r?.text ?? ''}
                            data={r?.diffusion_token_probabilities}
                            kind="diffusion"
                          />
                        ) : (
                          r?.text
                        )}
                        {showRefinement && (
                          <DiffusionPreview
                            key={`${previewRun}:${selected}:${r!.preview!.block_index}`}
                            prediction={r!.preview!}
                            continuation={Boolean(r?.text)}
                          />
                        )}
                      </div>
                    ) : (
                      <div className="empty">
                        <span className="empty-symbol">{i ? '→' : '≋'}</span>
                        <p>
                          {r?.status === 'running'
                            ? 'Generating…'
                            : 'Ready when you are'}
                        </p>
                        <small>
                          {r?.status === 'running'
                            ? previewRunning
                              ? 'Waiting for the first model prediction.'
                              : 'Waiting for the first content block.'
                            : 'Your model’s response will appear here.'}
                        </small>
                      </div>
                    )}
                    {r?.reasoning && (
                      <details>
                        <summary>Reasoning output</summary>
                        <pre>{r.reasoning}</pre>
                      </details>
                    )}
                  </div>
                  {previewRunning && (
                    <div className="diffusion-animation-caption">
                      <span title={r?.previewUnavailable}>
                        {r?.previewUnavailable
                          ? 'Live preview unavailable'
                          : 'Live model predictions · provisional'}
                      </span>
                      <span>
                        {r?.preview
                          ? `Block ${r.preview.block_index} · step ${r.preview.denoising_step}`
                          : pendingBlock == null
                            ? 'Waiting for a prediction'
                            : `Waiting for block ${pendingBlock}`}
                      </span>
                    </div>
                  )}
                  {m.id === 'diffusion' && <DenoisingReadout result={r} />}
                  {m.id === 'autoregressive' && showTokenProbabilities && (
                    <TokenProbabilityLegend
                      data={r?.ar_token_probabilities}
                      running={r?.status === 'running'}
                    />
                  )}
                  {m.id === 'diffusion' && showDiffusionTokenProbabilities && (
                    <TokenProbabilityLegend
                      data={r?.diffusion_token_probabilities}
                      running={r?.status === 'running'}
                      kind="diffusion"
                    />
                  )}
                  <footer>
                    <span
                      className={
                        'dot ' + (r?.status === 'complete' ? 'on' : '')
                      }
                    />
                    {r?.status === 'complete'
                      ? `${fmt(r.completion_tokens, 0)} output tokens · ${r.finish_reason || 'complete'}`
                      : r?.status || 'Idle'}
                    <span className="footer-right">FP8 W8A8 · no MTP</span>
                  </footer>
                </article>
              );
            })}
          </div>
          {ratio && (
            <div className="verdict">
              <span>
                <Check size={18} />
                Measured completion time
              </span>
              <strong>{fmt(ratio, 2)}×</strong>
              <p>
                Gemma 4 time ÷ DiffusionGemma time{' '}
                <small>
                  Compare output lengths above; this is not a quality score.
                </small>
              </p>
            </div>
          )}
          <p className="method-note">
            Configured context budget: {fmt(contextBudget, 0)} tokens for prompt
            plus answer. Temperature and seed apply to Gemma 4 only; Live
            comparisons use adaptive diffusion with a maximum of 48 denoising
            steps per block. Actual steps are shown only when reported by the
            server. Timers include the request and network round trip. Text
            appears as the server emits it, including diffusion blocks. The
            optional live preview shows the model’s intermediate predictions;
            these can change until a block is committed and do not count as
            output tokens or first committed output. Demo timings include any
            preview overhead{displayedPreview ? ' (enabled for this demo)' : ''}
            . Token colors show raw model probabilities for AR and final
            denoising probabilities for committed diffusion output, with red
            for low, orange for intermediate and green for high likelihood.
            Hover for exact values. Demo timings include probability collection when
            enabled. Profiling always disables previews and token probability
            collection. The after-first-block rate excludes prefill and the
            entire first output block; it is unavailable for a single block.
            Batch size dispatches that many concurrent requests per model; the
            inference scheduler controls the actual GPU batch.
          </p>
        </TabsContent>
        <TabsContent value="profile">
          <div className="profile-grid">
            <section className="profile-controls">
              <h2>Measure the frontier</h2>
              <div className="twocol">
                <label htmlFor="profile-input-length">
                  Input length (tokens)
                  <Input
                    id="profile-input-length"
                    type="number"
                    min={1}
                    max={Math.max(1, contextBudget - tokens)}
                    step={1}
                    placeholder="Original length"
                    value={inputTokens}
                    onChange={(e) => setInputTokens(e.target.value)}
                    disabled={busy}
                    aria-describedby="input-length-help"
                  />
                </label>
                <label htmlFor="profile-output-length">
                  Output limit (tokens)
                  <Input
                    id="profile-output-length"
                    type="number"
                    min={1}
                    max={contextBudget - 1}
                    value={tokens}
                    onChange={(e) => setTokens(+e.target.value)}
                    disabled={busy}
                  />
                </label>
              </div>
              <p className="method-note" id="input-length-help">
                Set input length to trim or repeat the source text before
                timing. Leave blank for its original length. Chat formatting
                adds tokens; actual totals appear in the results.
              </p>
              <p>
                Matched prompts, warmups, repeated waves. Live diffusion
                previews are always disabled here, including during warmups.
              </p>
              <label>
                Diffusion canvas tokens
                <select
                  aria-label="Diffusion canvas tokens"
                  value={canvasLength}
                  onChange={(e) => setCanvasLength(Number(e.target.value))}
                  disabled={busy}
                >
                  {[8, 16, 32, 64, 128, 256, 512].map((size) => (
                    <option key={size} value={size}>
                      {size}
                      {size === 256 ? ' (default)' : ''}
                    </option>
                  ))}
                </select>
              </label>
              <p className="method-note">
                Changing canvas size reloads DiffusionGemma and warms it up
                before measurement. Allow a few minutes. Sizes other than 256
                are experimental.
              </p>
              <label>
                Batch sizes
                <Input
                  value={batches}
                  onChange={(e) => setBatches(e.target.value)}
                  disabled={busy}
                />
              </label>
              <div className="twocol">
                <label>
                  Diffusion denoising
                  <select
                    value={profileDenoisingMode}
                    onChange={(e) =>
                      setProfileDenoisingMode(e.target.value as DenoisingMode)
                    }
                    disabled={busy}
                  >
                    <option value="adaptive">Adaptive (max 48)</option>
                    <option value="fixed">Fixed step count</option>
                  </select>
                </label>
                {profileDenoisingMode === 'fixed' && (
                  <label htmlFor="fixed-denoising-steps">
                    Steps per block
                    <Input
                      id="fixed-denoising-steps"
                      type="number"
                      min={1}
                      max={48}
                      step={1}
                      value={fixedDenoisingSteps}
                      onChange={(e) => setFixedDenoisingSteps(+e.target.value)}
                      disabled={busy}
                    />
                  </label>
                )}
              </div>
              {profileDenoisingMode === 'fixed' && (
                <p className="denoising-warning">
                  Fixed steps change the sampling schedule and may reduce output
                  quality. Compare the generated text as well as timing.
                </p>
              )}
              <div className="twocol">
                <label>
                  Requests per condition
                  <Input
                    type="number"
                    min={1}
                    max={10000}
                    value={requestCount}
                    onChange={(e) => setRequestCount(+e.target.value)}
                    disabled={busy}
                  />
                </label>
                <label>
                  Warmup waves
                  <Input
                    type="number"
                    min={0}
                    max={10}
                    value={warmups}
                    onChange={(e) => setWarmups(+e.target.value)}
                    disabled={busy}
                  />
                </label>
              </div>
              <p className="method-note">
                Input, formatting and output must fit the{' '}
                {fmt(contextBudget, 0)}-token context. Repeated excerpts are
                recorded as repeated text, not longer book passages.
              </p>
              <div className="twocol">
                <label>
                  Dataset task
                  <select
                    value={workload}
                    onChange={(e) => setWorkload(e.target.value)}
                    disabled={busy}
                  >
                    <option value="continuation">
                      Continue passage (chat)
                    </option>
                    <option value="raw">Raw completion</option>
                  </select>
                </label>
                <label>
                  Output stopping
                  <select
                    value={outputMode}
                    onChange={(e) => setOutputMode(e.target.value)}
                    disabled={busy}
                  >
                    <option value="natural">Natural EOS / token limit</option>
                    <option value="fixed">Force token count</option>
                  </select>
                </label>
              </div>
              <label>
                Prompt dataset
                <input
                  type="file"
                  accept=".jsonl,.ndjson"
                  disabled={busy}
                  onChange={(e) =>
                    e.target.files?.[0] && loadDataset(e.target.files[0])
                  }
                />
              </label>
              <Button
                variant="outline"
                disabled={busy || !bridge}
                onClick={loadPreparedDataset}
              >
                Use prepared PG19 bank
              </Button>
              <div className="dataset">
                <span>{datasetName}</span>
                <small>
                  {dataset.length
                    ? `${dataset.length} excerpts · ${workload === 'raw' ? 'raw completion' : 'chat continuation'}`
                    : `Uses the live-demo prompt · ${workload === 'raw' ? 'raw completion' : 'chat API'}`}
                </small>
                {dataset.length > 0 && (
                  <Button
                    variant="ghost"
                    disabled={busy}
                    onClick={() => {
                      setDataset([]);
                      setDatasetName('Current prompt');
                    }}
                  >
                    Clear dataset
                  </Button>
                )}
              </div>
              <p className="method-note">
                Natural stopping counts the tokens actually returned. Forcing
                length can generate special-token tails and overstate
                useful-text throughput. Chat continuation adds the same
                instruction around each excerpt. Each model uses the same
                measured prompts at every batch size. The request count must
                divide evenly by each batch size. Warmups are excluded. Start
                with 1, 2, 4 while checking GPU memory.
              </p>
              {busy && (running || runProgress?.kind === 'profile') ? (
                <Button
                  variant="outline"
                  onClick={() => void stopRun()}
                  disabled={
                    cancelling ||
                    (runProgress?.kind === 'profile' && !runProgress.run_status)
                  }
                >
                  <Square />
                  {cancelling ? 'Stopping…' : 'Stop sweep'}
                </Button>
              ) : (
                <Button disabled={!canRun} onClick={() => execute('profile')}>
                  <Play />
                  Run sweep
                </Button>
              )}
              <p className="phase">
                {busy && runProgress ? runProgress.message : phase} ·{' '}
                {displayedDenoising}
              </p>
            </section>
            <section className="chart-panel">
              <div className="section-head">
                <div>
                  <p className="eyebrow">
                    {runSynthetic ? 'SYNTHETIC TEST DATA' : 'MEASURED RESULTS'}
                  </p>
                  <h2>Throughput frontier</h2>
                </div>
                {run && (
                  <Button
                    variant="outline"
                    onClick={() =>
                      download(run.id + '.json', JSON.stringify(run, null, 2))
                    }
                  >
                    <Download />
                    JSON
                  </Button>
                )}
              </div>
              <div className="legend">
                <span className="purple">▼ DiffusionGemma</span>
                <span className="red">■ Gemma 4 AR</span>
              </div>
              {summaries.some((s) => s.valid) ? (
                <ResponsiveContainer width="100%" height={390}>
                  <ScatterChart
                    margin={{ top: 30, right: 35, bottom: 35, left: 25 }}
                  >
                    <CartesianGrid stroke="#e6e8eb" />
                    <XAxis
                      dataKey="mean_user_tps"
                      type="number"
                      name="Per-user throughput"
                      label={{
                        value: 'Per-user throughput (tok/s/user)',
                        position: 'bottom',
                        offset: 10,
                      }}
                    />
                    <YAxis
                      dataKey="aggregate_tps"
                      type="number"
                      name="Aggregate throughput"
                      label={{
                        value: 'Aggregate throughput (tok/s)',
                        angle: -90,
                        position: 'insideLeft',
                        offset: -15,
                      }}
                    />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                    {models.map((m, i) => (
                      <Scatter
                        key={m.id}
                        name={m.label}
                        data={summaries
                          .filter((s) => s.model === m.id && s.valid)
                          .map((s) => ({ ...s, label: `c=${s.batch_size}` }))}
                        fill={i ? '#c64840' : '#7851b7'}
                        line
                        shape={i ? 'square' : 'triangle'}
                      >
                        <LabelList dataKey="label" position="top" />
                      </Scatter>
                    ))}
                  </ScatterChart>
                </ResponsiveContainer>
              ) : (
                <div className="chart-empty">
                  <Activity size={40} />
                  <h3>Your measurements belong here.</h3>
                  <p>
                    Run a sweep to plot real throughput.
                    <br />
                    No reference or estimated points are preloaded.
                  </p>
                </div>
              )}
              <p className="method-note">
                c = concurrent requests per model. Includes prefill, queueing,
                and transport. Aggregate is measured per endpoint, not a GPU
                kernel counter. Failed or incomplete waves are excluded from the
                frontier. After-first-block rates pool token/time intervals; *
                marks requests with token-count increases after the final
                visible text.
              </p>
            </section>
          </div>
          {summaries.length > 0 && (
            <section className="results-table">
              <Table>
                <TableHeader>
                  <TableRow>
                    {[
                      'Model',
                      'Batch',
                      'Input tok (total)',
                      'Per user tok/s',
                      'Aggregate tok/s',
                      'After first block tok/s',
                      'P50 latency',
                      'P95 latency',
                      'Successes',
                    ].map((x) => (
                      <TableHead key={x}>{x}</TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {summaries.map((s, i) => (
                    <TableRow key={i}>
                      <TableCell>{s.model}</TableCell>
                      <TableCell>{s.batch_size}</TableCell>
                      <TableCell title="Observed input tokens including instructions and chat formatting">
                        {fmt(s.min_prompt_tokens, 0)}
                        {s.min_prompt_tokens !== s.max_prompt_tokens
                          ? `–${fmt(s.max_prompt_tokens, 0)}`
                          : ''}
                      </TableCell>
                      <TableCell>{fmt(s.mean_user_tps)}</TableCell>
                      <TableCell>{fmt(s.aggregate_tps)}</TableCell>
                      <TableCell title="Pooled token/time intervals; excludes the complete first block">
                        {fmt(s.pooled_post_first_block_tps)}
                        {s.requests_without_post_first_visible_text ? ' *' : ''}
                      </TableCell>
                      <TableCell>
                        {fmt(
                          s.p50_latency_ms == null
                            ? null
                            : s.p50_latency_ms / 1000,
                          2,
                        )}{' '}
                        s
                      </TableCell>
                      <TableCell>
                        {fmt(
                          s.p95_latency_ms == null
                            ? null
                            : s.p95_latency_ms / 1000,
                          2,
                        )}{' '}
                        s
                      </TableCell>
                      <TableCell>
                        {s.successes}/{s.requests}
                        {!s.valid ? ' · incomplete' : ''}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </section>
          )}
          {results.some((r) => r.model === 'diffusion') && (
            <details className="profile-denoising">
              <summary>Actual diffusion steps by request</summary>
              <div className="profile-denoising-requests">
                {(run?.results ?? results)
                  .filter((r) => r.model === 'diffusion')
                  .map((r, index) => (
                    <div key={r.request_id ?? index}>
                      <p>
                        Request {index + 1}
                        {r.batch_size ? ` · c=${r.batch_size}` : ''}
                      </p>
                      <DenoisingReadout result={r} />
                    </div>
                  ))}
              </div>
            </details>
          )}
        </TabsContent>
        <TabsContent value="setup">
          <section className="setup">
            <h2>Connected model servers</h2>
            <p>
              The local bridge reads <code>config/models.json</code>. SSH
              tunnels keep the model servers private.
            </p>
            {models.map((m) => (
              <div className="endpoint" key={m.id}>
                <div>
                  <h3>
                    {m.label} <span className="host">{m.host}</span>
                  </h3>
                  <code>{m.model}</code>
                  <p>{m.baseUrl}</p>
                </div>
                <span className={health[m.id]?.ok ? 'good' : 'error'}>
                  {health[m.id]?.ok
                    ? 'Ready'
                    : health[m.id]?.error || 'Not checked'}
                </span>
              </div>
            ))}
            <Button variant="outline" onClick={connect}>
              <RefreshCw />
              Check connections
            </Button>
            <h2 className="history-heading">Saved runs</h2>
            {history.length === 0 ? (
              <p>No saved runs yet.</p>
            ) : (
              history.map((r) => (
                <div className="history" key={r.id}>
                  <span>
                    <strong>{r.kind}</strong> ·{' '}
                    {new Date(r.created_at).toLocaleString()}
                    {r.synthetic ? ' · synthetic test' : ''}
                    <small className="history-denoising">
                      {denoisingLabel(r.settings, r.configuration)}
                      {r.kind === 'profile' &&
                        (r.settings?.input_tokens != null
                          ? ` · ${fmt(r.settings.input_tokens, 0)} input text tokens`
                          : ' · original input length')}
                    </small>
                  </span>
                  <Button
                    variant="ghost"
                    onClick={async () => {
                      const full = await fetch(
                        api() + '/api/runs/' + r.id,
                      ).then((v) => v.json());
                      download(r.id + '.json', JSON.stringify(full, null, 2));
                    }}
                  >
                    <Download />
                    Export
                  </Button>
                </div>
              ))
            )}
          </section>
        </TabsContent>
      </Tabs>
      {error && (
        <div className="error-banner" role="alert">
          {error}
        </div>
      )}
      <footer className="page-footer">
        <span>Spark / Gemma Lab</span>
        <span>
          FP8 · 26B-A4B · observed performance only <ArrowUpRight size={14} />
        </span>
      </footer>
    </main>
  );
}
