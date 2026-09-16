#!/usr/bin/env python3
"""Plot saved measurements and denoising settings without consulting current config."""
import argparse, csv, json
from pathlib import Path


def positive_int(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def saved_canvas_metadata(run, result=None):
    """Resolve only saved evidence; a selected/current UI value is not actual CL."""
    requested = positive_int((run.get('settings') or {}).get('canvas_length'))
    runtime = positive_int(((run.get('configuration') or {}).get('runtime') or {}).get('canvas_length'))
    results = [result] if result is not None else run.get('results', [])
    reported = {positive_int((row.get('denoising') or {}).get('canvas_length'))
                for row in results if row.get('model') == 'diffusion'}
    reported.discard(None)
    actual_values = reported | ({runtime} if runtime else set())
    consistent = len(actual_values) <= 1
    actual = next(iter(actual_values)) if len(actual_values) == 1 else None
    source = ('conflicting_saved_metadata' if not consistent else
              'per_request_metadata' if reported else
              'saved_runtime' if runtime else 'unrecorded')
    return {
        'requested_canvas_length': requested,
        'actual_canvas_length': actual,
        'canvas_length_source': source,
        'canvas_length_consistent': consistent,
        'canvas_length_matches_request': requested == actual if requested and actual else None,
    }


def canvas_caption(run):
    info = saved_canvas_metadata(run)
    requested, actual = info['requested_canvas_length'], info['actual_canvas_length']
    if not info['canvas_length_consistent']:
        return 'Canvas: conflicting saved runtime/request metadata'
    if actual is None:
        return (f'Canvas: {requested} requested; actual length not recorded' if requested
                else 'Canvas length not recorded')
    if requested and requested != actual:
        return f'Canvas MISMATCH: {requested} requested; {actual} recorded'
    return f'Canvas: {actual} tokens' + (' (saved runtime; request not recorded)' if requested is None else '')


def summary_export_rows(run):
    canvas = saved_canvas_metadata(run)
    # These fields describe the Diffusion side of the paired run, including
    # when they appear on the AR summary row; AR itself has no canvas.
    return [{**row, **{'diffusion_' + key: value for key, value in canvas.items()}}
            for row in run.get('summaries', [])]


def saved_denoising_settings(run):
    settings = run.get('settings') or {}
    runtime = (run.get('configuration') or {}).get('runtime') or {}
    mode = settings.get('denoising_mode')
    steps = positive_int(settings.get('denoising_steps'))
    if steps is None and mode != 'fixed':
        steps = positive_int(runtime.get('max_denoising_steps'))
    if mode == 'fixed':
        caption = (f'Diffusion: fixed {steps} denoising steps/block; early convergence disabled'
                   if steps else 'Diffusion: fixed denoising; step count not recorded')
    elif mode == 'adaptive':
        caption = (f'Diffusion: adaptive convergence; cap {steps} denoising steps/block'
                   if steps else 'Diffusion: adaptive convergence; step cap not recorded')
    else:
        mode = None
        caption = (f'Diffusion: cap {steps}; convergence mode not recorded'
                   if steps else 'Diffusion: denoising cap and convergence mode not recorded')
    return mode, steps, caption + '\n' + canvas_caption(run)


def measured_denoising_blocks(run):
    """Only complete measured requests with an available final attribution."""
    rows = []
    for result in run.get('results', []):
        if result.get('warmup') or result.get('model') != 'diffusion' or result.get('status') != 'complete':
            continue
        info = result.get('denoising') or {}
        available = info.get('status') == 'available' or (not info.get('status') and info.get('available') is True)
        if not available or info.get('final') is False:
            continue
        for block in info.get('blocks', []):
            steps = positive_int(block.get('denoising_steps'))
            concurrency = positive_int(result.get('batch_size'))
            if steps is None or concurrency is None:
                continue
            rows.append({
                'run_id': run.get('id', ''), 'model': 'diffusion',
                'request_id': result.get('request_id', ''),
                'wave': result.get('wave'), 'repeat': result.get('repeat'),
                'batch_size': concurrency, 'block_index': block.get('block_index'),
                'denoising_steps': steps, 'canvas_tokens': block.get('canvas_tokens'),
                'emitted_tokens': block.get('emitted_tokens'), 'mode': info.get('mode'),
                'max_steps': info.get('max_steps'), 'source': info.get('source'),
                **saved_canvas_metadata(run, result),
                'synthetic': bool(run.get('synthetic')),
            })
    return rows


def decorate(fig, ax, run, caption):
    source_tokens = (run.get('settings') or {}).get('input_tokens')
    if source_tokens is not None:
        totals = []
        for model in sorted({r.get('model', '') for r in run.get('results', [])}):
            counts = [r['prompt_tokens'] for r in run.get('results', [])
                      if r.get('model') == model and isinstance(r.get('prompt_tokens'), int)]
            if counts:
                observed = str(min(counts)) if min(counts) == max(counts) else f'{min(counts)}–{max(counts)}'
                totals.append(f'{model}: {observed}')
        caption += f'\nInput text: {source_tokens:,} tokens · Total with formatting: ' + '; '.join(totals)
    fig.suptitle(caption, fontsize=10, color='#444444')
    if run.get('synthetic'):
        ax.text(.5, .5, 'SYNTHETIC TEST DATA', transform=ax.transAxes,
                ha='center', rotation=25, fontsize=25, alpha=.3, zorder=20)


def plot_denoising(run, rows, output, plt, caption, saved_cap):
    fields = ['run_id', 'model', 'request_id', 'wave', 'repeat', 'batch_size',
              'block_index', 'denoising_steps', 'canvas_tokens', 'emitted_tokens',
              'requested_canvas_length', 'actual_canvas_length', 'canvas_length_source',
              'canvas_length_consistent', 'canvas_length_matches_request',
              'mode', 'max_steps', 'source', 'synthetic']
    with (output / 'denoising-blocks.csv').open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    if rows:
        concurrencies = sorted({row['batch_size'] for row in rows})
        values = [[row['denoising_steps'] for row in rows if row['batch_size'] == c]
                  for c in concurrencies]
        positions = list(range(len(concurrencies)))
        ax.boxplot(values, positions=positions, widths=.4, patch_artist=True,
                   showfliers=False, boxprops={'facecolor': '#e7dbf5', 'edgecolor': '#7851b7'},
                   medianprops={'color': '#332044', 'linewidth': 2},
                   whiskerprops={'color': '#7851b7'}, capprops={'color': '#7851b7'})
        for position, samples in zip(positions, values):
            # Deterministic spreading shows coincident blocks; the y values remain exact.
            offsets = [0] if len(samples) == 1 else [(.3 * (i / (len(samples) - 1) - .5)) for i in range(len(samples))]
            ax.scatter([position + offset for offset in offsets], samples, s=23,
                       color='#7851b7', alpha=.6, edgecolors='none', zorder=3)
        ax.set_xticks(positions, [f'c={c}\n{len(samples)} blocks' for c, samples in zip(concurrencies, values)])
        caps = {positive_int(row['max_steps']) for row in rows}
        caps.discard(None)
        if saved_cap:
            caps.add(saved_cap)
        for cap in sorted(caps):
            ax.axhline(cap, color='#888888', linestyle='--', linewidth=1,
                       label=f'Saved step cap: {cap}', zorder=1)
        if caps:
            ax.legend(loc='upper right')
        maximum = max([row['denoising_steps'] for row in rows] + list(caps))
        ax.set_ylim(0, maximum * 1.15 + .5)
        ax.set_xlim(-.6, len(concurrencies) - .4)
        ax.text(.98, .03, 'One point per committed block; commit passes excluded',
                transform=ax.transAxes, ha='right', fontsize=9, color='gray')
    else:
        ax.text(.5, .5, 'No available per-block denoising measurements in this run.',
                transform=ax.transAxes, ha='center', va='center', color='#666666')
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set(xlabel='Client concurrency · completed measured requests',
           ylabel='Actual denoising steps per emitted block',
           title='Denoising-step distribution')
    ax.grid(axis='y', alpha=.2)
    decorate(fig, ax, run, caption)
    fig.savefig(output / 'denoising-steps.png', dpi=200)
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, default=Path('results/plots'))
    parser.add_argument('--allow-synthetic', action='store_true', help='For integration tests only; watermarks all plots')
    args=parser.parse_args()
    run=json.loads(args.run.read_text())
    if run.get('synthetic') and not args.allow_synthetic:
        parser.error('Synthetic runs are not hardware measurements. Use --allow-synthetic only to test plotting.')
    summaries=run['summaries']
    if not summaries: parser.error('No measured summaries in this run')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _, saved_cap, caption = saved_denoising_settings(run)
    args.output.mkdir(parents=True,exist_ok=True)
    exported_summaries = summary_export_rows(run)
    with (args.output/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(exported_summaries[0]));writer.writeheader();writer.writerows(exported_summaries)
    canvas = saved_canvas_metadata(run)
    if not canvas['canvas_length_consistent'] or canvas['canvas_length_matches_request'] is False:
        parser.error('Saved canvas metadata is inconsistent; CSV preserves the conflicting evidence, but no performance plots were made.')
    fig,ax=plt.subplots(figsize=(10,7),layout='constrained')
    styles={'diffusion':('#7851b7','v','DiffusionGemma FP8'),'autoregressive':('#c64840','s','Gemma 4 AR FP8')}
    valid=[s for s in summaries if s['valid']]
    if not valid: parser.error('No complete measurements with verified token counts to plot; CSV contains failures')
    for model,(color,marker,label) in styles.items():
        rows=sorted([s for s in valid if s['model']==model],key=lambda s:s['batch_size'])
        if not rows: continue
        x=[s['mean_user_tps'] for s in rows];y=[s['aggregate_tps'] for s in rows]
        ax.plot(x,y,color=color,marker=marker,markersize=9,linewidth=2,label=label,linestyle='-' if model=='diffusion' else '--')
        for s in rows: ax.annotate(f"c={s['batch_size']}",(s['mean_user_tps'],s['aggregate_tps']),xytext=((8,8) if model=='diffusion' else (-8,8)),ha=('left' if model=='diffusion' else 'right'),textcoords='offset points',color=color)
    ax.set(xlabel='Mean per-user served output throughput (tok/s/user)',ylabel='Aggregate served output throughput (tok/s)',title='Throughput frontier',xlim=(0,None),ylim=(0,None))
    ax.set_xlim(0,max(s['mean_user_tps'] for s in valid)*1.18);ax.set_ylim(0,max(s['aggregate_tps'] for s in valid)*1.15)
    ax.grid(alpha=.2);ax.legend();ax.text(.98,.03,'c = client concurrency · end-to-end timing',transform=ax.transAxes,ha='right',fontsize=9,color='gray')
    decorate(fig, ax, run, caption)
    fig.savefig(args.output/'frontier.png',dpi=200);fig.savefig(args.output/'frontier.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
    for model,(color,marker,label) in styles.items():
        rows=sorted([s for s in valid if s['model']==model],key=lambda s:s['batch_size'])
        if rows:
            ax.plot([r['batch_size'] for r in rows],[r['p50_latency_ms']/1000 for r in rows],color=color,marker=marker,label=label+' p50')
            ax.plot([r['batch_size'] for r in rows],[r['p95_latency_ms']/1000 for r in rows],color=color,linestyle=':',label=label+' p95')
    ax.set(xlabel='Client concurrency',ylabel='Complete-answer latency (s)',title='Latency by concurrency',ylim=(0,None));ax.grid(alpha=.2);ax.legend()
    decorate(fig, ax, run, caption)
    fig.savefig(args.output/'latency.png',dpi=200);plt.close(fig)
    if any(r.get('pooled_post_first_block_tps') is not None for r in valid):
        fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
        for model,(color,marker,label) in styles.items():
            rows=sorted([r for r in valid if r['model']==model and r.get('pooled_post_first_block_tps') is not None],key=lambda r:r['batch_size'])
            if rows: ax.plot([r['batch_size'] for r in rows],[r['pooled_post_first_block_tps'] for r in rows],color=color,marker=marker,label=label)
        ax.set(xlabel='Client concurrency',ylabel='Pooled generation rate after first block (tok/s/user)',title='Generation after the first emitted block',ylim=(0,None));ax.grid(alpha=.2);ax.legend()
        ax.text(.98,.03,'Excludes prefill AND first output block; client arrival timing',transform=ax.transAxes,ha='right',fontsize=9,color='gray')
        decorate(fig, ax, run, caption)
        fig.savefig(args.output/'generation.png',dpi=200);plt.close(fig)
    plot_denoising(run, measured_denoising_blocks(run), args.output, plt, caption, saved_cap)
    print(args.output.resolve())
if __name__=='__main__':main()
