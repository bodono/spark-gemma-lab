#!/usr/bin/env python3
"""CPU fixture checks for staged exports; no network, GPU or project writes."""
import copy
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parent.parent / 'scripts/plot_results.py'
spec = importlib.util.spec_from_file_location('canvas_plot_results', SCRIPT)
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)


def fixture(canvas=512):
    return {
        'id': 'canvas-fixture', 'synthetic': True,
        'settings': {'canvas_length': canvas, 'denoising_mode': 'adaptive', 'denoising_steps': 48},
        'configuration': {'runtime': {'canvas_length': canvas}},
        'results': [{
            'model': 'diffusion', 'status': 'complete', 'batch_size': 1,
            'request_id': 'request-one', 'prompt_tokens': 100,
            'denoising': {'status': 'available', 'final': True, 'canvas_length': canvas,
                         'mode': 'adaptive', 'max_steps': 48, 'source': 'fixture',
                         'blocks': [{'block_index': 1, 'denoising_steps': 7,
                                     'canvas_tokens': canvas, 'emitted_tokens': 64}]},
        }],
        'summaries': [
            {'model': model, 'batch_size': 1, 'valid': True, 'requests': 1,
             'mean_user_tps': 20, 'aggregate_tps': 20,
             'p50_latency_ms': 3200, 'p95_latency_ms': 3200,
             'pooled_post_first_block_tps': None}
            for model in ['diffusion', 'autoregressive']
        ],
    }


class CanvasExportTests(unittest.TestCase):
    def test_saved_canvas_overrides_current_environment(self):
        before = os.environ.get('CANVAS_LENGTH')
        os.environ['CANVAS_LENGTH'] = '128'
        try:
            run = fixture(512)
            self.assertEqual(plots.saved_canvas_metadata(run)['actual_canvas_length'], 512)
            self.assertIn('Canvas: 512 tokens', plots.saved_denoising_settings(run)[2])
            self.assertNotIn('128', plots.canvas_caption(run))
        finally:
            if before is None:
                os.environ.pop('CANVAS_LENGTH', None)
            else:
                os.environ['CANVAS_LENGTH'] = before

    def test_historical_runtime_and_missing_provenance_are_distinct(self):
        run = fixture(256)
        del run['settings']['canvas_length']
        del run['results'][0]['denoising']['canvas_length']
        info = plots.saved_canvas_metadata(run)
        self.assertIsNone(info['requested_canvas_length'])
        self.assertEqual(info['actual_canvas_length'], 256)
        self.assertEqual(info['canvas_length_source'], 'saved_runtime')
        self.assertIn('request not recorded', plots.canvas_caption(run))
        del run['configuration']['runtime']['canvas_length']
        self.assertIsNone(plots.saved_canvas_metadata(run)['actual_canvas_length'])
        self.assertEqual(plots.canvas_caption(run), 'Canvas length not recorded')
        # A partial block's count must not be mistaken for configured canvas size.
        self.assertEqual(plots.measured_denoising_blocks(run)[0]['canvas_tokens'], 256)

    def test_summary_and_block_csv_keep_actual_and_emitted_separate(self):
        run = fixture(512)
        original = copy.deepcopy(run)
        summaries = plots.summary_export_rows(run)
        for row in summaries:
            self.assertEqual(row['diffusion_requested_canvas_length'], 512)
            self.assertEqual(row['diffusion_actual_canvas_length'], 512)
            self.assertTrue(row['diffusion_canvas_length_matches_request'])
            self.assertNotIn('actual_canvas_length', row)
        blocks = plots.measured_denoising_blocks(run)
        self.assertEqual(len(blocks), 1)
        self.assertEqual((blocks[0]['actual_canvas_length'], blocks[0]['canvas_tokens'], blocks[0]['emitted_tokens']),
                         (512, 512, 64))
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(blocks[0]))
        writer.writeheader()
        writer.writerows(blocks)
        row = next(csv.DictReader(io.StringIO(buffer.getvalue())))
        self.assertEqual(row['requested_canvas_length'], '512')
        self.assertEqual(row['actual_canvas_length'], '512')
        self.assertEqual(row['emitted_tokens'], '64')
        self.assertEqual(run, original, 'Export must not mutate saved run evidence')

    def test_canvas_disagreement_is_explicit(self):
        run = fixture(256)
        run['settings']['canvas_length'] = 128
        self.assertFalse(plots.saved_canvas_metadata(run)['canvas_length_matches_request'])
        self.assertIn('MISMATCH', plots.canvas_caption(run))
        run['results'][0]['denoising']['canvas_length'] = 512
        info = plots.saved_canvas_metadata(run)
        self.assertFalse(info['canvas_length_consistent'])
        self.assertIsNone(info['actual_canvas_length'])
        self.assertIn('conflicting', plots.canvas_caption(run))

    def test_real_export_outputs_with_one_block_and_no_generation_interval(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / 'fixture.json'
            source.write_text(json.dumps(fixture(512)))
            output = root / 'plots'
            result = subprocess.run([sys.executable, str(SCRIPT), str(source), '--allow-synthetic',
                                     '--output', str(output)], capture_output=True, text=True,
                                    env={**os.environ, 'CANVAS_LENGTH': '128'}, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((output / 'frontier.png').exists())
            self.assertTrue((output / 'denoising-steps.png').exists())
            self.assertFalse((output / 'generation.png').exists())
            with (output / 'summary.csv').open() as handle:
                self.assertEqual(next(csv.DictReader(handle))['diffusion_actual_canvas_length'], '512')
            with (output / 'denoising-blocks.csv').open() as handle:
                row = next(csv.DictReader(handle))
                self.assertEqual(row['actual_canvas_length'], '512')
                self.assertEqual(row['emitted_tokens'], '64')
            self.assertIn('Canvas: 512 tokens', (output / 'frontier.svg').read_text())

    def test_mismatch_exports_csv_but_refuses_performance_plots(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            run = fixture(256)
            run['settings']['canvas_length'] = 128
            source = root / 'mismatch.json'
            source.write_text(json.dumps(run))
            output = root / 'plots'
            result = subprocess.run([sys.executable, str(SCRIPT), str(source), '--allow-synthetic',
                                     '--output', str(output)], capture_output=True, text=True, timeout=60)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Saved canvas metadata is inconsistent', result.stderr)
            self.assertTrue((output / 'summary.csv').exists())
            self.assertFalse((output / 'frontier.png').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
