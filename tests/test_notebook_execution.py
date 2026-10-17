"""Checks for notebook failures that occur in a fresh checkout."""

from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[1]


def notebook_cell(marker):
    notebook = json.loads((ROOT / 'app.ipynb').read_text())
    return next(''.join(cell['source']) for cell in notebook['cells']
                if cell['cell_type'] == 'code' and marker in ''.join(cell['source']))


class NotebookExecutionTests(unittest.TestCase):
    def test_bridge_exports_create_output_directory(self):
        raw = pd.DataFrame({'day': ['2025-01-01'], 'gross_volume_usd': [123.]})
        engineered = pd.DataFrame({'day': ['2025-01-01'], 'BII': [0.5]})
        source = notebook_cell("bridge_raw.to_csv(")
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                for _ in range(2):
                    exec(source, {'bridge_raw': raw, 'bridge_engineered': engineered})
                assert_frame_equal(pd.read_csv('bridges/bridge_raw.csv'), raw)
                assert_frame_equal(pd.read_csv('bridges/bridge_engineered.csv'), engineered)
            finally:
                os.chdir(previous)

    def test_regime_date_comparison_uses_common_predictions(self):
        dates = pd.date_range('2025-08-01', periods=5)
        namespace = {
            'pd': pd,
            'results_h3_m1': pd.DataFrame({'feature_date': dates[:4]}),
            'results_h3_m3': pd.DataFrame({'feature_date': dates[1:]}),
            'regime_info': pd.DataFrame({
                'feature_date': dates,
                'bridge_regime': ['High', 'High', 'Normal', 'High', 'High'],
            }),
            'quintile_info': pd.DataFrame({
                'feature_date': dates,
                'bridge_quintile': ['Q4', 'Q4', 'Q4', None, 'Q4'],
            }),
        }
        output = StringIO()
        with redirect_stdout(output):
            exec(notebook_cell('# Dates in the High regime'), namespace)
        self.assertEqual(namespace['high_dates'], {dates[1], dates[3]})
        self.assertEqual(namespace['q4_dates'], {dates[1], dates[2]})
        self.assertEqual(output.getvalue(),
                         'High n: 2\nQ4 n: 2\nCommon: 1\nHigh only: 1\n'
                         'Q4 only: 1\nExactly same dates: False\n')


if __name__ == '__main__':
    unittest.main()
