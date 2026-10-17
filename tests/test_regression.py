"""Offline regression checks against results captured before extraction."""

import ast
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from eth_bridge.data import fetch_yfinance_ohlcv_df, rolling_zscore
from eth_bridge.forecasting import test_volume

FIXTURES = Path(__file__).parent / 'fixtures'
ROOT = Path(__file__).resolve().parents[1]


def sample_data():
    dates = pd.date_range('2025-04-01', '2025-08-23', name='Date')
    t = np.arange(len(dates), dtype=float)
    return pd.DataFrame({
        'volume': 1000 + 2 * t + 100 * np.sin(t / 3),
        'close': 100 + np.cos(t / 5),
        'bridge': 20 + np.sin(t / 7),
    }, index=dates)


def serializable_results(result):
    forecasts, monthly, summary = result
    forecasts = forecasts.copy()
    forecasts['month'] = forecasts['month'].astype(str)
    for col in ['feature_date', 'target_date']:
        forecasts[col] = forecasts[col].astype(str)
    return {
        'forecasts': forecasts.to_dict(orient='list'),
        'monthly': monthly.to_dict(orient='list'),
        'summary': summary,
    }


class RegressionTests(unittest.TestCase):
    def test_function_bodies_match_reference(self):
        fingerprints = json.loads((FIXTURES / 'function_fingerprints.json').read_text())
        for qualified_name, expected in fingerprints.items():
            module, name = qualified_name.split('.')
            tree = ast.parse((ROOT / 'eth_bridge' / f'{module}.py').read_text())
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            actual = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
            self.assertEqual(actual, expected, qualified_name)

    def test_forecasts_match_original_for_all_horizons(self):
        expected = json.loads((FIXTURES / 'volume_forecasts.json').read_text())
        for horizon in (1, 3, 7):
            with self.subTest(horizon=horizon):
                data = sample_data()
                before = data.copy(deep=True)
                stdout = StringIO()
                with redirect_stdout(stdout):
                    result = test_volume(data, horizon=horizon, model_name='Regression')
                actual = serializable_results(result)
                reference = expected[str(horizon)]
                for key in ('forecasts', 'monthly'):
                    assert_frame_equal(pd.DataFrame(actual[key]), pd.DataFrame(reference[key]),
                                       check_exact=False, rtol=1e-12, atol=1e-12)
                for key, value in reference['summary'].items():
                    if isinstance(value, float):
                        self.assertAlmostEqual(actual['summary'][key], value, places=12)
                    else:
                        self.assertEqual(actual['summary'][key], value)
                self.assertEqual(stdout.getvalue(), reference['stdout'])
                assert_frame_equal(data, before)

    def test_rolling_zscore_constant_and_missing_values(self):
        series = pd.Series([1., 1., 1., np.nan, 2., 3., 4.])
        expected = (series - series.rolling(3).mean()) / series.rolling(3).std().replace(0, np.nan)
        assert_series_equal(rolling_zscore(series, window=3), expected)

    def test_download_normalization_without_network(self):
        columns = pd.MultiIndex.from_product([
            ['Open', 'High', 'Low', 'Close', 'Volume'], ['ETH-USD']
        ])
        index = pd.date_range('2025-01-01', periods=2, tz='UTC', name='Date')
        downloaded = pd.DataFrame([[1, 2, 0, 1, 10], [2, 3, 1, 2, 20]],
                                  index=index, columns=columns)
        with patch('eth_bridge.data.yf.download', return_value=downloaded) as download:
            actual = fetch_yfinance_ohlcv_df('ETH-USD')
        expected = pd.DataFrame([[1, 2, 0, 1, 10], [2, 3, 1, 2, 20]],
                                index=index.tz_localize(None),
                                columns=['open', 'high', 'low', 'close', 'volume'])
        assert_frame_equal(actual, expected)
        download.assert_called_once_with('ETH-USD', start='2023-01-01', end='2026-08-31',
                                         interval='1d', auto_adjust=False, progress=False)


if __name__ == '__main__':
    unittest.main()
