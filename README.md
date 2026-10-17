# ETH Bridge Volume Prediction

A research project that predicts changes in Ethereum trading volume using market data and Layer 2 cross-chain bridge activity. Experiments include price-direction classification baselines and trading-volume regression models.

This repository provides a simplified prototype developed as part of a broader research effort. It illustrates the modeling workflow and selected experiments, rather than serving as a complete reproduction of the research. Data, preprocessing, model specifications, and evaluation settings may differ from those used in the full study, so results should be interpreted within the scope of this implementation and may not match findings reported elsewhere.

## Getting started

Requirements: Python 3.13 or later and `uv`. Run these commands from the repository root:

```sh
uv sync --frozen
uv run --with jupyterlab jupyter lab app.ipynb
```

Restart the notebook kernel and run the cells from top to bottom. Later experiments use data and settings created by earlier cells. Running all experiments can take time because the models are trained repeatedly.

## Project structure

| Path | Purpose |
| --- | --- |
| `app.ipynb` | Data preparation, M1–M4 comparisons, statistical tests, regime analysis, and momentum ablation |
| `eth_bridge/data.py` | Yahoo Finance OHLCV normalization and rolling z-scores |
| `eth_bridge/forecasting.py` | Trading-volume Random Forest forecasts with periodic retraining |
| `eth_bridge/evaluation.py` | MAE, RMSE, R², and direction accuracy |
| `bridge_raw_data.csv` | Raw bridge data collected from Dune |
| `bridge_indicators2.csv` | Generated bridge indicators |
| `bridges/bridge_raw.csv` | Generated raw bridge features |
| `bridges/bridge_engineered.csv` | Generated engineered bridge features |
| `*eth_rf_weekly_retrain_*.csv` | Daily predictions and monthly metrics from classification experiments |
| `tests/` | Offline regression tests and reference results |

The models use the following feature sets:

- **M1:** Market features only.
- **M2:** Market and raw bridge features.
- **M3:** Market and engineered bridge features.
- **M4:** Market, raw bridge, and engineered bridge features.

The notebook defines the exact columns used by each model.

## Data source

`bridge_raw_data.csv` contains cross-chain bridge data collected from Dune, including transfer amounts and transaction counts by date and chain. Other CSV files are generated during execution and excluded from Git. The `bridges/` output directory is created automatically when exporting features.

## Experiment settings and reproducibility

For the volume regression experiments, training starts on `2023-08-01` and evaluation runs from `2025-08-01` through `2026-07-31`. Forecast horizons are 1, 3, and 7 days. The default retraining interval is 7 days, and the Random Forest seed is 42. The target is `log(volume[t+h]) - log(volume[t])`. Training uses only labels observable at the forecast date.

Use the committed `uv.lock` to reuse dependency versions. ETH prices and trading volumes are downloaded from Yahoo Finance during execution, so an internet connection is required. Provider revisions and download timing can affect full-notebook results.

The notebook retains experiment-specific versions of `dm_test_hac` to preserve their existing behavior. Running the notebook overwrites generated CSV files in the repository root and `bridges/`. Local HTML exports (`app.html`) are excluded from Git.

## Validation

```sh
uv run python -m unittest discover -s tests -v
```

Tests run offline. They compare 1-, 3-, and 7-day forecasts, monthly metrics, summary statistics, and console output against fixed-input reference results captured before the refactoring. Numerical comparisons allow a tolerance of `1e-12` for minor floating-point differences from parallel computation. Function AST fingerprints check for unintended changes; the forecasting fingerprint includes the English error messages. Additional checks cover output-directory creation and regime-date comparisons.

These tests do not execute the entire notebook or verify live data downloads.

## License

[MIT](LICENSE)
