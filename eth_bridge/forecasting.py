"""Reusable forecasting helpers from the research notebook."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def test_volume(input_data, horizon=3, model_name='Model', retrain_days=7):
    TRAIN_START = pd.Timestamp('2023-08-01')
    TEST_START = pd.Timestamp('2025-08-01')
    TEST_END = pd.Timestamp('2026-07-31')
    VOLUME_COL = 'volume'
    TARGET_COL = 'future_volume_change'
    HORIZON = horizon
    RETRAIN_DAYS = retrain_days
    RANDOM_STATE = 42
    data = input_data.copy()
    if data.index.name == 'Date':
        data = data.reset_index()
    if 'Date' in data.columns:
        date_col = 'Date'
        data[date_col] = pd.to_datetime(data[date_col])
    elif 'day' in data.columns:
        date_col = 'day'
        data[date_col] = pd.to_datetime(data[date_col])
    elif isinstance(data.index, pd.DatetimeIndex):
        data = data.reset_index()
        first_col = data.columns[0]
        data = data.rename(columns={first_col: 'Date'})
        date_col = 'Date'
        data[date_col] = pd.to_datetime(data[date_col])
    else:
        raise ValueError('Could not find a Date or day column.')
    data = data.sort_values(date_col).reset_index(drop=True)
    if VOLUME_COL not in data.columns:
        raise ValueError("Missing 'volume' column.")
    data[VOLUME_COL] = pd.to_numeric(data[VOLUME_COL], errors='coerce')
    data.loc[data[VOLUME_COL] <= 0, VOLUME_COL] = np.nan
    data[TARGET_COL] = np.log(data[VOLUME_COL].shift(-HORIZON)) - np.log(data[VOLUME_COL])
    data['future_volume'] = data[VOLUME_COL].shift(-HORIZON)
    exclude_cols = [date_col, TARGET_COL, 'future_volume', 'target', 'future_return_d']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    for col in feature_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data[TARGET_COL] = pd.to_numeric(data[TARGET_COL], errors='coerce')
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=feature_cols + [TARGET_COL, VOLUME_COL, 'future_volume']).reset_index(drop=True)
    data = data[(data[date_col] >= TRAIN_START) & (data[date_col] <= TEST_END)].copy()
    data = data.sort_values(date_col).reset_index(drop=True)
    print()
    print('=' * 70)
    print(f'{model_name} | ETH Volume t+{HORIZON}')
    print('=' * 70)
    print('Number of features:', len(feature_cols))
    print('Data period:', data[date_col].min(), '->', data[date_col].max())
    test_indices = data.index[(data[date_col] >= TEST_START) & (data[date_col] <= TEST_END)].tolist()
    if len(test_indices) == 0:
        raise ValueError('No data available in the test period.')
    RF_PARAMS = {'n_estimators': 300, 'max_depth': 8, 'min_samples_leaf': 5, 'max_features': 'sqrt', 'random_state': RANDOM_STATE, 'n_jobs': -1}
    predictions = []
    model = None
    last_retrain_date = None
    for idx in test_indices:
        current_row = data.loc[idx]
        current_date = current_row[date_col]
        need_retrain = model is None or (current_date - last_retrain_date).days >= RETRAIN_DAYS
        if need_retrain:
            observable_cutoff = current_date - pd.Timedelta(days=HORIZON)
            train = data[(data[date_col] >= TRAIN_START) & (data[date_col] <= observable_cutoff)].copy()
            if len(train) == 0:
                continue
            X_train = train[feature_cols]
            y_train = train[TARGET_COL]
            model = RandomForestRegressor(**RF_PARAMS)
            model.fit(X_train, y_train)
            last_retrain_date = current_date
            print(f'Retrain: {current_date.date()} | Train through: {train[date_col].max().date()} | n={len(train)}')
        X_today = data.loc[[idx], feature_cols]
        predicted_change = model.predict(X_today)[0]
        actual_change = float(current_row[TARGET_COL])
        current_volume = float(current_row[VOLUME_COL])
        actual_future_volume = float(current_row['future_volume'])
        predicted_future_volume = current_volume * np.exp(predicted_change)
        predictions.append({'feature_date': current_date, 'target_date': current_date + pd.Timedelta(days=HORIZON), 'horizon': HORIZON, 'actual_volume_change': actual_change, 'predicted_volume_change': predicted_change, 'current_volume': current_volume, 'actual_future_volume': actual_future_volume, 'predicted_future_volume': predicted_future_volume})
    results = pd.DataFrame(predictions)
    if len(results) == 0:
        raise ValueError('No prediction results available.')
    y_true = results['actual_volume_change']
    y_pred = results['predicted_volume_change']
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    actual_direction = (y_true > 0).astype(int)
    predicted_direction = (y_pred > 0).astype(int)
    direction_accuracy = (actual_direction == predicted_direction).mean()
    results['month'] = results['feature_date'].dt.to_period('M')
    monthly_results = []
    for month, g in results.groupby('month'):
        monthly_mae = mean_absolute_error(g['actual_volume_change'], g['predicted_volume_change'])
        monthly_rmse = np.sqrt(mean_squared_error(g['actual_volume_change'], g['predicted_volume_change']))
        monthly_r2 = r2_score(g['actual_volume_change'], g['predicted_volume_change'])
        actual_dir = (g['actual_volume_change'] > 0).astype(int)
        predicted_dir = (g['predicted_volume_change'] > 0).astype(int)
        monthly_direction_accuracy = (actual_dir == predicted_dir).mean()
        monthly_results.append({'month': str(month), 'n': len(g), 'mae': monthly_mae, 'rmse': monthly_rmse, 'r2': monthly_r2, 'direction_accuracy': monthly_direction_accuracy})
    monthly_results = pd.DataFrame(monthly_results)
    summary = {'model': model_name, 'horizon': HORIZON, 'n_features': len(feature_cols), 'n_test': len(results), 'mae': mae, 'rmse': rmse, 'r2': r2, 'direction_accuracy': direction_accuracy}
    print()
    print('-' * 70)
    print(f'Overall MAE         : {mae:.6f}')
    print(f'Overall RMSE        : {rmse:.6f}')
    print(f'Overall R²          : {r2:.6f}')
    print(f'Direction Accuracy  : {direction_accuracy:.4f}')
    return (results, monthly_results, summary)
