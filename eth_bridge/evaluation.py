"""Reusable evaluation helpers from the research notebook."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    actual_direction = (y_true > 0).astype(int)
    predicted_direction = (y_pred > 0).astype(int)
    direction_accuracy = (actual_direction.values == predicted_direction.values).mean()
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'direction_accuracy': direction_accuracy}
