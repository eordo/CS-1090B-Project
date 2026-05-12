import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


CRISIS_WINDOWS = {
    'GFC':   (pd.Timestamp('2008-08-31'), pd.Timestamp('2009-05-31')),
    'COVID': (pd.Timestamp('2020-02-29'), pd.Timestamp('2020-05-31')),
}
# Expansion windows for additional context (optional in table)
EXPANSION_WINDOWS = {
    'pre_GFC':    (pd.Timestamp('2005-02-28'), pd.Timestamp('2007-11-30')),
    'post_GFC':   (pd.Timestamp('2010-02-28'), pd.Timestamp('2019-11-30')),
    'post_COVID': (pd.Timestamp('2021-02-28'), pd.Timestamp('2024-11-30')),
}


def compute_metrics(
    results_df: pd.DataFrame,
    label: str = '',
) -> dict:
    """
    Compute standard nowcasting evaluation metrics for one model.
    """
    df = results_df.dropna(subset=['y_true', 'y_pred'])
    if len(df) == 0:
        warnings.warn(
            f"compute_metrics({label!r}): no valid predictions after dropping NaNs.",
            UserWarning,
            stacklevel=2,
        )
        return {
            'model': label,
            'n': 0,
            'rmse': np.nan,
            'mae': np.nan,
            'mda': np.nan
        }

    y_true = df['y_true'].values
    y_pred = df['y_pred'].values

    rmse = root_mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    mda = float(np.mean(np.sign(y_pred) == np.sign(y_true)))

    return {'model': label, 'n': len(df), 'rmse': rmse, 'mae': mae, 'mda': mda}


def crisis_window(
    results_df: pd.DataFrame,
    window: str,
) -> pd.DataFrame:
    """
    Subset a results DataFrame to a named crisis window.
    """
    windows = {**CRISIS_WINDOWS, **EXPANSION_WINDOWS}
    if window not in windows:
        raise ValueError(
            f"Unknown window {window!r}. "
            f"Available: {list(windows.keys())}"
        )
    start, end = windows[window]
    mask = (
        (results_df['nowcast_date'] >= start) &
        (results_df['nowcast_date'] <= end)
    )
    return results_df.loc[mask].reset_index(drop=True)


def build_comparison_table(
    results_dict: dict,
    include_crisis: bool = True,
) -> pd.DataFrame:
    """
    Build a formatted comparison table across all models.

    Example output:
                      n    RMSE     MAE     MDA  RMSE_GFC  RMSE_COVID
    Ridge            76   0.xxx   0.xxx   0.xxx     0.xxx       0.xxx
    RF               76   0.xxx   0.xxx   0.xxx     0.xxx       0.xxx
    NCDENow (clean)  76   0.xxx   0.xxx   0.xxx     0.xxx       0.xxx
    NCDENow (ragged) 76   0.xxx   0.xxx   0.xxx     0.xxx       0.xxx
    """
    rows = []
    for label, df in results_dict.items():
        m = compute_metrics(df, label=label)
        row = {
            'Model': label,
            'n': m['n'],
            'RMSE': m['rmse'],
            'MAE': m['mae'],
            'MDA': m['mda'],
        }

        if include_crisis:
            for crisis_name in CRISIS_WINDOWS:
                sub = crisis_window(df, crisis_name)
                if len(sub) > 0:
                    cm = compute_metrics(sub)
                    row[f'RMSE_{crisis_name}'] = cm['rmse']
                    row[f'n_{crisis_name}']    = cm['n']
                else:
                    row[f'RMSE_{crisis_name}'] = np.nan
                    row[f'n_{crisis_name}']    = 0

        rows.append(row)

    table = pd.DataFrame(rows).set_index('Model')
    return table
