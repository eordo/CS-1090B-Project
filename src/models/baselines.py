"""
baselines.py
------------
Ridge and Random Forest models evaluated on an expanding window.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler


# Train for a minimum of 20 quarters (5 years).
DEFAULT_MIN_TRAIN_Q = 20

RIDGE_ALPHAS = np.logspace(-2, 3, 30)
RF_N_ESTIMATORS  = 300
RF_MAX_FEATURES  = 'sqrt'
RF_MIN_SAMPLES_L = 2
RF_RANDOM_STATE  = 109

RANDOM_SEED = 109


def expanding_window_ridge(
    X: np.ndarray,
    y: np.ndarray,
    nowcast_dates: list[pd.Timestamp],
    min_train_quarters: int = DEFAULT_MIN_TRAIN_Q,
    alphas: np.ndarray = RIDGE_ALPHAS,
    cv: int = 5,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Expanding-window Ridge regression with cross-validated alpha selection.

    At each step t:
        1. Fit StandardScaler on X[:t]
        2. Fit RidgeCV on scaled X[:t], y[:t] with leave-one-out or k-fold CV.
        3. Predict X[t] using the training scaler and fitted model.
    """
    T = len(nowcast_dates)
    assert X.shape[0] == T
    assert len(y) == T

    records = []
    for t in range(min_train_quarters, T):
        if np.isnan(y[t]):
            warnings.warn(
                f"Ridge: GDP NaN at t={t} ({nowcast_dates[t].date()}), skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue

        # Training slice. Drop any rows where y is NaN.
        X_train, y_train = X[:t], y[:t]
        valid = ~np.isnan(y_train)
        X_train, y_train = X_train[valid], y_train[valid]

        if len(y_train) < 5:
            warnings.warn(
                f"Ridge: fewer than 5 valid training samples at t={t}, skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X[t:t+1])

        # RidgeCV with leave-one-out.
        model = RidgeCV(alphas=alphas, cv=cv, scoring='neg_mean_squared_error')
        model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)[0]
        y_true = float(y[t])

        if verbose:
            print(
                f"[Ridge t={t:>3}]  {nowcast_dates[t].strftime('%Y-%m')}  "
                f"alpha={model.alpha_:.3f}  "
                f"true={y_true:+.4f}  pred={y_pred:+.4f}"
            )

        records.append({
            'nowcast_date': nowcast_dates[t],
            'y_true': y_true,
            'y_pred': y_pred,
            'alpha_selected': model.alpha_,
        })

    return pd.DataFrame(records)


def expanding_window_rf(
    X: np.ndarray,
    y: np.ndarray,
    nowcast_dates: list[pd.Timestamp],
    min_train_quarters: int = DEFAULT_MIN_TRAIN_Q,
    n_estimators: int       = RF_N_ESTIMATORS,
    max_features: str       = RF_MAX_FEATURES,
    min_samples_leaf: int   = RF_MIN_SAMPLES_L,
    verbose: bool           = False,
) -> pd.DataFrame:
    """
    Expanding-window Random Forest regressor.

    Hyperparameters are fixed (no per-window tuning) to keep runtime 
    manageable.
    """
    T = len(nowcast_dates)
    assert X.shape[0] == T
    assert len(y) == T

    records = []
    for t in range(min_train_quarters, T):
        if np.isnan(y[t]):
            warnings.warn(
                f"RF: GDP NaN at t={t} ({nowcast_dates[t].date()}), skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue

        X_train, y_train = X[:t], y[:t]
        valid = ~np.isnan(y_train)
        X_train, y_train = X_train[valid], y_train[valid]

        if len(y_train) < 5:
            warnings.warn(
                f"RF: fewer than 5 valid training samples at t={t}, skipping.",
                UserWarning, stacklevel=2,
            )
            continue

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X[t:t+1])[0]
        y_true = float(y[t])

        if verbose:
            print(
                f"[RF    t={t:>3}]  {nowcast_dates[t].strftime('%Y-%m')}  "
                f"true={y_true:+.4f}  pred={y_pred:+.4f}"
            )

        records.append({
            'nowcast_date': nowcast_dates[t],
            'y_true': y_true,
            'y_pred': y_pred,
        })

    return pd.DataFrame(records)
