"""
Dynamic Factor Model (DFM) factor extraction for NCDENow preprocessing.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


N_FACTORS = 6
ROLLING_WINDOW = 12


def impute_ragged_edge(
    panel_df: pd.DataFrame,
    rolling_window: int = ROLLING_WINDOW
) -> pd.DataFrame:
    """
    Impute ragged-edge NaNs.
    """
    df = panel_df.copy()
    
    # Compute rolling means on available history, then backfill leading NaNs 
    # with the global column mean.
    rolling_means = df.rolling(window=rolling_window, min_periods=1).mean()
    df = df.fillna(rolling_means)
    df = df.fillna(df.mean())

    if df.isna().any().any():
        warnings.warn(
            "impute_ragged_edge: residual NaNs after imputation. "
            "Check for series that are entirely NaN in this snapshot.",
            UserWarning,
            stacklevel=2
        )
    
    return df


def fit_pca_factors(
    panel_df: pd.DataFrame,
    n_factors: int = N_FACTORS,
    impute: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit PCA on the full training panel and extract latent factors.
    """
    if impute:
        panel_df = impute_ragged_edge(panel_df)
    
    # Standardize for PCA.
    scaler = StandardScaler()
    X = panel_df.values # (T, N)
    X_scaled = scaler.fit_transform(X)

    # Perform PCA.
    pca = PCA(n_components=n_factors, random_state=109)
    factors = pca.fit_transform(X_scaled)
    factors_df = pd.DataFrame(
        factors,
        index=panel_df.index,
        columns=[f'F{i+1}' for i in range(n_factors)]
    )

    # Diagnostic prints.
    evr = pca.explained_variance_ratio_
    cum_evr = np.cumsum(evr)
    print(f"PCA factor extraction: n_factors={n_factors}, "
          f"(T,N)={panel_df.shape}")
    print("  Variance explained per factor: "
          + ' '.join(f"F{i+1}={v:.1%}" for i, v in enumerate(evr)))
    print("  Cumulative:                    "
          + ' '.join(f"F1-{i+1}={v:.1%}" for i, v in enumerate(cum_evr)))
    
    return factors_df, pca.components_, evr, scaler


def extract_factors_from_snapshot(
    snapshot: np.ndarray,
    series_names: list[str],
    nowcast_date: pd.Timestamp,
    fitted_scaler: StandardScaler,
    fitted_loadings: np.ndarray
):
    """
    Project a single real-time snapshot onto pre-fitted PCA factors.
    """
    lookback, n_series = snapshot.shape
    assert n_series == fitted_loadings.shape[1], (
        f"Snapshot has {n_series} series but loadings expect "
        f"{fitted_loadings.shape[1]}."
    )

    # Build a dated DataFrame with ragged-edge imputation.
    obs_start = nowcast_date - pd.DateOffset(months=lookback - 1)
    obs_start = obs_start.replace(day=1)
    date_index = pd.date_range(start=obs_start, periods=lookback, freq='MS')
    snap_df = pd.DataFrame(snapshot, index=date_index, columns=series_names)
    snap_df = impute_ragged_edge(snap_df)

    # Note that we use an already fitted training scaler!
    X_scaled = fitted_scaler.transform(snap_df.values)
    
    # Project onto factors.
    F = X_scaled @ fitted_loadings.T # (lookback, n_factors)

    return F


def build_factor_panel(
    fredmd_transformed: pd.DataFrame,
    n_factors: int = N_FACTORS,
    train_end: str = '2018-12-01',
) -> dict:
    """
    Fit PCA on training data and extract factors for the full panel.
    """
    train_df = fredmd_transformed.loc[:train_end]
    full_df = fredmd_transformed

    # Fit on the training window only.
    factors_train, loadings, evr, scaler = fit_pca_factors(
        train_df,
        n_factors=n_factors,
    )

    # Project the full sample as if it were out-of-sample.
    full_df_imputed = impute_ragged_edge(full_df)
    X_full_scaled = scaler.transform(full_df_imputed.values)
    F_full = X_full_scaled @ loadings.T

    factors_df = pd.DataFrame(
        F_full,
        index=full_df.index,
        columns=[f'F{i+1}' for i in range(n_factors)]
    )

    return {
        'factors_df': factors_df,
        'factors_train': factors_train,
        'factors_test': factors_df.loc[
                            pd.Timestamp(train_end)
                            + pd.DateOffset(months=1)
                        ],
        'loadings': loadings,
        'explained_variance_ratio': evr,
        'scaler': scaler,
        'n_factors': n_factors,
        'train_end': train_end
    }
