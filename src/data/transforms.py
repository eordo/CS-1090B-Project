"""
FRED-MD transformation codes and stationarity utilities.

Transformation codes follow McCracken & Ng (2016), Table 1. All functions 
operate on pandas Series or DataFrames and preserve the index.
"""

import warnings
import numpy as np
import pandas as pd
from .series_catalog import CORE_SERIES


def apply_tcode(series: pd.Series, tcode: int) -> pd.Series:
    """
    Apply a stationarity transformation to a single series according to the 
    McCracken & Ng (2016) transformation code.
    """
    match tcode:
        case 1:
            return series
        case 2:
            return series.diff()
        case 3:
            return series.diff().diff()
        case 4:
            return np.log(series.clip(lower=1e-8))
        case 5:
            return np.log(series.clip(lower=1e-8)).diff()
        case 6:
            return np.log(series.clip(lower=1e-8)).diff().diff()
        case 7:
            return series.pct_change()
        case _:
            raise ValueError(
                f"Unknown tcode {tcode!r}. Valid codes are 1 through 7."
            )


def transform_panel(
    raw_df: pd.DataFrame,
    core_series: dict = None,
) -> pd.DataFrame:
    """
    Apply FRED-MD transformation codes to all columns in raw_df.

    Only columns present in both raw_df and core_series are transformed. Any 
    unrecognized columns are dropped with a warning. This makes the function 
    safe to call on subsets of the full panel.
    """
    if core_series is None:
        core_series = CORE_SERIES

    transformed = {}
    skipped = []

    for col in raw_df.columns:
        if col not in core_series:
            skipped.append(col)
            continue
        _, tcode, _ = core_series[col]
        transformed[col] = apply_tcode(raw_df[col], tcode)

    if skipped:
        warnings.warn(
            f"transform_panel: {len(skipped)} column(s) not found in "
            f"core_series and were dropped: {skipped}",
            UserWarning,
            stacklevel=2,
        )

    return pd.DataFrame(transformed, index=raw_df.index)


def trim_sample(
    df: pd.DataFrame,
    start: str = '2000-01-01',
    end: str = '2024-12-01',
) -> pd.DataFrame:
    """
    Trim a transformed panel to the sample window.

    Always call this *after* transform_panel so that differencing-induced
    leading NaNs at the raw-data boundary are excluded before trimming.
    """
    return df.loc[start:end]


def check_mid_sample_nans(
    df: pd.DataFrame,
    skip_leading: int = 5,
    raise_on_fail: bool = False,
) -> pd.Series:
    """
    Check for unexpected NaN values in the interior of the transformed panel.

    Leading NaNs from differencing (rows 0 to skip_leading-1) are ignored.
    Returns a Series of NaN counts per column; non-zero entries warrant
    investigation (e.g., data gaps in source, series discontinued mid-sample).
    """
    mid = df.iloc[skip_leading:]
    counts = mid.isna().sum()
    problem = counts[counts > 0]

    if len(problem) == 0:
        print("✓ No unexpected mid-sample NaNs. Panel looks clean.")
    else:
        msg = (
            f"⚠ Mid-sample NaNs detected in {len(problem)} series:\n"
            + problem.to_string()
        )
        if raise_on_fail:
            raise ValueError(msg)
        else:
            warnings.warn(msg, UserWarning, stacklevel=2)

    return counts
