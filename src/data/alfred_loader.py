"""
Build a pseudo-real-time ragged-edge dataset from ALFRED vintage data.

## Background

ALFRED (Archival FRED) stores every historical vintage of each series, i.e.,
what each series looked like on a specific real-time date before subsequent
revisions. The ragged-edge structure arises because different series are
released on different schedules within a quarter: for example, payrolls appear 
~4 days after month-end while PCE may lag by ~4 weeks.

This module constructs a real-time snapshot dataset. For each nowcasting
reference date, we pull the vintage that was actually available to a 
forecaster on that date, leaving NaN where the series had not yet been
released. This dataset is the key input for the NCDENow ragged-edge and is the 
treatment variable in the research question.

## Design

Our nowcasting convention is that we evaluate at the END OF MONTH 2 of each 
quarter, e.g., for 2005 Q1, the nowcasting date is 2005-02-28. This is the 
most information-rich mid-quarter point: month-1 data for fast-releasing series
has arrived, and month-2 data is starting to trickle in. This matches the 
"nowcast at month 2" convention used in Giannone et al. (2008).

For each nowcasting date t_now and each series i:
    1. Fetch all vintage dates for series i.
    2. Find the most recent vintage that is <= t_now, i.e., actually available
       to a forecaster on that date.
    3. From that vintage, extract the last LOOKBACK_MONTHS of observations
       up to and including the current quarter's month-2 reference period.
    4. If no vintage is available by t_now, the series is NaN for this date
       (ragged edge).

The output is a list of snapshots, one per nowcasting date, each a
(LOOKBACK_MONTHS, n_series) array. These snapshots are then consumed by
the DFM factor extractor (dfm.py).
"""

import pickle
import time
import warnings
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from .series_catalog import CORE_SERIES


ALFRED_BASE_URL  = 'https://api.stlouisfed.org/fred'
REQUEST_TIMEOUT  = 20   # Seconds per API call
RETRY_ATTEMPTS   = 3    # Retries on transient HTTP errors
RETRY_DELAY      = 2.0  # Seconds between retries
INTER_CALL_DELAY = 0.15 # Seconds between API calls (rate-limit courtesy)

# Lookback window, or how many months of history to include in each snapshot.
# By default, one year of monthly observations per series per snapshot.
DEFAULT_LOOKBACK_MONTHS = 12

# Pickle the dataset because this is a time-consuming process!
RAGGED_EDGE_FILENAME = 'alfred_ragged_edge.pkl'


def fetch_vintage_dates(fred_id: str, api_key: str) -> list[pd.Timestamp]:
    """
    Fetch all available vintage dates for a FRED series from ALFRED.
    """
    url = (
        f"{ALFRED_BASE_URL}/series/vintagedates"
        f"?series_id={fred_id}&api_key={api_key}&file_type=json"
    )

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            dates_raw = resp.json().get('vintage_dates', [])
            return sorted(pd.to_datetime(dates_raw))
        except requests.exceptions.HTTPError as e:
            # 429 is a rate limit error, 5xx are server errors. 
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise RuntimeError(
                    f"fetch_vintage_dates failed for {fred_id} "
                    f"after {RETRY_ATTEMPTS} attempts: {e}"
                ) from e
        except Exception as e:
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(
                    f"fetch_vintage_dates failed for {fred_id}: {e}"
                ) from e

    # Just to satisfy type checkers.
    return []


def fetch_obs_at_vintage(
    fred_id: str,
    vintage_date: str | pd.Timestamp,
    obs_start: str,
    obs_end: str,
    api_key: str,
) -> pd.DataFrame | None:
    """
    Fetch observations for fred_id as they existed on a specific vintage date.

    Uses the ALFRED realtime_start=realtime_end=vintage_date query pattern,
    which returns the state of the series on that exact date.
    """
    if isinstance(vintage_date, pd.Timestamp):
        vintage_date = vintage_date.strftime('%Y-%m-%d')

    url = (
        f"{ALFRED_BASE_URL}/series/observations"
        f"?series_id={fred_id}&api_key={api_key}&file_type=json"
        f"&realtime_start={vintage_date}&realtime_end={vintage_date}"
        f"&observation_start={obs_start}&observation_end={obs_end}"
    )

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            obs = resp.json().get('observations', [])
            if not obs:
                return None
            df = pd.DataFrame(obs)
            df['date']  = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            return (df[['date', 'value']]
                    .dropna(subset=['value'])
                    .reset_index(drop=True))
        except requests.exceptions.HTTPError as e:
            # 429 is a rate limit error, 5xx are server errors. 
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY * attempt)
            else:
                warnings.warn(
                    f"fetch_obs_at_vintage failed for {fred_id} @ "
                    f"{vintage_date}: {e}",
                    UserWarning, stacklevel=2,
                )
                return None
        except Exception as e:
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY)
            else:
                warnings.warn(
                    f"fetch_obs_at_vintage failed for {fred_id} @ "
                    f"{vintage_date}: {e}",
                    UserWarning, stacklevel=2,
                )
                return None

    # Just to satisfy type checkers.
    return None


def generate_nowcast_dates(
    sample_start: str = '2000-01-01',
    sample_end:   str = '2024-12-01',
) -> list[pd.Timestamp]:
    """
    Generate the sequence of nowcasting reference dates.

    Our convention is to use END OF MONTH 2 of each quarter. Specifically, 
    that is the last calendar day of February, May, August, and November.
    These are the months when the most informative mid-quarter balance of
    fast-releasing (labor, financial) and slower (PCE, sales) data is
    available for a real-time forecaster.
    """
    # Month-2 of each quarter: Feb (Q1), May (Q2), Aug (Q3), Nov (Q4).
    nowcast_months = {2, 5, 8, 11}
    all_month_ends = pd.date_range(
        start=sample_start,
        end=sample_end,
        freq='ME'
    )
    return [d for d in all_month_ends if d.month in nowcast_months]


def build_ragged_edge_dataset(
    core_series: dict = None,
    nowcast_dates: list[pd.Timestamp] = None,
    lookback_months: int = DEFAULT_LOOKBACK_MONTHS,
    data_dir: str | Path = None,
    api_key: str = None,
    sample_start: str = '2000-01-01',
    sample_end: str = '2024-12-01',
    verbose: bool = True,
) -> dict:
    """
    Build the ragged-edge snapshot dataset from ALFRED vintage data.

    For each nowcasting date in nowcast_dates, and for each series in
    core_series, this function:
      1. Retrieves all vintage dates for the series.
      2. Finds the most recent vintage available on or before the nowcasting
         date (the "real-time available" vintage).
      3. Pulls LOOKBACK_MONTHS of observations from that vintage.
      4. Stores them in a (lookback_months, n_series) snapshot array.
         Missing (not-yet-released) series entries are NaN.

    This is the structural ragged-edge: different series have different NaN
    patterns at each nowcasting date, reflecting real publication timing.
    """
    if core_series is None:
        core_series = CORE_SERIES
    if nowcast_dates is None:
        nowcast_dates = generate_nowcast_dates(sample_start, sample_end)

    series_names = list(core_series.keys())
    n_series     = len(series_names)
    n_dates      = len(nowcast_dates)

    # Pre-fetch all vintage date lists to avoid redundant API calls. Each 
    # series needs its vintage list looked up once.
    if verbose:
        print(f"Pre-fetching vintage date lists for {n_series} series...")

    vintage_map = {}
    for name, (fred_id, _, desc) in core_series.items():
        try:
            vintage_dates = fetch_vintage_dates(fred_id, api_key)
            vintage_map[name] = vintage_dates
            if verbose:
                print(f"  ✓ {name:<22}  {len(vintage_dates):>5} vintages")
        except Exception as e:
            vintage_map[name] = []
            warnings.warn(
                f"Could not fetch vintage dates for {name} ({fred_id}): {e}. "
                "This series will be NaN in all snapshots.",
                UserWarning, stacklevel=2,
            )
        time.sleep(INTER_CALL_DELAY)

    # Allocate output array. NaN means not yet available at that snapshot.
    snapshots = np.full((n_dates, lookback_months, n_series), np.nan)

    if verbose:
        print(
            f"\nBuilding ragged-edge snapshots for ({n_dates} nowcast dates, "
            f"{n_series} series, {lookback_months} months) lookback...\n"
            f"{'Date':<14} {'Available':>10}  {'NaN (not released)':>20}"
        )
        print('-' * 50)

    for t_idx, t_now in enumerate(nowcast_dates):
        # Observation window is length `lookback_months`, ending at `t_now`.
        obs_end   = t_now
        obs_start = t_now - pd.DateOffset(months=lookback_months - 1)
        obs_start = obs_start.replace(day=1) # First of the month
        obs_end_str   = obs_end.strftime('%Y-%m-%d')
        obs_start_str = obs_start.strftime('%Y-%m-%d')

        n_available = 0
        n_missing   = 0
        for s_idx, name in enumerate(series_names):
            fred_id = core_series[name][0]
            vintage_dates = vintage_map[name]
            if not vintage_dates:
                n_missing += 1
                continue

            # Find the most recent vintage available on or before `t_now`.
            vdates_series = pd.Series(vintage_dates)
            available = vdates_series[vdates_series <= t_now]
            if available.empty:
                # Series not yet created or released by this nowcast date.
                n_missing += 1
                continue

            # Fetch observations from the most recently available vintage.
            best_vintage = available.iloc[-1]
            df_obs = fetch_obs_at_vintage(
                fred_id,
                best_vintage,
                obs_start_str,
                obs_end_str,
                api_key,
            )
            time.sleep(INTER_CALL_DELAY)
            if df_obs is None or df_obs.empty:
                n_missing += 1
                continue

            # Align to the expected monthly date index. Build a reference 
            # index of month-start dates for the window.
            ref_index = pd.date_range(
                start=obs_start, end=obs_end, freq='MS'
            )[:lookback_months]

            df_obs = df_obs.set_index('date').reindex(ref_index)
            values = df_obs['value'].values

            # Store in a snapshot. If there is a length mismatch, pad/trim to 
            # length `lookback_months`.
            n_vals = min(len(values), lookback_months)
            snapshots[t_idx,:n_vals,s_idx] = values[:n_vals]
            n_available += 1

        if verbose:
            print(
                f"  {t_now.strftime('%Y-%m-%d')}   "
                f"{n_available:>10}  {n_missing:>20}"
            )

    result = {
        'snapshots': snapshots,             # (T, lookback_months, N)
        'nowcast_dates': nowcast_dates,     # (T,)
        'series_names': series_names,       # (N,)
        'lookback_months': lookback_months,
    }

    # Save to disk.
    if data_dir is not None:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        cache_path = data_dir / RAGGED_EDGE_FILENAME
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        if verbose:
            print(f"\nRagged-edge dataset saved -> {cache_path}")
            print(f"  Array shape: {snapshots.shape}  "
                  f"(T={n_dates}, L={lookback_months}, N={n_series})")

    return result


def load_or_build_ragged_edge(
    data_dir: str | Path,
    api_key: str,
    core_series: dict = None,
    nowcast_dates: list[pd.Timestamp] = None,
    lookback_months: int = DEFAULT_LOOKBACK_MONTHS,
    sample_start: str = '2000-01-01',
    sample_end: str = '2024-12-01',
    force_rebuild: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Load the ragged-edge dataset from cache, or build it if not present.
    """
    data_dir = Path(data_dir)
    cache_path = data_dir / RAGGED_EDGE_FILENAME

    if not force_rebuild and cache_path.exists():
        if verbose:
            print(f"Loading ragged-edge dataset from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            result = pickle.load(f)
        if verbose:
            shape = result['snapshots'].shape
            print(f"  Loaded: shape={shape}  "
                  f"({shape[0]} dates, {shape[1]} months, {shape[2]} series)")
        return result

    if verbose:
        print("Building ragged-edge dataset from ALFRED API...")
        print("  ⚠ This will take 20-40 minutes. Do not interrupt.")
        print("  The result will be cached to avoid repeating this step.\n")

    return build_ragged_edge_dataset(
        core_series=core_series,
        nowcast_dates=nowcast_dates,
        lookback_months=lookback_months,
        data_dir=data_dir,
        api_key=api_key,
        sample_start=sample_start,
        sample_end=sample_end,
        verbose=verbose,
    )
