"""
Fetch and cache FRED-MD and FRED-QD series via the FRED API.
"""

import os
import warnings
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from fredapi import Fred
from .series_catalog import CORE_SERIES, FREDQD_SERIES


# Fetch from this date to ensure differencing doesn't eat the sample start.
PRE_SAMPLE_START = '1995-01-01'

# Minimum rows a cached CSV must have to be considered valid.
_MIN_ROWS = 10

# Default file names.
FREDMD_FILENAME = 'fredmd_raw_core.csv'
FREDQD_FILENAME = 'fredqd_raw.csv'


def get_api_key(
    env_path: str | Path = None,
    for_staff_evaluation: bool =False
):
    """
    Load the FRED API key from a .env file or the environment.

    Looks for the variable FRED_API_KEY. Searches in order:
        1. env_path (if provided)
        2. .env in the current working directory
        3. .env in the parent of the current working directory (handles running
        code from the notebooks/ directory)
        4. Environment variable already set in the shell
    """
    if for_staff_evaluation:
        print("Executing for staff evaluation.\n"
              "Raw data is already saved locally - no API key needed.")
        return
    elif env_path is not None:
        load_dotenv(dotenv_path=Path(env_path))
    else:
        # Try cwd first, then parent directory (handles running code in the 
        # notebooks/ subdirectory).
        for candidate in [Path.cwd() / '.env', Path.cwd().parent / '.env']:
            if candidate.exists():
                load_dotenv(dotenv_path=candidate)
                break

    key = os.getenv('FRED_API_KEY')
    if not key:
        raise EnvironmentError(
            "FRED_API_KEY not found. "
            "Add it to a .env file in the project root:\n"
            "   export FRED_API_KEY=your_key_here\n"
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return key


def load_fredmd(
    data_dir: str | Path,
    core_series: dict = None,
    api_key: str = None,
    force_refresh: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load the FRED-MD core panel (raw, untransformed).

    Loads from what fredapi caches to disk if available and valid; otherwise, 
    fetches from FRED and saves the result. Note that the returned DataFrame 
    uses new names, e.g., 'CLAIMSx' as column names, NOT the raw FRED series 
    IDs, e.g., 'ICSA'. Apply transformation codes separately via 
    transforms.transform_panel().
    """
    if core_series is None:
        core_series = CORE_SERIES

    data_dir = Path(data_dir)
    cache_path = data_dir / FREDMD_FILENAME

    # Load from cache if available.
    if not force_refresh and _is_valid_csv(cache_path, expected_cols=len(core_series)):
        df = pd.read_csv(cache_path, index_col='date', parse_dates=True)
        if verbose:
            print(f"FRED-MD loaded from cache: {cache_path}")
            _summarize('FRED-MD', df)
        return df

    # Else, fetch from API.
    if force_refresh and cache_path.exists():
        cache_path.unlink()
        if verbose:
            print("FRED-MD: cache deleted, re-fetching...")
    elif cache_path.exists():
        if verbose:
            print("FRED-MD: cached file is incomplete/corrupt - re-fetching...")
        cache_path.unlink()

    if api_key is None:
        api_key = get_api_key()
    fred = _make_fred_client(api_key)
    
    if verbose:
        print(f"Fetching {len(core_series)} FRED-MD series from API...")

    frames = {}
    failed = {}
    for name, (fred_id, tcode, desc) in core_series.items():
        try:
            s = fred.get_series(
                    fred_id,
                    observation_start=PRE_SAMPLE_START,
                    frequency='m'
                )
            frames[name] = s
            if verbose:
                print(f"  ✓ {name:<15}  tcode={tcode}  {desc}")
        except Exception as e:
            failed[name] = str(e)
            if verbose:
                print(f"  ✗ {name:<15}  ERROR: {e}")

    if failed:
        warnings.warn(
            f"load_fredmd: {len(failed)} series failed to fetch: "
            f"{list(failed.keys())}. "
            "They will be absent from the returned DataFrame.",
            UserWarning,
            stacklevel=2,
        )

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df.index.name = 'date'

    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)

    if verbose:
        print(f"\nFetched and saved -> {cache_path}")
        _summarize('FRED-MD', df)

    return df


def load_fredqd(
    data_dir: str | Path,
    fredqd_series: dict = None,
    api_key: str = None,
    force_refresh: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load the FRED-QD target series (raw, untransformed).

    The primary target is GDPC1 (real GDP). GDPDEF is fetched as supplementary
    context but should not be used as a model input. Apply transformation codes
    separately via transforms.transform_panel() or apply_tcode() directly.
    """
    if fredqd_series is None:
        fredqd_series = FREDQD_SERIES

    data_dir = Path(data_dir)
    cache_path = data_dir / FREDQD_FILENAME

    # Load from cache, if available.
    if not force_refresh and _is_valid_csv(cache_path, expected_cols=len(fredqd_series)):
        df = pd.read_csv(cache_path, index_col='date', parse_dates=True)
        if verbose:
            print(f"FRED-QD loaded from cache: {cache_path}")
            _summarize('FRED-QD', df)
        return df

    # Else, fetch from API.
    if force_refresh and cache_path.exists():
        cache_path.unlink()
    elif cache_path.exists():
        if verbose:
            print("FRED-QD: cached file is incomplete/corrupt - re-fetching...")
        cache_path.unlink()

    if api_key is None:
        api_key = get_api_key()
    fred = _make_fred_client(api_key)

    if verbose:
        print(f"Fetching {len(fredqd_series)} FRED-QD series from API...")

    frames = {}
    for name, (fred_id, tcode, desc) in fredqd_series.items():
        try:
            s = fred.get_series(
                    fred_id,
                    observation_start=PRE_SAMPLE_START,
                    frequency='q',
                )
            frames[name] = s
            if verbose:
                print(f"  ✓ {name:<15}  tcode={tcode}  {desc}")
        except Exception as e:
            if verbose:
                print(f"  ✗ {name:<15}  ERROR: {e}")

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df.index.name = 'date'

    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)

    if verbose:
        print(f"\nFetched and saved -> {cache_path}")
        _summarize('FRED-QD', df)

    return df


def _is_valid_csv(path: Path, expected_cols: int = None) -> bool:
    """
    Return True only if the CSV exists, is non-empty, and has the expected
    number of columns (if specified).
    """
    if not path.exists():
        return False
    try:
        probe = pd.read_csv(path, index_col=0, nrows=_MIN_ROWS + 1)
        if len(probe) < _MIN_ROWS:
            return False
        if expected_cols is not None and len(probe.columns) < expected_cols:
            return False
        return True
    except Exception:
        return False


def _make_fred_client(api_key) -> Fred:
    """Instantiate a fredapi.Fred client."""
    return Fred(api_key=api_key)


def _summarize(name: str, df: pd.DataFrame) -> None:
    """Print a one-line summary of a loaded DataFrame."""
    if df.empty:
        print(f"  {name}: EMPTY — check file path or API key and re-run")
    else:
        print(
            f"  {name}: shape={df.shape}\t"
            f"{df.index.min().date()} -> {df.index.max().date()}"
        )
