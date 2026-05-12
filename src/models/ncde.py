"""
ncde.py
-------
Adapter and training loop for NCDENow (Lim et al., 2024).

A note on NCDENow: The source lives at src/ncde_now/ as a git submodule.
Initialize it with `git submodule update --init --recursive`.

A note on torchcde: NCDENow uses a subclassed CubicSpline (CustomCubicSpline) 
in a way that is incompatible with torchcde 0.2.x. The project requirements 
match the version that is in pinned in src/ncde_now/environment.yaml.
"""

import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import NamedTuple, Optional
from tqdm.auto import tqdm
from ..data.series_catalog import GDP_TARGET


# Add src/ to sys.path so that `ncde_now` resolves as a package.
# src/ncde_now/ must also be on sys.path so that its internal imports resolve.
_SRC_DIR  = Path(__file__).resolve().parent.parent
_NCDE_DIR = _SRC_DIR / 'ncde_now'

for _p in [str(_SRC_DIR), str(_NCDE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# See the note about torchcde.
try:
    import torchcde
except ImportError as e:
    raise ImportError(
        "torchcde is required. Install with pip and pin the version from "
        "src/ncde_now/environment.yaml."
    ) from e


# Architecture parameters.
DEFAULT_FE_TYPE            = 'FactorAnalysisEncoder'
DEFAULT_CDE_TYPE           = 'MLPCDEFunc'
DEFAULT_HIDDEN_SIZE        = 32
DEFAULT_HIDDEN_HIDDEN_SIZE = 64
DEFAULT_OUTPUT_SIZE        = 1
DEFAULT_N_FACTORS          = 6  # Equal to N_FACTORS in `dfm.py`
DEFAULT_N_LAYERS           = 2
DEFAULT_ODE_METHOD         = 'rk4'

# Training parameters.
DEFAULT_LR              = 1e-3
DEFAULT_EPOCHS          = 100
DEFAULT_BATCH_SIZE      = 16
DEFAULT_GRAD_CLIP       = 1.0
DEFAULT_WEIGHT_DECAY    = 1e-4
DEFAULT_MIN_TRAIN_Q     = 20    # Minimum 5 years for training

RANDOM_SEED = 109


class NCDEResult(NamedTuple):
    """
    Return value of expanding_window_ncde().
    """
    predictions: pd.DataFrame
    history:     pd.DataFrame


def align_gdp_to_nowcast_dates(
    fredqd_transformed: pd.DataFrame,
    nowcast_dates: list[pd.Timestamp],
    gdp_col: str = GDP_TARGET,
) -> np.ndarray:
    """
    Align quarterly GDP growth to the nowcasting date sequence.

    Nowcast dates are end-of-month-2 (Feb/May/Aug/Nov), but the target 'GDPC1' 
    is indexed at quarter-start (Jan/Apr/Jul/Oct). This function maps each 
    nowcast date to its corresponding quarter-start date in the FRED-QD index.
    """
    _M2_TO_QS = {2: 1, 5: 4, 8: 7, 11: 10}
    gdp = fredqd_transformed[gdp_col]
    aligned = []
    for d in nowcast_dates:
        qs_month = _M2_TO_QS.get(d.month)
        if qs_month is None:
            raise ValueError(
                f"Unexpected nowcast month {d.month} ({d.date()}). "
                "generate_nowcast_dates() should only produce Feb/May/Aug/Nov."
            )
        qs = pd.Timestamp(year=d.year, month=qs_month, day=1)
        aligned.append(gdp.get(qs, np.nan))

    return np.array(aligned, dtype=np.float64)


def build_ncde_model(
    n_series: int,
    n_factors: int           = DEFAULT_N_FACTORS,
    hidden_size: int         = DEFAULT_HIDDEN_SIZE,
    hidden_hidden_size: int  = DEFAULT_HIDDEN_HIDDEN_SIZE,
    output_size: int         = DEFAULT_OUTPUT_SIZE,
    n_layers: int            = DEFAULT_N_LAYERS,
    fe_type: str             = DEFAULT_FE_TYPE,
    cde_type: str            = DEFAULT_CDE_TYPE,
    ode_method: str          = DEFAULT_ODE_METHOD,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Instantiate the NCDENow model.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NCDENow = _import_ncde()
    return NCDENow(
        fe_type=fe_type,
        cde_type=cde_type,
        input_size=n_series + 1, # +1 for prepended time channel
        hidden_size=hidden_size,
        hidden_hidden_size=hidden_hidden_size,
        output_size=output_size,
        n_factors=n_factors,
        n_layers=n_layers,
        ode_method=ode_method,
        device=device,
    ).to(device)


def expanding_window_ncde(
    panels: np.ndarray,
    factors: np.ndarray,
    gdp_growth: np.ndarray,
    nowcast_dates: list[pd.Timestamp],
    n_factors: int           = DEFAULT_N_FACTORS,
    hidden_size: int         = DEFAULT_HIDDEN_SIZE,
    hidden_hidden_size: int  = DEFAULT_HIDDEN_HIDDEN_SIZE,
    n_layers: int            = DEFAULT_N_LAYERS,
    fe_type: str             = DEFAULT_FE_TYPE,
    cde_type: str            = DEFAULT_CDE_TYPE,
    ode_method: str          = DEFAULT_ODE_METHOD,
    min_train_quarters: int  = DEFAULT_MIN_TRAIN_Q,
    lr: float                = DEFAULT_LR,
    epochs: int              = DEFAULT_EPOCHS,
    batch_size: int          = DEFAULT_BATCH_SIZE,
    grad_clip: float         = DEFAULT_GRAD_CLIP,
    weight_decay: float      = DEFAULT_WEIGHT_DECAY,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_label: str    = 'ncde',
    device: Optional[torch.device] = None,
    verbose: bool            = True,
) -> pd.DataFrame:
    """
    Expanding-window pseudo-real-time evaluation of NCDENow.

    At step t, train on quarters [0, t-1] and predict quarter t. The model is 
    re-initialized from scratch at each step (no leakage). Evaluation begins 
    at min_train_quarters.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    T, L, N = panels.shape
    assert factors.shape == (T, n_factors), (
        f"factors shape {factors.shape} does not match (T={T}, n_factors={n_factors})"
    )
    assert len(gdp_growth) == T
    assert len(nowcast_dates) == T

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Prediction rows and per-epoch loss rows.
    records = []
    history_rows = []
    for t in range(min_train_quarters, T):
        if np.isnan(gdp_growth[t]):
            warnings.warn(
                f"GDP NaN at t={t} ({nowcast_dates[t].date()}), skipping.",
                UserWarning,
                stacklevel=2
            )
            continue

        qtr = _date_to_quarter_str(nowcast_dates[t])

        # Prepare training data.
        X_train = _prepend_time_channel(panels[:t]) # (t, L, N+1)
        F_train = factors[:t].astype(np.float32)    # (t, n_factors)
        y_train = gdp_growth[:t].astype(np.float32) # (t,)

        path_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        # Pre-compute spline coefficients once per window, not per epoch.
        train_coeffs  = torchcde.natural_cubic_spline_coeffs(path_t)
        train_factors = torch.tensor(F_train, dtype=torch.float32, device=device)
        train_y       = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)

        # Initialize model.
        model = build_ncde_model(
            n_series=N,
            n_factors=n_factors,
            hidden_size=hidden_size,
            hidden_hidden_size=hidden_hidden_size,
            n_layers=n_layers,
            fe_type=fe_type,
            cde_type=cde_type,
            ode_method=ode_method,
            device=device,
        )
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        criterion = nn.MSELoss()

        # Training loop.
        model.train()
        n_train = t
        final_loss = np.nan
        for epoch in tqdm(range(1, epochs + 1), desc='Training', leave=False):
            perm       = torch.randperm(n_train, device=device)
            epoch_loss = 0.0
            n_batches  = 0
            for start in range(0, n_train, batch_size):
                idx = perm[start:start+batch_size]
                if len(idx) == 0:
                    continue

                b_x      = path_t[idx]
                b_coeffs = train_coeffs[idx]
                b_f      = train_factors[idx]
                b_y      = train_y[idx]

                optimizer.zero_grad()
                pred_y, _, _ = model(b_x, b_coeffs, b_f)
                loss = criterion(pred_y, b_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1
            
            avg_loss   = epoch_loss / max(n_batches, 1)
            final_loss = avg_loss

            # Record epoch history.
            history_rows.append({
                'step': t,
                'nowcast_date': nowcast_dates[t],
                'epoch': epoch + 1,
                'train_loss': avg_loss
            })
            if verbose and (epoch + 1) % 25 == 0:
                print(f"  [t={t:>3}]  epoch {epoch+1:>3}/{epochs}  loss={avg_loss:.5f}")

        # Out-of-sample prediction.
        model.eval()
        with torch.no_grad():
            x_snap = _prepend_time_channel(panels[t])[None]  # (1, L, N+1)
            x_t    = torch.tensor(x_snap, dtype=torch.float32, device=device)
            c_snap = torchcde.natural_cubic_spline_coeffs(x_t)
            f_snap = torch.tensor(
                factors[t][None].astype(np.float32),
                dtype=torch.float32,
                device=device
            )
            pred_y, _, _ = model(x_t, c_snap, f_snap)
            y_pred = pred_y.item()

        y_true = float(gdp_growth[t])

        if verbose:
            print(
                f"[t={t:>3}]  {qtr}  "
                f"true={y_true:+.4f}  pred={y_pred:+.4f}  "
                f"|err|={abs(y_pred - y_true):.4f}"
            )

        records.append({
            'nowcast_date': nowcast_dates[t],
            'y_true': y_true,
            'y_pred': y_pred,
        })

        # Save checkpoint.
        if checkpoint_dir is not None:
            ckpt_path = checkpoint_dir / f'{checkpoint_label}_step{t:03d}_{qtr}.pt'
            torch.save({
                'step': t,
                'nowcast_date': nowcast_dates[t].isoformat(),
                'quarter': qtr,
                'y_true': y_true,
                'y_pred': y_pred,
                'final_train_loss': final_loss,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'arch': {
                    'n_series': N,
                    'n_factors': n_factors,
                    'hidden_size': hidden_size,
                    'hidden_hidden_size': hidden_hidden_size,
                    'n_layers': n_layers,
                    'fe_type': fe_type,
                    'cde_type': cde_type,
                    'ode_method': ode_method
                }
            }, ckpt_path)

    predictions = pd.DataFrame(records)
    history     = pd.DataFrame(history_rows)

    return NCDEResult(predictions=predictions, history=history)


def load_checkpoint(ckpt_path: str | Path, device: torch.device = None) -> dict:
    """
    Load a saved checkpoint and reconstruct the model.
    """
    if device is None:
        device = torch.device('cpu')
    ckpt = torch.load(ckpt_path, map_locatoin=device)
    arch = ckpt['arch']
    model = build_ncde_model(
        n_series=arch['n_series'],
        n_factors=arch['n_factors'],
        hidden_size=arch['hidden_size'],
        hidden_hidden_size=arch['hidden_hidden_size'],
        n_layers=arch['n_layers'],
        fe_type=arch['fe_type'],
        cde_type=arch['cde_type'],
        ode_method=arch['ode_method'],
        device=device
    )
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return {
        'model': model,
        'step': ckpt['step'],
        'nowcast_date': ckpt['nowcast_date'],
        'quarter': ckpt['quarter'],
        'y_true': ckpt['y_true'],
        'y_pred': ckpt['y_pred'],
        'final_train_loss': ckpt['final_train_loss']
    }


def _check_torchcde_compat() -> None:
    """
    Verify that the installed torchcde exposes the private CubicSpline
    attributes that NCDENow's CustomCubicSpline subclass depends on.
    """
    has_old_attrs = all(
        hasattr(torchcde.interpolation_cubic.CubicSpline, a)
        for a in ('_b', '_two_c', '_three_d')
    )
    has_public_deriv = hasattr(
        torchcde.interpolation_cubic.CubicSpline, 'derivative'
    )
    if has_old_attrs:
        print(f"torchcde {torchcde.__version__}: no patch needed.")
    elif has_public_deriv:
        print(f"torchcde {torchcde.__version__}: public derivative() detected, "
              "patch will be applied on first model import.")
    else:
        raise RuntimeError(
            f"torchcde {torchcde.__version__} is not compatible with NCDENow. "
            "CubicSpline has neither the old private coefficient attributes "
            "nor a public derivative() method. Install torchcde 0.2.5."
        )


def _patch_custom_cubic_spline() -> None:
    from sub_module.neural_cde import CustomCubicSpline
 
    # Already patched? Nothing to do.
    if getattr(CustomCubicSpline, '_patched_for_torchcde_025', False):
        return
 
    # Old torchcde versions have the private attrs. No patch needed.
    _OLD_ATTRS = ('_b', '_two_c', '_three_d')
    if all(hasattr(torchcde.interpolation_cubic.CubicSpline, a) for a in _OLD_ATTRS):
        return
 
    # Confirm the public derivative() exists to replace them.
    if not hasattr(torchcde.interpolation_cubic.CubicSpline, 'derivative'):
        raise RuntimeError(
            f"torchcde {torchcde.__version__} CubicSpline has neither the old "
            "private coefficient attributes nor a public derivative() method. "
            "This torchcde version is not supported."
        )
 
    def _derivative_025(self, t):
        """
        Delegate to the torchcde 0.2.5 public derivative() and return as
        (deriv, deriv) to match the (alpha, beta) hidden state tuple that
        cdeint expects from the NeuralCDE vector field.
        """
        deriv = super(CustomCubicSpline, self).derivative(t)
        return deriv, deriv
 
    CustomCubicSpline.derivative = _derivative_025
    CustomCubicSpline._patched_for_torchcde_025 = True


def _date_to_quarter_str(dt: pd.Timestamp) -> str:
    _M2Q = {2: 'Q1', 5: 'Q2', 8: 'Q3', 11: 'Q4'}
    return f"{dt.year}-{_M2Q.get(dt.month, f'M{dt.month}')}"


def _import_ncde():
    """Lazily import NCDENow. Raises an error if the submodule is absent."""
    try:
        from ncde_now.model import NCDENow
    except ImportError:
        raise ImportError("Cannot import NCDENow from src/ncde_now/model.py.")
    _patch_custom_cubic_spline()
    return NCDENow


def _prepend_time_channel(panel: np.ndarray) -> np.ndarray:
    """
    Prepend a normalized [0, 1] time channel to a (L, N) panel.
    """
    if panel.ndim == 2:
        L, N = panel.shape
        t = np.linspace(0, 1, L, dtype=np.float32)
        return np.concatenate([t[:,None], panel.astype(np.float32)], axis=1)
    elif panel.ndim == 3:
        B, L, N = panel.shape
        t = np.linspace(0, 1, L, dtype=np.float32)
        t_tiled = np.tile(t[None,:,None], (B, 1, 1))
        return np.concatenate([t_tiled, panel.astype(np.float32)], axis=2)
    else:
        raise ValueError(f"panel must be 2D or 3D, got shape {panel.shape}")
