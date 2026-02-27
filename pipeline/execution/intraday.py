"""
execution/intraday.py

Intraday execution models for SAFE trades:
- VWAP
- TWAP
- POV (participation of volume)
- Fill probability (ADV / order size constraint)

All functions are deterministic and designed to plug into your existing
SAFE → execution flow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal, Optional


ExecutionModel = Literal["VWAP", "TWAP", "POV"]


# ----------------------------------------------------------------------
# Synthetic intraday curves
# ----------------------------------------------------------------------

def synthetic_volume_curve(n: int = 10) -> np.ndarray:
    """
    Return a simple U-shaped intraday volume curve that sums to 1.0.
    """
    # You can tweak this shape later if you like
    u = np.array([0.06, 0.05, 0.04, 0.03, 0.02,
                  0.02, 0.03, 0.04, 0.05, 0.06], dtype=float)
    if n != len(u):
        # Resample to n buckets if needed
        x_old = np.linspace(0, 1, len(u))
        x_new = np.linspace(0, 1, n)
        u = np.interp(x_new, x_old, u)
    u = np.clip(u, 1e-6, None)
    return u / u.sum()


def synthetic_price_curve(
    mid_price: float,
    vol_daily: float,
    n: int = 10,
    drift_bps: float = 0.0,
    noise_scale: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a synthetic intraday price path around a mid price.

    Parameters
    ----------
    mid_price : float
        Reference price (e.g., today's close or mid).
    vol_daily : float
        Daily volatility proxy (e.g., vol_20d).
    n : int
        Number of intraday buckets.
    drift_bps : float
        Total drift over the day in basis points.
    noise_scale : float
        Scale of intraday noise relative to vol_daily.
    rng : np.random.Generator, optional
        Optional RNG for reproducibility.

    Returns
    -------
    np.ndarray
        Length-n array of synthetic intraday prices.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Linear drift in price over the day
    drift = (drift_bps / 1e4) * mid_price
    start = mid_price - 0.5 * drift
    end = mid_price + 0.5 * drift
    base = np.linspace(start, end, n)

    # Noise scaled by volatility
    noise_std = vol_daily * noise_scale * mid_price
    noise = rng.normal(0.0, noise_std, size=n)

    prices = base + noise
    return np.maximum(prices, 1e-6)


# ----------------------------------------------------------------------
# Core helpers
# ----------------------------------------------------------------------

def _compute_order_size_shares(row: pd.Series) -> float:
    """
    Compute order size in shares from portfolio_notional, weight, and price.
    """
    notional = float(row["portfolio_notional"]) * float(row["weight"])
    side = float(row["side"])  # ±1
    price = float(row["price"])
    if price <= 0:
        return 0.0
    shares = (notional * side) / price
    return shares


def _apply_adv_cap(order_shares: float, adv_shares: float, adv_cap: float) -> float:
    """
    Cap order size by a fraction of ADV.
    """
    max_shares = adv_cap * adv_shares
    if max_shares <= 0:
        return 0.0
    # Preserve sign of order_shares
    sign = np.sign(order_shares) if order_shares != 0 else 1.0
    return float(sign * min(abs(order_shares), max_shares))


def _fill_fraction(order_shares: float, executed_shares: float) -> float:
    """
    Compute fill fraction, safe for zero order size.
    """
    if abs(order_shares) < 1e-9:
        return 0.0
    return float(np.clip(executed_shares / order_shares, 0.0, 1.0))


def _slippage_bps(effective_price: float, ref_price: float, side: float) -> float:
    """
    Compute signed slippage in basis points relative to a reference price.

    For buys (side=+1), positive slippage means worse (higher) price.
    For sells (side=-1), positive slippage means worse (lower) price.
    """
    if ref_price <= 0:
        return 0.0
    # For buys: (eff - ref) / ref; for sells: (ref - eff) / ref
    raw = (effective_price - ref_price) / ref_price
    signed = raw * side
    return float(signed * 1e4)


# ----------------------------------------------------------------------
# Execution models
# ----------------------------------------------------------------------

def exec_vwap(
    row: pd.Series,
    vol_curve: np.ndarray,
    prices: np.ndarray,
    adv_cap: float = 0.1,
) -> dict:
    """
    VWAP execution: slice order proportional to intraday volume curve.

    Parameters
    ----------
    row : pd.Series
        Single SAFE trade row with fields:
        - side, weight, price, vol_daily, adv_shares, portfolio_notional
    vol_curve : np.ndarray
        Intraday volume curve (sums to 1).
    prices : np.ndarray
        Intraday prices aligned with vol_curve.
    adv_cap : float
        Max fraction of ADV allowed for the order.

    Returns
    -------
    dict
        Execution summary for this trade.
    """
    side = float(row["side"])
    mid_price = float(row["price"])
    adv_shares = float(row["adv_shares"])

    order_shares_raw = _compute_order_size_shares(row)
    order_shares = _apply_adv_cap(order_shares_raw, adv_shares, adv_cap)

    # Child orders proportional to volume
    child_shares = order_shares * vol_curve
    executed_shares = child_shares.sum()

    # Effective price is volume-weighted by child_shares
    if abs(executed_shares) < 1e-9:
        effective_price = mid_price
    else:
        effective_price = float(np.sum(prices * child_shares) / executed_shares)

    fill_frac = _fill_fraction(order_shares_raw, executed_shares)
    slippage = _slippage_bps(effective_price, mid_price, side)

    executed_notional = executed_shares * mid_price  # PnL uses mid or eff; you choose

    return {
        "symbol": row["symbol"],
        "side": side,
        "weight": row["weight"],
        "price": mid_price,
        "order_size_shares": order_shares_raw,
        "order_size_shares_capped": order_shares,
        "executed_shares": executed_shares,
        "executed_notional": executed_notional,
        "effective_price": effective_price,
        "slippage_bps": slippage,
        "fill_fraction": fill_frac,
        "adv_shares": adv_shares,
        "vol_daily": row["vol_daily"],
        "model": "VWAP",
    }


def exec_twap(
    row: pd.Series,
    prices: np.ndarray,
    n_slices: int,
    adv_cap: float = 0.1,
) -> dict:
    """
    TWAP execution: slice order evenly across time buckets.

    Parameters
    ----------
    row : pd.Series
        Single SAFE trade row.
    prices : np.ndarray
        Intraday prices.
    n_slices : int
        Number of equal time slices.
    adv_cap : float
        Max fraction of ADV allowed for the order.

    Returns
    -------
    dict
        Execution summary for this trade.
    """
    side = float(row["side"])
    mid_price = float(row["price"])
    adv_shares = float(row["adv_shares"])

    order_shares_raw = _compute_order_size_shares(row)
    order_shares = _apply_adv_cap(order_shares_raw, adv_shares, adv_cap)

    if n_slices <= 0:
        n_slices = len(prices)
    child_shares = np.full(n_slices, order_shares / n_slices, dtype=float)

    executed_shares = child_shares.sum()

    if abs(executed_shares) < 1e-9:
        effective_price = mid_price
    else:
        # Align prices length with n_slices
        if len(prices) != n_slices:
            x_old = np.linspace(0, 1, len(prices))
            x_new = np.linspace(0, 1, n_slices)
            prices_resampled = np.interp(x_new, x_old, prices)
        else:
            prices_resampled = prices
        effective_price = float(np.sum(prices_resampled * child_shares) / executed_shares)

    fill_frac = _fill_fraction(order_shares_raw, executed_shares)
    slippage = _slippage_bps(effective_price, mid_price, side)
    executed_notional = executed_shares * mid_price

    return {
        "symbol": row["symbol"],
        "side": side,
        "weight": row["weight"],
        "price": mid_price,
        "order_size_shares": order_shares_raw,
        "order_size_shares_capped": order_shares,
        "executed_shares": executed_shares,
        "executed_notional": executed_notional,
        "effective_price": effective_price,
        "slippage_bps": slippage,
        "fill_fraction": fill_frac,
        "adv_shares": adv_shares,
        "vol_daily": row["vol_daily"],
        "model": "TWAP",
    }


def exec_pov(
    row: pd.Series,
    vol_curve: np.ndarray,
    prices: np.ndarray,
    pov_rate: float = 0.1,
    adv_cap: float = 0.1,
) -> dict:
    """
    POV execution: participate in a fixed fraction of volume.

    Parameters
    ----------
    row : pd.Series
        Single SAFE trade row.
    vol_curve : np.ndarray
        Intraday volume curve (sums to 1).
    prices : np.ndarray
        Intraday prices aligned with vol_curve.
    pov_rate : float
        Fraction of volume to participate in (e.g., 0.1 = 10%).
    adv_cap : float
        Max fraction of ADV allowed for the order.

    Returns
    -------
    dict
        Execution summary for this trade.
    """
    side = float(row["side"])
    mid_price = float(row["price"])
    adv_shares = float(row["adv_shares"])

    order_shares_raw = _compute_order_size_shares(row)
    order_shares = _apply_adv_cap(order_shares_raw, adv_shares, adv_cap)

    # Volume per slice in shares
    vol_shares = adv_shares * vol_curve
    child_shares = pov_rate * vol_shares * np.sign(order_shares)

    # Cannot exceed total order_shares
    total_child = child_shares.sum()
    if abs(total_child) > abs(order_shares):
        scale = abs(order_shares) / abs(total_child)
        child_shares *= scale

    executed_shares = child_shares.sum()

    if abs(executed_shares) < 1e-9:
        effective_price = mid_price
    else:
        effective_price = float(np.sum(prices * child_shares) / executed_shares)

    fill_frac = _fill_fraction(order_shares_raw, executed_shares)
    slippage = _slippage_bps(effective_price, mid_price, side)
    executed_notional = executed_shares * mid_price

    return {
        "symbol": row["symbol"],
        "side": side,
        "weight": row["weight"],
        "price": mid_price,
        "order_size_shares": order_shares_raw,
        "order_size_shares_capped": order_shares,
        "executed_shares": executed_shares,
        "executed_notional": executed_notional,
        "effective_price": effective_price,
        "slippage_bps": slippage,
        "fill_fraction": fill_frac,
        "adv_shares": adv_shares,
        "vol_daily": row["vol_daily"],
        "model": "POV",
        "pov_rate": pov_rate,
    }


# ----------------------------------------------------------------------
# Public interface
# ----------------------------------------------------------------------

def simulate_execution_intraday(
    trades: pd.DataFrame,
    model: ExecutionModel = "VWAP",
    pov_rate: float = 0.1,
    adv_cap: float = 0.1,
    n_buckets: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Simulate intraday execution for a set of SAFE trades using VWAP/TWAP/POV.

    Parameters
    ----------
    trades : pd.DataFrame
        SAFE trades with at least:
        - symbol, side, weight, price, vol_daily, adv_shares, portfolio_notional
    model : {"VWAP", "TWAP", "POV"}
        Execution model to use.
    pov_rate : float
        Participation rate for POV model.
    adv_cap : float
        Max fraction of ADV allowed for each order.
    n_buckets : int
        Number of intraday buckets for synthetic curves.
    rng : np.random.Generator, optional
        RNG for synthetic price paths.

    Returns
    -------
    pd.DataFrame
        Execution results, one row per trade.
    """
    if rng is None:
        rng = np.random.default_rng()

    vol_curve = synthetic_volume_curve(n_buckets)

    results = []
    for _, row in trades.iterrows():
        mid_price = float(row["price"])
        vol_daily = float(row["vol_daily"])

        prices = synthetic_price_curve(
            mid_price=mid_price,
            vol_daily=vol_daily,
            n=n_buckets,
            drift_bps=0.0,
            noise_scale=0.1,
            rng=rng,
        )

        if model == "VWAP":
            out = exec_vwap(row, vol_curve, prices, adv_cap=adv_cap)
        elif model == "TWAP":
            out = exec_twap(row, prices, n_slices=n_buckets, adv_cap=adv_cap)
        elif model == "POV":
            out = exec_pov(row, vol_curve, prices, pov_rate=pov_rate, adv_cap=adv_cap)
        else:
            raise ValueError(f"Unknown execution model: {model}")

        results.append(out)

    return pd.DataFrame(results)