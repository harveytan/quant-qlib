# pipeline/safe_entry.py
import pandas as pd
from pipeline.utils import prints


def base_model_entry_logic(row):
    score = row["score"]
    mom = row["mom_label"]
    crash = row["crash_label"]
    direction = row["direction"]  # +1 long, -1 short

    # ---------- BASE (MODEL / SYSTEM VIEW) ----------
    if direction == 1:
        if score < 0.015:
            entry_raw = "BLOCKED"
            reason = "long_weak_score"
        elif crash in ["3x", "4x"]:
            entry_raw = "BLOCKED"
            reason = "long_extreme_crash"
        elif mom == "0x":
            entry_raw = "BLOCKED"
            reason = "long_zero_momentum"
        elif crash == "2x" or mom == "0.5x":
            entry_raw = "RISKY"
            reason = "long_crash_or_soft_momentum"
        else:
            entry_raw = "SAFE"
            reason = "long_clean_trend"
    else:
        if score > -0.01:
            entry_raw = "BLOCKED"
            reason = "short_weak_score"
        elif mom == "0x":
            entry_raw = "BLOCKED"
            reason = "short_zero_momentum"
        elif mom == "0.5x":
            entry_raw = "RISKY"
            reason = "short_soft_momentum"
        else:
            entry_raw = "SAFE"
            reason = "short_clean_trend"

    # Late-crash override for shorts
    if direction == -1 and crash in ["3x", "4x"] and entry_raw in ["SAFE", "RISKY"]:
        entry_raw = "RISKY_LATE_CRASH"
        reason = "short_late_crash"

    return {
        "entry_raw": entry_raw,
        "entry_reason": reason,
    }


def classify_entry(row):
    """
    Human-safe entry classifier.

    ALWAYS returns: entry_raw, entry_human, entry_reason
    """
    direction = row["direction"]
    score = row["score"]
    mom_raw = row["mom_raw"]
    dist_ma20 = row["dist_ma20"]
    crash_score = row["crash_score"]
    vol_ratio = row["vol_ratio"]
    ret_1d = row["ret_1d"]
    close = row["$close"]
    ma_20 = row["ma_20"]

    if row["instrument"] == "HOOD":
        print("HOOD DEBUG:", {
            "mom_3d": row.get("mom_3d"),
            "trend_20d": row.get("trend_20d"),
            "trend_60d": row.get("trend_60d"),
            "vol_ratio": row.get("vol_ratio"),
            "gap_down_pct": row.get("gap_down_pct"),
            "crash_score": row.get("crash_score"),
            "close_1d_change": row.get("close_1d_change"),
            "close_2d_change": row.get("close_2d_change"),
            "close_3d_change": row.get("close_3d_change"),
        })
    # ============================================================
    # A. ORDERLY PULLBACK OVERRIDE (fixes HOOD)
    # ============================================================
    # A multi-day selloff that is controlled, trend-intact, and not a panic gap.
    # This override neutralizes crash_score so the base model doesn't block it.
    is_orderly_pullback = (
        direction == 1
        and "mom_3d" in row
        and "trend_20d" in row
        and "trend_60d" in row
        and "gap_down_pct" in row
        and row["mom_3d"] < 0
        and row["trend_20d"] > 0
        and row["trend_60d"] > 0
        # Volatility: allow normal (<1.3) OR slightly elevated (<1.6) for crash_score==4
        and (
            vol_ratio < 1.3
            or (crash_score == 4 and vol_ratio < 1.6)
        )

        # Gap: normal (<3%) OR very small (<1.5%) for crash_score==4
        and (
            abs(row["gap_down_pct"]) < 0.03
            or (crash_score == 4 and abs(row["gap_down_pct"]) < 0.015)
        )

        # Crash score logic: allow 0–2 normally, allow 4 only under tight vol+gap conditions
        and (
            crash_score <= 2
            or (crash_score == 4 and vol_ratio < 1.6 and abs(row["gap_down_pct"]) < 0.015)
        )
    )
    if is_orderly_pullback:
        crash_score = 0  # for numeric checks
        # also patch crash_label for base model so it doesn't see "3x/4x"
        row_for_base = row.copy()
        row_for_base["crash_label"] = "0x"
    else:
        row_for_base = row


    # === SAFE: Counter-trend exhaustion reversal (HOOD-style) ===
    is_countertrend_exhaustion = (
        direction == 1
        and row["crash_score"] == 4
        and row["trend_20d"] < 0
        and row["trend_60d"] < 0
        and row["gap_down_pct"] < -0.05
        and row["vol_ratio"] < 1.6
        and row["close_1d_change"] > row["close_2d_change"] > row["close_3d_change"]
    )

    # ============================================================
    # B. BASE MODEL VIEW (after orderly-pullback override)
    # ============================================================
    base = base_model_entry_logic(row_for_base)
    entry_raw = base["entry_raw"]
    base_reason = base["entry_reason"]

    # ============================================================
    # C. EXHAUSTION PULLBACK LONG (HOOD pattern)
    # ============================================================
    #  - Long
    #  - Trend intact (20d, 60d up)
    #  - 3d negative momentum (selloff)
    #  - Yesterday deepest of last 3 days
    #  - Vol normal, no big gap panic
    #  - Not crashing numerically (crash_score == 0 after override)
    has_exhaustion_fields = all(
        c in row
        for c in [
            "mom_3d",
            "trend_20d",
            "trend_60d",
            "close_1d_change",
            "close_2d_change",
            "close_3d_change",
            "gap_down_pct",
        ]
    )

    is_exhaustion_pullback = (
        direction == 1
        and has_exhaustion_fields
        and crash_score == 0
        and row["trend_20d"] > 0
        and row["trend_60d"] > 0
        and row["mom_3d"] < 0
        and row["close_1d_change"] < row["close_2d_change"] < row["close_3d_change"]
        and vol_ratio < 1.3
        and row["gap_down_pct"] > -0.03
    )

    if is_exhaustion_pullback:
        return {
            "entry_raw": "SAFE",
            "entry_human": "SAFE_EXHAUSTION_PULLBACK",
            "entry_reason": "long_exhaustion_pullback",
        }

    if is_countertrend_exhaustion:
        return {
            "entry_raw": "WATCH",
            "entry_human": "WATCH_REVERSAL",
            "entry_reason": "countertrend_exhaustion_reversal",
        }

    # ============================================================
    # D. If base model BLOCKED and exhaustion pullback didn't override
    # ============================================================
    if entry_raw == "BLOCKED":
        return {
            "entry_raw": entry_raw,
            "entry_human": "BLOCKED",
            "entry_reason": base_reason,
        }

    # ============================================================
    # E. Hard late-crash override (numeric)
    # ============================================================
    if crash_score >= 3:
        return {
            "entry_raw": entry_raw,
            "entry_human": "WATCH_LATECRASH",
            "entry_reason": "short_late_crash" if direction == -1 else "long_late_crash",
        }

    # ============================================================
    # F. Value setup logic
    # ============================================================
    is_high_score = abs(score) > 0.02
    is_rising_score = mom_raw > 0
    is_price_weak = dist_ma20 < 0
    is_not_crashing = crash_score <= 1

    if is_high_score and is_rising_score and is_price_weak and is_not_crashing:
        # Mark as SAFE_* so it shows up in SAFE list
        return {
            "entry_raw": entry_raw,
            "entry_human": "SAFE_VAL_SETUP",
            "entry_reason": "high_score_rising_price_weak_no_crash",
        }

    # ============================================================
    # G. Trend filter
    # ============================================================
    if direction == 1:
        trend_ok = close > ma_20
    else:
        trend_ok = close < ma_20

    if not trend_ok:
        return {
            "entry_raw": entry_raw,
            "entry_human": "BLOCKED",
            "entry_reason": "trend_misaligned",
        }

    # ============================================================
    # H. Overextension guardrail
    # ============================================================
    if direction == 1 and ret_1d > 0.04:
        return {
            "entry_raw": entry_raw,
            "entry_human": "BLOCKED",
            "entry_reason": "overextended_up",
        }

    if direction == -1 and ret_1d < -0.04:
        return {
            "entry_raw": entry_raw,
            "entry_human": "BLOCKED",
            "entry_reason": "overextended_down",
        }

    # ============================================================
    # I. Volatility guardrail
    # ============================================================
    if vol_ratio > 1.5:
        return {
            "entry_raw": entry_raw,
            "entry_human": "SAFE_HI_SPREAD" if entry_raw == "SAFE" else "WATCH_HI_SPREAD",
            "entry_reason": "high_volatility_use_spread_limit",
        }

    # ============================================================
    # J. Default clean entry
    # ============================================================
    if entry_raw == "SAFE":
        entry_human = "SAFE"
    elif entry_raw in ["RISKY", "RISKY_LATE_CRASH"]:
        entry_human = "BLOCKED"
    else:
        entry_human = "BLOCKED"

    return {
        "entry_raw": entry_raw,
        "entry_human": entry_human,
        "entry_reason": base_reason,
    }


def apply_safe_entry(portfolio: pd.DataFrame) -> pd.DataFrame:
    entry_df = portfolio.apply(classify_entry, axis=1, result_type="expand")
    return pd.concat([portfolio, entry_df], axis=1)


def add_safe_features(df: pd.DataFrame) -> pd.DataFrame:
    # df indexed by (instrument, datetime)
    df = df.sort_index()

    # 1. Momentum pullback features
    df["close_1d_change"] = df.groupby("instrument")["$close"].pct_change(1)
    df["close_2d_change"] = df.groupby("instrument")["$close"].pct_change(2)
    df["close_3d_change"] = df.groupby("instrument")["$close"].pct_change(3)

    df["mom_3d"] = df.groupby("instrument")["$close"].pct_change(3)

    # 2. Trend features
    df["trend_20d"] = df.groupby("instrument")["$close"].transform(
        lambda x: x / x.shift(20) - 1
    )
    df["trend_60d"] = df.groupby("instrument")["$close"].transform(
        lambda x: x / x.shift(60) - 1
    )

    # 3. Gap feature (negative = gap down)
    df["gap_down_pct"] = df.groupby("instrument")["$open"].pct_change(1)

    return df