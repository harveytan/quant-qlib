# pipeline/portfolio_builder.py
import pandas as pd

from pipeline.safe_entry import apply_safe_entry, add_safe_features


def select_long_short(df_today, top_k_long, top_k_short):
    df_sorted = df_today.sort_values("score", ascending=False)

    longs = df_sorted.head(top_k_long).copy()
    shorts = df_sorted.tail(top_k_short).copy()

    long_names = set(longs["instrument"])
    shorts = shorts[~shorts["instrument"].isin(long_names)]

    if len(shorts) < top_k_short:
        needed = top_k_short - len(shorts)
        remaining = df_sorted[
            ~df_sorted["instrument"].isin(long_names | set(shorts["instrument"]))
        ]
        refill = remaining.tail(needed)
        shorts = pd.concat([shorts, refill], axis=0)

    return longs, shorts


def apply_volatility_weights(longs, shorts, df_today):
    median_vol = df_today["$vol_20d"].median()

    longs["$vol_20d"] = longs["$vol_20d"].fillna(median_vol)
    shorts["$vol_20d"] = shorts["$vol_20d"].fillna(median_vol)

    vol_floor = median_vol * 0.5
    vol_cap = median_vol * 3.0

    longs["$vol_20d"] = longs["$vol_20d"].clip(lower=vol_floor, upper=vol_cap)
    shorts["$vol_20d"] = shorts["$vol_20d"].clip(lower=vol_floor, upper=vol_cap)

    longs["inv_vol"] = 1.0 / longs["$vol_20d"]
    shorts["inv_vol"] = 1.0 / shorts["$vol_20d"]

    longs["weight"] = longs["inv_vol"] / longs["inv_vol"].sum()
    shorts["weight"] = -shorts["inv_vol"] / shorts["inv_vol"].sum()

    portfolio = pd.concat([longs, shorts], axis=0)

    total_weight = portfolio["weight"].sum()
    if abs(total_weight) > 1e-6:
        portfolio["weight"] -= total_weight / len(portfolio)

    return portfolio


def add_derived_fields(portfolio, momentum_label_fn):
    portfolio["direction"] = portfolio["weight"].apply(lambda w: 1 if w > 0 else -1)

    portfolio["mom_value"] = portfolio["mom_raw"] * portfolio["direction"]
    portfolio["mom_label"] = portfolio["mom_value"].apply(momentum_label_fn)

    portfolio["ret_1d"] = portfolio["$close"] / portfolio["prev_close"] - 1

    portfolio["trend_ok"] = (
        (portfolio["direction"] == 1) & (portfolio["$close"] > portfolio["ma_20"])
    ) | (
        (portfolio["direction"] == -1) & (portfolio["$close"] < portfolio["ma_20"])
    )

    median_vol = portfolio["$vol_20d"].median()
    portfolio["vol_ratio"] = portfolio["$vol_20d"] / median_vol

    return portfolio

def build_long_short_portfolio(
    df_today: pd.DataFrame,
    df_full: pd.DataFrame,
    latest_date,
    top_k_long=20,
    top_k_short=20,
    momentum_label_fn=None,
):
    if momentum_label_fn is None:
        raise ValueError("momentum_label_fn must be provided")

    # 1. SAFE features on full history (df_full has close/open/high/low)
    df_full = df_full.sort_index()
    df_full = add_safe_features(df_full)

    # 2. Slice today's rows WITH SAFE features
    df_today_safe = df_full.xs(latest_date, level="datetime").reset_index()
    df_today_safe["datetime"] = latest_date

    # 3. Select long/short candidates from df_today (your existing daily slice)
    longs, shorts = select_long_short(df_today, top_k_long, top_k_short)

    # 4. Apply volatility weights
    portfolio = apply_volatility_weights(longs, shorts, df_today)

    # 5. Add derived fields (mom_label, ret_1d, dist_ma20, vol_ratio, etc.)
    portfolio = add_derived_fields(portfolio, momentum_label_fn)

    # 6. Merge SAFE features from df_today_safe into portfolio
    safe_cols = [
        "mom_3d", "trend_20d", "trend_60d",
        "gap_down_pct",
        "close_1d_change", "close_2d_change", "close_3d_change",
    ]
    portfolio["datetime"] = latest_date

    portfolio = portfolio.merge(
        df_today_safe[["instrument", "datetime"] + safe_cols],
        on=["instrument", "datetime"],
        how="left",
        suffixes=("", "_safe"),
    )

    for col in safe_cols:
        safe_col = col + "_safe"
        if safe_col in portfolio.columns:
            portfolio[col] = portfolio[safe_col]
            portfolio.drop(columns=[safe_col], inplace=True)

    # 7. Apply SAFE classifier
    portfolio = apply_safe_entry(portfolio)

    return portfolio