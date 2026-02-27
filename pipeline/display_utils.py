# pipeline/display_utils.py
import pandas as pd

from pipeline.utils import prints


def print_safe_trades(portfolio, safe_df_path: str | None = None):
    safe = portfolio[portfolio["entry_human"].str.startswith("SAFE")].copy()

    prints("\n=== SAFE TRADES ===")
    if safe.empty:
        prints("No SAFE trades today.")
        # Always return an empty DataFrame with the correct schema
        return pd.DataFrame(columns=[
            "symbol", "direction", "score", "momentum", "crash", "reason"
        ])

    safe = safe.sort_values(
        by=["direction", "score", "mom_label"],
        ascending=[False, False, False],
    )

    prints(" ***====  SAFE TRADES  (sorted by LONG/SHORT → score → momentum)  ====***")
    # -----------------------------
    # Header (printed once)
    # -----------------------------
    prints(f"{'L/S':<5} {'Code':<6} {'Score':<7} {'MOM':<4} {'Crash':<6} {'VWAP':<6} {'Reason'}")
    prints("-" * 74)
    # -----------------------------
    # Rows
    # -----------------------------
    for _, row in safe.iterrows():
        side = "LONG" if row["direction"] == 1 else "SHORT"

        # Round VWAP to whole dollars

        prints(
            f"{side:<5} "
            f"{row['instrument']:<6} "
            f"{row['score']:<8.4f} "
            f"{row['mom_label']:<4} "
            f"{row['crash_score']}x   "
            f"${row['vwap']:<6.2f} "
            f"{row['entry_reason']}"
        )

    safe_norm = pd.DataFrame({
        "symbol": safe["instrument"].astype(str),
        "direction": safe["direction"].apply(lambda x: "LONG" if x == 1 else "SHORT"),
        "score": safe["score"],
        "momentum": safe["mom_value"],
        "crash": safe["crash_score"],
        "reason": safe["entry_reason"],
    })

    safe_df_path.parent.mkdir(parents=True, exist_ok=True)
    safe_norm.to_parquet(safe_df_path, index=False)
    return safe_norm