import pandas as pd
from pathlib import Path
import json


def compute_daily_ic(df: pd.DataFrame) -> pd.Series:
    """
    df must contain: ['date', 'symbol', 'pred', 'label']
    """
    ic = df.groupby("date").apply(
        lambda x: x["pred"].corr(x["label"])
    ).rename("IC")
    return ic


def rolling_ic_metrics(ic_series: pd.Series) -> pd.DataFrame:
    ic_series = ic_series.sort_index()
    return pd.DataFrame(
        {
            "IC": ic_series,
            "IC_20": ic_series.rolling(20).mean(),
            "IC_60": ic_series.rolling(60).mean(),
            "IC_120": ic_series.rolling(120).mean(),
            "IC_vol_20": ic_series.rolling(20).std(),
            "IC_autocorr_20": ic_series.rolling(20).apply(
                lambda x: x.autocorr(), raw=False
            ),
        }
    )


def ic_alerts(roll_df: pd.DataFrame) -> pd.DataFrame:
    if roll_df.empty:
        return roll_df

    last_date = roll_df.index.max()
    row = roll_df.loc[last_date]

    low_mean = row["IC_20"] < 0
    high_vol = row["IC_vol_20"] > roll_df["IC_vol_20"].median() * 2

    if low_mean or high_vol:
        return roll_df.loc[[last_date]]
    else:
        return roll_df.iloc[0:0]  # empty DF

def run_rolling_ic_monitor(
    ic_series: pd.Series,
    out_dir: str | Path,
) -> dict:
    """
    ic_series: daily IC indexed by date
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roll_df = rolling_ic_metrics(ic_series)
    alerts_df = ic_alerts(roll_df)

    roll_df.to_csv(out_dir / "rolling_ic_metrics.csv")
    alerts_df.to_csv(out_dir / "rolling_ic_alerts.csv")

    latest_date = roll_df.index.max()
    latest_row = roll_df.loc[latest_date]

    summary = {
        "last_date": str(latest_date),
        "IC_last": float(latest_row["IC"]),
        "IC_20_last": float(latest_row["IC_20"]),
        "IC_vol_20_last": float(latest_row["IC_vol_20"]),
        "n_alerts_total": int(len(alerts_df)),
    }

    with open(out_dir / "rolling_ic_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary