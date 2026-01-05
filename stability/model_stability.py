import pandas as pd
from pathlib import Path
import json


def compute_feature_importance_jaccard(
    old_importance: pd.Series,
    new_importance: pd.Series,
    k: int = 20,
) -> float:
    old_top = set(old_importance.sort_values(ascending=False).head(k).index)
    new_top = set(new_importance.sort_values(ascending=False).head(k).index)
    if not old_top and not new_top:
        return float("nan")
    return len(old_top & new_top) / len(old_top | new_top)


def prediction_stability(old_pred: pd.Series, new_pred: pd.Series) -> float:
    aligned = pd.concat([old_pred, new_pred], axis=1, join="inner").dropna()
    if aligned.shape[0] < 2:
        return float("nan")
    return aligned.iloc[:, 0].corr(aligned.iloc[:, 1])


def rank_stability(old_pred: pd.Series, new_pred: pd.Series, k: int = 50) -> float:
    aligned = pd.concat([old_pred, new_pred], axis=1, join="inner").dropna()
    if aligned.shape[0] < k:
        return float("nan")

    old_top_idx = aligned.iloc[:, 0].nlargest(k).index
    new_top_idx = aligned.iloc[:, 1].nlargest(k).index
    return len(set(old_top_idx) & set(new_top_idx)) / k


def run_model_stability_check(
    old_importance: pd.Series,
    new_importance: pd.Series,
    old_pred: pd.Series,
    new_pred: pd.Series,
    out_dir: str | Path,
    k_importance: int = 20,
    k_rank: int = 50,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jaccard = compute_feature_importance_jaccard(
        old_importance, new_importance, k=k_importance
    )
    pred_corr = prediction_stability(old_pred, new_pred)
    rank_overlap = rank_stability(old_pred, new_pred, k=k_rank)

    summary = {
        "feature_importance_jaccard": float(jaccard),
        "prediction_corr": float(pred_corr),
        "rank_overlap_topk": float(rank_overlap),
        "feature_importance_low": jaccard < 0.3,
        "prediction_corr_low": pred_corr < 0.7,
        "rank_overlap_low": rank_overlap < 0.5,
    }

    with open(out_dir / "model_stability_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame(
        {
            "metric": [
                "feature_importance_jaccard",
                "prediction_corr",
                "rank_overlap_topk",
            ],
            "value": [jaccard, pred_corr, rank_overlap],
        }
    ).to_csv(out_dir / "model_stability_metrics.csv", index=False)

    return summary