import pickle
import pandas as pd

# Load predictions
with open("mlruns/818494821712337451/13aedcba5cfe4a7a9ae6aaf8ef85b13f/artifacts/pred.pkl", "rb") as f:
    preds = pickle.load(f)


# Convert to DataFrame
ranked_df = pd.DataFrame([
    {"datetime": dt, "instrument": inst, "prediction": score}
    for (dt, inst), score in preds.items()
])

# Sort and rank
ranked_df = ranked_df.sort_values(by=["datetime", "prediction"], ascending=[True, False])
top10 = ranked_df.groupby("datetime").head(10)

print(top10)





# Flatten and rank
# ranked = []
# for dt, df in preds.items():
#     df = df.reset_index()
#     df["datetime"] = dt
#     ranked.append(df)

# import pandas as pd
# ranked_df = pd.concat(ranked)
# ranked_df = ranked_df.rename(columns={"score": "prediction"})
# ranked_df = ranked_df.sort_values(by=["datetime", "prediction"], ascending=[True, False])

# # Top 10 predictions per day
# top10 = ranked_df.groupby("datetime").head(10)
# print(top10)