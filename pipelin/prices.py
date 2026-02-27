# pipeline/prices.py

import pandas as pd

try:
    import qlib
    from qlib.data import D
except ImportError:
    qlib = None
    D = None


def load_price_df(symbols, start_date, end_date):
    """
    Returns a MultiIndex DataFrame indexed by (symbol, date)
    with a 'close' column. Dates are normalized to pandas.Timestamp.
    """
    if qlib is None or D is None:
        raise ImportError("Qlib is not available. Ensure qlib is installed and initialized.")

    if isinstance(symbols, str):
        symbols = [symbols]

    dfs = []
    for sym in symbols:
        # Adjust this if your feature naming differs
        df = D.features([sym], ["$close"], start_date, end_date)
        df = df.rename(columns={"$close": "close"})
        df["symbol"] = sym
        dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=["symbol", "date", "close"]).set_index(["symbol", "date"])

    out = pd.concat(dfs)
    out = out.reset_index().rename(columns={"datetime": "date"})
    out["date"] = pd.to_datetime(out["date"])
    out = out.set_index(["symbol", "date"]).sort_index()
    return out