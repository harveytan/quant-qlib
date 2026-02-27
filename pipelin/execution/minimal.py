def simulate_execution_minimal_v2(df, portfolio_notional=100000):
    """
    Execution simulator v2:
    - volatility-scaled slippage
    - spread-based fill probability
    - ADV-aware fill fraction
    - explicit, safe, no silent errors
    """

    out = df.copy()

    # === ORDER SIZE (shares) ===
    out["order_notional"] = out["weight"].abs() * portfolio_notional
    out["order_size_shares"] = out["order_notional"] / out["price"]

    # === RELATIVE ORDER SIZE ===
    # how large the order is relative to synthetic ADV
    out["rel_adv"] = out["order_size_shares"] / out["adv_shares"]

    # === BASE SLIPPAGE (bps) ===
    # spread_bps is already synthetic (2 + 0.1 * crash)
    base_slip = out["spread_bps"]

    # === VOLATILITY IMPACT (bps) ===
    # vol_daily is vol_20d from your df_today
    vol_impact = 5 * out["vol_daily"]  # tune later

    # === SIZE IMPACT (bps) ===
    size_impact = 50 * out["rel_adv"].clip(lower=0, upper=1)

    # === TOTAL SLIPPAGE (bps) ===
    out["slippage_bps"] = base_slip + vol_impact + size_impact

    # === FILL PROBABILITY ===
    # lower fills when:
    # - spread is wide
    # - volatility is high
    # - order is large relative to ADV
    fill_prob = (
        1
        - 0.3 * out["rel_adv"].clip(0, 1)
        - 0.2 * (out["spread_bps"] / 10).clip(0, 1)
        - 0.2 * out["vol_daily"].clip(0, 1)
    )

    out["fill_fraction"] = fill_prob.clip(lower=0.1, upper=1.0)

    # === EFFECTIVE PRICE ===
    # slippage_bps is applied in direction of trade
    slip_mult = 1 + (out["slippage_bps"] / 10000) * out["side"].map({"LONG": 1, "SHORT": -1, 1: 1, -1: -1})
    out["effective_price"] = out["price"] * slip_mult

    # === EXECUTED NOTIONAL ===
    out["executed_notional"] = out["order_notional"] * out["fill_fraction"]

    return out