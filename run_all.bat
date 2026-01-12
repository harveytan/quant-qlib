@echo on
call collect_data.bat
call normalize_data.bat
start "" /WAIT python enrich_data.py
start "" /WAIT python -m baselines.run_baselines
call dump_to_bin.bat

:: here manually run training
:: call train.bat (train_ensemble.py)

:: after training completed, run evaluation -> see model's IC (Information Coefficient) performance:
:: IC between 0.20–0.30 -> exceptional
:: IC greater than 0.30+ -> Institutional Grade alpha
:: code: python diagnostics_long_short.py

:: Rebalance Daily: run top_long_short.py
:: Log daily rule-base trades: run daily_inference.py
:: code:python top_long_short.py
:: code:python daily_inference.py

:: Backtest:
:: On each day t:
:: - Use features at date t
:: - Predict scores with your model
:: - Build long/short weights (same build_long_short_portfolio)
:: - Realized 1‑day PnL ≈ weighted sum of next‑day return \$ ret\_ 1d\$ 
:: - Repeat over all days, accumulate PnL
:: code: python backtest_long_short.py


:: This is also a self documentation
:: first step as always run collect_data.bat
:: second step is to run normalize_data.bat
:: third step is to run dump_to_bin.bat
:: 
:: Latest train code is in train.bat
:: Path to improving predictive trading

:: You’ve built the core: train → reload → evaluate.
::   To turn that into “very high profitable returns,” shift from model metrics to 
::   a disciplined trading framework that converts predictions into portfolios, measures economics with costs, and iterates toward robustness.

:: Define a clear objective and label horizon
:: • 	Target: Predict cross‑sectional returns, not absolute prices.
:: • 	Horizon: Start with 5‑day forward returns (less noisy than 1‑day), hold for 5 days, rebalance weekly.
:: • 	Universe: Keep your 40‑stock sandbox for speed; expand only after the pipeline is stable.

:: Build a rigorous evaluation loop
:: • 	Core metrics:
:: • 	IC and Rank IC: Per‑day mean and stability.
:: • 	Economics: CAGR, Sharpe, max drawdown, hit rate, turnover.
:: • 	Cost‑adjusted returns: Slippage + commission deducted.
:: • 	Validation design:
:: • 	Walk‑forward: Train on period A, validate on B, test on C; roll forward.
:: • 	Robustness: Try different months, universes, and horizons; check consistency.
:: • 	Leakage guards: Ensure features don’t use future info; cross‑sectional z‑score normalization is OK, but keep it within each date.

:: From predictions to a tradable portfolio
:: • 	Ranking: Each day, rank stocks by predicted return.
:: • 	Weights:
:: • 	Long-only: Top decile; weights ∝ z‑score of predictions.
:: • 	Long‑short: Top decile long, bottom decile short; dollar‑neutral or beta‑neutral.
:: • 	Risk controls:
:: • 	Position caps: e.g., max 5% per name.
:: • 	Sector/industry neutralization: Prevent concentration.
:: • 	Volatility scaling: Reduce exposure on high‑volatility names/days.
:: • 	Costs: Model per‑trade commission and slippage; apply to turnover each rebalance.

::🔧 1. Feature Engineering
::• 	Lagged features: Add lagged returns, volatility, or rr_ratio to capture momentum or mean-reversion.
::• 	Sector indicators: Encode sector membership to help the model learn sector-specific behavior.
::• 	Macro overlays: Include macro signals like interest rates, VIX, or CPI if available — even as dummy features.

::🧠 2. Model Enhancements
::• 	Stacked models: Combine LightGBM with a linear model or neural net for hybrid learning.
::• 	Interaction terms: Use polynomial features or tree-based interaction constraints to capture nonlinear relationships.
::• 	Regularization tuning: Try stronger L1/L2 penalties to reduce overfitting and improve generalization.

::🧪 3. Label Refinement
::• 	Smoothed returns: Use exponentially weighted returns to reduce noise.
::• 	Risk-adjusted labels: Normalize returns by volatility or drawdown to teach the model about risk.
::• 	Cohort-aware labels: Penalize stocks that consistently underperform their cohort, even if raw return is positive.

::📊 4. Data Augmentation
::• 	Synthetic instruments: Create pseudo-stocks by bootstrapping features and returns from real ones.
::• 	Rolling windows: Train on overlapping windows to increase sample size without adding symbols.
::• 	Dropout masking: Randomly mask features during training to improve robustness.
::🧭 5. Diagnostic Extensions
::- Log IC per date: Track rank correlation to validate score quality.
::- Visualize score vs realized return: Confirm that higher scores lead to better outcomes.
::- Track rr_ratio by score bucket: Validate that your model is separating winners from losers.

::🚀 Bonus: Optuna + Early Stopping + Score Filtering
::- Use Optuna to tune not just hyperparameters, but also:
::- Score thresholds for trade inclusion
::- rr_ratio filters for attribution
::- Drawdown guards for survivability


@REM 📅 2025-09-29 — Trade List — 10 holding days
@REM   Buys: BABA(+0.7403), GOOG(+0.7037), AMD(+0.4196), RDDT(+0.3979), MSTR(+0.3163), APP(+0.3099), AAPL(+0.3046), PLTR(+0.2726), DVN(+0.2447), NVDA(+0.2308)
@REM   Sells: ABT(-0.2281), MSFT(-0.2445), SSO(-0.2511), VWO(-0.2519), SPUU(-0.2713), MCD(-0.3503), COST(-0.3851), LLY(-0.3864), SPY(-0.4578), ADBE(-0.5822)
@REM     ⏱ 5d → Buy: +1.2% (10), Sell: +2.2% (10), Spread: -1.0%, Vol: 9.41%, R/R: 0.11 ⚠️
@REM     ⏱ 10d → Buy: -2.0% (10), Sell: +0.6% (10), Spread: -2.5%, Vol: 10.95%, R/R: 0.23 ⚠️

@REM 📅 2025-10-13 — Trade List — 10 holding days
@REM   Buys: BABA(+1.3572), INTC(+0.7305), AVGO(+0.2189), NET(+0.0654), ADBE(-0.0153), NVDA(-0.0480), NFLX(-0.0681), DVN(-0.0682), APP(-0.0955), GOOG(-0.1011)
@REM   Sells: ABT(-0.3521), SPUU(-0.3562), UNH(-0.3688), QQQ(-0.4080), LLY(-0.4147), MSTR(-0.4769), COST(-0.5164), SPY(-0.5392), VWO(-0.5399), RDDT(-0.5823)

@REM 📅 2025-10-13 — Trade List — 10 holding days ( on new model by adding new std() params )
@REM   Buys: NET(-0.3123), NVDA(-0.3682)
@REM   Sells: PEP(-0.5929), APP(-0.6003)

@REM 📅 2025-10-14 — Trade List — 10 holding days
@REM   Buys: NET(-0.3328), SPY(-0.3754)
@REM   Sells: APP(-0.5891), PEP(-0.6083)