@echo on
:: FIRST TIME CODE
rem start "" /WAIT python scripts/dump_bin.py dump_all ^
rem   --data_path C:/Users/harve/.qlib/stock_data/normalize/us_data ^
rem   --qlib_dir C:/Users/harve/.qlib/qlib_data/us_data ^
rem   --freq day ^
rem   --date_field_name date ^
rem   --symbol_field_name symbol ^
rem   --exclude_fields symbol,date

:: Second time and next
rem start "" /WAIT python scripts/dump_bin.py dump_update ^
rem   --data_path C:/Users/harve/.qlib/stock_data/normalize/us_data ^
rem   --qlib_dir C:/Users/harve/.qlib/qlib_data/us_data ^
rem   --freq day ^
rem   --date_field_name date ^
rem   --symbol_field_name symbol ^
rem   --exclude_fields symbol,date

:: This section is to not skip normalize and have normalized data available
@REM start "" /WAIT python scripts/dump_bin.py dump_all ^
@REM --data_path C:/Users/harve/.qlib/stock_data/normalize/us_data ^
@REM --freq day ^
@REM --qlib_dir C:/Users/harve/.qlib/qlib_data/us_data ^
@REM --symbol_field_name symbol ^
@REM --exclude_fields symbol

:: original Working code
@REM start "" /WAIT python scripts/dump_bin.py dump_all ^
@REM --data_path "C:/Users/harve/.qlib/stock_data/normalize/us_data" ^
@REM --qlib_dir "C:/Users/harve/.qlib/qlib_data/us_data" ^
@REM --include_fields open,high,low,close,volume ^
@REM --date_field_name date ^
@REM --symbol_field_name symbol ^
@REM --freq day ^
@REM --region us

@REM start "" /WAIT python scripts/dump_bin.py dump_all ^
@REM --data_path "C:/Users/harve/.qlib/stock_data/normalize/us_data_enriched" ^
@REM --qlib_dir "C:/Users/harve/.qlib/qlib_data/us_data" ^
@REM --include_fields open,high,low,close,volume_log,vol_5_20,vol_10_20,ret_5d,vol_5d,rank_ret_5d,rank_vol_5d,ensemble_label,ret_10d,vol_10d,rank_ret_10d,rank_vol_10d,ret_20d,vol_20d,rank_ret_20d,rank_vol_20d,days_since_ipo ^
@REM --date_field_name date ^
@REM --symbol_field_name symbol ^
@REM --freq day ^
@REM --region us

start "" /WAIT python scripts/dump_bin.py dump_all ^
--data_path "C:/Users/harve/.qlib/stock_data/normalize/us_data_enriched" ^
--qlib_dir "C:/Users/harve/.qlib/qlib_data/us_data" ^
--include_fields open,high,low,close,vol_5d,vol_10d,vol_20d,vol_5_20,vol_10_20,vol_20_60,vol_5_60,ret_5d,ret_10d,ret_20d,mom_60d,mom_5d_z,mom_20d_z,price_above_ma20,price_above_ma60,trend_5_20,intraday_range,intraday_body,range_ma5,range_ma20,volume_log,volume_shock,volume_z,volume_vol,days_since_ipo_cont,ipo_bucket,ret_5d_vol_scaled,ret_10d_vol_scaled,ret_20d_vol_scaled,intraday_vol_ratio,intraday_range_vol_norm,overnight_ret,overnight_ret_z_20,trend_persist_ma20,trend_persist_ma60,micro_imbalance,micro_imbalance_z_20,vol_20d_resid_liq,rank_vol_5d,rank_vol_10d,rank_vol_20d,rank_mom_20d,rank_mom_60d,rank_intraday_range,rank_volume_log,rank_volume_shock,eps_actual_lag3,eps_est_lag3,eps_surprise_lag3,eps_ttm,eps_growth_yoy,surprise_std,surprise_pct,beat_streak,revision_trend,eps_momentum,earnings_yield,ensemble_label ^
--date_field_name date ^
--symbol_field_name symbol ^
--freq day ^
--region us

