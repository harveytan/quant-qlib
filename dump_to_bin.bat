@echo off
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

start "" /WAIT python scripts/dump_bin.py dump_all ^
--data_path "C:/Users/harve/.qlib/stock_data/normalize/us_data_enriched" ^
--qlib_dir "C:/Users/harve/.qlib/qlib_data/us_data" ^
--include_fields open,high,low,close,volume,ret_5d,vol_5d,rank_ret_5d,rank_vol_5d,ensemble_label,ret_10d,vol_10d,rank_ret_10d,rank_vol_10d,ret_20d,vol_20d,rank_ret_20d,rank_vol_20d,days_since_ipo ^
--date_field_name date ^
--symbol_field_name symbol ^
--freq day ^
--region us