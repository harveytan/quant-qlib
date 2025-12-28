setlocal
for /f %%i in ('powershell -NoProfile -Command "(Get-Date).AddDays(-12).ToString('yyyy-MM-dd')"') do set YESTERDAY=%%i
:: start "" /WAIT python scripts/data_collector/yahoo/collector.py download_data --source_dir c:\Users\harve\.qlib\stock_data\source\us_data --region us --start %YESTERDAY% --limit_symbols c:\ws\qlib\my_symbols.txt

:: load data starting from 2019
start "" /WAIT python scripts/data_collector/yahoo/collector.py download_data --source_dir c:\Users\harve\.qlib\stock_data\source\us_data --region us --start 2018-01-01 --limit_symbols c:\ws\qlib\my_symbols.txt
endlocal