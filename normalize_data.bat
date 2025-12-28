@echo on
start "" /WAIT python scripts\data_collector\yahoo\collector.py normalize_data ^
--source_dir "C:\Users\harve\.qlib\stock_data\source\us_data" ^
--normalize_dir "C:\Users\harve\.qlib\stock_data\normalize\us_data" ^
--symbol_field_name symbol ^
--date_field_name date ^
--region us