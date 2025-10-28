import pandas as pd
import pandas_market_calendars as mcal

# Get NYSE calendar
nyse = mcal.get_calendar('NYSE')

# Define date range
schedule = nyse.schedule(start_date='2019-01-01', end_date='2025-09-27')

# Extract trading days
calendar = mcal.date_range(schedule, frequency='1D')

# Save to CSV
calendar.strftime('%Y-%m-%d').to_series().to_csv('calendar.csv', index=False, header=False)