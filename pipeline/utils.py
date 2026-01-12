import qlib
from qlib.data import D
import pandas as pd

def get_last_trading_day():
    # Get all trading days from Qlib calendar
    cal = D.calendar(start_time="2020-01-01", end_time=pd.Timestamp.today())
    return cal[-1].strftime("%Y-%m-%d")