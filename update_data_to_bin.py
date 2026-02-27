# update_bin_from_csv.py
import os
from qlib.data import D
from qlib.data.dataset.utils import exists_qlib_data
from qlib.config import REG_CN, REG_US
from qlib.utils import init_instance_by_config

# Path to your CSV-imported binary data
qlib_data_1d_dir = "C:/Users/harve/.qlib/qlib_data/us_data"

# Check if data already exists
if not exists_qlib_data(qlib_data_1d_dir):
    print(f"QLib 1d data not found at {qlib_data_1d_dir}")
else:
    print(f"QLib 1d data detected at {qlib_data_1d_dir}")

# Set QLib to use your binary data
D.set_data_path(qlib_data_1d_dir)

# Verify instruments
all_instruments = D.list_instruments(instruments=None, as_list=True)
print("All instruments recognized by QLib:")
print(all_instruments)
print(f"Total instruments: {len(all_instruments)}")
