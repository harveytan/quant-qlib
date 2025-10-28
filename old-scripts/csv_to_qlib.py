import subprocess
from pathlib import Path

csv_dir = Path("C:/Users/harve/.qlib/stock_data/source/us_data")
qlib_data_dir = Path("C:/Users/harve/.qlib/qlib_data/us_data")

# Step 1: Convert CSVs to QLib raw format
subprocess.run([
    "python", "csv_to_qlib.py",
    str(csv_dir)
], check=True)

# Step 2: Normalize the data
subprocess.run([
    "python", "scripts/normalize.py",
    "--input_dir", str(qlib_data_dir),
    "--output_dir", str(qlib_data_dir / "normalize"),
    "--freq", "day"
], check=True)

# Step 3: Dump binary data
subprocess.run([
    "python", "scripts/dump_bin.py", "dump_all",
    "--data_path", str(qlib_data_dir),
    "--qlib_dir", str(qlib_data_dir),
    "--freq", "day",
    "--symbol_field_name", "symbol",
    "--exclude_fields", "symbol"
], check=True)
