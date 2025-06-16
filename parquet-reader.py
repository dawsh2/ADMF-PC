import pandas as pd
import glob
import argparse
import os

def read_parquet_preview(file_path, limit=100):
    """
    Reads the first `limit` rows from one or more Parquet files.

    Parameters:
        file_path (str): Parquet file path or glob pattern (e.g., *.parquet)
        limit (int): Number of rows to preview

    Returns:
        None: Prints output to console
    """
    file_paths = glob.glob(file_path) if '*' in file_path or file_path.endswith('/') else [file_path]
    pd.set_option("display.max_rows", 100)   
    
    for file in file_paths:
        if not os.path.isfile(file):
            print(f"File not found: {file}")
            continue
        try:
            df = pd.read_parquet(file, engine='pyarrow')
            print(f"\n--- Preview: {file} ({min(len(df), limit)} rows) ---")
            print(df.head(limit))
        except Exception as e:
            print(f"Error reading {file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read and preview Parquet files.")
    parser.add_argument("path", type=str, help="Path to a Parquet file or a glob pattern (e.g., *.parquet)")
    parser.add_argument("--limit", type=int, default=100, help="Number of rows to preview (default: 100)")
    args = parser.parse_args()

    read_parquet_preview(args.path, args.limit)

