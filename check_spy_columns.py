"""Check SPY data columns"""
import pandas as pd

df = pd.read_csv("./data/SPY_1m.csv")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nData types:")
print(df.dtypes)