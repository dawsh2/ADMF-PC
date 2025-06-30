#!/usr/bin/env python3
import pandas as pd

fills = pd.read_parquet("/Users/daws/ADMF-PC/config/bollinger/results/20250628_194812/traces/execution/fills/execution_fills.parquet")
print("Fills columns:", list(fills.columns))
print("\nFirst fill:")
print(fills.iloc[0])