#!/usr/bin/env python3
"""Check for any filters or additional constraints in execution."""

import pandas as pd
from pathlib import Path
import json

execution_run = Path("/Users/daws/ADMF-PC/config/bollinger/results/20250628_194812")

# Check portfolio orders for any filtering
orders_path = execution_run / "traces/portfolio/orders/portfolio_orders.parquet"
if orders_path.exists():
    orders = pd.read_parquet(orders_path)
    print(f"Portfolio orders: {len(orders)}")
    
    # Check metadata for filters
    if 'metadata' in orders.columns and len(orders) > 0:
        # Get first order metadata
        first_meta = orders.iloc[0]['metadata']
        if isinstance(first_meta, dict):
            print("\nFirst order metadata keys:")
            for key in sorted(first_meta.keys()):
                print(f"  {key}: {first_meta[key]}")

# Check fills
fills_path = execution_run / "traces/execution/fills/execution_fills.parquet"
if fills_path.exists():
    fills = pd.read_parquet(fills_path)
    print(f"\nExecution fills: {len(fills)}")

# Check position events
pos_open = execution_run / "traces/portfolio/positions_open/positions_open.parquet"
pos_close = execution_run / "traces/portfolio/positions_close/positions_close.parquet"

if pos_open.exists() and pos_close.exists():
    opens = pd.read_parquet(pos_open)
    closes = pd.read_parquet(pos_close)
    print(f"\nPosition opens: {len(opens)}")
    print(f"Position closes: {len(closes)}")
    
    # Check exit types
    if 'exit_type' in closes.columns:
        exit_counts = closes['exit_type'].value_counts()
        print("\nExit type breakdown:")
        for exit_type, count in exit_counts.items():
            pct = count/len(closes)*100
            print(f"  {exit_type}: {count} ({pct:.1f}%)")
    
    # Check for intraday constraints
    if 'exit_reason' in closes.columns:
        eod_exits = closes[closes['exit_reason'].str.contains('EOD|end of day|intraday', case=False, na=False)]
        if len(eod_exits) > 0:
            print(f"\nFound {len(eod_exits)} potential intraday/EOD exits")
            print("Sample reasons:", eod_exits['exit_reason'].head().tolist())

# Load and check logs
log_path = execution_run / "backtest.log"
if log_path.exists():
    with open(log_path) as f:
        log_content = f.read()
        
    # Check for filters mentioned
    if "filter" in log_content.lower():
        print("\n⚠️ Found 'filter' mentioned in logs")
        # Get lines with filter
        filter_lines = [line for line in log_content.split('\n') if 'filter' in line.lower()]
        print(f"Found {len(filter_lines)} lines mentioning filters")
        for line in filter_lines[:5]:
            print(f"  {line[:100]}...")
            
    # Check for EOD/intraday constraints
    if any(word in log_content.lower() for word in ['eod', 'end of day', 'intraday']):
        print("\n⚠️ Found EOD/intraday constraints in logs")