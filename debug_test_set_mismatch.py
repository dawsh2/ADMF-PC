#!/usr/bin/env python3
"""Debug why test set results don't match notebook expectations."""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

print("=== Test Set Performance Debugging ===")
print(f"Analysis run at: {datetime.now()}")

# First, let's verify the notebook's calculations
print("\n1. NOTEBOOK TEST SET ANALYSIS")
print("-" * 50)

# Load notebook signals
notebook_signals = pd.read_parquet("config/bollinger/results/20250625_170201/traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet")
notebook_signals['ts'] = pd.to_datetime(notebook_signals['ts'])
notebook_signals = notebook_signals.sort_values('ts')

# Count trades from signals
notebook_signals['prev_val'] = notebook_signals['val'].shift(1).fillna(0)
notebook_entries = notebook_signals[(notebook_signals['prev_val'] == 0) & (notebook_signals['val'] != 0)]
print(f"Notebook entries: {len(notebook_entries)}")
print(f"Signal period: {notebook_signals['ts'].min()} to {notebook_signals['ts'].max()}")

# Load market data for the test period
market_data = pd.read_csv("config/bollinger/results/20250625_170201/data/SPY_5m.csv")
market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])
test_start = pd.Timestamp('2025-01-28', tz='UTC')
test_data = market_data[market_data['timestamp'] >= test_start]
print(f"Test period bars: {len(test_data)}")

# Now check your test set results
print("\n2. YOUR TEST SET RESULTS")
print("-" * 50)

# Find your test results directory
test_results_dirs = list(Path("config/bollinger/results").glob("*"))
test_results_dirs = [d for d in test_results_dirs if d.is_dir() and d.name != "latest" and d.name != "20250625_170201"]
test_results_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

if test_results_dirs:
    latest_test_dir = test_results_dirs[0]
    print(f"Found test results: {latest_test_dir}")
    
    # Check if it has position data
    test_positions_file = latest_test_dir / "traces/portfolio/positions_close/positions_close.parquet"
    if test_positions_file.exists():
        test_positions = pd.read_parquet(test_positions_file)
        print(f"Test positions found: {len(test_positions)}")
        
        # Analyze performance
        returns = []
        exit_types = {}
        
        for _, row in test_positions.iterrows():
            metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
            entry_price = metadata.get('entry_price', 0)
            exit_price = metadata.get('exit_price', 0)
            exit_type = metadata.get('exit_type', 'unknown')
            
            if entry_price > 0:
                ret = (exit_price - entry_price) / entry_price - 0.0001  # 1bp cost
                returns.append(ret)
                exit_types[exit_type] = exit_types.get(exit_type, 0) + 1
        
        if returns:
            total_return = (1 + pd.Series(returns)).prod() - 1
            win_rate = (pd.Series(returns) > 0).mean()
            avg_return = pd.Series(returns).mean()
            
            print(f"\nYour Test Performance:")
            print(f"  Total Return: {total_return*100:.2f}%")
            print(f"  Win Rate: {win_rate*100:.1f}%")
            print(f"  Avg Return/Trade: {avg_return*100:.3f}%")
            print(f"  Exit Types: {exit_types}")
    
    # Check signals
    test_signals_file = latest_test_dir / "traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet"
    if test_signals_file.exists():
        your_test_signals = pd.read_parquet(test_signals_file)
        your_test_signals['ts'] = pd.to_datetime(your_test_signals['ts'])
        
        print(f"\nYour test signals: {len(your_test_signals)}")
        print(f"Your test period: {your_test_signals['ts'].min()} to {your_test_signals['ts'].max()}")
        
        # Compare with notebook signals
        print("\n3. SIGNAL COMPARISON")
        print("-" * 50)
        
        # Align timestamps
        notebook_min = notebook_signals['ts'].min()
        your_min = your_test_signals['ts'].min()
        
        if abs((notebook_min - your_min).total_seconds()) < 3600:  # Within 1 hour
            print("✓ Start times match (within 1 hour)")
            
            # Compare first 100 signals
            print("\nComparing first 20 signal values:")
            for i in range(min(20, len(notebook_signals), len(your_test_signals))):
                nb_sig = notebook_signals.iloc[i]
                yr_sig = your_test_signals.iloc[i]
                
                time_diff = abs((nb_sig['ts'] - yr_sig['ts']).total_seconds())
                value_match = nb_sig['val'] == yr_sig['val']
                
                if not value_match or time_diff > 300:  # 5 min tolerance
                    print(f"  {i}: NB: {nb_sig['ts']} val={nb_sig['val']} | YOU: {yr_sig['ts']} val={yr_sig['val']} {'❌' if not value_match else '⚠️'}")
                elif i < 5:  # Show first few even if matching
                    print(f"  {i}: {nb_sig['ts']} val={nb_sig['val']} ✓")
        else:
            print(f"❌ Start times don't match:")
            print(f"  Notebook: {notebook_min}")
            print(f"  Yours: {your_min}")

print("\n4. CONFIGURATION CHECK")
print("-" * 50)

# Check if you're using the same config
if test_results_dirs:
    # Look for config in your test directory
    config_files = list(latest_test_dir.glob("*.yaml"))
    if config_files:
        print(f"Found config: {config_files[0].name}")
        with open(config_files[0], 'r') as f:
            config_content = f.read()
            if "stop_loss: 0.00075" in config_content and "take_profit: 0.001" in config_content:
                print("✓ Stop loss and profit target match notebook (0.075% / 0.1%)")
            else:
                print("❌ Configuration mismatch!")
                print("Config content:")
                print(config_content[:500])

print("\n5. POSSIBLE ISSUES")
print("-" * 50)
print("Common reasons for performance mismatch:")
print("1. Different data source or price adjustments")
print("2. Signal generation differences (check Bollinger calculation)")
print("3. Execution timing (bar close vs intrabar)")
print("4. Constraint differences (intraday vs overnight)")
print("5. Missing configurations or parameters")

# Show me what directory to analyze
print(f"\nTo help further, please run:")
print(f"1. Share your test config: cat [your_test_config].yaml")
print(f"2. Share your test results: python3 analyze_exits.py [your_test_results_dir]")
else:
    print("❌ No test results directory found. Please share the path to your test results.")