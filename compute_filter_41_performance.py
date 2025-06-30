#!/usr/bin/env python3
"""Compute exact performance of Filter 41 strategy on test set"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# First, let's find the exact Filter 41 strategy from the original results
original_results_dir = Path("config/keltner/results/20250622_180858")
traces_dir = original_results_dir / "traces" / "keltner_bands"

# Strategy 1029 = Period 50, Mult 1.0, Filter 41
# Strategy ID 1029 corresponds to this specific combination
strategy_1029_file = traces_dir / "SPY_5m_compiled_strategy_1029.parquet"

print("COMPUTING EXACT FILTER 41 PERFORMANCE")
print("="*80)

# Load the winning strategy trace
if strategy_1029_file.exists():
    df_1029 = pd.read_parquet(strategy_1029_file)
    print(f"Loaded Filter 41 strategy trace: {len(df_1029)} signal changes")
    
    # Analyze the exact trades
    trades = 0
    trade_returns = []
    in_trade = False
    entry_price = None
    trade_direction = 0
    
    for i in range(len(df_1029)):
        signal = df_1029.iloc[i]['val']
        price = df_1029.iloc[i]['px']
        
        if not in_trade and signal != 0:
            trades += 1
            in_trade = True
            entry_price = price
            trade_direction = signal
        elif in_trade and signal == 0:
            in_trade = False
            if entry_price is not None:
                if trade_direction > 0:  # Long
                    ret = (price - entry_price) / entry_price
                else:  # Short
                    ret = (entry_price - price) / entry_price
                trade_returns.append(ret)
    
    # Calculate exact metrics
    win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) * 100
    avg_return = np.mean(trade_returns) * 100
    total_return = sum(trade_returns) * 100
    
    # Compound return
    cumulative = 1.0
    for r in trade_returns:
        cumulative *= (1 + r)
    compound_return = (cumulative - 1) * 100
    
    print(f"\nFilter 41 Strategy (Training Period) - EXACT RESULTS:")
    print(f"  Completed trades: {trades}")
    print(f"  Win rate: {win_rate:.1f}%")
    print(f"  Avg return per trade: {avg_return:.4f}%")
    print(f"  Total return: {total_return:.2f}%")
    print(f"  Compound return: {compound_return:.2f}%")
    
    # Now we need to check if we ran Filter 41 on test set
    # Let's look for any test results with similar parameters
    
    print("\n" + "-"*80)
    print("CHECKING FOR FILTER 41 IN TEST SET...")
    
    # The test set config didn't include filters, so we can't compute exact performance
    # We need to acknowledge this limitation
    
    print("\nIMPORTANT: The test set was run WITHOUT filters.")
    print("To compute exact Filter 41 performance on test set, we would need to:")
    print("1. Re-run the test with the exact Filter 41 configuration")
    print("2. The filter expression for Filter 41 is not available in the current data")
    
    print("\nWHAT WE KNOW FOR CERTAIN:")
    print(f"- Filter 41 on train: {trades} trades, {compound_return:.2f}% return")
    print(f"- No filter on train: 1131 trades, 4.60% return") 
    print(f"- No filter on test: 292 trades, 0.03% return")
    print(f"- Filter 41 selected {trades/1131*100:.1f}% of available trades in training")
    
else:
    print(f"ERROR: Could not find strategy file at {strategy_1029_file}")
    print("Cannot compute exact Filter 41 performance.")

print("\nCONCLUSION:")
print("To compute exact test performance, we need to re-run with Filter 41.")
print("The filter configuration is embedded in the original parameter sweep")