#!/usr/bin/env python3
"""Analyze ensemble performance"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load the signal trace
trace_path = Path("results/latest/traces/ensemble/SPY_5m_compiled_strategy_0.parquet")
signals = pd.read_parquet(trace_path)

print("Signal Trace Analysis")
print("=" * 50)
print(f"Total signal changes: {len(signals)}")
print(f"\nSignal value distribution:")
print(signals['val'].value_counts().sort_index())

# Check for values outside -1, 0, 1
unique_vals = signals['val'].unique()
print(f"\nUnique signal values: {sorted(unique_vals)}")

if any(abs(v) > 1 for v in unique_vals):
    print("\n⚠️  WARNING: Signal values outside [-1, 0, 1] range detected!")
    print("This might indicate multiple strategies voting in the same direction")

# Load market data to calculate performance
data_path = Path("../../../../data/SPY_5m.parquet")
if data_path.exists():
    market_data = pd.read_parquet(data_path)
    
    # Merge signals with market data
    # Convert sparse signals to full series
    full_signals = pd.Series(index=range(max(signals['idx']) + 1), dtype=float)
    for _, row in signals.iterrows():
        full_signals.iloc[int(row['idx'])] = row['val']
    full_signals = full_signals.fillna(method='ffill').fillna(0)
    
    # Calculate returns
    prices = market_data['close'].iloc[:len(full_signals)]
    returns = prices.pct_change()
    
    # Calculate strategy returns
    positions = full_signals.shift(1).fillna(0)  # Enter position on next bar
    strategy_returns = positions * returns
    
    # Performance metrics
    total_return = (1 + strategy_returns).prod() - 1
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252 * 78)  # Annualized
    
    # Trade statistics
    trades = positions.diff().abs() / 2  # Each round trip = 1 trade
    num_trades = int(trades.sum())
    
    print(f"\nPerformance Metrics:")
    print(f"Total Return: {total_return:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Number of Trades: {num_trades}")
    print(f"Avg Trades per Day: {num_trades / (len(market_data) / 78):.1f}")
    
    # Win/loss analysis
    trade_returns = []
    in_position = False
    entry_price = 0
    
    for i in range(1, len(positions)):
        if positions.iloc[i] != 0 and not in_position:
            # Enter position
            in_position = True
            entry_price = prices.iloc[i]
        elif positions.iloc[i] == 0 and in_position:
            # Exit position
            exit_price = prices.iloc[i]
            trade_return = (exit_price - entry_price) / entry_price * positions.iloc[i-1]
            trade_returns.append(trade_return)
            in_position = False
    
    if trade_returns:
        trade_returns = pd.Series(trade_returns)
        win_rate = (trade_returns > 0).mean()
        avg_win = trade_returns[trade_returns > 0].mean() if any(trade_returns > 0) else 0
        avg_loss = trade_returns[trade_returns < 0].mean() if any(trade_returns < 0) else 0
        
        print(f"\nTrade Analysis:")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Avg Win: {avg_win:.3%}")
        print(f"Avg Loss: {avg_loss:.3%}")
        print(f"Win/Loss Ratio: {abs(avg_win/avg_loss) if avg_loss != 0 else 'N/A':.2f}")
else:
    print("\nMarket data not found for performance calculation")