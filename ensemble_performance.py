#!/usr/bin/env python3
import pandas as pd
import numpy as np

# Load metadata to determine which data file to use
import json
with open('config/ensemble/results/latest/metadata.json', 'r') as f:
    metadata = json.load(f)

# Get the component ID (could be SPY_1m or SPY_5m)
component_id = list(metadata['components'].keys())[0]
symbol_timeframe = component_id.split('_compiled')[0]  # e.g., "SPY_1m" or "SPY_5m"

# Load signals
signal_path = f'config/ensemble/results/latest/traces/ensemble/{component_id}.parquet'
signals = pd.read_parquet(signal_path)

# Load appropriate market data
timeframe = symbol_timeframe.split('_')[1]  # Extract "1m" or "5m"
market_data = pd.read_parquet(f'data/SPY_{timeframe}.parquet')

print(f'Ensemble Strategy Performance ({timeframe} data)')
print('=' * 50)

# Convert sparse signals to full series
max_idx = signals['idx'].max()
full_signals = pd.Series(index=range(max_idx + 1), dtype=float)
for _, row in signals.iterrows():
    full_signals.iloc[int(row['idx'])] = row['val']
full_signals = full_signals.fillna(method='ffill').fillna(0)

# Align with market data
prices = market_data['close'].iloc[:len(full_signals)]
returns = prices.pct_change()

# Calculate strategy returns (enter position on next bar)
positions = full_signals.shift(1).fillna(0)
strategy_returns = positions * returns

# Performance metrics
cumulative_returns = (1 + strategy_returns).cumprod()
total_return = cumulative_returns.iloc[-1] - 1

# Adjust annualization for timeframe
bars_per_day = 390 if timeframe == '1m' else 78 if timeframe == '5m' else 78
annualized_return = (1 + total_return) ** (252 * bars_per_day / len(prices)) - 1
sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252 * bars_per_day)
max_dd = (cumulative_returns / cumulative_returns.cummax() - 1).min()

print(f"Total Return: {total_return:.2%}")
print(f"Annualized Return: {annualized_return:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:.2%}")

# Trade statistics
position_changes = positions.diff().fillna(0)
trades = position_changes[position_changes != 0]
num_trades = len(trades) // 2  # Round trips

print(f"\nNumber of Trades: {num_trades}")
print(f"Avg Trades per Day: {num_trades / (len(market_data) / 78):.1f}")

# Signal statistics
print(f"\nSignal Distribution:")
print(f"Long signals: {(positions == 1).sum()} bars ({(positions == 1).mean():.1%})")
print(f"Short signals: {(positions == -1).sum()} bars ({(positions == -1).mean():.1%})")
print(f"Neutral: {(positions == 0).sum()} bars ({(positions == 0).mean():.1%})")

# Win/loss analysis
trade_returns = []
current_return = 0
entry_idx = None
entry_price = None
position_type = None

for i in range(len(positions)):
    if positions.iloc[i] != 0 and (i == 0 or positions.iloc[i-1] == 0):
        # Enter position
        entry_idx = i
        entry_price = prices.iloc[i]
        position_type = positions.iloc[i]  # 1 for long, -1 for short
    elif positions.iloc[i] == 0 and i > 0 and positions.iloc[i-1] != 0:
        # Exit position
        if entry_idx is not None and entry_price is not None:
            exit_price = prices.iloc[i]
            # Calculate return based on position type
            if position_type > 0:  # Long position
                trade_return = (exit_price / entry_price) - 1
            else:  # Short position
                trade_return = (entry_price / exit_price) - 1
            trade_returns.append(trade_return)

if trade_returns:
    trade_returns = pd.Series(trade_returns)
    win_rate = (trade_returns > 0).mean()
    avg_win = trade_returns[trade_returns > 0].mean() if any(trade_returns > 0) else 0
    avg_loss = trade_returns[trade_returns < 0].mean() if any(trade_returns < 0) else 0
    
    print(f"\nTrade Analysis:")
    print(f"Total Completed Trades: {len(trade_returns)}")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Number of Winners: {(trade_returns > 0).sum()}")
    print(f"Number of Losers: {(trade_returns < 0).sum()}")
    print(f"Avg Win: {avg_win:.3%}")
    print(f"Avg Loss: {avg_loss:.3%}")
    if avg_loss != 0:
        print(f"Win/Loss Ratio: {abs(avg_win/avg_loss):.2f}")
    
    # Best and worst trades
    print(f"\nBest Trade: {trade_returns.max():.2%}")
    print(f"Worst Trade: {trade_returns.min():.2%}")
    
    # Add distribution analysis
    print(f"\nTrade Distribution:")
    print(f"Trades > 0.1%: {(trade_returns > 0.001).sum()}")
    print(f"Trades < -0.1%: {(trade_returns < -0.001).sum()}")
    print(f"Flat trades (-0.1% to 0.1%): {((trade_returns >= -0.001) & (trade_returns <= 0.001)).sum()}")
