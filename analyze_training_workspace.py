"""Analyze Bollinger RSI Simple Signals on training data (before VWAP filter)"""
import pandas as pd
import numpy as np
from pathlib import Path

# Training workspace path
workspace = Path("workspaces/signal_generation_a033b74d")
signal_file = workspace / "traces/SPY_1m/signals/bollinger_rsi_simple_signals/SPY_compiled_strategy_0.parquet"

# Read the sparse signal data
signals_df = pd.read_parquet(signal_file)
signals_df['ts'] = pd.to_datetime(signals_df['ts'])

print("=== Bollinger RSI Simple Signals - TRAINING DATA Analysis ===")
print(f"Total signal changes: {len(signals_df)}")
print(f"Date range: {signals_df['ts'].min()} to {signals_df['ts'].max()}")
print(f"Total bars covered: 81,768")

# Convert sparse signals to trades
trades = []
current_position = 0

for i in range(len(signals_df)):
    row = signals_df.iloc[i]
    new_signal = row['val']
    
    if current_position != 0 and new_signal != current_position:
        entry_idx = i - 1
        entry_row = signals_df.iloc[entry_idx]
        
        # Calculate PnL
        entry_price = entry_row['px']
        exit_price = row['px']
        pnl_pct = (exit_price / entry_price - 1) * current_position * 100
        bars_held = row['idx'] - entry_row['idx']
        
        trades.append({
            'entry_time': entry_row['ts'],
            'exit_time': row['ts'],
            'entry_bar': entry_row['idx'],
            'exit_bar': row['idx'],
            'direction': 'long' if current_position > 0 else 'short',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'bars_held': bars_held,
            'hour': pd.to_datetime(entry_row['ts']).hour
        })
    
    current_position = new_signal

trades_df = pd.DataFrame(trades)

print(f"\nTotal completed trades: {len(trades_df)}")

if len(trades_df) > 0:
    # Overall performance
    print(f"\n=== Overall Performance (Training Data) ===")
    print(f"Average return per trade: {trades_df['pnl_pct'].mean():.3f}%")
    print(f"Win rate: {(trades_df['pnl_pct'] > 0).mean():.1%}")
    print(f"Average bars held: {trades_df['bars_held'].mean():.1f}")
    print(f"Std dev of returns: {trades_df['pnl_pct'].std():.3f}%")
    
    # Winners vs Losers
    winners = trades_df[trades_df['pnl_pct'] > 0]
    losers = trades_df[trades_df['pnl_pct'] < 0]
    
    print(f"\n=== Winners vs Losers ===")
    print(f"Winners: {len(winners)} trades, avg return: {winners['pnl_pct'].mean():.3f}%, avg bars: {winners['bars_held'].mean():.1f}")
    print(f"Losers: {len(losers)} trades, avg return: {losers['pnl_pct'].mean():.3f}%, avg bars: {losers['bars_held'].mean():.1f}")
    if len(losers) > 0 and len(winners) > 0:
        print(f"Losers hold {losers['bars_held'].mean() / winners['bars_held'].mean():.1f}x longer than winners")
    
    # Long vs Short
    longs = trades_df[trades_df['direction'] == 'long']
    shorts = trades_df[trades_df['direction'] == 'short']
    
    print(f"\n=== Long vs Short Performance ===")
    print(f"Longs: {len(longs)} trades ({len(longs)/len(trades_df)*100:.1f}%), avg: {longs['pnl_pct'].mean():.3f}%, win rate: {(longs['pnl_pct'] > 0).mean():.1%}")
    print(f"Shorts: {len(shorts)} trades ({len(shorts)/len(trades_df)*100:.1f}%), avg: {shorts['pnl_pct'].mean():.3f}%, win rate: {(shorts['pnl_pct'] > 0).mean():.1%}")
    
    # What-if with stop loss
    trades_with_stop = trades_df.copy()
    trades_with_stop.loc[trades_with_stop['pnl_pct'] < -0.1, 'pnl_pct'] = -0.1
    print(f"\n=== Impact of -0.1% Stop Loss ===")
    print(f"Without stop: {trades_df['pnl_pct'].mean():.3f}% avg return")
    print(f"With stop: {trades_with_stop['pnl_pct'].mean():.3f}% avg return")
    improvement = (trades_with_stop['pnl_pct'].mean() - trades_df['pnl_pct'].mean()) / abs(trades_df['pnl_pct'].mean()) * 100 if trades_df['pnl_pct'].mean() != 0 else 0
    print(f"Improvement: {improvement:.0f}%")
    
    # Calculate annualized returns
    # Estimate trading days
    date_range = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days
    trades_per_day = len(trades_df) / date_range if date_range > 0 else 0
    trades_per_year = trades_per_day * 252
    
    print(f"\n=== Annualized Return Projections ===")
    print(f"Date range: {date_range} days")
    print(f"Trades per day: {trades_per_day:.2f}")
    print(f"Projected trades per year: {trades_per_year:.0f}")
    
    # With 1bp execution cost
    exec_cost = 0.0002
    net_return = trades_df['pnl_pct'].mean() / 100 - exec_cost
    net_return_stop = trades_with_stop['pnl_pct'].mean() / 100 - exec_cost
    
    if net_return > 0:
        annual_net = (1 + net_return) ** trades_per_year - 1
        print(f"\nWithout stop (1bp cost): {annual_net*100:.1f}% annual")
    else:
        print(f"\nWithout stop (1bp cost): UNPROFITABLE")
    
    if net_return_stop > 0:
        annual_net_stop = (1 + net_return_stop) ** trades_per_year - 1
        print(f"With stop (1bp cost): {annual_net_stop*100:.1f}% annual")
    
    # Compare with test data results
    print(f"\n=== Training vs Test Comparison ===")
    print(f"Training avg return: {trades_df['pnl_pct'].mean():.3f}%")
    print(f"Test avg return: 0.015% (from our analysis)")
    print(f"Training seems {'better' if trades_df['pnl_pct'].mean() > 0.015 else 'worse'} than test")
    
    # Distribution analysis
    print(f"\n=== Return Distribution ===")
    print(f"Best trade: {trades_df['pnl_pct'].max():.3f}%")
    print(f"Worst trade: {trades_df['pnl_pct'].min():.3f}%")
    print(f"25th percentile: {trades_df['pnl_pct'].quantile(0.25):.3f}%")
    print(f"75th percentile: {trades_df['pnl_pct'].quantile(0.75):.3f}%")
    
    # How many trades would benefit from VWAP filter?
    print(f"\n=== Potential VWAP Filter Impact ===")
    shorts_below_01 = shorts[shorts['pnl_pct'] < -0.1]
    print(f"Shorts with >0.1% loss: {len(shorts_below_01)} trades")
    print(f"These represent {len(shorts_below_01)/len(shorts)*100:.1f}% of all shorts")
    print(f"Average loss of these bad shorts: {shorts_below_01['pnl_pct'].mean():.3f}%")
    print(f"\nIf VWAP filter removes ~63% of shorts (as in test data):")
    print(f"Would remove ~{int(len(shorts) * 0.63)} shorts")
    print(f"Keeping ~{int(len(shorts) * 0.37)} shorts")