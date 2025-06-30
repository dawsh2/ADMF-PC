"""Analyze win rate improvements and max-bars limit strategy"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load source market data
market_df = pd.read_parquet('./data/SPY_1m.parquet')
market_df['timestamp'] = pd.to_datetime(market_df['timestamp'])
market_df = market_df.set_index('timestamp').sort_index()

# Workspace path
workspace = Path("workspaces/signal_generation_7ecda4b8")
signal_file = workspace / "traces/SPY_1m/signals/bollinger_rsi_simple_signals/SPY_compiled_strategy_0.parquet"

# Read the sparse signal data
signals_df = pd.read_parquet(signal_file)
signals_df['ts'] = pd.to_datetime(signals_df['ts'])

print("=== Win Rate Improvement Analysis ===")
print(f"Market data range: {market_df.index.min()} to {market_df.index.max()}")
print(f"Signal data range: {signals_df['ts'].min()} to {signals_df['ts'].max()}")

# Convert sparse signals to trades with full price data
trades = []
current_position = 0
entry_time = None
entry_price = None
entry_idx = None

for i in range(len(signals_df)):
    row = signals_df.iloc[i]
    new_signal = row['val']
    
    # Close existing position if changing
    if current_position != 0 and new_signal != current_position:
        if entry_time is not None:
            # Get market data between entry and exit
            exit_time = row['ts']
            mask = (market_df.index >= entry_time) & (market_df.index <= exit_time)
            trade_bars = market_df[mask]
            
            if len(trade_bars) > 0:
                # Track high/low during trade
                if current_position > 0:  # Long
                    high_price = trade_bars['high'].max()
                    low_price = trade_bars['low'].min()
                    max_favorable = (high_price / entry_price - 1) * 100
                    max_adverse = (low_price / entry_price - 1) * 100
                else:  # Short
                    high_price = trade_bars['high'].max()
                    low_price = trade_bars['low'].min()
                    max_favorable = (entry_price / low_price - 1) * 100
                    max_adverse = (entry_price / high_price - 1) * 100
                
                exit_price = row['px']
                pnl_pct = (exit_price / entry_price - 1) * current_position * 100
                bars_held = row['idx'] - entry_idx
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'direction': 'long' if current_position > 0 else 'short',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'bars_held': bars_held,
                    'max_favorable': max_favorable,
                    'max_adverse': max_adverse,
                    'num_bars': len(trade_bars)
                })
    
    # Update position
    if new_signal != 0 and new_signal != current_position:
        entry_time = row['ts']
        entry_price = row['px']
        entry_idx = row['idx']
    
    current_position = new_signal

trades_df = pd.DataFrame(trades)

if len(trades_df) > 0:
    print(f"\nTotal trades analyzed: {len(trades_df)}")
    
    # 1. Analyze trades by holding period to find optimal max bars
    print("\n=== Performance by Holding Period ===")
    print(f"{'Bars Held':<15} {'Count':<8} {'Avg Return':<12} {'Win Rate':<10} {'Cum Impact':<12}")
    print("-" * 60)
    
    # Group by bars held ranges
    bar_ranges = [(1, 5), (6, 10), (11, 20), (21, 30), (31, 50), (51, 100), (101, 200), (201, 10000)]
    
    cumulative_trades = 0
    cumulative_return = 0
    
    for min_bars, max_bars in bar_ranges:
        mask = (trades_df['bars_held'] >= min_bars) & (trades_df['bars_held'] <= max_bars)
        subset = trades_df[mask]
        
        if len(subset) > 0:
            avg_return = subset['pnl_pct'].mean()
            win_rate = (subset['pnl_pct'] > 0).mean()
            cumulative_trades += len(subset)
            cumulative_return += subset['pnl_pct'].sum()
            
            label = f"{min_bars}-{max_bars}" if max_bars < 10000 else f"{min_bars}+"
            print(f"{label:<15} {len(subset):<8} {avg_return:>10.3f}%  {win_rate:>8.1%}  {cumulative_return/cumulative_trades:>10.3f}%")
    
    # 2. Test max-bars exit strategies
    print("\n=== Max Bars Exit Strategy Performance ===")
    print(f"{'Max Bars':<10} {'Avg Return':<12} {'Win Rate':<10} {'Trades/Day':<12} {'Annual Net':<12} {'Sharpe':<8}")
    print("-" * 75)
    
    max_bar_limits = [None, 200, 100, 50, 30, 20, 15, 10, 5]
    
    date_range_days = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days
    base_trades_per_day = len(trades_df) / date_range_days
    exec_cost = 0.0001
    
    best_sharpe = -999
    best_max_bars = None
    
    for max_bars in max_bar_limits:
        # Apply max bars limit
        trades_limited = trades_df.copy()
        
        if max_bars is not None:
            # For trades exceeding max bars, calculate early exit
            long_trades = (trades_limited['bars_held'] > max_bars)
            
            if long_trades.any():
                # Estimate PnL if exited at max_bars
                # This is approximate - would need tick data for exact calculation
                trades_limited.loc[long_trades, 'pnl_pct'] = (
                    trades_limited.loc[long_trades, 'pnl_pct'] * 
                    (max_bars / trades_limited.loc[long_trades, 'bars_held'])
                )
                trades_limited.loc[long_trades, 'bars_held'] = max_bars
            
            max_bars_label = f"{max_bars}"
        else:
            max_bars_label = "None"
        
        # Calculate metrics
        avg_return = trades_limited['pnl_pct'].mean()
        win_rate = (trades_limited['pnl_pct'] > 0).mean()
        
        # Approximate trades per day (would increase with early exits)
        if max_bars is not None:
            avg_bars_reduction = trades_df['bars_held'].mean() / trades_limited['bars_held'].mean()
            est_trades_per_day = base_trades_per_day * avg_bars_reduction
        else:
            est_trades_per_day = base_trades_per_day
        
        trades_per_year = est_trades_per_day * 252
        
        # Annual return
        net_return_per_trade = (avg_return / 100) - (2 * exec_cost)
        if net_return_per_trade > -1:
            annual_net = (1 + net_return_per_trade) ** trades_per_year - 1
            annual_net_str = f"{annual_net*100:.1f}%"
        else:
            annual_net_str = "LOSS"
        
        # Sharpe calculation
        trades_limited['date'] = trades_limited['exit_time'].dt.date
        trades_limited['pnl_decimal_net'] = (trades_limited['pnl_pct'] / 100) - (2 * exec_cost)
        
        daily_returns = trades_limited.groupby('date')['pnl_decimal_net'].sum()
        date_range = pd.date_range(start=daily_returns.index.min(), 
                                  end=daily_returns.index.max(), 
                                  freq='D')
        daily_returns = daily_returns.reindex(date_range.date, fill_value=0)
        
        daily_mean = daily_returns.mean()
        daily_std = daily_returns.std()
        
        if daily_std > 0:
            sharpe = (daily_mean * 252) / (daily_std * np.sqrt(252))
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_max_bars = max_bars
        else:
            sharpe = 0
        
        print(f"{max_bars_label:<10} {avg_return:>10.3f}%  {win_rate:>8.1%}  {est_trades_per_day:>10.2f}  {annual_net_str:>11}  {sharpe:>6.2f}")
    
    if best_max_bars is not None:
        print(f"\n*** Best Sharpe ratio of {best_sharpe:.2f} achieved with {best_max_bars} max bars ***")
    
    # 3. Analyze why trades lose
    print("\n=== Loss Analysis ===")
    losing_trades = trades_df[trades_df['pnl_pct'] < 0]
    print(f"Total losing trades: {len(losing_trades)} ({len(losing_trades)/len(trades_df)*100:.1f}%)")
    
    # Trades that were winning but turned to losers
    turned_losers = losing_trades[losing_trades['max_favorable'] > 0.05]  # Had 5bp profit at some point
    print(f"Losing trades that were profitable (>5bp) at some point: {len(turned_losers)} ({len(turned_losers)/len(losing_trades)*100:.1f}%)")
    
    if len(turned_losers) > 0:
        print(f"Average max favorable excursion for these trades: {turned_losers['max_favorable'].mean():.3f}%")
        print(f"Average final loss for these trades: {turned_losers['pnl_pct'].mean():.3f}%")
    
    # 4. Filter analysis to improve win rate
    print("\n=== Win Rate Improvement Strategies ===")
    
    # Strategy 1: Exit when trade reaches small profit
    profit_targets = [0.02, 0.03, 0.05, 0.10]
    print("\nProfit Target Exit:")
    
    for target in profit_targets:
        # Estimate trades that would hit target
        trades_hit_target = trades_df[trades_df['max_favorable'] >= target]
        new_win_rate = len(trades_hit_target) / len(trades_df)
        print(f"  {target:.2f}% target: ~{new_win_rate*100:.1f}% win rate (from {(trades_df['pnl_pct'] > 0).mean()*100:.1f}%)")
    
    # Strategy 2: Combine stop loss with max bars
    print("\nCombined Strategy (0.005% stop + max bars):")
    for max_bars in [30, 20, 10]:
        trades_combined = trades_df.copy()
        
        # Apply stop loss
        trades_combined.loc[trades_combined['pnl_pct'] < -0.005, 'pnl_pct'] = -0.005
        
        # Apply max bars
        long_trades = (trades_combined['bars_held'] > max_bars)
        if long_trades.any():
            trades_combined.loc[long_trades, 'pnl_pct'] = (
                trades_combined.loc[long_trades, 'pnl_pct'] * 
                (max_bars / trades_combined.loc[long_trades, 'bars_held'])
            )
        
        win_rate = (trades_combined['pnl_pct'] > 0).mean()
        avg_return = trades_combined['pnl_pct'].mean()
        print(f"  Max {max_bars} bars: {win_rate*100:.1f}% win rate, {avg_return:.3f}% avg return")