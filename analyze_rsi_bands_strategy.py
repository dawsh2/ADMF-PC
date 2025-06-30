"""Analyze RSI Bands strategy performance"""
import pandas as pd
import numpy as np
from pathlib import Path

# Workspace path
workspace = Path("workspaces/signal_generation_4119862e")
signal_file = workspace / "traces/SPY_1m/signals/rsi_bands/SPY_compiled_strategy_0.parquet"

# Read the sparse signal data
signals_df = pd.read_parquet(signal_file)
signals_df['ts'] = pd.to_datetime(signals_df['ts'])

print("=== RSI Bands Strategy Analysis ===")
print(f"Workspace: signal_generation_4119862e")
print(f"Strategy Type: RSI Bands")
print(f"Date range: {signals_df['ts'].min()} to {signals_df['ts'].max()}")
print(f"Total signal changes: {len(signals_df)}")

# Convert sparse signals to trades
trades = []
current_position = 0

for i in range(len(signals_df)):
    row = signals_df.iloc[i]
    new_signal = row['val']
    
    # Close existing position if changing
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
            'direction': 'long' if current_position > 0 else 'short',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'bars_held': bars_held,
            'pnl_decimal': pnl_pct / 100
        })
    
    current_position = new_signal

trades_df = pd.DataFrame(trades)

if len(trades_df) > 0:
    # Basic metrics
    print(f"\n=== PERFORMANCE METRICS ===")
    total_trades = len(trades_df)
    avg_return = trades_df['pnl_pct'].mean()
    win_rate = (trades_df['pnl_pct'] > 0).mean()
    
    print(f"Total completed trades: {total_trades}")
    print(f"Average return per trade: {avg_return:.4f}%")
    print(f"Win rate: {win_rate:.1%}")
    print(f"Avg bars held: {trades_df['bars_held'].mean():.1f}")
    
    # By direction
    longs = trades_df[trades_df['direction'] == 'long']
    shorts = trades_df[trades_df['direction'] == 'short']
    
    print(f"\nLongs: {len(longs)} trades, avg: {longs['pnl_pct'].mean():.4f}%, win rate: {(longs['pnl_pct'] > 0).mean():.1%}")
    print(f"Shorts: {len(shorts)} trades, avg: {shorts['pnl_pct'].mean():.4f}%, win rate: {(shorts['pnl_pct'] > 0).mean():.1%}")
    
    # Calculate annualized metrics
    date_range_days = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days
    trades_per_day = total_trades / date_range_days if date_range_days > 0 else 0
    trades_per_year = trades_per_day * 252
    
    print(f"\n=== ANNUALIZED METRICS ===")
    print(f"Trading period: {date_range_days} days")
    print(f"Trades per day: {trades_per_day:.2f}")
    print(f"Trades per year: {trades_per_year:.0f}")
    
    # No cost returns
    avg_return_decimal = avg_return / 100
    if avg_return_decimal > -1:
        annual_return_gross = (1 + avg_return_decimal) ** trades_per_year - 1
        print(f"\nAnnualized return (no costs): {annual_return_gross*100:.1f}%")
    
    # With various execution costs
    print(f"\nWith execution costs:")
    for cost_bp in [0.5, 1.0, 2.0]:
        exec_cost = cost_bp / 10000
        net_return_per_trade = avg_return_decimal - (2 * exec_cost)
        if net_return_per_trade > -1:
            annual_return_net = (1 + net_return_per_trade) ** trades_per_year - 1
            print(f"  {cost_bp}bp cost: {annual_return_net*100:.1f}%")
        else:
            print(f"  {cost_bp}bp cost: LOSS")
    
    # Risk metrics
    print(f"\n=== RISK METRICS ===")
    print(f"Best trade: {trades_df['pnl_pct'].max():.3f}%")
    print(f"Worst trade: {trades_df['pnl_pct'].min():.3f}%")
    print(f"Std dev of returns: {trades_df['pnl_pct'].std():.3f}%")
    
    # Calculate Sharpe
    trades_df['date'] = trades_df['exit_time'].dt.date
    daily_returns = trades_df.groupby('date')['pnl_decimal'].sum()
    
    date_range = pd.date_range(start=daily_returns.index.min(), 
                              end=daily_returns.index.max(), 
                              freq='D')
    daily_returns = daily_returns.reindex(date_range.date, fill_value=0)
    
    daily_mean = daily_returns.mean()
    daily_std = daily_returns.std()
    
    if daily_std > 0:
        sharpe = (daily_mean * 252) / (daily_std * np.sqrt(252))
        print(f"Sharpe ratio (no costs): {sharpe:.2f}")
    
    # Return distribution
    print(f"\n=== RETURN DISTRIBUTION ===")
    percentiles = [5, 25, 50, 75, 95]
    for p in percentiles:
        print(f"{p}th percentile: {trades_df['pnl_pct'].quantile(p/100):.3f}%")
    
    # Holding period analysis
    print(f"\n=== HOLDING PERIOD ANALYSIS ===")
    bar_ranges = [(1, 10), (11, 30), (31, 60), (61, 120), (121, 10000)]
    for min_bars, max_bars in bar_ranges:
        mask = (trades_df['bars_held'] >= min_bars) & (trades_df['bars_held'] <= max_bars)
        subset = trades_df[mask]
        if len(subset) > 0:
            label = f"{min_bars}-{max_bars}" if max_bars < 10000 else f"{min_bars}+"
            print(f"{label:<10} bars: {len(subset):>4} trades, avg: {subset['pnl_pct'].mean():>7.3f}%, win rate: {(subset['pnl_pct'] > 0).mean():>6.1%}")
    
    # Compare with Bollinger RSI strategy
    print(f"\n=== COMPARISON WITH BOLLINGER RSI SIMPLE SIGNALS ===")
    print(f"                        RSI Bands    Bollinger RSI")
    print(f"Trades per year:        {trades_per_year:>9.0f}    699")
    print(f"Avg return per trade:   {avg_return:>9.4f}%    0.0078%")
    print(f"Win rate:               {win_rate*100:>9.1f}%    65.7%")
    print(f"Annual (no costs):      {annual_return_gross*100:>9.1f}%    8.3%")
    
else:
    print("\nNo completed trades found in the signal data.")

# Look at signal distribution
print(f"\n=== SIGNAL DISTRIBUTION ===")
signal_counts = signals_df['val'].value_counts().sort_index()
print(f"Signal values: {signal_counts.to_dict()}")

# Save trades for further analysis
if len(trades_df) > 0:
    trades_df.to_csv('rsi_bands_trades.csv', index=False)
    print(f"\nSaved {len(trades_df)} trades to 'rsi_bands_trades.csv'")