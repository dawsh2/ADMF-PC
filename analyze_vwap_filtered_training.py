"""Analyze VWAP-filtered training data and compare with non-filtered"""
import pandas as pd
import numpy as np
from pathlib import Path

# VWAP-filtered workspace
filtered_workspace = Path("workspaces/signal_generation_e73fe3c9")
filtered_signal_file = filtered_workspace / "traces/SPY_1m/signals/bollinger_rsi_simple_signals/SPY_compiled_strategy_0.parquet"

# Read the sparse signal data
signals_df = pd.read_parquet(filtered_signal_file)
signals_df['ts'] = pd.to_datetime(signals_df['ts'])

print("=== VWAP-Filtered Bollinger RSI - TRAINING DATA Analysis ===")
print(f"Total signal changes: {len(signals_df)} (vs 1698 without filter)")
print(f"Reduction in signals: {(1698 - len(signals_df))/1698*100:.1f}%")
print(f"Date range: {signals_df['ts'].min()} to {signals_df['ts'].max()}")

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
            'direction': 'long' if current_position > 0 else 'short',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'bars_held': bars_held
        })
    
    current_position = new_signal

trades_df = pd.DataFrame(trades)

print(f"\nTotal completed trades: {len(trades_df)}")

if len(trades_df) > 0:
    # Load non-filtered trades for comparison
    no_filter_trades = pd.read_csv('training_trades_no_vwap.csv')
    no_filter_trades['entry_time'] = pd.to_datetime(no_filter_trades['entry_time'])
    
    # Overall performance comparison
    print(f"\n=== Performance Comparison ===")
    print(f"{'Metric':<30} {'No Filter':<15} {'VWAP Filter':<15} {'Change':<15}")
    print("-" * 75)
    
    # Average return
    avg_no_filter = no_filter_trades['pnl_pct'].mean()
    avg_filtered = trades_df['pnl_pct'].mean()
    print(f"{'Average return per trade':<30} {avg_no_filter:>10.3f}%    {avg_filtered:>10.3f}%    {avg_filtered - avg_no_filter:>+10.3f}%")
    
    # Trade count
    print(f"{'Total trades':<30} {len(no_filter_trades):>10}       {len(trades_df):>10}       {len(trades_df) - len(no_filter_trades):>+10}")
    
    # Win rate
    wr_no_filter = (no_filter_trades['pnl_pct'] > 0).mean()
    wr_filtered = (trades_df['pnl_pct'] > 0).mean()
    print(f"{'Win rate':<30} {wr_no_filter*100:>10.1f}%      {wr_filtered*100:>10.1f}%      {(wr_filtered - wr_no_filter)*100:>+10.1f}%")
    
    # By direction
    print(f"\n=== Direction Breakdown ===")
    for direction in ['long', 'short']:
        nf_dir = no_filter_trades[no_filter_trades['direction'] == direction]
        f_dir = trades_df[trades_df['direction'] == direction]
        
        print(f"\n{direction.capitalize()}s:")
        print(f"  Count: {len(nf_dir)} → {len(f_dir)} ({len(f_dir) - len(nf_dir):+d})")
        print(f"  Avg return: {nf_dir['pnl_pct'].mean():.3f}% → {f_dir['pnl_pct'].mean():.3f}% ({f_dir['pnl_pct'].mean() - nf_dir['pnl_pct'].mean():+.3f}%)")
        print(f"  Win rate: {(nf_dir['pnl_pct'] > 0).mean()*100:.1f}% → {(f_dir['pnl_pct'] > 0).mean()*100:.1f}%")
    
    # Which trades were filtered out?
    print(f"\n=== Filter Analysis ===")
    shorts_removed = len(no_filter_trades[no_filter_trades['direction'] == 'short']) - len(trades_df[trades_df['direction'] == 'short'])
    longs_removed = len(no_filter_trades[no_filter_trades['direction'] == 'long']) - len(trades_df[trades_df['direction'] == 'long'])
    
    print(f"Shorts removed: {shorts_removed} ({shorts_removed/len(no_filter_trades[no_filter_trades['direction'] == 'short'])*100:.1f}%)")
    print(f"Longs removed: {longs_removed} ({longs_removed/len(no_filter_trades[no_filter_trades['direction'] == 'long'])*100:.1f}% - should be 0)")
    
    # Stop loss analysis
    print(f"\n=== With 0.01% Stop Loss ===")
    trades_with_stop = trades_df.copy()
    trades_with_stop.loc[trades_with_stop['pnl_pct'] < -0.01, 'pnl_pct'] = -0.01
    
    print(f"VWAP filtered avg: {trades_df['pnl_pct'].mean():.3f}%")
    print(f"VWAP + 0.01% stop: {trades_with_stop['pnl_pct'].mean():.3f}%")
    print(f"Trades hitting stop: {len(trades_df[trades_df['pnl_pct'] < -0.01])}")
    
    # Annualized returns
    date_range = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days
    trades_per_day = len(trades_df) / date_range if date_range > 0 else 0
    trades_per_year = trades_per_day * 252
    
    print(f"\n=== Annualized Projections (1bp cost) ===")
    print(f"Trades per year: {trades_per_year:.0f}")
    
    # Calculate for different scenarios
    scenarios = [
        ("No filter, no stop", avg_no_filter, 699),  # Original trades per year
        ("VWAP filter only", avg_filtered, trades_per_year),
        ("VWAP + 0.01% stop", trades_with_stop['pnl_pct'].mean(), trades_per_year)
    ]
    
    print(f"\n{'Scenario':<20} {'Avg Return':<12} {'Annual (1bp)':<15}")
    print("-" * 47)
    
    exec_cost = 0.0002
    for scenario, avg_ret, tpy in scenarios:
        net_return = avg_ret / 100 - exec_cost
        if net_return > 0:
            annual_net = (1 + net_return) ** tpy - 1
            annual_str = f"{annual_net*100:.1f}%"
        else:
            annual_str = "LOSS"
        print(f"{scenario:<20} {avg_ret:>8.3f}%    {annual_str:>14}")
    
    # Best/worst trades
    print(f"\n=== Trade Quality ===")
    print(f"Best trade (no filter): {no_filter_trades['pnl_pct'].max():.3f}%")
    print(f"Best trade (filtered): {trades_df['pnl_pct'].max():.3f}%")
    print(f"Worst trade (no filter): {no_filter_trades['pnl_pct'].min():.3f}%")
    print(f"Worst trade (filtered): {trades_df['pnl_pct'].min():.3f}%")
    
    # Summary
    print(f"\n=== KEY FINDINGS ===")
    print(f"1. VWAP filter removed {shorts_removed} shorts ({shorts_removed/len(no_filter_trades[no_filter_trades['direction'] == 'short'])*100:.1f}%)")
    print(f"2. Average return {'improved' if avg_filtered > avg_no_filter else 'worsened'} from {avg_no_filter:.3f}% to {avg_filtered:.3f}%")
    print(f"3. With 0.01% stop, VWAP-filtered strategy achieves {trades_with_stop['pnl_pct'].mean():.3f}% avg return")
    
    # Save filtered trades
    trades_df.to_csv('training_trades_with_vwap.csv', index=False)
    print(f"\nSaved {len(trades_df)} VWAP-filtered trades to training_trades_with_vwap.csv")