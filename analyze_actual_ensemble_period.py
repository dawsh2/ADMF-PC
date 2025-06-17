#!/usr/bin/env python3
"""
Analyze the actual period that was processed in the restored ensemble.
"""

import pandas as pd
import numpy as np
from pathlib import Path

WORKSPACE_PATH = "/Users/daws/ADMF-PC/workspaces/duckdb_ensemble_v1_c6dcf7c0"
DATA_PATH = "/Users/daws/ADMF-PC/data/SPY_1m.parquet"
TRANSACTION_COST = 0.00005  # 0.5 basis points per trade ONE WAY

def analyze_actual_period():
    """Analyze the actual period processed by the ensemble."""
    
    print("="*80)
    print("📊 RESTORED ENSEMBLE ANALYSIS - Actual Period Processed")
    print("="*80)
    
    # Load full SPY data
    data = pd.read_parquet(DATA_PATH)
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.set_index('timestamp')
    data = data.sort_index()
    
    # Load signal file
    signal_file = Path(WORKSPACE_PATH) / "traces/SPY_1m/signals/unknown/SPY_adaptive_ensemble_default.parquet"
    signals_df = pd.read_parquet(signal_file)
    
    # Get the actual data that was processed
    min_idx = signals_df['idx'].min()
    max_idx = signals_df['idx'].max()
    actual_data = data.iloc[min_idx:max_idx+1].copy()
    
    print(f"\n📅 ACTUAL PERIOD ANALYZED:")
    print(f"  Bar indices: {min_idx:,} to {max_idx:,}")
    print(f"  Total bars: {len(actual_data):,}")
    print(f"  Period: {actual_data.index[0]} to {actual_data.index[-1]}")
    print(f"  Days: {(actual_data.index[-1] - actual_data.index[0]).days}")
    
    # Reconstruct signal timeline for actual period
    print("\n🎬 Reconstructing signal timeline...")
    signal_timeline = pd.Series(index=actual_data.index, dtype=float)
    signal_timeline[:] = 0.0  # Start flat
    
    for _, row in signals_df.iterrows():
        bar_idx = row['idx']
        signal_value = row['val']
        # Convert global index to local index
        local_idx = bar_idx - min_idx
        
        if 0 <= local_idx < len(signal_timeline):
            signal_timeline.iloc[local_idx:] = signal_value
    
    # Calculate trades and holding periods
    position_changes = signal_timeline.diff()
    trades = position_changes[position_changes != 0].dropna()
    num_trades = len(trades)
    
    # Calculate holding periods
    if num_trades > 1:
        trade_indices = trades.index
        holding_periods = []
        for i in range(len(trade_indices)-1):
            bars_held = len(signal_timeline.loc[trade_indices[i]:trade_indices[i+1]]) - 1
            holding_periods.append(bars_held)
        
        avg_holding_period = np.mean(holding_periods)
        median_holding_period = np.median(holding_periods)
        min_holding_period = np.min(holding_periods)
        max_holding_period = np.max(holding_periods)
    else:
        avg_holding_period = median_holding_period = min_holding_period = max_holding_period = 0
    
    # Calculate returns
    prices = actual_data['close']
    price_returns = prices.pct_change()
    positions = signal_timeline.shift(1)
    strategy_returns = positions * price_returns
    
    # Apply transaction costs
    transaction_costs = 0
    for i in range(1, len(signal_timeline)):
        prev_pos = signal_timeline.iloc[i-1]
        curr_pos = signal_timeline.iloc[i]
        
        if prev_pos != curr_pos:
            position_size_change = abs(curr_pos - prev_pos)
            trade_cost = position_size_change * TRANSACTION_COST
            transaction_costs += trade_cost
    
    avg_tc_per_bar = transaction_costs / len(signal_timeline)
    net_returns = strategy_returns - avg_tc_per_bar
    
    # Performance metrics
    strategy_returns = strategy_returns.dropna()
    net_returns = net_returns.dropna()
    
    gross_total_return = (1 + strategy_returns).prod() - 1
    net_total_return = (1 + net_returns).prod() - 1
    market_return = (prices.iloc[-1] / prices.iloc[0]) - 1
    
    # Risk metrics
    volatility = net_returns.std() * np.sqrt(252 * 390)
    daily_returns = net_returns.resample('D').sum()
    
    if len(daily_returns) > 1:
        sharpe = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
    else:
        sharpe = 0
    
    # Drawdown
    cumulative = (1 + net_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win rate
    positive_returns = net_returns[net_returns > 0]
    negative_returns = net_returns[net_returns < 0]
    win_rate = len(positive_returns) / (len(positive_returns) + len(negative_returns)) * 100 if (len(positive_returns) + len(negative_returns)) > 0 else 0
    
    # Signal distribution
    signal_dist = signal_timeline.value_counts()
    
    # Print results
    print("\n" + "="*60)
    print("📊 PERFORMANCE SUMMARY - RESTORED ENSEMBLE")
    print("="*60)
    
    print(f"\n💵 RETURNS:")
    print(f"  Gross Return: {gross_total_return:.2%}")
    print(f"  Transaction Costs: {transaction_costs:.3%}")
    print(f"  Net Return: {net_total_return:.2%}")
    print(f"  Market Return: {market_return:.2%}")
    print(f"  Outperformance: {(net_total_return - market_return):.2%}")
    
    print(f"\n📈 RISK METRICS:")
    print(f"  Volatility: {volatility:.2%}")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_drawdown:.2%}")
    print(f"  Win Rate: {win_rate:.1f}%")
    
    print(f"\n🔄 TRADING ACTIVITY:")
    print(f"  Total Trades: {num_trades:,}")
    print(f"  Trades per 1000 bars: {num_trades / len(actual_data) * 1000:.1f}")
    print(f"  Avg Holding Period: {avg_holding_period:.1f} bars")
    print(f"  Median Hold: {median_holding_period:.0f} bars")
    print(f"  Min/Max Hold: {min_holding_period:.0f} / {max_holding_period:.0f} bars")
    
    print(f"\n📊 SIGNAL DISTRIBUTION:")
    for signal, count in signal_dist.items():
        direction = "LONG" if signal == 1 else ("SHORT" if signal == -1 else "FLAT")
        pct = count / len(signal_timeline) * 100
        print(f"  {direction}: {pct:.1f}% of time")
    
    # Compare with cost-optimized
    print("\n" + "="*60)
    print("📊 COMPARISON:")
    print("="*60)
    print("\nCost-Optimized (22k bars): -33.22% net, 112 trades/day")
    print(f"Restored ({len(actual_data):,} bars): {net_total_return:.2%} net, {num_trades / len(actual_data) * 390:.1f} trades/day")
    
    # Check regime changes
    classifier_file = Path(WORKSPACE_PATH) / "traces/SPY_1m/classifiers/unknown/SPY_vol_mom_classifier.parquet"
    if classifier_file.exists():
        classifier_df = pd.read_parquet(classifier_file)
        regime_changes = len(classifier_df)
        print(f"\nRegime changes: {regime_changes} ({regime_changes / len(actual_data) * 1000:.1f} per 1000 bars)")

if __name__ == "__main__":
    analyze_actual_period()