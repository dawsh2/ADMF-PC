#!/usr/bin/env python3
"""
Corrected Cost-Optimized Ensemble Analysis - Last 22k Bars (Out-of-Sample)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Constants
WORKSPACE_PATH = "/Users/daws/ADMF-PC/workspaces/duckdb_ensemble_cost_optimized_v1_014a539f"
DATA_PATH = "/Users/daws/ADMF-PC/data/SPY_1m.parquet"
ANALYSIS_BARS = 22000  # Out-of-sample period only
TRANSACTION_COST = 0.00005  # 0.5 basis points (0.005%) per trade ONE WAY

def analyze_ensemble_corrected():
    """Corrected analysis with proper transaction cost logic."""
    
    # Load source data
    print("📊 Loading SPY data...")
    data = pd.read_parquet(DATA_PATH)
    
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.set_index('timestamp')
    
    data = data.sort_index()
    
    # Get last 22k bars
    total_bars = len(data)
    analysis_start_idx = max(0, total_bars - ANALYSIS_BARS)
    analysis_data = data.iloc[analysis_start_idx:].copy()
    
    print(f"✅ Total bars in dataset: {total_bars:,}")
    print(f"📈 Analyzing last {ANALYSIS_BARS:,} bars (out-of-sample)")
    print(f"🗓️  Period: {analysis_data.index[0]} to {analysis_data.index[-1]}")
    
    # Load ensemble signals
    signal_file = Path(WORKSPACE_PATH) / "traces/SPY_1m/signals/unknown/SPY_adaptive_ensemble_cost_optimized.parquet"
    
    print("\n🔄 Loading ensemble signals...")
    signals_df = pd.read_parquet(signal_file)
    print(f"✅ Total signal changes: {len(signals_df):,}")
    
    # Filter to analysis window
    signals_filtered = signals_df[signals_df['idx'] >= analysis_start_idx].copy()
    print(f"📊 Signals in analysis window: {len(signals_filtered):,}")
    
    # Reconstruct signal timeline
    print("\n🎬 Reconstructing signal timeline...")
    signal_timeline = pd.Series(index=analysis_data.index, dtype=float)
    signal_timeline[:] = 0.0  # Start flat
    
    for _, row in signals_filtered.iterrows():
        bar_idx = row['idx']
        signal_value = row['val']
        analysis_idx = bar_idx - analysis_start_idx
        
        if 0 <= analysis_idx < len(signal_timeline):
            signal_timeline.iloc[analysis_idx:] = signal_value
    
    # Calculate returns with CORRECTED transaction costs
    print("\n💰 Calculating returns with corrected transaction costs...")
    prices = analysis_data['close']
    price_returns = prices.pct_change()
    
    # Calculate strategy returns (before costs)
    positions = signal_timeline.shift(1)  # Positions are based on previous bar's signal
    strategy_returns = positions * price_returns
    
    # Track position changes for transaction costs
    position_changes = signal_timeline.diff()
    
    # Count actual trades and calculate transaction costs properly
    transaction_costs = 0
    total_trades = 0
    
    for i in range(1, len(signal_timeline)):
        prev_pos = signal_timeline.iloc[i-1]
        curr_pos = signal_timeline.iloc[i]
        
        if prev_pos != curr_pos:
            # Position changed
            position_size_change = abs(curr_pos - prev_pos)
            
            # Transaction cost is applied to the change in position
            # For a reversal from +1 to -1, position_size_change = 2
            # This correctly charges 2 * 0.005% = 0.01% (1 basis point)
            trade_cost = position_size_change * TRANSACTION_COST
            transaction_costs += trade_cost
            total_trades += 1
    
    # Apply transaction costs
    avg_tc_per_bar = transaction_costs / len(signal_timeline)
    net_returns = strategy_returns - avg_tc_per_bar
    
    # Remove NaN values
    strategy_returns = strategy_returns.dropna()
    net_returns = net_returns.dropna()
    
    # Calculate cumulative returns
    gross_total_return = (1 + strategy_returns).prod() - 1
    net_total_return = (1 + net_returns).prod() - 1
    market_return = (prices.iloc[-1] / prices.iloc[0]) - 1
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"\n💵 RETURNS:")
    print(f"  Gross Return (before costs): {gross_total_return:.2%}")
    print(f"  Total Transaction Costs: {transaction_costs:.2%} ({transaction_costs*100:.3f}%)")
    print(f"  Net Return (after costs): {net_total_return:.2%}")
    print(f"  Market Return (SPY): {market_return:.2%}")
    print(f"  Outperformance: {(net_total_return - market_return):.2%}")
    
    # Risk metrics
    volatility = net_returns.std() * np.sqrt(252 * 390)  # Annualized
    daily_returns = net_returns.resample('D').sum()  # Aggregate to daily
    
    if len(daily_returns) > 1:
        sharpe = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
    else:
        sharpe = 0
    
    # Drawdown calculation
    cumulative = (1 + net_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    print(f"\n📈 RISK METRICS:")
    print(f"  Annualized Volatility: {volatility:.2%}")
    print(f"  Sharpe Ratio (daily): {sharpe:.2f}")
    print(f"  Max Drawdown: {max_drawdown:.2%}")
    
    print(f"\n🔄 TRADING ACTIVITY:")
    print(f"  Total Trades: {total_trades:,}")
    print(f"  Avg Trades per Day: {total_trades / (len(analysis_data) / 390):.1f}")
    print(f"  Transaction Cost per Trade: {TRANSACTION_COST:.3%} one-way")
    
    # Win rate based on positive return bars
    positive_returns = net_returns[net_returns > 0]
    negative_returns = net_returns[net_returns < 0]
    
    if len(positive_returns) + len(negative_returns) > 0:
        win_rate = len(positive_returns) / (len(positive_returns) + len(negative_returns)) * 100
        print(f"\n🎯 WIN/LOSS STATISTICS:")
        print(f"  Win Rate (bars): {win_rate:.1f}%")
        if len(positive_returns) > 0:
            print(f"  Avg Win: {positive_returns.mean():.3%}")
        if len(negative_returns) > 0:
            print(f"  Avg Loss: {negative_returns.mean():.3%}")
    
    # Signal distribution
    signal_dist = signal_timeline.value_counts()
    print(f"\n📊 SIGNAL DISTRIBUTION:")
    for signal, count in signal_dist.items():
        direction = "LONG" if signal == 1 else ("SHORT" if signal == -1 else "FLAT")
        pct = count / len(signal_timeline) * 100
        print(f"  {direction}: {pct:.1f}% of time")
    
    # Sanity check on transaction costs
    print(f"\n✅ SANITY CHECK:")
    print(f"  Total trades: {total_trades}")
    print(f"  Expected max TC (if all reversals): {total_trades * 0.01:.2%}")
    print(f"  Expected min TC (if all entries/exits): {total_trades * 0.005:.2%}")
    print(f"  Actual total TC: {transaction_costs:.3%}")
    
    print("\n" + "="*80)
    print("✅ Analysis complete!")

if __name__ == "__main__":
    analyze_ensemble_corrected()