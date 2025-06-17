#!/usr/bin/env python3
"""
Analyze the restored ensemble strategy performance with all strategies added back.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pyarrow.parquet as pq

# Constants
WORKSPACE_PATH = "/Users/daws/ADMF-PC/workspaces/duckdb_ensemble_v1_c6dcf7c0"
DATA_PATH = "/Users/daws/ADMF-PC/data/SPY_1m.parquet"
ANALYSIS_BARS = 22000  # Out-of-sample period
TRANSACTION_COST = 0.00005  # 0.5 basis points per trade ONE WAY

def analyze_restored_ensemble():
    """Analyze the restored ensemble with all strategies added back."""
    
    print("="*80)
    print("📊 RESTORED ENSEMBLE ANALYSIS - With All Strategies Added Back")
    print("="*80)
    
    # Load source data
    print("\n📊 Loading SPY data...")
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
    
    # Find signal files
    traces_dir = Path(WORKSPACE_PATH) / "traces/SPY_1m"
    
    # Look for ensemble signal file
    signal_files = list(traces_dir.rglob("*/SPY_adaptive_ensemble_default.parquet"))
    if not signal_files:
        print("\n❌ No ensemble signal files found!")
        print(f"Searched in: {traces_dir}")
        # List what's actually there
        all_parquet = list(traces_dir.rglob("*.parquet"))
        if all_parquet:
            print("\nFound these files instead:")
            for f in all_parquet[:10]:
                print(f"  - {f.relative_to(traces_dir)}")
        return
    
    signal_file = signal_files[0]
    print(f"\n🔄 Loading ensemble signals from: {signal_file.relative_to(WORKSPACE_PATH)}")
    
    # Load signals
    signals_df = pd.read_parquet(signal_file)
    print(f"✅ Total signal changes: {len(signals_df):,}")
    
    # Get metadata from parquet file
    try:
        parquet_file = pq.ParquetFile(signal_file)
        metadata = parquet_file.metadata
        if metadata and metadata.metadata:
            print("\n📋 Signal File Metadata:")
            for key, value in metadata.metadata.items():
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                if key in ['total_bars', 'total_changes', 'compression_ratio']:
                    print(f"  {key}: {value}")
    except Exception as e:
        print(f"Could not read metadata: {e}")
    
    # Filter to analysis window
    signals_filtered = signals_df[signals_df['idx'] >= analysis_start_idx].copy()
    print(f"\n📊 Signals in analysis window: {len(signals_filtered):,}")
    
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
    
    # Calculate position changes and trades
    position_changes = signal_timeline.diff()
    trades = position_changes[position_changes != 0].dropna()
    num_trades = len(trades)
    
    # Calculate average holding period
    if num_trades > 1:
        # Calculate bars between trades
        trade_indices = trades.index
        holding_periods = []
        for i in range(len(trade_indices)-1):
            bars_held = len(signal_timeline.loc[trade_indices[i]:trade_indices[i+1]]) - 1
            holding_periods.append(bars_held)
        
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0
        median_holding_period = np.median(holding_periods) if holding_periods else 0
        min_holding_period = np.min(holding_periods) if holding_periods else 0
        max_holding_period = np.max(holding_periods) if holding_periods else 0
    else:
        avg_holding_period = median_holding_period = min_holding_period = max_holding_period = 0
    
    # Calculate returns with transaction costs
    print("\n💰 Calculating returns with transaction costs...")
    prices = analysis_data['close']
    price_returns = prices.pct_change()
    
    # Calculate strategy returns (before costs)
    positions = signal_timeline.shift(1)
    strategy_returns = positions * price_returns
    
    # Apply transaction costs properly
    transaction_costs = 0
    for i in range(1, len(signal_timeline)):
        prev_pos = signal_timeline.iloc[i-1]
        curr_pos = signal_timeline.iloc[i]
        
        if prev_pos != curr_pos:
            position_size_change = abs(curr_pos - prev_pos)
            trade_cost = position_size_change * TRANSACTION_COST
            transaction_costs += trade_cost
    
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
    
    if len(positive_returns) + len(negative_returns) > 0:
        win_rate = len(positive_returns) / (len(positive_returns) + len(negative_returns)) * 100
    else:
        win_rate = 0
    
    # Signal distribution
    signal_dist = signal_timeline.value_counts()
    
    # Print results
    print("\n" + "="*60)
    print("📊 PERFORMANCE SUMMARY - RESTORED ENSEMBLE")
    print("="*60)
    
    print(f"\n💵 RETURNS:")
    print(f"  Gross Return (before costs): {gross_total_return:.2%}")
    print(f"  Transaction Costs: {transaction_costs:.3%} total")
    print(f"  Net Return (after costs): {net_total_return:.2%}")
    print(f"  Market Return (SPY): {market_return:.2%}")
    print(f"  Outperformance: {(net_total_return - market_return):.2%}")
    
    print(f"\n📈 RISK METRICS:")
    print(f"  Annualized Volatility: {volatility:.2%}")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_drawdown:.2%}")
    print(f"  Win Rate: {win_rate:.1f}%")
    
    print(f"\n🔄 TRADING ACTIVITY:")
    print(f"  Total Trades: {num_trades:,}")
    print(f"  Avg Trades per Day: {num_trades / (len(analysis_data) / 390):.1f}")
    print(f"  Avg Holding Period: {avg_holding_period:.1f} bars ({avg_holding_period:.1f} minutes)")
    print(f"  Median Holding Period: {median_holding_period:.0f} bars")
    print(f"  Min/Max Hold: {min_holding_period:.0f} / {max_holding_period:.0f} bars")
    
    print(f"\n📊 SIGNAL DISTRIBUTION:")
    for signal, count in signal_dist.items():
        direction = "LONG" if signal == 1 else ("SHORT" if signal == -1 else "FLAT")
        pct = count / len(signal_timeline) * 100
        print(f"  {direction}: {pct:.1f}% of time")
    
    # Check classifier file
    classifier_files = list(traces_dir.rglob("*/SPY_vol_mom_classifier.parquet"))
    if classifier_files:
        classifier_file = classifier_files[0]
        print(f"\n🌍 REGIME ANALYSIS:")
        classifier_df = pd.read_parquet(classifier_file)
        classifier_filtered = classifier_df[classifier_df['idx'] >= analysis_start_idx]
        
        print(f"  Regime changes in period: {len(classifier_filtered)}")
        
        if len(classifier_filtered) > 0:
            regime_counts = classifier_filtered['val'].value_counts()
            print(f"\n  Regime distribution:")
            for regime, count in regime_counts.items():
                print(f"    {regime}: {count} occurrences")
    
    print("\n" + "="*60)
    print("✅ Analysis complete!")
    print("="*60)
    
    # Compare with cost-optimized version
    print("\n📊 COMPARISON WITH COST-OPTIMIZED VERSION:")
    print("  Cost-Optimized: -33.22% net return, 112 trades/day")
    print(f"  Restored: {net_total_return:.2%} net return, {num_trades / (len(analysis_data) / 390):.1f} trades/day")
    if avg_holding_period > 0:
        print(f"  Improvement: Holding period {avg_holding_period:.1f} min (restored) vs ~13 min (cost-opt)")

if __name__ == "__main__":
    analyze_restored_ensemble()