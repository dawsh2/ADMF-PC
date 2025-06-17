#!/usr/bin/env python3
"""
Cost-Optimized Ensemble Strategy Performance Analysis - Last 22k Bars (Out-of-Sample)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Constants
WORKSPACE_PATH = "/Users/daws/ADMF-PC/workspaces/duckdb_ensemble_cost_optimized_v1_014a539f"
DATA_PATH = "/Users/daws/ADMF-PC/data/SPY_1m.parquet"
ANALYSIS_BARS = 22000  # Out-of-sample period only
TRANSACTION_COST = 0.0001  # 0.01% per trade (0.5 bps each way)

def analyze_ensemble_performance():
    """Analyze the cost-optimized ensemble performance over last 22k bars."""
    
    # Load source data
    print("📊 Loading SPY data...")
    data = pd.read_parquet(DATA_PATH)
    
    # Ensure datetime index
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
    classifier_file = Path(WORKSPACE_PATH) / "traces/SPY_1m/classifiers/unknown/SPY_vol_mom_classifier.parquet"
    
    if not signal_file.exists():
        print(f"❌ Signal file not found: {signal_file}")
        return
    
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
    
    # Calculate returns with transaction costs
    print("\n💰 Calculating returns with transaction costs...")
    prices = analysis_data['close']
    price_returns = prices.pct_change()
    
    # Track position changes for transaction costs
    position_changes = signal_timeline.diff()
    trades = position_changes[position_changes != 0].dropna()
    num_trades = len(trades)
    
    # Calculate strategy returns
    strategy_returns = signal_timeline.shift(1) * price_returns
    
    # Apply transaction costs
    tc_series = pd.Series(index=signal_timeline.index, data=0.0)
    for timestamp in trades.index:
        tc_series[timestamp] = abs(trades[timestamp]) * TRANSACTION_COST
    
    # Net returns after transaction costs
    net_returns = strategy_returns - tc_series
    net_returns = net_returns.dropna()
    
    # Calculate performance metrics
    print("\n📊 Performance Metrics:")
    
    # Cumulative returns
    gross_cumulative = (1 + strategy_returns).cumprod() - 1
    net_cumulative = (1 + net_returns).cumprod() - 1
    market_return = (prices.iloc[-1] / prices.iloc[0]) - 1
    
    gross_total_return = gross_cumulative.iloc[-1]
    net_total_return = net_cumulative.iloc[-1]
    
    print(f"\n💵 RETURNS:")
    print(f"  Gross Return (before costs): {gross_total_return:.2%}")
    print(f"  Transaction Costs: {(gross_total_return - net_total_return):.2%}")
    print(f"  Net Return (after costs): {net_total_return:.2%}")
    print(f"  Market Return (SPY): {market_return:.2%}")
    print(f"  Outperformance: {(net_total_return - market_return):.2%}")
    
    # Risk metrics
    volatility = net_returns.std() * np.sqrt(252 * 390)  # Annualized for 1-min bars
    sharpe = (net_returns.mean() * 252 * 390) / volatility if volatility > 0 else 0
    
    # Drawdown
    cumulative = (1 + net_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    print(f"\n📈 RISK METRICS:")
    print(f"  Annualized Volatility: {volatility:.2%}")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_drawdown:.2%}")
    
    # Trade statistics
    print(f"\n🔄 TRADING ACTIVITY:")
    print(f"  Total Trades: {num_trades:,}")
    print(f"  Avg Trades per Day: {num_trades / (len(analysis_data) / 390):.1f}")
    
    # Win rate
    winning_days = net_returns[net_returns > 0]
    losing_days = net_returns[net_returns < 0]
    win_rate = len(winning_days) / (len(winning_days) + len(losing_days)) * 100
    
    print(f"\n🎯 WIN/LOSS STATISTICS:")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Avg Win: {winning_days.mean():.3%}" if len(winning_days) > 0 else "  Avg Win: N/A")
    print(f"  Avg Loss: {losing_days.mean():.3%}" if len(losing_days) > 0 else "  Avg Loss: N/A")
    
    # Signal distribution
    signal_dist = signal_timeline.value_counts()
    print(f"\n📊 SIGNAL DISTRIBUTION:")
    for signal, count in signal_dist.items():
        direction = "LONG" if signal == 1 else ("SHORT" if signal == -1 else "FLAT")
        pct = count / len(signal_timeline) * 100
        print(f"  {direction}: {pct:.1f}% of time")
    
    # Regime analysis if available
    if classifier_file.exists():
        print(f"\n🌍 REGIME ANALYSIS:")
        classifier_df = pd.read_parquet(classifier_file)
        classifier_filtered = classifier_df[classifier_df['idx'] >= analysis_start_idx]
        
        print(f"  Regime changes in period: {len(classifier_filtered)}")
        
        # Show regime distribution
        if len(classifier_filtered) > 0:
            regime_counts = classifier_filtered['val'].value_counts()
            print(f"\n  Regime occurrences:")
            for regime, count in regime_counts.items():
                print(f"    {regime}: {count} times")
    
    print("\n" + "="*80)
    print("✅ Analysis complete!")
    
    return {
        'net_return': net_total_return,
        'gross_return': gross_total_return, 
        'market_return': market_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'num_trades': num_trades,
        'win_rate': win_rate
    }

if __name__ == "__main__":
    results = analyze_ensemble_performance()