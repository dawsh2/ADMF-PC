#!/usr/bin/env python3
"""
Analyze trade patterns to identify potential filtering opportunities
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def load_price_data(workspace_path):
    """Load the original price data to calculate indicators."""
    # First, let's find a sample parquet file to get the price data structure
    sample_file = workspace_path / "traces" / "SPY_1m" / "signals" / "bollinger_bands" / "SPY_compiled_strategy_6.parquet"
    if sample_file.exists():
        df = pd.read_parquet(sample_file)
        print(f"Sample data structure:")
        print(df.head())
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nData range: {df['idx'].min()} to {df['idx'].max()}")
        return df
    return None

def analyze_entry_timing(trades_df, strategy_name):
    """Analyze when trades are entered (bar index patterns)."""
    print(f"\n{'='*60}")
    print(f"Entry Timing Analysis for {strategy_name}")
    print(f"{'='*60}")
    
    # Analyze entry bar distribution
    if 'entry_idx' in trades_df.columns:
        # Time of day analysis (assuming ~390 bars per day for regular hours)
        trades_df['bars_into_day'] = trades_df['entry_idx'] % 390
        trades_df['minutes_into_day'] = trades_df['bars_into_day']  # 1-minute bars
        
        # First/last 30 minutes
        first_30 = trades_df[trades_df['minutes_into_day'] < 30]
        last_30 = trades_df[trades_df['minutes_into_day'] > 360]
        middle = trades_df[(trades_df['minutes_into_day'] >= 30) & (trades_df['minutes_into_day'] <= 360)]
        
        print(f"\nTime of Day Analysis:")
        print(f"  First 30 min: {len(first_30)} trades, {first_30['pnl_pct'].mean():.3f}% avg PnL, {(first_30['pnl_pct'] > 0).mean():.1%} win rate")
        print(f"  Middle hours: {len(middle)} trades, {middle['pnl_pct'].mean():.3f}% avg PnL, {(middle['pnl_pct'] > 0).mean():.1%} win rate")
        print(f"  Last 30 min: {len(last_30)} trades, {last_30['pnl_pct'].mean():.3f}% avg PnL, {(last_30['pnl_pct'] > 0).mean():.1%} win rate")
        
        # Net returns after costs
        print(f"\nNet Returns (after 1bp costs):")
        print(f"  First 30 min: {first_30['pnl_pct'].sum() - len(first_30) * 0.01:.2f}%")
        print(f"  Middle hours: {middle['pnl_pct'].sum() - len(middle) * 0.01:.2f}%")
        print(f"  Last 30 min: {last_30['pnl_pct'].sum() - len(last_30) * 0.01:.2f}%")

def analyze_price_extremes(trades_df, signals_df, strategy_name):
    """Analyze relationship between entry price extremes and performance."""
    print(f"\n{'='*60}")
    print(f"Price Extreme Analysis for {strategy_name}")
    print(f"{'='*60}")
    
    # For each trade, calculate how far price was from recent average
    trades_with_context = []
    
    for idx, trade in trades_df.iterrows():
        entry_idx = int(trade['entry_idx'])
        
        # Get price context at entry
        entry_signal = signals_df[signals_df['idx'] == entry_idx]
        if len(entry_signal) > 0:
            entry_price = entry_signal.iloc[0]['px']
            
            # Calculate price deviation from recent prices
            # We'll use a simple approach: deviation from last 20 bar average
            recent_prices = signals_df[signals_df['idx'] <= entry_idx]['px'].tail(20)
            if len(recent_prices) > 0:
                avg_price = recent_prices.mean()
                price_deviation = (entry_price - avg_price) / avg_price * 100
                
                trade_context = trade.to_dict()
                trade_context['price_deviation'] = price_deviation
                trade_context['entry_price'] = entry_price
                trades_with_context.append(trade_context)
    
    if trades_with_context:
        context_df = pd.DataFrame(trades_with_context)
        
        # Bucket by deviation
        context_df['deviation_bucket'] = pd.cut(context_df['price_deviation'], 
                                                bins=[-np.inf, -0.2, -0.1, 0, 0.1, 0.2, np.inf],
                                                labels=['<-0.2%', '-0.2 to -0.1%', '-0.1 to 0%', '0 to 0.1%', '0.1 to 0.2%', '>0.2%'])
        
        print("\nPerformance by Price Deviation from 20-bar average:")
        for bucket in context_df['deviation_bucket'].unique():
            bucket_trades = context_df[context_df['deviation_bucket'] == bucket]
            if len(bucket_trades) > 0:
                avg_pnl = bucket_trades['pnl_pct'].mean()
                win_rate = (bucket_trades['pnl_pct'] > 0).mean()
                count = len(bucket_trades)
                net_return = bucket_trades['pnl_pct'].sum() - count * 0.01
                print(f"  {bucket}: {count:4d} trades, {avg_pnl:6.3f}% avg PnL, {win_rate:.1%} win rate, {net_return:6.2f}% net")

def analyze_consecutive_signals(signals_df, strategy_name):
    """Analyze rapid signal changes that might indicate noise."""
    print(f"\n{'='*60}")
    print(f"Signal Stability Analysis for {strategy_name}")
    print(f"{'='*60}")
    
    # Look for rapid signal changes
    signal_changes = []
    for i in range(1, len(signals_df)):
        prev_row = signals_df.iloc[i-1]
        curr_row = signals_df.iloc[i]
        
        bars_between = curr_row['idx'] - prev_row['idx']
        if bars_between < 10:  # Rapid change
            signal_changes.append({
                'bars_between': bars_between,
                'prev_signal': prev_row['val'],
                'curr_signal': curr_row['val'],
                'price_change': (curr_row['px'] - prev_row['px']) / prev_row['px'] * 100
            })
    
    if signal_changes:
        changes_df = pd.DataFrame(signal_changes)
        print(f"\nRapid signal changes (<10 bars apart): {len(changes_df)}")
        
        # Group by bars between
        for bars in sorted(changes_df['bars_between'].unique()):
            subset = changes_df[changes_df['bars_between'] == bars]
            avg_price_change = subset['price_change'].abs().mean()
            print(f"  {bars} bars apart: {len(subset)} occurrences, {avg_price_change:.3f}% avg price change")

def analyze_strategy_patterns(workspace_path, strategy_type, strategy_file):
    """Comprehensive pattern analysis for a strategy."""
    
    file_path = workspace_path / "traces" / "SPY_1m" / "signals" / strategy_type / f"{strategy_file}.parquet"
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return None
        
    # Load signals
    signals_df = pd.read_parquet(file_path)
    
    # Extract trades
    trades = []
    entry_price = None
    entry_signal = None
    entry_idx = None
    
    for _, row in signals_df.iterrows():
        signal = row['val']
        price = row['px']
        bar_idx = row['idx']
        
        if entry_price is None and signal != 0:
            entry_price = price
            entry_signal = signal
            entry_idx = bar_idx
        elif entry_price is not None and (signal == 0 or signal != entry_signal):
            if entry_signal > 0:
                pnl_pct = (price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - price) / entry_price * 100
            
            trades.append({
                'entry_idx': entry_idx,
                'exit_idx': bar_idx,
                'duration': bar_idx - entry_idx,
                'entry_price': entry_price,
                'exit_price': price,
                'pnl_pct': pnl_pct,
                'signal_type': 'long' if entry_signal > 0 else 'short'
            })
            
            if signal != 0:
                entry_price = price
                entry_signal = signal
                entry_idx = bar_idx
            else:
                entry_price = None
    
    if not trades:
        return None
        
    trades_df = pd.DataFrame(trades)
    
    # Run analyses
    analyze_entry_timing(trades_df, f"{strategy_type} - {strategy_file}")
    analyze_price_extremes(trades_df, signals_df, f"{strategy_type} - {strategy_file}")
    analyze_consecutive_signals(signals_df, f"{strategy_type} - {strategy_file}")
    
    return trades_df

def main():
    workspace_path = Path("/Users/daws/ADMF-PC/workspaces/signal_generation_45c186a6")
    
    # Load sample data first
    sample_data = load_price_data(workspace_path)
    
    # Analyze key strategies
    strategies_to_analyze = [
        ("bollinger_bands", "SPY_compiled_strategy_6"),  # High frequency, good returns
        ("pivot_bounces", "SPY_compiled_strategy_77"),   # Very high frequency
        ("vwap_deviation", "SPY_compiled_strategy_39"),  # Low frequency, high quality
        ("donchian_bands", "SPY_compiled_strategy_18"),  # Problematic strategy
    ]
    
    print("\n" + "="*60)
    print("PATTERN ANALYSIS ACROSS STRATEGIES")
    print("="*60)
    
    all_results = {}
    for strat_type, strat_file in strategies_to_analyze:
        print(f"\n\nAnalyzing {strat_type} - {strat_file}...")
        trades_df = analyze_strategy_patterns(workspace_path, strat_type, strat_file)
        if trades_df is not None:
            all_results[strat_type] = trades_df
    
    # Summary recommendations
    print("\n" + "="*60)
    print("DATA-DRIVEN RECOMMENDATIONS")
    print("="*60)
    
    print("\n1. TIME-OF-DAY FILTER:")
    print("   Look at the net returns by time period above")
    print("   If first/last 30 min are negative after costs, filter them")
    
    print("\n2. PRICE EXTREME FILTER:")
    print("   Check if extreme deviations (>0.2% from average) perform differently")
    print("   May indicate overextended moves that revert poorly")
    
    print("\n3. SIGNAL STABILITY:")
    print("   Rapid signal changes (<5 bars) may indicate noise")
    print("   Consider requiring signal persistence")

if __name__ == "__main__":
    main()