#!/usr/bin/env python3
"""
Analyze Strategy 20 with full OHLC stop loss testing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Import the analyzer class
sys.path.append('/Users/daws/ADMF-PC')
from analyze_keltner_with_full_data import FullDataStopAnalyzer

def main():
    workspace = "/Users/daws/ADMF-PC/workspaces/optimize_keltner_with_filters_20250622_102448"
    
    print("=== Keltner Strategy_20 Analysis with Full OHLC Data ===\n")
    
    analyzer = FullDataStopAnalyzer(workspace)
    
    # Load source OHLC data
    try:
        ohlc_df = analyzer.load_source_data()
        print(f"Loaded {len(ohlc_df)} bars of OHLC data\n")
    except Exception as e:
        print(f"Error loading OHLC data: {e}")
        return
    
    # Load strategy 20 signals
    strategy_file = analyzer.signals_path / "SPY_5m_compiled_strategy_20.parquet"
    if strategy_file.exists():
        signals_df = analyzer.load_signal_file(strategy_file)
        print(f"Strategy 20: {len(signals_df)} signals (23 trades expected)")
        
        # Basic performance check
        signals_sorted = signals_df.sort_values('idx').reset_index(drop=True)
        position_changes = 0
        prev_val = 0
        for _, row in signals_sorted.iterrows():
            if row['val'] != prev_val:
                position_changes += 1
                prev_val = row['val']
        print(f"Position changes: {position_changes}")
        
        # Test without stops first
        print("\nTesting baseline performance...")
        baseline = analyzer.simulate_with_stops_full_data(
            analyzer.union_signals_with_ohlc(signals_df, ohlc_df), 
            stop_loss_pct=float('inf')
        )
        print(f"Baseline: {baseline['avg_return_per_trade_bps']:.2f} bps/trade, "
              f"{baseline['num_trades']} trades, "
              f"{baseline['win_rate']*100:.1f}% win rate")
        
        # Test with various stops
        stop_losses = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]  # 10-500 bps
        
        print("\n" + "="*70)
        print("STOP LOSS ANALYSIS")
        print("="*70)
        print(f"{'Stop Loss':<12} {'RPT (bps)':<12} {'Win Rate':<10} {'Stop Rate':<12} {'Winners Stopped %':<18}")
        print("-" * 70)
        
        for stop_loss in stop_losses:
            result = analyzer.simulate_with_stops_full_data(
                analyzer.union_signals_with_ohlc(signals_df, ohlc_df),
                stop_loss_pct=stop_loss
            )
            
            stop_str = f"{stop_loss*10000:.0f} bps"
            print(f"{stop_str:<12} {result['avg_return_per_trade_bps']:>10.2f} "
                  f"{result['win_rate']*100:>8.1f}% {result['stop_rate']*100:>11.1f}% "
                  f"{result['pct_stopped_were_winners']*100:>17.1f}%")
    
    # Also check the other high-performing strategies
    print("\n\n=== Checking Other Top Strategies ===")
    for strategy_num in [3, 2, 4]:  # The other good performers
        strategy_file = analyzer.signals_path / f"SPY_5m_compiled_strategy_{strategy_num}.parquet"
        if strategy_file.exists():
            signals_df = analyzer.load_signal_file(strategy_file)
            baseline = analyzer.simulate_with_stops_full_data(
                analyzer.union_signals_with_ohlc(signals_df, ohlc_df), 
                stop_loss_pct=float('inf')
            )
            print(f"\nStrategy {strategy_num}: {baseline['avg_return_per_trade_bps']:.2f} bps/trade, "
                  f"{baseline['num_trades']} trades, "
                  f"{baseline['win_rate']*100:.1f}% win rate")
            
            # Test with 20 bps stop (previously identified as optimal)
            with_stop = analyzer.simulate_with_stops_full_data(
                analyzer.union_signals_with_ohlc(signals_df, ohlc_df),
                stop_loss_pct=0.002  # 20 bps
            )
            print(f"  With 20 bps stop: {with_stop['avg_return_per_trade_bps']:.2f} bps/trade")

if __name__ == "__main__":
    main()