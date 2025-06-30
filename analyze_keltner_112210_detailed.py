#!/usr/bin/env python3
"""
Detailed analysis of top Keltner strategies in workspace 112210 with full OHLC stop testing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append('/Users/daws/ADMF-PC')
from analyze_keltner_with_full_data import FullDataStopAnalyzer

def analyze_top_strategies_with_stops():
    """Analyze top performing strategies with proper stop loss testing."""
    
    workspace = "/Users/daws/ADMF-PC/configs/optimize_keltner_with_filters/20250622_112210"
    signals_path = Path(workspace) / "traces" / "SPY_5m_1m" / "signals" / "keltner_bands"
    
    # Top strategies to analyze based on previous results
    # Strategy 20: 4.09 bps (23 trades) - too few trades
    # Strategy 3: 0.68 bps (1429 trades) - good frequency
    # Strategy 4: 0.42 bps (1173 trades) - good frequency
    strategies_to_test = [3, 4, 2, 0, 21]  # Testing the viable ones
    
    print("=== Detailed Analysis of Keltner Workspace 112210 ===\n")
    
    # Initialize stop analyzer
    stop_analyzer = FullDataStopAnalyzer(workspace)
    
    # Load OHLC data
    try:
        ohlc_df = stop_analyzer.load_source_data()
        print(f"Loaded {len(ohlc_df)} bars of OHLC data\n")
    except Exception as e:
        print(f"Error loading OHLC data: {e}")
        return
    
    results_summary = []
    
    for strategy_num in strategies_to_test:
        print(f"\n{'='*70}")
        print(f"STRATEGY {strategy_num} ANALYSIS")
        print(f"{'='*70}")
        
        strategy_file = signals_path / f"SPY_5m_compiled_strategy_{strategy_num}.parquet"
        if not strategy_file.exists():
            print(f"Strategy {strategy_num} file not found")
            continue
            
        signals_df = pd.read_parquet(strategy_file)
        
        # Basic performance
        print(f"\nSignals: {len(signals_df)}")
        
        # Calculate basic metrics
        signals_sorted = signals_df.sort_values('idx').reset_index(drop=True)
        trades = []
        long_trades = []
        short_trades = []
        current_position = None
        
        for i in range(len(signals_sorted)):
            row = signals_sorted.iloc[i]
            signal = row['val']
            price = row['px']
            
            if signal != 0:
                if current_position is not None:
                    if current_position['direction'] == 'long':
                        ret = np.log(price / current_position['entry_price']) * 10000
                        long_trades.append(ret)
                    else:
                        ret = -np.log(price / current_position['entry_price']) * 10000
                        short_trades.append(ret)
                    trades.append(ret)
                
                current_position = {
                    'entry_price': price,
                    'direction': 'long' if signal > 0 else 'short'
                }
            elif signal == 0 and current_position is not None:
                if current_position['direction'] == 'long':
                    ret = np.log(price / current_position['entry_price']) * 10000
                    long_trades.append(ret)
                else:
                    ret = -np.log(price / current_position['entry_price']) * 10000
                    short_trades.append(ret)
                trades.append(ret)
                current_position = None
        
        if not trades:
            print("No trades found")
            continue
        
        # Apply execution costs
        exec_mult = 1 - (0.5 / 10000)
        trades_adj = [t * exec_mult for t in trades]
        long_trades_adj = [t * exec_mult for t in long_trades]
        short_trades_adj = [t * exec_mult for t in short_trades]
        
        print(f"\nBaseline Performance:")
        print(f"  Total trades: {len(trades)}")
        print(f"  Avg return: {np.mean(trades_adj):.2f} bps/trade")
        print(f"  Win rate: {len([t for t in trades_adj if t > 0]) / len(trades_adj) * 100:.1f}%")
        print(f"  Daily trades: {len(trades) / 213:.1f}")
        
        print(f"\nLong/Short Breakdown:")
        print(f"  Long trades: {len(long_trades)} ({len(long_trades)/len(trades)*100:.0f}%)")
        if long_trades:
            print(f"    Avg return: {np.mean(long_trades_adj):.2f} bps")
            print(f"    Win rate: {len([t for t in long_trades_adj if t > 0]) / len(long_trades_adj) * 100:.1f}%")
        
        print(f"  Short trades: {len(short_trades)} ({len(short_trades)/len(trades)*100:.0f}%)")
        if short_trades:
            print(f"    Avg return: {np.mean(short_trades_adj):.2f} bps")
            print(f"    Win rate: {len([t for t in short_trades_adj if t > 0]) / len(short_trades_adj) * 100:.1f}%")
        
        # Test stop losses
        print(f"\nStop Loss Analysis:")
        stop_losses = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
        
        print(f"{'Stop':<8} {'RPT':<8} {'Win%':<8} {'Stop%':<8} {'Winners Stopped':<15}")
        print("-" * 50)
        
        best_stop = None
        best_return = np.mean(trades_adj)
        
        # Test without stop first
        full_data = stop_analyzer.union_signals_with_ohlc(signals_df, ohlc_df)
        baseline_result = stop_analyzer.simulate_with_stops_full_data(full_data, float('inf'))
        print(f"{'None':<8} {baseline_result['avg_return_per_trade_bps']:>6.2f} "
              f"{baseline_result['win_rate']*100:>6.1f}% {'0.0':>6}% {'N/A':>14}")
        
        for stop_loss in stop_losses:
            result = stop_analyzer.simulate_with_stops_full_data(full_data, stop_loss)
            
            stop_str = f"{stop_loss*10000:.0f}bps"
            print(f"{stop_str:<8} {result['avg_return_per_trade_bps']:>6.2f} "
                  f"{result['win_rate']*100:>6.1f}% {result['stop_rate']*100:>6.1f}% "
                  f"{result['pct_stopped_were_winners']*100:>14.0f}%")
            
            if result['avg_return_per_trade_bps'] > best_return:
                best_return = result['avg_return_per_trade_bps']
                best_stop = stop_loss * 10000
        
        # Summary for this strategy
        improvement = (best_return - baseline_result['avg_return_per_trade_bps']) / baseline_result['avg_return_per_trade_bps'] * 100 if baseline_result['avg_return_per_trade_bps'] != 0 else 0
        
        results_summary.append({
            'strategy': strategy_num,
            'trades': len(trades),
            'daily_trades': len(trades) / 213,
            'baseline_rpt': baseline_result['avg_return_per_trade_bps'],
            'best_stop_bps': best_stop,
            'optimized_rpt': best_return,
            'improvement_pct': improvement,
            'long_rpt': np.mean(long_trades_adj) if long_trades else 0,
            'short_rpt': np.mean(short_trades_adj) if short_trades else 0
        })
    
    # Print final summary
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY - WORKSPACE 112210")
    print(f"{'='*80}\n")
    
    summary_df = pd.DataFrame(results_summary)
    summary_df = summary_df.sort_values('optimized_rpt', ascending=False)
    
    print(f"{'Strategy':<10} {'Trades/Day':<12} {'Baseline':<12} {'Best Stop':<12} {'Optimized':<12} {'Improvement':<12}")
    print("-" * 75)
    
    for idx, row in summary_df.iterrows():
        print(f"Strategy {row['strategy']:<3} {row['daily_trades']:>10.1f} "
              f"{row['baseline_rpt']:>11.2f} {row['best_stop_bps']:>11.0f} "
              f"{row['optimized_rpt']:>11.2f} {row['improvement_pct']:>11.0f}%")
    
    # Best overall strategy
    best = summary_df.iloc[0]
    print(f"\n{'='*50}")
    print("OPTIMAL CONFIGURATION")
    print(f"{'='*50}")
    print(f"Strategy: {int(best['strategy'])}")
    print(f"Trades per day: {best['daily_trades']:.1f}")
    print(f"Return per trade: {best['optimized_rpt']:.2f} bps")
    print(f"Stop loss: {best['best_stop_bps']:.0f} bps")
    print(f"Annual return estimate: {best['optimized_rpt'] * best['daily_trades'] * 252 / 100:.1f}%")
    
    # Compare to previous workspace
    print(f"\n{'='*50}")
    print("COMPARISON TO PREVIOUS WORKSPACE (102448)")
    print(f"{'='*50}")
    print("Previous best: Strategy 4, 0.59 bps/trade with 20 bps stop")
    print(f"Current best: Strategy {int(best['strategy'])}, {best['optimized_rpt']:.2f} bps/trade with {best['best_stop_bps']:.0f} bps stop")
    
    if best['optimized_rpt'] > 0.59:
        print(f"✓ This workspace is {best['optimized_rpt']/0.59:.1f}x better!")
    else:
        print("✗ Previous workspace performed better")
    
    # Save results
    summary_df.to_csv("keltner_112210_detailed_summary.csv", index=False)
    print(f"\nDetailed summary saved to keltner_112210_detailed_summary.csv")

if __name__ == "__main__":
    analyze_top_strategies_with_stops()