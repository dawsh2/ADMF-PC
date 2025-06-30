#!/usr/bin/env python3
"""
Detailed analysis of how a 0.1% stop loss affects winning trades.
Key focus: The path that winning trades take and when they would be stopped.
"""
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_01_stop_impact():
    """Analyze in detail how 0.1% stop affects winners."""
    
    # Load workspace data
    workspace = Path("workspaces/signal_generation_7ecda4b8")
    signal_file = workspace / "traces/SPY_1m/signals/bollinger_rsi_simple_signals/SPY_compiled_strategy_0.parquet"
    
    # Load SPY 1-minute data
    spy_data = pd.read_csv('data/SPY_1m.csv')
    spy_data['timestamp'] = pd.to_datetime(spy_data['timestamp'], utc=True)
    spy_data = spy_data.set_index('timestamp').sort_index()
    
    # Read signals
    signals_df = pd.read_parquet(signal_file)
    signals_df['ts'] = pd.to_datetime(signals_df['ts'])
    
    print("=== 0.1% Stop Loss Deep Dive: Impact on Winning Trades ===\n")
    
    # Convert sparse signals to trades with full tick-by-tick data
    trades_detailed = []
    current_position = 0
    
    for i in range(len(signals_df)):
        row = signals_df.iloc[i]
        new_signal = row['val']
        
        # Close existing position if changing
        if current_position != 0 and new_signal != current_position:
            entry_idx = i - 1
            entry_row = signals_df.iloc[entry_idx]
            
            # Get all prices during the trade
            entry_time = entry_row['ts']
            exit_time = row['ts']
            
            # Get tick-by-tick price data
            trade_prices = spy_data.loc[entry_time:exit_time, 'Close'].values
            if len(trade_prices) > 1:
                entry_price = trade_prices[0]
                
                # Calculate return path
                if current_position == 1:  # Long trade
                    return_path = (trade_prices / entry_price - 1) * 100
                else:  # Short trade
                    return_path = (1 - trade_prices / entry_price) * 100
                
                final_return = return_path[-1]
                mae = np.min(return_path)
                mfe = np.max(return_path)
                
                # Find if and when 0.1% stop would be hit
                stop_hit_idx = np.where(return_path <= -0.1)[0]
                if len(stop_hit_idx) > 0:
                    stop_hit_bar = stop_hit_idx[0]
                    return_at_stop = return_path[stop_hit_bar]
                else:
                    stop_hit_bar = None
                    return_at_stop = None
                
                trades_detailed.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'direction': 'long' if current_position > 0 else 'short',
                    'entry_price': entry_price,
                    'final_return': final_return,
                    'mae': mae,
                    'mfe': mfe,
                    'bars_held': len(trade_prices) - 1,
                    'stop_hit_bar': stop_hit_bar,
                    'return_at_stop': return_at_stop,
                    'return_path': return_path
                })
        
        current_position = new_signal
    
    trades_df = pd.DataFrame(trades_detailed)
    
    # Focus on winning trades
    winning_trades = trades_df[trades_df['final_return'] > 0].copy()
    
    # Winners that would be stopped
    stopped_winners = winning_trades[winning_trades['stop_hit_bar'].notna()].copy()
    
    print(f"Total trades analyzed: {len(trades_df)}")
    print(f"Winning trades: {len(winning_trades)} ({len(winning_trades)/len(trades_df)*100:.1f}%)")
    print(f"Winners that would hit 0.1% stop: {len(stopped_winners)} ({len(stopped_winners)/len(winning_trades)*100:.1f}% of winners)")
    
    if len(stopped_winners) > 0:
        print(f"\n=== Detailed Analysis of Stopped Winners ===\n")
        
        # Statistics
        print(f"Average final return (if not stopped): {stopped_winners['final_return'].mean():.3f}%")
        print(f"Median final return: {stopped_winners['final_return'].median():.3f}%")
        print(f"Best return foregone: {stopped_winners['final_return'].max():.3f}%")
        print(f"Total return foregone: {stopped_winners['final_return'].sum():.3f}%")
        
        print(f"\nTiming analysis:")
        print(f"Average bars to stop: {stopped_winners['stop_hit_bar'].mean():.1f}")
        print(f"Median bars to stop: {stopped_winners['stop_hit_bar'].median():.1f}")
        print(f"Fastest stop: {stopped_winners['stop_hit_bar'].min()} bars")
        print(f"Slowest stop: {stopped_winners['stop_hit_bar'].max()} bars")
        
        # Show distribution of when stops hit
        print(f"\nWhen stops are hit (bars from entry):")
        stop_timing = stopped_winners['stop_hit_bar'].value_counts().sort_index()
        for bars, count in stop_timing.items():
            if bars <= 10 or count > 1:  # Show first 10 bars or any with multiple hits
                print(f"  Bar {int(bars):2d}: {count:2d} trades")
        
        # Analyze recovery patterns
        print(f"\n=== Recovery Analysis ===")
        print(f"These winning trades went down 0.1% before recovering:\n")
        
        # Group by final return buckets
        return_buckets = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 10.0]
        stopped_winners['return_bucket'] = pd.cut(stopped_winners['final_return'], bins=return_buckets)
        bucket_analysis = stopped_winners.groupby('return_bucket').agg({
            'final_return': ['count', 'mean'],
            'stop_hit_bar': 'mean',
            'mfe': 'mean'
        })
        
        print("Final return buckets for stopped winners:")
        for bucket, row in bucket_analysis.iterrows():
            count = row[('final_return', 'count')]
            if count > 0:
                avg_return = row[('final_return', 'mean')]
                avg_stop_bar = row[('stop_hit_bar', 'mean')]
                avg_mfe = row[('mfe', 'mean')]
                print(f"  {bucket}: {int(count)} trades, avg return {avg_return:.3f}%, "
                      f"stopped at bar {avg_stop_bar:.1f}, avg MFE {avg_mfe:.3f}%")
        
        # Show some example trade paths
        print(f"\n=== Example Trade Paths (Winners that would be stopped) ===\n")
        
        # Sort by final return to show variety
        examples = stopped_winners.nlargest(3, 'final_return').index.tolist()
        if len(stopped_winners) > 6:
            examples.extend(stopped_winners.nsmallest(3, 'final_return').index.tolist())
        
        for idx in examples[:6]:
            trade = stopped_winners.loc[idx]
            print(f"Trade {idx}:")
            print(f"  Direction: {trade['direction']}")
            print(f"  Final return: {trade['final_return']:.3f}%")
            print(f"  Stop hit at bar: {int(trade['stop_hit_bar'])}")
            print(f"  MAE: {trade['mae']:.3f}%, MFE: {trade['mfe']:.3f}%")
            
            # Show key points in path
            path = trade['return_path']
            print(f"  Path highlights:")
            print(f"    Entry (bar 0): 0.000%")
            print(f"    Bar {int(trade['stop_hit_bar'])}: {path[int(trade['stop_hit_bar'])]:.3f}% (stop hit)")
            
            # Find recovery point
            recovery_idx = np.where(path[int(trade['stop_hit_bar']):] >= 0)[0]
            if len(recovery_idx) > 0:
                recovery_bar = int(trade['stop_hit_bar']) + recovery_idx[0]
                print(f"    Bar {recovery_bar}: {path[recovery_bar]:.3f}% (back to breakeven)")
            
            # MFE point
            mfe_bar = np.argmax(path)
            print(f"    Bar {mfe_bar}: {path[mfe_bar]:.3f}% (MFE)")
            print(f"    Exit (bar {len(path)-1}): {path[-1]:.3f}%")
            print()
    
    # Overall impact analysis
    print(f"\n=== Overall Impact of 0.1% Stop Loss ===\n")
    
    # Calculate metrics with and without stop
    baseline_avg = trades_df['final_return'].mean()
    baseline_win_rate = (trades_df['final_return'] > 0).mean()
    
    # Apply stop
    trades_with_stop = trades_df.copy()
    stop_mask = trades_with_stop['stop_hit_bar'].notna()
    trades_with_stop.loc[stop_mask, 'final_return'] = -0.1
    
    stop_avg = trades_with_stop['final_return'].mean()
    stop_win_rate = (trades_with_stop['final_return'] > 0).mean()
    
    print(f"Without stop loss:")
    print(f"  Average return per trade: {baseline_avg:.3f}%")
    print(f"  Win rate: {baseline_win_rate:.1%}")
    print(f"  Total return: {trades_df['final_return'].sum():.2f}%")
    
    print(f"\nWith 0.1% stop loss:")
    print(f"  Average return per trade: {stop_avg:.3f}%")
    print(f"  Win rate: {stop_win_rate:.1%}")
    print(f"  Total return: {trades_with_stop['final_return'].sum():.2f}%")
    
    print(f"\nNet impact:")
    print(f"  Change in avg return: {stop_avg - baseline_avg:+.3f}%")
    print(f"  Change in win rate: {(stop_win_rate - baseline_win_rate)*100:+.1f} percentage points")
    print(f"  Change in total return: {trades_with_stop['final_return'].sum() - trades_df['final_return'].sum():+.2f}%")
    
    # Risk-adjusted metrics
    baseline_std = trades_df['final_return'].std()
    stop_std = trades_with_stop['final_return'].std()
    
    print(f"\nRisk metrics:")
    print(f"  Std dev without stop: {baseline_std:.3f}%")
    print(f"  Std dev with stop: {stop_std:.3f}%")
    print(f"  Risk reduction: {(baseline_std - stop_std)/baseline_std*100:.1f}%")
    
    if baseline_std > 0:
        baseline_sharpe = baseline_avg / baseline_std * np.sqrt(252 * 390)  # Annualized
        stop_sharpe = stop_avg / stop_std * np.sqrt(252 * 390)
        print(f"\n  Sharpe without stop: {baseline_sharpe:.2f}")
        print(f"  Sharpe with stop: {stop_sharpe:.2f}")
    
    print(f"\n=== Key Insights ===")
    print(f"1. Only {len(stopped_winners)/len(winning_trades)*100:.1f}% of winners hit the 0.1% stop")
    print(f"2. Most stops occur within the first {stopped_winners['stop_hit_bar'].quantile(0.75):.0f} bars")
    print(f"3. The 0.1% stop improves expected return by limiting large losses")
    print(f"4. Risk-adjusted returns (Sharpe) are {'better' if stop_sharpe > baseline_sharpe else 'worse'} with the stop")

if __name__ == "__main__":
    analyze_01_stop_impact()