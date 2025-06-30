#!/usr/bin/env python3
"""
Properly analyze stop losses for 2826 strategy by tracking full equity curves.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def analyze_stops_with_equity_curves():
    print("=== PROPER STOP LOSS ANALYSIS FOR 2826 STRATEGY ===\n")
    
    # Load OHLC data
    ohlc_df = pd.read_csv("/Users/daws/ADMF-PC/data/SPY_5m.csv")
    ohlc_df['idx'] = range(len(ohlc_df))
    
    # Load a 2826 strategy
    workspace = "/Users/daws/ADMF-PC/config/keltner/results/latest"
    strategy_file = Path(workspace) / "traces" / "keltner_bands" / "SPY_5m_compiled_strategy_3.parquet"
    
    signals_df = pd.read_parquet(strategy_file)
    
    # Create full dataset with signals at every bar
    print("Merging signal data with OHLC...")
    merged_df = pd.merge_asof(
        ohlc_df.sort_values('idx'),
        signals_df[['idx', 'val']].sort_values('idx').rename(columns={'val': 'signal'}),
        on='idx',
        direction='backward'
    )
    merged_df['signal'] = merged_df['signal'].fillna(0)
    
    # Baseline performance (no stops)
    print("\nCalculating baseline performance...")
    baseline_trades = simulate_trades(merged_df, stop_loss_pct=None)
    baseline_df = pd.DataFrame(baseline_trades)
    baseline_df['return_net'] = baseline_df['return_bps'] - 0.5  # execution cost
    baseline_return = baseline_df['return_net'].mean()
    baseline_annual = baseline_return * len(baseline_df) / 100
    
    print(f"\nBASELINE PERFORMANCE (No Stops):")
    print(f"  Trades: {len(baseline_df)}")
    print(f"  Return: {baseline_return:.2f} bps/trade")
    print(f"  Win rate: {(baseline_df['return_net'] > 0).mean()*100:.1f}%")
    print(f"  Annual: {baseline_annual:.1f}%")
    
    # Test different stop levels
    stop_levels = [5, 10, 15, 20, 25, 30, 40, 50]
    results = []
    
    print("\nTesting stop losses...")
    for stop_bps in stop_levels:
        stop_pct = stop_bps / 10000
        
        trades = simulate_trades(merged_df, stop_loss_pct=stop_pct)
        trades_df = pd.DataFrame(trades)
        
        if len(trades_df) == 0:
            continue
            
        # Apply execution costs
        trades_df['return_net'] = trades_df['return_bps'] - 0.5
        
        # Calculate metrics
        avg_return = trades_df['return_net'].mean()
        total_trades = len(trades_df)
        stopped_trades = trades_df['stopped'].sum()
        stop_rate = stopped_trades / total_trades * 100 if total_trades > 0 else 0
        
        # Analyze stopped trades
        stopped_df = trades_df[trades_df['stopped']]
        if len(stopped_df) > 0:
            # Check if we would have been winners without stop
            stopped_winners = (stopped_df['max_return_bps'] > 0).sum()
            stopped_losers = len(stopped_df) - stopped_winners
            avg_stopped_return = stopped_df['return_net'].mean()
        else:
            stopped_winners = 0
            stopped_losers = 0
            avg_stopped_return = 0
        
        # Win rate
        win_rate = (trades_df['return_net'] > 0).mean() * 100
        
        # Annual return
        annual_return = avg_return * total_trades / 100
        
        results.append({
            'stop_bps': stop_bps,
            'avg_return': avg_return,
            'total_trades': total_trades,
            'stop_rate': stop_rate,
            'stopped_winners': stopped_winners,
            'stopped_losers': stopped_losers,
            'win_rate': win_rate,
            'annual_return': annual_return,
            'improvement': (avg_return / baseline_return - 1) * 100 if baseline_return != 0 else 0
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*100)
    print("STOP LOSS IMPACT ANALYSIS:")
    print("="*100)
    print(f"{'Stop':<8} {'RPT':<10} {'Trades':<10} {'Stop%':<10} {'Winners':<10} {'Losers':<10} {'Win%':<10} {'Annual%':<10} {'Improve%':<10}")
    print("-"*100)
    
    for _, row in results_df.iterrows():
        print(f"{row['stop_bps']:<8.0f} {row['avg_return']:<10.2f} {row['total_trades']:<10.0f} "
              f"{row['stop_rate']:<10.1f} {row['stopped_winners']:<10.0f} {row['stopped_losers']:<10.0f} "
              f"{row['win_rate']:<10.1f} {row['annual_return']:<10.1f} {row['improvement']:>+9.1f}%")
    
    # Find optimal stop
    if len(results_df) > 0:
        optimal_idx = results_df['avg_return'].idxmax()
        optimal = results_df.loc[optimal_idx]
        
        print(f"\n{'='*60}")
        print("OPTIMAL STOP LOSS ANALYSIS:")
        print(f"{'='*60}")
        print(f"Stop level: {optimal['stop_bps']:.0f} bps")
        print(f"Return improvement: {optimal['improvement']:.1f}%")
        print(f"New return per trade: {optimal['avg_return']:.2f} bps")
        print(f"New annual return: {optimal['annual_return']:.1f}%")
        print(f"Trades stopped: {optimal['stop_rate']:.1f}%")
        print(f"  - Winners that would have been: {optimal['stopped_winners']:.0f}")
        print(f"  - Losers prevented: {optimal['stopped_losers']:.0f}")
    
    # Group average clarification
    print(f"\n{'='*60}")
    print("GROUP AVERAGE CLARIFICATION:")
    print(f"{'='*60}")
    print("The '9.7% annual' figure comes from:")
    print("  - 0.68 bps/trade (gross, before costs)")
    print("  - 1,429 trades")
    print("  - 0.68 × 1,429 / 100 = 9.7%")
    print("\nAfter 0.5 bps execution costs:")
    print("  - 0.18 bps/trade net")
    print("  - 0.18 × 1,429 / 100 = 2.6% annual")
    print("\nWith optimal stops:")
    print(f"  - Expected: {optimal['annual_return']:.1f}% annual")

def simulate_trades(data, stop_loss_pct=None):
    """Simulate trades with proper equity curve tracking."""
    trades = []
    current_position = None
    
    for i in range(len(data)):
        row = data.iloc[i]
        
        # Track position if we have one
        if current_position is not None:
            # Update high/low watermarks
            if current_position['direction'] == 'long':
                current_return = np.log(row['close'] / current_position['entry_price']) * 10000
                high_return = np.log(row['high'] / current_position['entry_price']) * 10000
                low_return = np.log(row['low'] / current_position['entry_price']) * 10000
                
                current_position['max_return'] = max(current_position['max_return'], high_return)
                current_position['min_return'] = min(current_position['min_return'], low_return)
                
                # Check stop loss
                if stop_loss_pct and low_return <= -stop_loss_pct * 10000:
                    # Stop hit
                    trades.append({
                        'return_bps': -stop_loss_pct * 10000,
                        'direction': 'long',
                        'stopped': True,
                        'duration': i - current_position['entry_idx'],
                        'max_return_bps': current_position['max_return']
                    })
                    current_position = None
                    continue
                    
            else:  # short
                current_return = -np.log(row['close'] / current_position['entry_price']) * 10000
                high_return = -np.log(row['low'] / current_position['entry_price']) * 10000
                low_return = -np.log(row['high'] / current_position['entry_price']) * 10000
                
                current_position['max_return'] = max(current_position['max_return'], high_return)
                current_position['min_return'] = min(current_position['min_return'], low_return)
                
                # Check stop loss
                if stop_loss_pct and low_return <= -stop_loss_pct * 10000:
                    # Stop hit
                    trades.append({
                        'return_bps': -stop_loss_pct * 10000,
                        'direction': 'short',
                        'stopped': True,
                        'duration': i - current_position['entry_idx'],
                        'max_return_bps': current_position['max_return']
                    })
                    current_position = None
                    continue
        
        # Process signals
        signal = row['signal']
        
        if signal != 0:
            if current_position is not None:
                # Close existing position
                if current_position['direction'] == 'long':
                    ret = np.log(row['close'] / current_position['entry_price']) * 10000
                else:
                    ret = -np.log(row['close'] / current_position['entry_price']) * 10000
                
                trades.append({
                    'return_bps': ret,
                    'direction': current_position['direction'],
                    'stopped': False,
                    'duration': i - current_position['entry_idx'],
                    'max_return_bps': current_position['max_return']
                })
            
            # Open new position
            current_position = {
                'entry_idx': i,
                'entry_price': row['close'],
                'direction': 'long' if signal > 0 else 'short',
                'max_return': 0,
                'min_return': 0
            }
            
        elif signal == 0 and current_position is not None:
            # Exit signal
            if current_position['direction'] == 'long':
                ret = np.log(row['close'] / current_position['entry_price']) * 10000
            else:
                ret = -np.log(row['close'] / current_position['entry_price']) * 10000
            
            trades.append({
                'return_bps': ret,
                'direction': current_position['direction'],
                'stopped': False,
                'duration': i - current_position['entry_idx'],
                'max_return_bps': current_position['max_return']
            })
            current_position = None
    
    return trades

if __name__ == "__main__":
    analyze_stops_with_equity_curves()