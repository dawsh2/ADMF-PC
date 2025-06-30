#!/usr/bin/env python3
"""
Complete picture of 0.1% stop loss impact - both winners and losers.
Shows exactly what happens to all trades with this stop level.
"""
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_complete_stop_impact():
    """Full analysis of 0.1% stop on all trades."""
    
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
    
    print("=== Complete 0.1% Stop Loss Analysis ===\n")
    
    # Convert sparse signals to trades
    trades = []
    current_position = 0
    stop_level = 0.1  # 0.1% stop
    
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
            
            # Get price path
            trade_prices = spy_data.loc[entry_time:exit_time, 'Close'].values
            if len(trade_prices) > 1:
                entry_price = trade_prices[0]
                
                # Calculate returns at each bar
                if current_position == 1:  # Long trade
                    returns = (trade_prices / entry_price - 1) * 100
                else:  # Short trade
                    returns = (1 - trade_prices / entry_price) * 100
                
                final_return = returns[-1]
                mae = np.min(returns)
                mfe = np.max(returns)
                
                # Check if stop hit
                stop_hit_idx = np.where(returns <= -stop_level)[0]
                stop_hit = len(stop_hit_idx) > 0
                if stop_hit:
                    bars_to_stop = stop_hit_idx[0]
                else:
                    bars_to_stop = None
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'direction': 'long' if current_position > 0 else 'short',
                    'entry_price': entry_price,
                    'final_return': final_return,
                    'mae': mae,
                    'mfe': mfe,
                    'bars_held': len(trade_prices) - 1,
                    'stop_hit': stop_hit,
                    'bars_to_stop': bars_to_stop,
                    'is_winner': final_return > 0
                })
        
        current_position = new_signal
    
    trades_df = pd.DataFrame(trades)
    
    # Categorize trades
    winners = trades_df[trades_df['is_winner']]
    losers = trades_df[~trades_df['is_winner']]
    
    winners_stopped = winners[winners['stop_hit']]
    losers_stopped = losers[losers['stop_hit']]
    
    print(f"Total trades: {len(trades_df)}")
    print(f"  Winners: {len(winners)} ({len(winners)/len(trades_df)*100:.1f}%)")
    print(f"  Losers: {len(losers)} ({len(losers)/len(trades_df)*100:.1f}%)")
    
    print(f"\n=== 0.1% Stop Loss Impact ===\n")
    
    print(f"Trades that hit 0.1% stop: {trades_df['stop_hit'].sum()} ({trades_df['stop_hit'].sum()/len(trades_df)*100:.1f}%)")
    print(f"  Winners stopped: {len(winners_stopped)} ({len(winners_stopped)/len(winners)*100:.1f}% of winners)")
    print(f"  Losers stopped: {len(losers_stopped)} ({len(losers_stopped)/len(losers)*100:.1f}% of losers)")
    
    # Detailed breakdown
    print(f"\n=== Winners That Would Be Stopped (n={len(winners_stopped)}) ===")
    if len(winners_stopped) > 0:
        print(f"Final returns they would have achieved:")
        print(f"  Mean: {winners_stopped['final_return'].mean():.3f}%")
        print(f"  Median: {winners_stopped['final_return'].median():.3f}%")
        print(f"  Min: {winners_stopped['final_return'].min():.3f}%")
        print(f"  Max: {winners_stopped['final_return'].max():.3f}%")
        print(f"  Total foregone: {winners_stopped['final_return'].sum():.3f}%")
        print(f"\nTiming: Average bars to stop = {winners_stopped['bars_to_stop'].mean():.1f}")
    
    print(f"\n=== Losers That Would Be Stopped (n={len(losers_stopped)}) ===")
    if len(losers_stopped) > 0:
        print(f"Losses they would have incurred without stop:")
        print(f"  Mean: {losers_stopped['final_return'].mean():.3f}%")
        print(f"  Median: {losers_stopped['final_return'].median():.3f}%")
        print(f"  Min: {losers_stopped['final_return'].min():.3f}%")
        print(f"  Max: {losers_stopped['final_return'].max():.3f}%")
        print(f"  Total avoided: {losers_stopped['final_return'].sum():.3f}%")
        print(f"\nTiming: Average bars to stop = {losers_stopped['bars_to_stop'].mean():.1f}")
    
    # Calculate net impact
    print(f"\n=== Net Impact Analysis ===")
    
    # Without stop
    total_return_no_stop = trades_df['final_return'].sum()
    avg_return_no_stop = trades_df['final_return'].mean()
    
    # With stop
    return_foregone = winners_stopped['final_return'].sum() if len(winners_stopped) > 0 else 0
    loss_avoided = -losers_stopped['final_return'].sum() if len(losers_stopped) > 0 else 0
    stop_cost = len(trades_df[trades_df['stop_hit']]) * stop_level
    
    net_impact = loss_avoided - return_foregone - stop_cost
    
    print(f"Without stop:")
    print(f"  Total return: {total_return_no_stop:.2f}%")
    print(f"  Average per trade: {avg_return_no_stop:.3f}%")
    
    print(f"\nWith 0.1% stop:")
    print(f"  Profit foregone from stopped winners: -{return_foregone:.2f}%")
    print(f"  Loss avoided from stopped losers: +{loss_avoided:.2f}%")
    print(f"  Cost of all stops: -{stop_cost:.2f}%")
    print(f"  Net impact: {'+' if net_impact > 0 else ''}{net_impact:.2f}%")
    
    total_with_stop = total_return_no_stop + net_impact
    avg_with_stop = total_with_stop / len(trades_df)
    
    print(f"\n  Total return with stop: {total_with_stop:.2f}%")
    print(f"  Average per trade with stop: {avg_with_stop:.3f}%")
    print(f"  Improvement: {'+' if net_impact > 0 else ''}{net_impact:.2f}% total, "
          f"{'+' if avg_with_stop > avg_return_no_stop else ''}{(avg_with_stop - avg_return_no_stop):.3f}% per trade")
    
    # Distribution analysis
    print(f"\n=== Loss Distribution Analysis ===")
    
    loss_buckets = [0, -0.05, -0.1, -0.15, -0.2, -0.3, -0.5, -1.0, -10.0]
    losers['loss_bucket'] = pd.cut(losers['final_return'], bins=loss_buckets[::-1])
    
    print(f"\nLosing trades by magnitude:")
    for bucket in losers['loss_bucket'].value_counts().sort_index().index:
        count = len(losers[losers['loss_bucket'] == bucket])
        saved = len(losers[(losers['loss_bucket'] == bucket) & (losers['stop_hit'])])
        if count > 0:
            print(f"  {bucket}: {count} trades ({saved} saved by stop)")
    
    # Risk metrics
    print(f"\n=== Risk Reduction ===")
    
    returns_no_stop = trades_df['final_return']
    returns_with_stop = trades_df['final_return'].copy()
    returns_with_stop[trades_df['stop_hit']] = -stop_level
    
    vol_no_stop = returns_no_stop.std()
    vol_with_stop = returns_with_stop.std()
    
    downside_no_stop = returns_no_stop[returns_no_stop < 0].std()
    downside_with_stop = returns_with_stop[returns_with_stop < 0].std()
    
    worst_no_stop = returns_no_stop.min()
    worst_with_stop = returns_with_stop.min()
    
    print(f"Volatility:")
    print(f"  Without stop: {vol_no_stop:.3f}%")
    print(f"  With stop: {vol_with_stop:.3f}%")
    print(f"  Reduction: {(vol_no_stop - vol_with_stop)/vol_no_stop*100:.1f}%")
    
    print(f"\nDownside volatility:")
    print(f"  Without stop: {downside_no_stop:.3f}%")
    print(f"  With stop: {downside_with_stop:.3f}%")
    print(f"  Reduction: {(downside_no_stop - downside_with_stop)/downside_no_stop*100:.1f}%")
    
    print(f"\nWorst loss:")
    print(f"  Without stop: {worst_no_stop:.3f}%")
    print(f"  With stop: {worst_with_stop:.3f}%")
    
    # Sharpe calculation
    if vol_no_stop > 0 and vol_with_stop > 0:
        sharpe_no_stop = avg_return_no_stop / vol_no_stop * np.sqrt(252 * 390)
        sharpe_with_stop = avg_with_stop / vol_with_stop * np.sqrt(252 * 390)
        
        print(f"\nSharpe ratio (annualized):")
        print(f"  Without stop: {sharpe_no_stop:.2f}")
        print(f"  With stop: {sharpe_with_stop:.2f}")
        print(f"  Improvement: {(sharpe_with_stop/sharpe_no_stop - 1)*100:+.1f}%")
    
    print(f"\n=== Summary ===")
    print(f"1. The 0.1% stop affects {trades_df['stop_hit'].sum()} trades ({trades_df['stop_hit'].mean()*100:.1f}%)")
    print(f"2. It stops {len(winners_stopped)} eventual winners but saves {len(losers_stopped)} from larger losses")
    print(f"3. Net impact: {'+' if net_impact > 0 else ''}{net_impact:.2f}% ({'+' if net_impact > 0 else ''}{(avg_with_stop - avg_return_no_stop):.3f}% per trade)")
    print(f"4. Risk reduction: {(vol_no_stop - vol_with_stop)/vol_no_stop*100:.1f}% lower volatility")
    print(f"5. Sharpe ratio {'improves' if sharpe_with_stop > sharpe_no_stop else 'worsens'} by {abs((sharpe_with_stop/sharpe_no_stop - 1)*100):.1f}%")

if __name__ == "__main__":
    analyze_complete_stop_impact()