#!/usr/bin/env python3
"""
Analyze how stop losses affect winning trades by examining their full price paths.
Key question: How many eventual winners would be stopped out at various levels?
"""
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_winning_trade_paths():
    """Analyze the intra-trade price movements of winning trades."""
    
    # Load workspace data
    workspace = Path("workspaces/signal_generation_7ecda4b8")
    signal_file = workspace / "traces/SPY_1m/signals/bollinger_rsi_simple_signals/SPY_compiled_strategy_0.parquet"
    
    # Load SPY 1-minute data
    spy_data = pd.read_csv('data/SPY_1m.csv')
    spy_data['timestamp'] = pd.to_datetime(spy_data['timestamp'])
    spy_data = spy_data.set_index('timestamp').sort_index()
    
    # Read signals
    signals_df = pd.read_parquet(signal_file)
    signals_df['ts'] = pd.to_datetime(signals_df['ts'])
    
    print("=== Winning Trade Path Analysis ===")
    print(f"Analyzing how stop losses would affect eventual winners\n")
    
    # Convert sparse signals to trades with full price paths
    trades = []
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
            
            # Get price path during trade
            trade_prices = spy_data.loc[entry_time:exit_time, 'Close'].values
            if len(trade_prices) > 1:
                entry_price = trade_prices[0]
                exit_price = trade_prices[-1]
                
                # Calculate returns at each point
                if current_position == 1:  # Long trade
                    returns = (trade_prices / entry_price - 1) * 100
                else:  # Short trade
                    returns = (1 - trade_prices / entry_price) * 100
                
                final_return = returns[-1]
                
                # Calculate MAE and MFE
                mae = np.min(returns)  # Maximum Adverse Excursion
                mfe = np.max(returns)  # Maximum Favorable Excursion
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'direction': 'long' if current_position > 0 else 'short',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'final_return': final_return,
                    'mae': mae,
                    'mfe': mfe,
                    'bars_held': len(trade_prices) - 1,
                    'price_path': returns
                })
        
        current_position = new_signal
    
    trades_df = pd.DataFrame(trades)
    
    # Separate winning and losing trades
    winning_trades = trades_df[trades_df['final_return'] > 0].copy()
    losing_trades = trades_df[trades_df['final_return'] <= 0].copy()
    
    print(f"Total trades: {len(trades_df)}")
    print(f"Winning trades: {len(winning_trades)} ({len(winning_trades)/len(trades_df)*100:.1f}%)")
    print(f"Losing trades: {len(losing_trades)} ({len(losing_trades)/len(trades_df)*100:.1f}%)")
    
    # Analyze stop loss impact on winning trades
    print("\n=== Stop Loss Impact on Winning Trades ===")
    print("How many eventual winners would be stopped out?\n")
    
    stop_levels = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]
    
    print(f"{'Stop Loss':<12} {'Winners Stopped':<16} {'% of Winners':<13} {'Avg Return Lost':<16} {'Profit Foregone':<15}")
    print("-" * 90)
    
    for stop_pct in stop_levels:
        # Count winners that would be stopped
        stopped_winners = winning_trades[winning_trades['mae'] <= -stop_pct]
        num_stopped = len(stopped_winners)
        pct_stopped = num_stopped / len(winning_trades) * 100 if len(winning_trades) > 0 else 0
        
        # Calculate foregone profits
        if num_stopped > 0:
            avg_return_lost = stopped_winners['final_return'].mean()
            total_profit_lost = stopped_winners['final_return'].sum()
        else:
            avg_return_lost = 0
            total_profit_lost = 0
        
        print(f"{stop_pct:>6.2f}%     {num_stopped:>14}   {pct_stopped:>11.1f}%   {avg_return_lost:>14.3f}%   {total_profit_lost:>13.2f}%")
    
    # Detailed analysis for tight stops
    print("\n=== Detailed Analysis for Tight Stops ===")
    print("Focus on 0.1% - 0.3% stop losses\n")
    
    for stop_pct in [0.1, 0.15, 0.2, 0.25, 0.3]:
        stopped_winners = winning_trades[winning_trades['mae'] <= -stop_pct]
        
        if len(stopped_winners) > 0:
            print(f"\n{stop_pct}% Stop Loss:")
            print(f"  - Would stop {len(stopped_winners)} winning trades out of {len(winning_trades)}")
            print(f"  - Average final return of stopped winners: {stopped_winners['final_return'].mean():.3f}%")
            print(f"  - Best return lost: {stopped_winners['final_return'].max():.3f}%")
            print(f"  - Average bars to stop: {stopped_winners.apply(lambda x: next((i for i, r in enumerate(x['price_path']) if r <= -stop_pct), -1), axis=1).mean():.1f}")
            
            # Distribution of returns lost
            return_bins = [0, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 100]
            return_dist = pd.cut(stopped_winners['final_return'], bins=return_bins)
            print(f"  - Distribution of returns foregone:")
            for bin_range, count in return_dist.value_counts().sort_index().items():
                if count > 0:
                    print(f"    {bin_range}: {count} trades")
    
    # MAE distribution for winning trades
    print("\n=== MAE Distribution for Winning Trades ===")
    print("What percentage of winners experience various drawdowns?\n")
    
    mae_thresholds = [0, -0.05, -0.1, -0.15, -0.2, -0.25, -0.3, -0.5, -1.0]
    
    for i in range(len(mae_thresholds) - 1):
        upper = mae_thresholds[i]
        lower = mae_thresholds[i + 1]
        
        if i == 0:
            count = len(winning_trades[winning_trades['mae'] > upper])
            label = f"No drawdown"
        else:
            count = len(winning_trades[(winning_trades['mae'] <= upper) & (winning_trades['mae'] > lower)])
            label = f"{-upper:.2f}% to {-lower:.2f}%"
        
        pct = count / len(winning_trades) * 100 if len(winning_trades) > 0 else 0
        print(f"{label:<20} {count:>5} trades ({pct:>5.1f}%)")
    
    # Extreme drawdowns
    extreme_count = len(winning_trades[winning_trades['mae'] <= -1.0])
    extreme_pct = extreme_count / len(winning_trades) * 100 if len(winning_trades) > 0 else 0
    print(f"{'Worse than -1.0%':<20} {extreme_count:>5} trades ({extreme_pct:>5.1f}%)")
    
    # Analyze time to recovery for winners with drawdowns
    print("\n=== Recovery Analysis ===")
    print("For winners that experience drawdowns, how quickly do they recover?\n")
    
    drawdown_winners = winning_trades[winning_trades['mae'] < -0.1]
    
    if len(drawdown_winners) > 0:
        recovery_times = []
        
        for _, trade in drawdown_winners.iterrows():
            path = trade['price_path']
            # Find first point below -0.1%
            drawdown_idx = next((i for i, r in enumerate(path) if r < -0.1), None)
            if drawdown_idx is not None:
                # Find recovery point (back to breakeven)
                recovery_idx = next((i for i in range(drawdown_idx, len(path)) if path[i] >= 0), len(path))
                recovery_time = recovery_idx - drawdown_idx
                recovery_times.append(recovery_time)
        
        if recovery_times:
            print(f"Winners experiencing >0.1% drawdown: {len(drawdown_winners)}")
            print(f"Average bars to recover to breakeven: {np.mean(recovery_times):.1f}")
            print(f"Median bars to recovery: {np.median(recovery_times):.1f}")
            print(f"Max bars to recovery: {max(recovery_times)}")
    
    # Compare MAE vs final return
    print("\n=== MAE vs Final Return Relationship ===")
    
    if len(winning_trades) > 0:
        # Correlation
        correlation = winning_trades['mae'].corr(winning_trades['final_return'])
        print(f"Correlation between MAE and final return: {correlation:.3f}")
        
        # Bucketize by MAE
        mae_buckets = pd.cut(winning_trades['mae'], 
                            bins=[-100, -1.0, -0.5, -0.3, -0.2, -0.1, -0.05, 0, 100],
                            labels=['< -1%', '-1% to -0.5%', '-0.5% to -0.3%', 
                                   '-0.3% to -0.2%', '-0.2% to -0.1%', '-0.1% to -0.05%',
                                   '-0.05% to 0%', '> 0%'])
        
        mae_analysis = winning_trades.groupby(mae_buckets).agg({
            'final_return': ['count', 'mean', 'median', 'std'],
            'bars_held': 'mean'
        }).round(3)
        
        print("\nFinal returns by MAE bucket:")
        print(mae_analysis)
    
    # Optimal stop loss analysis
    print("\n=== Optimal Stop Loss Analysis ===")
    print("Expected value impact of different stop losses\n")
    
    print(f"{'Stop Loss':<12} {'E[Return/Trade]':<16} {'Win Rate':<10} {'Avg Win':<10} {'Avg Loss':<10}")
    print("-" * 70)
    
    # No stop baseline
    baseline_return = trades_df['final_return'].mean()
    baseline_win_rate = (trades_df['final_return'] > 0).mean()
    baseline_avg_win = trades_df[trades_df['final_return'] > 0]['final_return'].mean()
    baseline_avg_loss = trades_df[trades_df['final_return'] <= 0]['final_return'].mean()
    
    print(f"{'No stop':<12} {baseline_return:>14.3f}%   {baseline_win_rate:>8.1%}   {baseline_avg_win:>8.3f}%   {baseline_avg_loss:>8.3f}%")
    
    for stop_pct in [0.1, 0.15, 0.2, 0.25, 0.3, 0.5]:
        # Apply stop loss
        trades_with_stop = trades_df.copy()
        
        # Trades that hit stop
        stopped_mask = trades_with_stop['mae'] <= -stop_pct
        trades_with_stop.loc[stopped_mask, 'final_return'] = -stop_pct
        
        # Recalculate metrics
        avg_return = trades_with_stop['final_return'].mean()
        win_rate = (trades_with_stop['final_return'] > 0).mean()
        
        winners = trades_with_stop[trades_with_stop['final_return'] > 0]
        losers = trades_with_stop[trades_with_stop['final_return'] <= 0]
        
        avg_win = winners['final_return'].mean() if len(winners) > 0 else 0
        avg_loss = losers['final_return'].mean() if len(losers) > 0 else 0
        
        print(f"{stop_pct:>6.2f}%     {avg_return:>14.3f}%   {win_rate:>8.1%}   {avg_win:>8.3f}%   {avg_loss:>8.3f}%")
    
    # Print expected returns table instead of visualization
    print("\n=== Expected Return by Stop Level ===")
    stop_levels = [None, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1.0]
    expected_returns = []
    
    print(f"{'Stop Loss':<12} {'Expected Return':<16} {'Change from Baseline':<20}")
    print("-" * 50)
    
    for stop in stop_levels:
        if stop is None:
            expected_returns.append(baseline_return)
            print(f"{'No stop':<12} {baseline_return:>14.3f}%   {'(baseline)':<20}")
        else:
            temp_trades = trades_df.copy()
            stopped_mask = temp_trades['mae'] <= -stop
            temp_trades.loc[stopped_mask, 'final_return'] = -stop
            exp_return = temp_trades['final_return'].mean()
            expected_returns.append(exp_return)
            change = exp_return - baseline_return
            print(f"{stop:>6.2f}%     {exp_return:>14.3f}%   {change:>18.3f}%")
    
    return trades_df, winning_trades

if __name__ == "__main__":
    trades_df, winning_trades = analyze_winning_trade_paths()