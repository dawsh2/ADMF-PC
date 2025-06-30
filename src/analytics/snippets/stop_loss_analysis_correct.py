# Correct Stop Loss Analysis Implementation
# This snippet provides the CORRECT way to implement stop loss analysis
# It checks intraday prices and properly simulates stop loss triggers

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_stop_loss_impact(trades_df, stop_loss_levels=None, market_data=None):
    """
    Calculate returns with various stop loss levels using PROPER intraday simulation.
    
    THIS IS THE CORRECT IMPLEMENTATION:
    - Checks actual intraday high/low prices
    - Exits immediately when stop is hit
    - Stops out trades that would have been winners too
    - Does NOT retrospectively cap losses
    
    Args:
        trades_df: DataFrame of trades (must include entry_idx and exit_idx)
        stop_loss_levels: List of stop loss percentages (default 0.05% to 1%)
        market_data: Market data for intraday price movements
    
    Returns:
        DataFrame with returns for each stop loss level
    """
    if stop_loss_levels is None:
        stop_loss_levels = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]
    
    if market_data is None:
        raise ValueError("Market data is required for proper stop loss analysis!")
    
    results = []
    
    for sl_pct in stop_loss_levels:
        sl_decimal = sl_pct / 100
        
        trades_with_sl = []
        stopped_out_count = 0
        stopped_winners = 0  # Trades that were stopped but would have been winners
        
        # Process each trade with intraday stop loss
        for _, trade in trades_df.iterrows():
            # Get intraday prices for this trade
            trade_prices = market_data.iloc[int(trade['entry_idx']):int(trade['exit_idx'])+1]
            
            if len(trade_prices) == 0:
                continue
                
            entry_price = trade['entry_price']
            direction = trade['direction']
            original_return = trade['net_return']  # What the trade actually returned
            
            # Calculate stop loss price
            if direction == 1:  # Long position
                stop_price = entry_price * (1 - sl_decimal)
            else:  # Short position  
                stop_price = entry_price * (1 + sl_decimal)
            
            # Check if stop loss is hit
            stopped = False
            exit_price = trade['exit_price']
            exit_time = trade['exit_time']
            
            for idx, bar in trade_prices.iterrows():
                if direction == 1:  # Long
                    # Check if low price hits stop
                    if bar['low'] <= stop_price:
                        stopped = True
                        stopped_out_count += 1
                        exit_price = stop_price
                        exit_time = bar['timestamp']
                        # Check if this would have been a winner
                        if original_return > 0:
                            stopped_winners += 1
                        break
                else:  # Short
                    # Check if high price hits stop
                    if bar['high'] >= stop_price:
                        stopped = True
                        stopped_out_count += 1
                        exit_price = stop_price
                        exit_time = bar['timestamp']
                        # Check if this would have been a winner
                        if original_return > 0:
                            stopped_winners += 1
                        break
            
            # Calculate return with actual or stopped exit
            if direction == 1:  # Long
                raw_return = (exit_price - entry_price) / entry_price
            else:  # Short
                raw_return = (entry_price - exit_price) / entry_price
                
            # Apply execution costs
            net_return = raw_return - trade['execution_cost']
            
            trade_result = trade.copy()
            trade_result['raw_return'] = raw_return
            trade_result['net_return'] = net_return
            trade_result['stopped_out'] = stopped
            if stopped:
                trade_result['exit_price'] = exit_price
                trade_result['exit_time'] = exit_time
                
            trades_with_sl.append(trade_result)
        
        trades_with_sl_df = pd.DataFrame(trades_with_sl)
        
        if len(trades_with_sl_df) > 0:
            # Calculate metrics with stop loss
            total_return = trades_with_sl_df['net_return'].sum()
            avg_return = trades_with_sl_df['net_return'].mean()
            win_rate = (trades_with_sl_df['net_return'] > 0).mean()
            
            results.append({
                'stop_loss_pct': sl_pct,
                'total_return': total_return,
                'avg_return_per_trade': avg_return,
                'win_rate': win_rate,
                'stopped_out_count': stopped_out_count,
                'stopped_out_rate': stopped_out_count / len(trades_with_sl_df),
                'stopped_winners': stopped_winners,
                'stopped_winners_pct': stopped_winners / len(trades_with_sl_df) * 100,
                'num_trades': len(trades_with_sl_df),
                'avg_winner': trades_with_sl_df[trades_with_sl_df['net_return'] > 0]['net_return'].mean() if (trades_with_sl_df['net_return'] > 0).any() else 0,
                'avg_loser': trades_with_sl_df[trades_with_sl_df['net_return'] <= 0]['net_return'].mean() if (trades_with_sl_df['net_return'] <= 0).any() else 0
            })
    
    return pd.DataFrame(results)


def visualize_stop_loss_impact(sl_results_df, strategy_name="Strategy"):
    """
    Visualize the impact of stop losses on strategy performance
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Total return vs stop loss
    ax = axes[0, 0]
    ax.plot(sl_results_df['stop_loss_pct'], sl_results_df['total_return'] * 100, 'o-', linewidth=2)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Stop Loss %')
    ax.set_ylabel('Total Return %')
    ax.set_title(f'{strategy_name}: Total Return vs Stop Loss Level')
    ax.grid(True, alpha=0.3)
    
    # 2. Win rate vs stop loss
    ax = axes[0, 1]
    ax.plot(sl_results_df['stop_loss_pct'], sl_results_df['win_rate'] * 100, 'o-', linewidth=2, color='green')
    ax.set_xlabel('Stop Loss %')
    ax.set_ylabel('Win Rate %')
    ax.set_title('Win Rate vs Stop Loss Level')
    ax.grid(True, alpha=0.3)
    
    # 3. Stopped out rate and stopped winners
    ax = axes[1, 0]
    ax.plot(sl_results_df['stop_loss_pct'], sl_results_df['stopped_out_rate'] * 100, 'o-', linewidth=2, label='Total Stopped', color='red')
    ax.plot(sl_results_df['stop_loss_pct'], sl_results_df['stopped_winners_pct'], 'o-', linewidth=2, label='Winners Stopped', color='orange')
    ax.set_xlabel('Stop Loss %')
    ax.set_ylabel('Percentage of Trades')
    ax.set_title('Trades Stopped Out')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Average winner/loser
    ax = axes[1, 1]
    ax.plot(sl_results_df['stop_loss_pct'], sl_results_df['avg_winner'] * 100, 'o-', linewidth=2, label='Avg Winner', color='green')
    ax.plot(sl_results_df['stop_loss_pct'], sl_results_df['avg_loser'] * 100, 'o-', linewidth=2, label='Avg Loser', color='red')
    ax.set_xlabel('Stop Loss %')
    ax.set_ylabel('Average Return %')
    ax.set_title('Average Winner vs Loser Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    optimal_idx = sl_results_df['total_return'].idxmax()
    optimal_sl = sl_results_df.iloc[optimal_idx]
    
    print(f"\n📊 Stop Loss Analysis Summary for {strategy_name}:")
    print(f"Optimal stop loss: {optimal_sl['stop_loss_pct']:.2f}%")
    print(f"Total return at optimal: {optimal_sl['total_return']*100:.2f}%")
    print(f"Trades stopped at optimal: {optimal_sl['stopped_out_count']} ({optimal_sl['stopped_out_rate']*100:.1f}%)")
    print(f"Winners stopped at optimal: {optimal_sl['stopped_winners']} ({optimal_sl['stopped_winners_pct']:.1f}% of all trades)")
    
    return optimal_sl


# Example usage:
# trades = extract_trades(strategy_hash, trace_path, market_data, execution_cost_bps=1.0)
# sl_results = calculate_stop_loss_impact(trades, stop_loss_levels=[0.1, 0.2, 0.3, 0.5, 1.0], market_data=market_data)
# optimal = visualize_stop_loss_impact(sl_results, "My Strategy")