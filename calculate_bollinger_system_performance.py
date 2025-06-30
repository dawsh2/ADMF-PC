#!/usr/bin/env python3
"""
Calculate actual performance from system-generated Bollinger Bands signals
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def calculate_performance_from_signals(trace_path, market_data_path, stop_loss=0.00075, take_profit=0.001, execution_cost_bps=1.0):
    """Calculate performance using the same methodology as the notebook"""
    
    # Load trace and market data
    trace = pd.read_parquet(trace_path)
    market_data = pd.read_csv(market_data_path)
    
    # Convert timestamps
    trace['ts'] = pd.to_datetime(trace['ts'])
    market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])
    market_data = market_data.sort_values('timestamp')
    
    # Merge signals with market data
    df = market_data.merge(
        trace[['ts', 'val', 'px']], 
        left_on='timestamp', 
        right_on='ts', 
        how='left'
    )
    
    # Forward fill signals
    df['signal'] = df['val'].ffill().fillna(0)
    df['position'] = df['signal'].replace({0: 0, 1: 1, -1: -1})
    df['position_change'] = df['position'].diff().fillna(0)
    
    # Extract trades
    trades = []
    current_trade = None
    
    for idx, row in df.iterrows():
        if row['position_change'] != 0 and row['position'] != 0:
            # New position opened
            if current_trade is None:
                current_trade = {
                    'entry_time': row['timestamp'],
                    'entry_price': row['px'] if pd.notna(row['px']) else row['close'],
                    'direction': row['position'],
                    'entry_idx': idx
                }
        elif current_trade is not None and (row['position'] == 0 or row['position_change'] != 0):
            # Position closed
            exit_price = row['px'] if pd.notna(row['px']) else row['close']
            
            # Calculate raw return
            if current_trade['direction'] == 1:  # Long
                raw_return = (exit_price - current_trade['entry_price']) / current_trade['entry_price']
            else:  # Short
                raw_return = (current_trade['entry_price'] - exit_price) / current_trade['entry_price']
            
            # Apply execution costs (1 bps round trip)
            cost_adjustment = execution_cost_bps / 10000
            net_return = raw_return - cost_adjustment
            
            trade = {
                'entry_time': current_trade['entry_time'],
                'exit_time': row['timestamp'],
                'entry_price': current_trade['entry_price'],
                'exit_price': exit_price,
                'direction': current_trade['direction'],
                'raw_return': raw_return,
                'net_return': net_return,
                'duration_minutes': (row['timestamp'] - current_trade['entry_time']).total_seconds() / 60,
                'entry_idx': current_trade['entry_idx'],
                'exit_idx': idx
            }
            trades.append(trade)
            
            # Reset for next trade
            current_trade = None
            if row['position'] != 0 and row['position_change'] != 0:
                # Immediately open new position (reversal)
                current_trade = {
                    'entry_time': row['timestamp'],
                    'entry_price': row['px'] if pd.notna(row['px']) else row['close'],
                    'direction': row['position'],
                    'entry_idx': idx
                }
    
    trades_df = pd.DataFrame(trades)
    print(f"\nExtracted {len(trades_df)} trades from signals")
    
    # Apply stop loss and take profit
    modified_returns = []
    exit_types = {'stop': 0, 'target': 0, 'signal': 0}
    
    for _, trade in trades_df.iterrows():
        trade_prices = market_data.iloc[int(trade['entry_idx']):int(trade['exit_idx'])+1]
        
        if len(trade_prices) == 0:
            modified_returns.append(trade['net_return'])
            exit_types['signal'] += 1
            continue
        
        entry_price = trade['entry_price']
        direction = trade['direction']
        
        # Set stop and target prices
        if direction == 1:  # Long
            stop_price = entry_price * (1 - stop_loss)
            target_price = entry_price * (1 + take_profit)
        else:  # Short
            stop_price = entry_price * (1 + stop_loss)
            target_price = entry_price * (1 - take_profit)
        
        # Check each bar for exit
        exit_price = trade['exit_price']
        exit_type = 'signal'
        
        for _, bar in trade_prices.iterrows():
            if direction == 1:  # Long
                if bar['low'] <= stop_price:
                    exit_price = stop_price
                    exit_type = 'stop'
                    break
                elif bar['high'] >= target_price:
                    exit_price = target_price
                    exit_type = 'target'
                    break
            else:  # Short
                if bar['high'] >= stop_price:
                    exit_price = stop_price
                    exit_type = 'stop'
                    break
                elif bar['low'] <= target_price:
                    exit_price = target_price
                    exit_type = 'target'
                    break
        
        exit_types[exit_type] += 1
        
        # Calculate return with stop/target
        if direction == 1:
            raw_return = (exit_price - entry_price) / entry_price
        else:
            raw_return = (entry_price - exit_price) / entry_price
        
        net_return = raw_return - (execution_cost_bps / 10000)
        modified_returns.append(net_return)
    
    modified_returns = np.array(modified_returns)
    
    # Calculate performance metrics
    total_return = (1 + modified_returns).prod() - 1
    win_rate = (modified_returns > 0).mean()
    
    # Calculate Sharpe ratio
    trading_days = 47  # From notebook analysis
    trades_per_day = len(trades_df) / trading_days
    
    if modified_returns.std() > 0:
        sharpe = modified_returns.mean() / modified_returns.std() * np.sqrt(252 * trades_per_day)
    else:
        sharpe = 0
    
    # Calculate average winner/loser
    winners = modified_returns[modified_returns > 0]
    losers = modified_returns[modified_returns <= 0]
    
    avg_winner = winners.mean() if len(winners) > 0 else 0
    avg_loser = losers.mean() if len(losers) > 0 else 0
    
    # Exit type percentages
    stop_rate = exit_types['stop'] / len(trades_df)
    target_rate = exit_types['target'] / len(trades_df)
    signal_rate = exit_types['signal'] / len(trades_df)
    
    print("\n" + "="*60)
    print("SYSTEM EXECUTION PERFORMANCE (CALCULATED)")
    print("="*60)
    
    print(f"\nStrategy Parameters:")
    print(f"  Period: 10")
    print(f"  Std Dev: 1.5")
    print(f"  Stop Loss: {stop_loss*100:.3f}%")
    print(f"  Take Profit: {take_profit*100:.1f}%")
    
    print(f"\nPerformance Metrics:")
    print(f"  Total Return: {total_return*100:.2f}%")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Win Rate: {win_rate*100:.1f}%")
    print(f"  Number of Trades: {len(trades_df)}")
    print(f"  Trades per Day: {trades_per_day:.2f}")
    
    print(f"\nExit Type Distribution:")
    print(f"  Stop hits: {stop_rate*100:.1f}% ({exit_types['stop']} trades)")
    print(f"  Target hits: {target_rate*100:.1f}% ({exit_types['target']} trades)")
    print(f"  Signal exits: {signal_rate*100:.1f}% ({exit_types['signal']} trades)")
    
    print(f"\nReturn Distribution:")
    print(f"  Average Return per Trade: {modified_returns.mean()*100:.4f}%")
    print(f"  Average Winner: {avg_winner*100:.2f}%")
    print(f"  Average Loser: {avg_loser*100:.2f}%")
    
    print("\n" + "="*60)
    print("COMPARISON TO NOTEBOOK")
    print("="*60)
    
    print(f"\nNotebook Performance:")
    print(f"  Return: 20.74% vs System: {total_return*100:.2f}%")
    print(f"  Sharpe: 12.81 vs System: {sharpe:.2f}")
    print(f"  Win Rate: 75.0% vs System: {win_rate*100:.1f}%")
    print(f"  Stop Rate: 20.7% vs System: {stop_rate*100:.1f}%")
    print(f"  Target Rate: 69.0% vs System: {target_rate*100:.1f}%")
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'win_rate': win_rate,
        'num_trades': len(trades_df),
        'stop_rate': stop_rate,
        'target_rate': target_rate,
        'signal_rate': signal_rate
    }

if __name__ == "__main__":
    # Paths
    trace_path = Path("config/bollinger/results/latest/traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet")
    market_data_path = Path("config/bollinger/results/20250625_173629/data/SPY_5m.csv")  # Use notebook's data
    
    if not trace_path.exists():
        # Try specific timestamp directory
        trace_path = Path("config/bollinger/results/20250627_175630/traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet")
        market_data_path = Path("config/bollinger/results/20250625_173629/data/SPY_5m.csv")
    
    if trace_path.exists() and market_data_path.exists():
        print(f"Loading data from: {trace_path.parent.parent.parent.parent}")
        results = calculate_performance_from_signals(
            trace_path, 
            market_data_path,
            stop_loss=0.00075,
            take_profit=0.001,
            execution_cost_bps=1.0
        )
    else:
        print(f"Error: Could not find required files")
        print(f"  Trace: {trace_path.exists()}")
        print(f"  Market data: {market_data_path.exists()}")