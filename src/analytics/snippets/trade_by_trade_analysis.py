"""
Trade-by-trade analysis snippet for comparing analysis notebook results with execution engine.

This snippet can be used in Jupyter notebooks to analyze trades in detail.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def analyze_trades_detailed(strategy_hash, trace_path, market_data, stop_pct=0.075, target_pct=0.1, 
                           execution_cost_bps=1.0, max_trades=100, run_dir=None):
    """
    Perform detailed trade-by-trade analysis showing exact entry/exit conditions.
    
    Args:
        strategy_hash: Strategy identifier
        trace_path: Path to signal trace file
        market_data: Market OHLCV data
        stop_pct: Stop loss percentage (default 0.075%)
        target_pct: Take profit percentage (default 0.1%)
        execution_cost_bps: Execution cost in basis points
        max_trades: Maximum number of trades to analyze
        run_dir: Base directory for trace files
    
    Returns:
        DataFrame with detailed trade information
    """
    # Load signals
    if run_dir:
        signals_path = Path(run_dir) / trace_path
    else:
        signals_path = Path(trace_path)
        
    signals = pd.read_parquet(signals_path)
    signals['ts'] = pd.to_datetime(signals['ts'])
    
    # Merge with market data
    df = market_data.merge(
        signals[['ts', 'val', 'px']], 
        left_on='timestamp', 
        right_on='ts', 
        how='left'
    )
    
    # Forward fill signals
    df['signal'] = df['val'].ffill().fillna(0)
    df['position'] = df['signal'].replace({0: 0, 1: 1, -1: -1})
    df['position_change'] = df['position'].diff().fillna(0)
    
    trades = []
    current_trade = None
    trade_count = 0
    
    for idx, row in df.iterrows():
        # Skip if we've collected enough trades
        if trade_count >= max_trades and current_trade is None:
            break
            
        # New position opened
        if row['position_change'] != 0 and row['position'] != 0:
            if current_trade is None:
                current_trade = {
                    'trade_num': trade_count + 1,
                    'entry_time': row['timestamp'],
                    'entry_price': row['px'] if pd.notna(row['px']) else row['close'],
                    'entry_signal': row['signal'],
                    'direction': row['position'],
                    'entry_idx': idx,
                    'entry_bar': {
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close']
                    }
                }
                
        # Position closed or reversed
        elif current_trade is not None and (row['position'] == 0 or row['position_change'] != 0):
            exit_price = row['px'] if pd.notna(row['px']) else row['close']
            
            # Calculate stop and target prices
            entry_price = current_trade['entry_price']
            if current_trade['direction'] == 1:  # Long
                stop_price = entry_price * (1 - stop_pct/100)
                target_price = entry_price * (1 + target_pct/100)
            else:  # Short
                stop_price = entry_price * (1 + stop_pct/100)
                target_price = entry_price * (1 - target_pct/100)
            
            # Check each bar between entry and exit for stop/target hits
            trade_bars = df.iloc[current_trade['entry_idx']:idx+1].copy()
            actual_exit_price = exit_price
            actual_exit_time = row['timestamp']
            exit_type = 'signal'
            exit_bar_num = len(trade_bars) - 1
            
            # Check for intrabar stop/target hits
            for bar_idx, bar in enumerate(trade_bars.iterrows()):
                _, bar_data = bar
                
                if current_trade['direction'] == 1:  # Long
                    # Check stop first
                    if bar_data['low'] <= stop_price:
                        actual_exit_price = stop_price
                        actual_exit_time = bar_data['timestamp']
                        exit_type = 'stop_loss'
                        exit_bar_num = bar_idx
                        break
                    # Then target
                    elif bar_data['high'] >= target_price:
                        actual_exit_price = target_price
                        actual_exit_time = bar_data['timestamp']
                        exit_type = 'take_profit'
                        exit_bar_num = bar_idx
                        break
                else:  # Short
                    # Check stop first
                    if bar_data['high'] >= stop_price:
                        actual_exit_price = stop_price
                        actual_exit_time = bar_data['timestamp']
                        exit_type = 'stop_loss'
                        exit_bar_num = bar_idx
                        break
                    # Then target
                    elif bar_data['low'] <= target_price:
                        actual_exit_price = target_price
                        actual_exit_time = bar_data['timestamp']
                        exit_type = 'take_profit'
                        exit_bar_num = bar_idx
                        break
            
            # Calculate returns
            if current_trade['direction'] == 1:  # Long
                raw_return = (actual_exit_price - entry_price) / entry_price
                raw_return_no_stops = (exit_price - entry_price) / entry_price
            else:  # Short
                raw_return = (entry_price - actual_exit_price) / entry_price
                raw_return_no_stops = (entry_price - exit_price) / entry_price
            
            # Apply execution costs
            cost_adjustment = execution_cost_bps / 10000
            net_return = raw_return - cost_adjustment
            net_return_no_stops = raw_return_no_stops - cost_adjustment
            
            trade = {
                'trade_num': current_trade['trade_num'],
                'entry_time': current_trade['entry_time'],
                'entry_price': entry_price,
                'entry_signal': current_trade['entry_signal'],
                'direction': 'LONG' if current_trade['direction'] == 1 else 'SHORT',
                
                # Original exit (no stops)
                'original_exit_time': row['timestamp'],
                'original_exit_price': exit_price,
                'original_exit_signal': row['signal'],
                'original_return': raw_return_no_stops,
                'original_net_return': net_return_no_stops,
                
                # Actual exit (with stops)
                'actual_exit_time': actual_exit_time,
                'actual_exit_price': actual_exit_price,
                'exit_type': exit_type,
                'actual_return': raw_return,
                'actual_net_return': net_return,
                
                # Trade details
                'stop_price': stop_price,
                'target_price': target_price,
                'bars_in_trade': exit_bar_num + 1,
                'total_bars_without_stops': len(trade_bars),
                
                # Bar details for debugging
                'entry_bar': current_trade['entry_bar'],
                'exit_bar': {
                    'open': bar_data['open'],
                    'high': bar_data['high'],
                    'low': bar_data['low'],
                    'close': bar_data['close']
                }
            }
            
            trades.append(trade)
            trade_count += 1
            
            # Reset for next trade
            current_trade = None
            if row['position'] != 0 and row['position_change'] != 0:
                # Immediately open new position (reversal)
                current_trade = {
                    'trade_num': trade_count + 1,
                    'entry_time': row['timestamp'],
                    'entry_price': row['px'] if pd.notna(row['px']) else row['close'],
                    'entry_signal': row['signal'],
                    'direction': row['position'],
                    'entry_idx': idx,
                    'entry_bar': {
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close']
                    }
                }
    
    return pd.DataFrame(trades)


def display_trade_comparison(trades_df, num_trades=20):
    """
    Display a formatted comparison of trades with and without stops.
    
    Args:
        trades_df: DataFrame from analyze_trades_detailed
        num_trades: Number of trades to display
    """
    print(f"TRADE-BY-TRADE ANALYSIS (First {num_trades} trades)")
    print("=" * 120)
    print(f"{'#':>3} {'Entry Time':>20} {'Dir':>5} {'Entry':>8} {'Stop':>8} {'Target':>8} "
          f"{'Exit Type':>10} {'Exit Price':>10} {'Return%':>8} {'Bars':>5}")
    print("-" * 120)
    
    for _, trade in trades_df.head(num_trades).iterrows():
        # Format return with color coding
        ret_pct = trade['actual_net_return'] * 100
        ret_str = f"{ret_pct:>7.3f}%"
        
        print(f"{trade['trade_num']:>3} {trade['entry_time'].strftime('%Y-%m-%d %H:%M'):>20} "
              f"{trade['direction']:>5} {trade['entry_price']:>8.2f} "
              f"{trade['stop_price']:>8.2f} {trade['target_price']:>8.2f} "
              f"{trade['exit_type']:>10} {trade['actual_exit_price']:>10.2f} "
              f"{ret_str} {trade['bars_in_trade']:>5}")
    
    # Summary statistics
    print("\n" + "=" * 120)
    print("SUMMARY STATISTICS")
    print("-" * 60)
    
    # Exit type breakdown
    exit_types = trades_df['exit_type'].value_counts()
    print("\nExit Type Breakdown:")
    for exit_type, count in exit_types.items():
        pct = count / len(trades_df) * 100
        print(f"  {exit_type:>12}: {count:>4} ({pct:>5.1f}%)")
    
    # Performance comparison
    print("\nPerformance Comparison:")
    print(f"  Without stops: {trades_df['original_net_return'].mean()*100:>6.3f}% avg return")
    print(f"  With stops:    {trades_df['actual_net_return'].mean()*100:>6.3f}% avg return")
    print(f"  Improvement:   {(trades_df['actual_net_return'].mean() - trades_df['original_net_return'].mean())*100:>6.3f}%")
    
    # Win rate comparison
    orig_win_rate = (trades_df['original_net_return'] > 0).mean() * 100
    actual_win_rate = (trades_df['actual_net_return'] > 0).mean() * 100
    print(f"\nWin Rate:")
    print(f"  Without stops: {orig_win_rate:>5.1f}%")
    print(f"  With stops:    {actual_win_rate:>5.1f}%")
    
    # Average bars in trade
    print(f"\nAverage bars in trade:")
    print(f"  Without stops: {trades_df['total_bars_without_stops'].mean():>5.1f}")
    print(f"  With stops:    {trades_df['bars_in_trade'].mean():>5.1f}")
    
    return trades_df


# Example usage in notebook:
"""
# In your Jupyter notebook, use like this:

# Load the analysis functions
from src.analytics.snippets.trade_by_trade_analysis import analyze_trades_detailed, display_trade_comparison

# For the analysis notebook (strategy 5edc4365):
strategy_hash = '5edc43651004'
trace_path = strategy_index[strategy_index['strategy_hash'] == strategy_hash].iloc[0]['trace_path']

# Analyze trades
trades = analyze_trades_detailed(
    strategy_hash=strategy_hash,
    trace_path=trace_path,
    market_data=market_data,
    stop_pct=0.075,
    target_pct=0.1,
    execution_cost_bps=1.0,
    max_trades=100,
    run_dir=run_dir
)

# Display comparison
display_trade_comparison(trades, num_trades=30)

# You can also export to CSV for detailed comparison
trades.to_csv('trades_analysis_5edc4365.csv', index=False)
"""