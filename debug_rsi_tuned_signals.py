#!/usr/bin/env python3
"""
Debug RSI tuned strategy signals to verify:
1. Return calculation logic is correct
2. Strategy generates both long and short signals
3. Position logic matches signal direction
"""

import duckdb
import pandas as pd

def debug_rsi_tuned_strategy(workspace_path, data_path):
    """Debug the RSI tuned strategy signals and return calculations."""
    
    con = duckdb.connect()
    
    signal_file = f'{workspace_path}/traces/SPY_1m/signals/rsi/SPY_rsi_tuned_test.parquet'
    
    print('DEBUGGING RSI TUNED STRATEGY SIGNALS')
    print('=' * 60)
    
    # Get all signals
    signals_df = con.execute(f'SELECT idx, val FROM read_parquet("{signal_file}") ORDER BY idx').df()
    print(f'Total signal changes: {len(signals_df)}')
    
    # Analyze signal distribution
    signal_counts = signals_df['val'].value_counts().sort_index()
    print(f'\\nSIGNAL DISTRIBUTION:')
    for signal, count in signal_counts.items():
        pct = count / len(signals_df) * 100
        signal_name = 'Long Entry' if signal == 1 else 'Short Entry' if signal == -1 else 'Exit/Flat'
        print(f'  {signal} ({signal_name}): {count} signals ({pct:.1f}%)')
    
    # Show sample signals
    print(f'\\nSAMPLE SIGNALS (first 20):')
    print(signals_df.head(20).to_string(index=False))
    
    # Get price data
    price_data = con.execute(f'SELECT bar_index, close FROM read_parquet("{data_path}")').df()
    price_dict = dict(zip(price_data['bar_index'], price_data['close']))
    
    # Manual trade reconstruction with detailed logging
    print(f'\\nTRADE RECONSTRUCTION (first 10 trades):')
    trades = []
    i = 0
    trade_count = 0
    
    while i < len(signals_df) and trade_count < 10:
        current_signal = signals_df.iloc[i]
        
        if current_signal['val'] in [1, -1]:  # Entry signal
            entry_idx = current_signal['idx']
            entry_direction = current_signal['val']
            
            # Find corresponding exit
            exit_idx = None
            for j in range(i + 1, len(signals_df)):
                if signals_df.iloc[j]['val'] == 0:
                    exit_idx = signals_df.iloc[j]['idx']
                    break
            
            if exit_idx is not None and entry_idx in price_dict and exit_idx in price_dict:
                entry_price = price_dict[entry_idx]
                exit_price = price_dict[exit_idx]
                holding_bars = exit_idx - entry_idx
                
                # Calculate return with detailed logging
                if entry_direction == 1:  # Long position
                    raw_return = (exit_price - entry_price) / entry_price
                    direction_name = "LONG"
                    calc_explanation = f"({exit_price:.4f} - {entry_price:.4f}) / {entry_price:.4f}"
                else:  # Short position
                    raw_return = (entry_price - exit_price) / entry_price
                    direction_name = "SHORT"
                    calc_explanation = f"({entry_price:.4f} - {exit_price:.4f}) / {entry_price:.4f}"
                
                trade_count += 1
                print(f'\\n  Trade {trade_count}:')
                print(f'    Direction: {direction_name} (signal={entry_direction})')
                print(f'    Entry: bar {entry_idx}, price ${entry_price:.4f}')
                print(f'    Exit:  bar {exit_idx}, price ${exit_price:.4f}')
                print(f'    Hold:  {holding_bars} bars')
                print(f'    Return calculation: {calc_explanation} = {raw_return:.6f} = {raw_return*100:.6f}%')
                
                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': exit_idx,
                    'direction': entry_direction,
                    'direction_name': direction_name,
                    'holding_bars': holding_bars,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return_pct': raw_return * 100,
                    'calc_explanation': calc_explanation
                })
        i += 1
    
    # Build all trades for summary statistics
    print(f'\\n\\nBUILDING ALL TRADES FOR STATISTICS...')
    all_trades = []
    i = 0
    
    while i < len(signals_df):
        current_signal = signals_df.iloc[i]
        
        if current_signal['val'] in [1, -1]:
            entry_idx = current_signal['idx']
            entry_direction = current_signal['val']
            
            exit_idx = None
            for j in range(i + 1, len(signals_df)):
                if signals_df.iloc[j]['val'] == 0:
                    exit_idx = signals_df.iloc[j]['idx']
                    break
            
            if exit_idx is not None and entry_idx in price_dict and exit_idx in price_dict:
                entry_price = price_dict[entry_idx]
                exit_price = price_dict[exit_idx]
                
                if entry_direction == 1:  # Long
                    raw_return = (exit_price - entry_price) / entry_price
                else:  # Short
                    raw_return = (entry_price - exit_price) / entry_price
                
                all_trades.append({
                    'direction': entry_direction,
                    'return_pct': raw_return * 100
                })
        i += 1
    
    # Statistics
    trades_df = pd.DataFrame(all_trades)
    
    print(f'\\nTRADE STATISTICS:')
    print(f'  Total trades: {len(trades_df)}')
    
    # By direction
    long_trades = trades_df[trades_df['direction'] == 1]
    short_trades = trades_df[trades_df['direction'] == -1]
    
    print(f'\\nBY DIRECTION:')
    print(f'  Long trades (signal=1): {len(long_trades)} ({len(long_trades)/len(trades_df)*100:.1f}%)')
    if len(long_trades) > 0:
        print(f'    Avg return: {long_trades["return_pct"].mean():.6f}%')
        print(f'    Win rate: {(long_trades["return_pct"] > 0).mean()*100:.1f}%')
        print(f'    Best: {long_trades["return_pct"].max():.4f}%')
        print(f'    Worst: {long_trades["return_pct"].min():.4f}%')
    
    print(f'  Short trades (signal=-1): {len(short_trades)} ({len(short_trades)/len(trades_df)*100:.1f}%)')
    if len(short_trades) > 0:
        print(f'    Avg return: {short_trades["return_pct"].mean():.6f}%')
        print(f'    Win rate: {(short_trades["return_pct"] > 0).mean()*100:.1f}%')
        print(f'    Best: {short_trades["return_pct"].max():.4f}%')
        print(f'    Worst: {short_trades["return_pct"].min():.4f}%')
    
    # Overall
    print(f'\\nOVERALL:')
    print(f'  Average return per trade: {trades_df["return_pct"].mean():.6f}%')
    print(f'  Total return: {trades_df["return_pct"].sum():.4f}%')
    print(f'  Win rate: {(trades_df["return_pct"] > 0).mean()*100:.1f}%')
    
    # Verify the strategy logic by checking the actual strategy file
    print(f'\\n\\nSTRATEGY VERIFICATION:')
    print(f'Let me check if the RSI tuned strategy actually generates both long and short signals...')
    
    return trades_df


if __name__ == "__main__":
    workspace_path = '/Users/daws/ADMF-PC/workspaces/test_rsi_tuned_oos_4805b412'
    data_path = '/Users/daws/ADMF-PC/data/SPY_1m.parquet'
    
    trades_df = debug_rsi_tuned_strategy(workspace_path, data_path)