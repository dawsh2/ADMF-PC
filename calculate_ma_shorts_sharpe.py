#!/usr/bin/env python3
"""
Calculate precise Sharpe ratio for MA crossover shorts-only strategy.
"""

import duckdb
import pandas as pd
import numpy as np

def calculate_ma_shorts_sharpe(workspace_path, data_path):
    """Calculate actual Sharpe ratio for shorts-only positions."""
    
    con = duckdb.connect()
    
    signal_file = f'{workspace_path}/traces/SPY_1m/signals/momentum/SPY_ma_crossover_momentum_test.parquet'
    
    print('CALCULATING PRECISE SHARPE FOR MA SHORTS-ONLY')
    print('=' * 60)
    
    # Get all signals
    signals_df = con.execute(f'SELECT idx, val FROM read_parquet("{signal_file}") ORDER BY idx').df()
    
    # Get price data
    price_data = con.execute(f'SELECT bar_index, close FROM read_parquet("{data_path}")').df()
    price_dict = dict(zip(price_data['bar_index'], price_data['close']))
    
    # Extract only short positions
    short_trades = []
    
    for i in range(len(signals_df) - 1):
        entry_signal = signals_df.iloc[i]
        exit_signal = signals_df.iloc[i + 1]
        
        # Only process short entries (-1)
        if entry_signal['val'] == -1:
            entry_idx = entry_signal['idx']
            exit_idx = exit_signal['idx']
            
            if entry_idx in price_dict and exit_idx in price_dict:
                entry_price = price_dict[entry_idx]
                exit_price = price_dict[exit_idx]
                
                # Short position return
                trade_return = (entry_price - exit_price) / entry_price * 100
                
                short_trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': exit_idx,
                    'return_pct': trade_return
                })
    
    if len(short_trades) == 0:
        print('No short trades found!')
        return None
    
    # Convert to DataFrame for analysis
    shorts_df = pd.DataFrame(short_trades)
    
    # Calculate metrics
    avg_return = shorts_df['return_pct'].mean()
    volatility = shorts_df['return_pct'].std()
    total_return = shorts_df['return_pct'].sum()
    win_rate = (shorts_df['return_pct'] > 0).mean() * 100
    
    # Sharpe calculation
    sharpe_per_trade = avg_return / volatility if volatility > 0 else 0
    
    print(f'SHORT POSITIONS ANALYSIS:')
    print(f'  Total short positions: {len(shorts_df)}')
    print(f'  Average return: {avg_return:.6f}%')
    print(f'  Volatility: {volatility:.6f}%')
    print(f'  Total return: {total_return:.4f}%')
    print(f'  Win rate: {win_rate:.2f}%')
    print(f'  Sharpe per trade: {sharpe_per_trade:.6f}')
    
    # Annualized calculations
    total_data_bars = signals_df['idx'].max() - signals_df['idx'].min()
    bars_per_day = 390
    data_days = total_data_bars / bars_per_day
    
    trades_per_day = len(shorts_df) / data_days
    trades_per_year = trades_per_day * 252
    
    annual_return = avg_return * trades_per_year
    annual_volatility = volatility * np.sqrt(trades_per_year)
    annual_sharpe = annual_return / annual_volatility if annual_volatility > 0 else 0
    
    print(f'\\nANNUALIZED METRICS:')
    print(f'  Trades per year: {trades_per_year:.0f}')
    print(f'  Annual return: {annual_return:.2f}%')
    print(f'  Annual volatility: {annual_volatility:.2f}%')
    print(f'  Annual Sharpe: {annual_sharpe:.4f}')
    
    # Return distribution
    print(f'\\nRETURN DISTRIBUTION:')
    print(f'  Best trade: {shorts_df["return_pct"].max():.4f}%')
    print(f'  Worst trade: {shorts_df["return_pct"].min():.4f}%')
    print(f'  25th percentile: {shorts_df["return_pct"].quantile(0.25):.4f}%')
    print(f'  Median: {shorts_df["return_pct"].median():.4f}%')
    print(f'  75th percentile: {shorts_df["return_pct"].quantile(0.75):.4f}%')
    
    # Compare all strategies
    print(f'\\n📊 COMPREHENSIVE SHARPE COMPARISON:')
    print(f'  MA Crossover Combined: 2.24 Sharpe, 34.39% return')
    print(f'  MA Crossover Shorts-only: {annual_sharpe:.2f} Sharpe, {annual_return:.2f}% return')
    print(f'  RSI Tuned Combined: 0.018 Sharpe, 1.56% return')
    
    return {
        'annual_sharpe': annual_sharpe,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'trades_per_year': trades_per_year
    }


if __name__ == "__main__":
    workspace_path = '/Users/daws/ADMF-PC/workspaces/test_ma_crossover_momentum_c94a270b'
    data_path = '/Users/daws/ADMF-PC/data/SPY_1m.parquet'
    
    results = calculate_ma_shorts_sharpe(workspace_path, data_path)