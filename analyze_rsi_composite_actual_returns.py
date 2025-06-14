#!/usr/bin/env python3
"""
Analyze actual RSI composite strategy returns from signal traces.

This script expands sparse signal storage and calculates the actual
performance of the RSI composite strategy.
"""

import duckdb
import pandas as pd

def analyze_rsi_composite_returns(workspace_path, data_path):
    """Calculate actual returns from RSI composite strategy signal traces."""
    
    con = duckdb.connect()
    
    signal_file = f'{workspace_path}/traces/SPY_1m/signals/rsi/SPY_rsi_composite_test.parquet'
    
    print('CALCULATING ACTUAL RSI COMPOSITE STRATEGY RETURNS')
    print('=' * 60)
    
    # Step 1: Get the signal changes and expand them
    print('Step 1: Expanding sparse signals...')
    
    # Get signal data
    signals_df = con.execute(f'SELECT idx, val FROM read_parquet("{signal_file}") ORDER BY idx').df()
    print(f'Raw signals: {len(signals_df)} signal changes')
    
    # Expand signals to get position for every bar
    expanded_positions = []
    for i in range(len(signals_df)):
        current_idx = signals_df.iloc[i]['idx']
        current_val = signals_df.iloc[i]['val']
        
        if i < len(signals_df) - 1:
            next_idx = signals_df.iloc[i + 1]['idx']
            end_idx = next_idx
        else:
            end_idx = current_idx + 1000  # Use reasonable end for last signal
        
        # Add position for each bar from current to next signal
        for bar_idx in range(current_idx, end_idx):
            expanded_positions.append({
                'bar_idx': bar_idx,
                'position': current_val
            })
    
    expanded_df = pd.DataFrame(expanded_positions)
    
    # Only keep bars where we have a position (not flat/0)
    positioned_df = expanded_df[expanded_df['position'] != 0].copy()
    print(f'Bars with positions: {len(positioned_df):,}')
    
    # Step 2: Calculate returns
    print('Step 2: Calculating returns...')
    
    # Get price data for relevant bars
    price_data = con.execute(f'SELECT bar_index, close FROM read_parquet("{data_path}")').df()
    price_dict = dict(zip(price_data['bar_index'], price_data['close']))
    
    # Calculate returns for each positioned bar
    returns = []
    prev_price = None
    
    for i, row in positioned_df.iterrows():
        bar_idx = row['bar_idx']
        position = row['position']
        
        if bar_idx in price_dict:
            current_price = price_dict[bar_idx]
            
            if prev_price is not None:
                price_return = (current_price - prev_price) / prev_price
                strategy_return = position * price_return
                
                returns.append({
                    'bar_idx': bar_idx,
                    'position': position,
                    'price_return': price_return,
                    'strategy_return': strategy_return
                })
            
            prev_price = current_price
    
    returns_df = pd.DataFrame(returns)
    
    if len(returns_df) > 0:
        # Calculate performance metrics
        total_bars = len(returns_df)
        avg_return_per_bar = returns_df['strategy_return'].mean()
        volatility_per_bar = returns_df['strategy_return'].std()
        total_return = returns_df['strategy_return'].sum()
        win_rate = (returns_df['strategy_return'] > 0).mean() * 100
        
        print(f'\nACTUAL STRATEGY PERFORMANCE:')
        print(f'  Total bars with positions: {total_bars:,}')
        print(f'  Average return per bar: {avg_return_per_bar*100:.6f}%')
        print(f'  Volatility per bar: {volatility_per_bar*100:.6f}%')
        print(f'  Total return: {total_return*100:.4f}%')
        print(f'  Win rate: {win_rate:.2f}%')
        print(f'  Best bar: {returns_df["strategy_return"].max()*100:.4f}%')
        print(f'  Worst bar: {returns_df["strategy_return"].min()*100:.4f}%')
        
        # Sharpe ratio
        if volatility_per_bar > 0:
            sharpe_ratio = avg_return_per_bar / volatility_per_bar
            print(f'  Sharpe ratio: {sharpe_ratio:.4f}')
        
        # Annualized metrics
        bars_per_day = 390
        trading_days_per_year = 252
        bars_per_year = bars_per_day * trading_days_per_year
        
        annual_return = avg_return_per_bar * bars_per_year * 100
        annual_volatility = volatility_per_bar * (bars_per_year ** 0.5) * 100
        annual_sharpe = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        print(f'\nANNUALIZED METRICS:')
        print(f'  Annualized return: {annual_return:.2f}%')
        print(f'  Annualized volatility: {annual_volatility:.2f}%')
        print(f'  Annualized Sharpe ratio: {annual_sharpe:.4f}')
        
        # Position distribution
        pos_dist = positioned_df['position'].value_counts().sort_index()
        print(f'\nPOSITION DISTRIBUTION:')
        for pos, count in pos_dist.items():
            pct = count / len(positioned_df) * 100
            pos_name = 'Long' if pos == 1 else 'Short' if pos == -1 else 'Flat'
            print(f'  {pos_name} ({pos}): {count:,} bars ({pct:.1f}%)')
        
        # Calculate some additional metrics
        print(f'\nADDITIONAL INSIGHTS:')
        
        # Data period info
        min_bar = positioned_df['bar_idx'].min()
        max_bar = positioned_df['bar_idx'].max()
        total_time_bars = max_bar - min_bar + 1
        position_coverage = len(positioned_df) / total_time_bars * 100
        
        print(f'  Data period: bar {min_bar:,} to {max_bar:,} ({total_time_bars:,} bars)')
        print(f'  Position coverage: {position_coverage:.1f}% of time')
        
        # Trading frequency
        signal_changes = len(signals_df)
        avg_hold_period = len(positioned_df) / signal_changes if signal_changes > 0 else 0
        
        print(f'  Signal changes: {signal_changes:,}')
        print(f'  Average holding period: {avg_hold_period:.1f} bars')
        
        return {
            'total_return_pct': total_return * 100,
            'annualized_return_pct': annual_return,
            'annualized_sharpe': annual_sharpe,
            'win_rate_pct': win_rate,
            'total_bars': total_bars,
            'avg_return_per_bar_pct': avg_return_per_bar * 100
        }
        
    else:
        print('No valid return data calculated')
        return None


if __name__ == "__main__":
    workspace_path = '/Users/daws/ADMF-PC/workspaces/test_rsi_composite_oos_02563d53'
    data_path = '/Users/daws/ADMF-PC/data/SPY_1m.parquet'
    
    results = analyze_rsi_composite_returns(workspace_path, data_path)
    
    if results:
        print(f'\n🎯 SUMMARY: {results["annualized_return_pct"]:.2f}% annualized return')
        print(f'📊 Sharpe ratio: {results["annualized_sharpe"]:.4f}')
        print(f'🎲 Win rate: {results["win_rate_pct"]:.1f}%')