#!/usr/bin/env python3
"""
Analyze MA crossover performance filtered by strong momentum regime only.
Shows performance for short only, long only, and combined.
"""

import duckdb
import pandas as pd
import numpy as np

def analyze_ma_crossover_regime_filtered(workspace_path, data_path):
    """Analyze MA crossover when filtered to only trade in strong momentum regime."""
    
    con = duckdb.connect()
    
    # Paths
    signal_file = f'{workspace_path}/traces/SPY_1m/signals/momentum/SPY_ma_crossover_momentum_test.parquet'
    # Use the best performing momentum regime classifier from our analysis
    classifier_file = f'/Users/daws/ADMF-PC/workspaces/expansive_grid_search_c410bba2/traces/SPY_1m/classifiers/momentum_regime_grid/SPY_momentum_regime_grid_65_35_015.parquet'
    
    print('MA CROSSOVER FILTERED BY STRONG MOMENTUM REGIME')
    print('=' * 60)
    
    # Get signals
    signals_df = con.execute(f'SELECT idx, val FROM read_parquet("{signal_file}") ORDER BY idx').df()
    
    # Get classifier data and expand sparse storage
    classifier_df = con.execute(f'SELECT idx, val FROM read_parquet("{classifier_file}") ORDER BY idx').df()
    
    # Expand classifier to full regime coverage
    max_idx = signals_df['idx'].max()
    expanded_regimes = []
    
    for i in range(len(classifier_df)):
        current_idx = classifier_df.iloc[i]['idx']
        current_regime = classifier_df.iloc[i]['val']
        
        if i < len(classifier_df) - 1:
            next_idx = classifier_df.iloc[i + 1]['idx']
        else:
            next_idx = max_idx + 1
        
        # Add regime for each bar from current to next change
        for bar_idx in range(current_idx, next_idx):
            expanded_regimes.append({
                'bar_idx': bar_idx,
                'regime': 'strong_momentum' if current_regime == 1 else 'weak_momentum'
            })
    
    regime_df = pd.DataFrame(expanded_regimes)
    
    # Merge signals with regimes
    signals_with_regime = []
    for i in range(len(signals_df) - 1):
        entry_signal = signals_df.iloc[i]
        exit_signal = signals_df.iloc[i + 1]
        
        entry_idx = entry_signal['idx']
        exit_idx = exit_signal['idx']
        direction = entry_signal['val']
        
        # Get regime at entry
        regime_at_entry = regime_df[regime_df['bar_idx'] == entry_idx]['regime'].iloc[0] if len(regime_df[regime_df['bar_idx'] == entry_idx]) > 0 else 'unknown'
        
        signals_with_regime.append({
            'entry_idx': entry_idx,
            'exit_idx': exit_idx,
            'direction': direction,
            'regime': regime_at_entry,
            'holding_bars': exit_idx - entry_idx
        })
    
    signals_regime_df = pd.DataFrame(signals_with_regime)
    
    # Get price data
    price_data = con.execute(f'SELECT bar_index, close FROM read_parquet("{data_path}")').df()
    price_dict = dict(zip(price_data['bar_index'], price_data['close']))
    
    # Calculate returns for all scenarios
    results = {}
    
    for scenario in ['all_trades', 'strong_momentum_only', 'strong_momentum_long', 'strong_momentum_short']:
        print(f'\\n{scenario.upper().replace("_", " ")}:')
        print('-' * 40)
        
        # Filter trades based on scenario
        if scenario == 'all_trades':
            filtered_df = signals_regime_df
        elif scenario == 'strong_momentum_only':
            filtered_df = signals_regime_df[signals_regime_df['regime'] == 'strong_momentum']
        elif scenario == 'strong_momentum_long':
            filtered_df = signals_regime_df[(signals_regime_df['regime'] == 'strong_momentum') & (signals_regime_df['direction'] == 1)]
        elif scenario == 'strong_momentum_short':
            filtered_df = signals_regime_df[(signals_regime_df['regime'] == 'strong_momentum') & (signals_regime_df['direction'] == -1)]
        
        if len(filtered_df) == 0:
            print('  No trades in this scenario')
            continue
        
        # Calculate returns
        trade_returns = []
        for _, trade in filtered_df.iterrows():
            entry_idx = trade['entry_idx']
            exit_idx = trade['exit_idx']
            direction = trade['direction']
            
            if entry_idx in price_dict and exit_idx in price_dict:
                entry_price = price_dict[entry_idx]
                exit_price = price_dict[exit_idx]
                
                if direction == 1:  # Long
                    trade_return = (exit_price - entry_price) / entry_price * 100
                else:  # Short
                    trade_return = (entry_price - exit_price) / entry_price * 100
                
                trade_returns.append(trade_return)
        
        if len(trade_returns) == 0:
            print('  No valid returns calculated')
            continue
        
        # Calculate metrics
        returns_array = np.array(trade_returns)
        avg_return = returns_array.mean()
        volatility = returns_array.std()
        total_return = returns_array.sum()
        win_rate = (returns_array > 0).mean() * 100
        num_trades = len(trade_returns)
        
        # Annualized metrics
        total_days = (signals_df['idx'].max() - signals_df['idx'].min()) / 390
        trades_per_day = num_trades / total_days
        trades_per_year = trades_per_day * 252
        
        annual_return = avg_return * trades_per_year
        annual_volatility = volatility * np.sqrt(trades_per_year)
        annual_sharpe = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        print(f'  Number of trades: {num_trades}')
        print(f'  Average return per trade: {avg_return:.6f}%')
        print(f'  Win rate: {win_rate:.2f}%')
        print(f'  Volatility: {volatility:.6f}%')
        print(f'  Total return: {total_return:.4f}%')
        print(f'  Trades per year: {trades_per_year:.0f}')
        print(f'  Annualized return: {annual_return:.2f}%')
        print(f'  Annualized Sharpe: {annual_sharpe:.4f}')
        
        # Direction breakdown
        if scenario in ['all_trades', 'strong_momentum_only']:
            long_count = len(filtered_df[filtered_df['direction'] == 1])
            short_count = len(filtered_df[filtered_df['direction'] == -1])
            print(f'  Long positions: {long_count} ({long_count/num_trades*100:.1f}%)')
            print(f'  Short positions: {short_count} ({short_count/num_trades*100:.1f}%)')
        
        results[scenario] = {
            'num_trades': num_trades,
            'avg_return': avg_return,
            'annual_return': annual_return,
            'annual_sharpe': annual_sharpe,
            'win_rate': win_rate
        }
    
    # Summary comparison
    print(f'\\n\\nSUMMARY COMPARISON:')
    print('=' * 80)
    print(f'{"Scenario":<30} {"Trades":<10} {"Avg Return":<12} {"Annual Return":<15} {"Sharpe":<10} {"Win Rate":<10}')
    print('-' * 80)
    
    for scenario, metrics in results.items():
        print(f'{scenario:<30} {metrics["num_trades"]:<10} {metrics["avg_return"]:>11.4f}% {metrics["annual_return"]:>13.2f}% {metrics["annual_sharpe"]:>9.4f} {metrics["win_rate"]:>8.1f}%')
    
    # Key insights
    print(f'\\n\\nKEY INSIGHTS:')
    if 'all_trades' in results and 'strong_momentum_only' in results:
        improvement = results['strong_momentum_only']['annual_return'] - results['all_trades']['annual_return']
        print(f'✅ Filtering for strong momentum improves annual return by {improvement:.2f} percentage points')
    
    if 'strong_momentum_long' in results and 'strong_momentum_short' in results:
        long_return = results['strong_momentum_long']['annual_return']
        short_return = results['strong_momentum_short']['annual_return']
        if abs(short_return) > abs(long_return):
            print(f'✅ Shorts outperform longs in strong momentum: {short_return:.2f}% vs {long_return:.2f}%')
        else:
            print(f'✅ Longs outperform shorts in strong momentum: {long_return:.2f}% vs {short_return:.2f}%')
    
    return results


if __name__ == "__main__":
    workspace_path = '/Users/daws/ADMF-PC/workspaces/test_ma_crossover_momentum_c94a270b'
    data_path = '/Users/daws/ADMF-PC/data/SPY_1m.parquet'
    
    results = analyze_ma_crossover_regime_filtered(workspace_path, data_path)