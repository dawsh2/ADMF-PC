#!/usr/bin/env python3
"""
Clarify the difference between annualization methods and their validity.
"""

import duckdb
import pandas as pd

def clarify_annualization_methods(workspace_path, data_path):
    """Compare different ways to annualize returns and explain the differences."""
    
    con = duckdb.connect()
    
    signal_file = f'{workspace_path}/traces/SPY_1m/signals/rsi/SPY_rsi_tuned_test.parquet'
    
    print('CLARIFYING ANNUALIZATION METHODS')
    print('=' * 60)
    
    # Get signals and trades (using same logic as before)
    signals_df = con.execute(f'SELECT idx, val FROM read_parquet("{signal_file}") ORDER BY idx').df()
    price_data = con.execute(f'SELECT bar_index, close FROM read_parquet("{data_path}")').df()
    price_dict = dict(zip(price_data['bar_index'], price_data['close']))
    
    # Build trades list
    trades = []
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
                
                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': exit_idx,
                    'direction': entry_direction,
                    'holding_bars': exit_idx - entry_idx,
                    'return_pct': raw_return * 100
                })
        i += 1
    
    trades_df = pd.DataFrame(trades)
    
    print(f'Total trades analyzed: {len(trades_df)}')
    print(f'Average return per trade: {trades_df["return_pct"].mean():.6f}%')
    print(f'Average holding period: {trades_df["holding_bars"].mean():.1f} bars')
    
    # Timeline analysis
    min_bar = trades_df['entry_idx'].min()
    max_bar = trades_df['exit_idx'].max()
    total_data_bars = max_bar - min_bar
    
    print(f'\\nDATA TIMELINE:')
    print(f'  First trade entry: bar {min_bar:,}')
    print(f'  Last trade exit: bar {max_bar:,}')
    print(f'  Total data span: {total_data_bars:,} bars')
    
    # Convert to time periods
    bars_per_day = 390  # 6.5 trading hours * 60 minutes
    trading_days_per_year = 252
    bars_per_year = bars_per_day * trading_days_per_year
    
    data_span_days = total_data_bars / bars_per_day
    data_span_years = total_data_bars / bars_per_year
    
    print(f'  Data span: {data_span_days:.1f} trading days = {data_span_years:.3f} years')
    
    # Method comparisons
    print(f'\\nANNUALIZATION METHOD COMPARISON:')
    
    avg_return_per_trade = trades_df['return_pct'].mean()
    total_return = trades_df['return_pct'].sum()
    avg_holding_bars = trades_df['holding_bars'].mean()
    
    # Method 1: Trade frequency scaling
    trades_per_day = bars_per_day / avg_holding_bars
    trades_per_year = trades_per_day * trading_days_per_year
    method1_annual = avg_return_per_trade * trades_per_year
    
    print(f'\\n  METHOD 1 - Trade Frequency Scaling:')
    print(f'    Logic: If avg trade makes {avg_return_per_trade:.6f}% in {avg_holding_bars:.1f} bars,')
    print(f'           and we can make {trades_per_year:.0f} such trades per year,')
    print(f'           then annual return = {avg_return_per_trade:.6f}% × {trades_per_year:.0f} = {method1_annual:.2f}%')
    print(f'    Assumption: Unlimited trading opportunities')
    print(f'    Problem: Ignores market capacity and transaction costs')
    
    # Method 2: Actual time period
    method2_annual = total_return / data_span_years
    
    print(f'\\n  METHOD 2 - Actual Time Period:')
    print(f'    Logic: Total return {total_return:.4f}% over {data_span_years:.3f} years')
    print(f'           Annual return = {total_return:.4f}% ÷ {data_span_years:.3f} = {method2_annual:.2f}%')
    print(f'    Reality: What actually happened in this time period')
    print(f'    Problem: Small sample size ({data_span_years:.3f} years)')
    
    # Method 3: Portfolio simulation (more realistic)
    print(f'\\n  METHOD 3 - Portfolio Simulation:')
    print(f'    Starting capital: $100,000')
    
    capital = 100000
    trade_history = []
    
    for _, trade in trades_df.iterrows():
        # Assume we invest full capital in each trade
        trade_return_decimal = trade['return_pct'] / 100
        capital_after_trade = capital * (1 + trade_return_decimal)
        profit = capital_after_trade - capital
        
        trade_history.append({
            'capital_before': capital,
            'return_pct': trade['return_pct'],
            'profit': profit,
            'capital_after': capital_after_trade
        })
        
        capital = capital_after_trade
    
    total_portfolio_return = (capital - 100000) / 100000 * 100
    method3_annual = total_portfolio_return / data_span_years
    
    print(f'    Final capital: ${capital:,.2f}')
    print(f'    Total portfolio return: {total_portfolio_return:.4f}%')
    print(f'    Annualized portfolio return: {method3_annual:.2f}%')
    print(f'    Reality: Compound returns with realistic capital constraints')
    
    # Which method is most appropriate?
    print(f'\\nWHICH METHOD IS MOST APPROPRIATE?')
    print(f'\\n  For BACKTESTING: Method 2 or 3 (actual performance)')
    print(f'    - Shows what really happened')
    print(f'    - Accounts for limited opportunities')
    print(f'    - More conservative and realistic')
    print(f'\\n  For THEORETICAL SCALING: Method 1 (with caveats)')
    print(f'    - Useful if strategy can scale to higher frequency')
    print(f'    - Must subtract realistic transaction costs')
    print(f'    - Must consider market impact and capacity')
    print(f'\\n  RECOMMENDATION: Use Method 2/3 for realistic assessment')
    print(f'                   Use Method 1 only for theoretical maximum')
    
    # Transaction cost break-even
    print(f'\\nTRANSACTION COST REALITY CHECK:')
    print(f'  Average return per trade: {avg_return_per_trade:.6f}%')
    print(f'  Round-trip break-even cost: {avg_return_per_trade/2:.6f}% per trade')
    print(f'  In basis points: {avg_return_per_trade/2*100:.2f} bps')
    print(f'  ⚠️  Even 1 bps (0.01%) cost makes strategy unprofitable!')
    
    return {
        'method1_annual': method1_annual,
        'method2_annual': method2_annual, 
        'method3_annual': method3_annual,
        'data_span_years': data_span_years,
        'trades_per_year_theoretical': trades_per_year,
        'avg_return_per_trade': avg_return_per_trade
    }


if __name__ == "__main__":
    workspace_path = '/Users/daws/ADMF-PC/workspaces/test_rsi_tuned_oos_4805b412'
    data_path = '/Users/daws/ADMF-PC/data/SPY_1m.parquet'
    
    results = clarify_annualization_methods(workspace_path, data_path)