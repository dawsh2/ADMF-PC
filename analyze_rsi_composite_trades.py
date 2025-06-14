#!/usr/bin/env python3
"""
Analyze RSI composite strategy by identifying complete trade cycles.

Each trade cycle: Entry (1 or -1) → Exit (0) → Next Entry
"""

import duckdb
import pandas as pd

def analyze_rsi_composite_trades(workspace_path, data_path):
    """Calculate returns by identifying complete trade cycles."""
    
    con = duckdb.connect()
    
    signal_file = f'{workspace_path}/traces/SPY_1m/signals/rsi/SPY_rsi_composite_test.parquet'
    
    print('ANALYZING RSI COMPOSITE COMPLETE TRADE CYCLES')
    print('=' * 60)
    
    # Get all signals
    signals_df = con.execute(f'SELECT idx, val FROM read_parquet("{signal_file}") ORDER BY idx').df()
    print(f'Total signal changes: {len(signals_df)}')
    
    # Identify complete trades (entry → exit cycles)
    trades = []
    
    i = 0
    while i < len(signals_df):
        current_signal = signals_df.iloc[i]
        
        # Look for entry signals (1 or -1)
        if current_signal['val'] in [1, -1]:
            entry_idx = current_signal['idx']
            entry_direction = current_signal['val']
            
            # Find the corresponding exit (next 0)
            exit_idx = None
            for j in range(i + 1, len(signals_df)):
                if signals_df.iloc[j]['val'] == 0:
                    exit_idx = signals_df.iloc[j]['idx']
                    break
            
            if exit_idx is not None:
                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': exit_idx,
                    'direction': entry_direction,
                    'holding_bars': exit_idx - entry_idx
                })
        
        i += 1
    
    print(f'\\nIdentified {len(trades)} complete trade cycles')
    
    if len(trades) == 0:
        print('No complete trades found!')
        return None
    
    # Analyze holding periods
    trades_df = pd.DataFrame(trades)
    print(f'\\nHolding period analysis:')
    print(f'  Average: {trades_df["holding_bars"].mean():.1f} bars')
    print(f'  Median: {trades_df["holding_bars"].median():.1f} bars')
    print(f'  Min: {trades_df["holding_bars"].min()} bars')
    print(f'  Max: {trades_df["holding_bars"].max()} bars')
    
    # Distribution of holding periods
    hold_dist = trades_df['holding_bars'].value_counts().sort_index()
    print(f'\\nHolding period distribution:')
    for period, count in hold_dist.head(10).items():
        pct = count / len(trades_df) * 100
        print(f'  {period} bars: {count} trades ({pct:.1f}%)')
    
    # Get price data
    price_data = con.execute(f'SELECT bar_index, close FROM read_parquet("{data_path}")').df()
    price_dict = dict(zip(price_data['bar_index'], price_data['close']))
    
    # Calculate returns for each trade
    trade_returns = []
    
    for _, trade in trades_df.iterrows():
        entry_idx = trade['entry_idx']
        exit_idx = trade['exit_idx']
        direction = trade['direction']
        
        if entry_idx in price_dict and exit_idx in price_dict:
            entry_price = price_dict[entry_idx]
            exit_price = price_dict[exit_idx]
            
            if direction == 1:  # Long trade
                trade_return = (exit_price - entry_price) / entry_price
            else:  # Short trade
                trade_return = (entry_price - exit_price) / entry_price
            
            trade_returns.append({
                'entry_idx': entry_idx,
                'exit_idx': exit_idx,
                'direction': direction,
                'holding_bars': trade['holding_bars'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'return_pct': trade_return * 100
            })
    
    if len(trade_returns) == 0:
        print('No valid trade returns calculated!')
        return None
    
    returns_df = pd.DataFrame(trade_returns)
    
    # Performance analysis
    total_trades = len(returns_df)
    avg_return_per_trade = returns_df['return_pct'].mean()
    volatility_per_trade = returns_df['return_pct'].std()
    total_return = returns_df['return_pct'].sum()
    win_rate = (returns_df['return_pct'] > 0).mean() * 100
    
    print(f'\\nTRADE-BY-TRADE PERFORMANCE:')
    print(f'  Total trades: {total_trades:,}')
    print(f'  Average return per trade: {avg_return_per_trade:.6f}%')
    print(f'  Volatility per trade: {volatility_per_trade:.6f}%')
    print(f'  Total return: {total_return:.4f}%')
    print(f'  Win rate: {win_rate:.2f}%')
    print(f'  Best trade: {returns_df["return_pct"].max():.4f}%')
    print(f'  Worst trade: {returns_df["return_pct"].min():.4f}%')
    
    # Sharpe ratio
    if volatility_per_trade > 0:
        sharpe_ratio = avg_return_per_trade / volatility_per_trade
        print(f'  Sharpe ratio: {sharpe_ratio:.4f}')
    
    # Calculate annualized metrics based on trade frequency
    avg_holding_bars = returns_df['holding_bars'].mean()
    bars_per_day = 390
    trading_days_per_year = 252
    
    # Calculate how many such trades could happen per year
    trades_per_day = bars_per_day / avg_holding_bars
    trades_per_year = trades_per_day * trading_days_per_year
    
    annual_return = avg_return_per_trade * trades_per_year
    annual_volatility = volatility_per_trade * (trades_per_year ** 0.5)
    annual_sharpe = annual_return / annual_volatility if annual_volatility > 0 else 0
    
    print(f'\\nANNUALIZED PROJECTIONS:')
    print(f'  Average holding period: {avg_holding_bars:.1f} bars')
    print(f'  Potential trades per year: {trades_per_year:.0f}')
    print(f'  Annualized return: {annual_return:.2f}%')
    print(f'  Annualized volatility: {annual_volatility:.2f}%')
    print(f'  Annualized Sharpe ratio: {annual_sharpe:.4f}')
    
    # Direction analysis
    long_trades = returns_df[returns_df['direction'] == 1]
    short_trades = returns_df[returns_df['direction'] == -1]
    
    print(f'\\nDIRECTIONAL ANALYSIS:')
    if len(long_trades) > 0:
        print(f'  Long trades: {len(long_trades)} ({len(long_trades)/total_trades*100:.1f}%)')
        print(f'    Avg return: {long_trades["return_pct"].mean():.4f}%')
        print(f'    Win rate: {(long_trades["return_pct"] > 0).mean()*100:.1f}%')
    
    if len(short_trades) > 0:
        print(f'  Short trades: {len(short_trades)} ({len(short_trades)/total_trades*100:.1f}%)')
        print(f'    Avg return: {short_trades["return_pct"].mean():.4f}%')
        print(f'    Win rate: {(short_trades["return_pct"] > 0).mean()*100:.1f}%')
    
    # Show sample trades
    print(f'\\nSAMPLE TRADES:')
    sample_trades = returns_df.head(10)[['entry_idx', 'exit_idx', 'direction', 'holding_bars', 'return_pct']]
    print(sample_trades.to_string(index=False))
    
    return {
        'total_trades': total_trades,
        'avg_return_per_trade': avg_return_per_trade,
        'win_rate': win_rate,
        'annualized_return': annual_return,
        'annualized_sharpe': annual_sharpe,
        'avg_holding_bars': avg_holding_bars
    }


if __name__ == "__main__":
    workspace_path = '/Users/daws/ADMF-PC/workspaces/test_rsi_composite_oos_02563d53'
    data_path = '/Users/daws/ADMF-PC/data/SPY_1m.parquet'
    
    results = analyze_rsi_composite_trades(workspace_path, data_path)
    
    if results:
        print(f'\\n🎯 CORRECTED ANALYSIS:')
        print(f'📈 Annualized return: {results["annualized_return"]:.2f}%')
        print(f'📊 Sharpe ratio: {results["annualized_sharpe"]:.4f}')
        print(f'🎲 Win rate: {results["win_rate"]:.1f}%')
        print(f'⏱️  Avg holding: {results["avg_holding_bars"]:.1f} bars')