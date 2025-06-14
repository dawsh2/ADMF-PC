#!/usr/bin/env python3
"""
Analyze MA crossover momentum strategy that's always in position.

This strategy alternates between long (1) and short (-1) without flat periods.
"""

import duckdb
import pandas as pd

def analyze_ma_crossover_always_in(workspace_path, data_path):
    """Calculate returns for always-in-position strategy."""
    
    con = duckdb.connect()
    
    signal_file = f'{workspace_path}/traces/SPY_1m/signals/momentum/SPY_ma_crossover_momentum_test.parquet'
    
    print('ANALYZING MA CROSSOVER MOMENTUM (ALWAYS IN POSITION)')
    print('=' * 60)
    
    # Get all signals
    signals_df = con.execute(f'SELECT idx, val FROM read_parquet("{signal_file}") ORDER BY idx').df()
    print(f'Total signal changes: {len(signals_df)}')
    
    # Show signal distribution
    signal_counts = signals_df['val'].value_counts().sort_index()
    print(f'\\nSignal distribution:')
    for signal, count in signal_counts.items():
        pct = count / len(signals_df) * 100
        signal_name = 'Long' if signal == 1 else 'Short' if signal == -1 else 'Flat'
        print(f'  {signal} ({signal_name}): {count} ({pct:.1f}%)')
    
    # Get price data
    price_data = con.execute(f'SELECT bar_index, close FROM read_parquet("{data_path}")').df()
    price_dict = dict(zip(price_data['bar_index'], price_data['close']))
    
    # For always-in strategy, each signal change is a position flip
    trades = []
    
    for i in range(len(signals_df) - 1):
        # Entry is current signal, exit is next signal change
        entry_signal = signals_df.iloc[i]
        exit_signal = signals_df.iloc[i + 1]
        
        entry_idx = entry_signal['idx']
        exit_idx = exit_signal['idx']
        direction = entry_signal['val']
        
        if entry_idx in price_dict and exit_idx in price_dict:
            entry_price = price_dict[entry_idx]
            exit_price = price_dict[exit_idx]
            holding_bars = exit_idx - entry_idx
            
            if direction == 1:  # Long position
                trade_return = (exit_price - entry_price) / entry_price * 100
            else:  # Short position
                trade_return = (entry_price - exit_price) / entry_price * 100
            
            trades.append({
                'entry_idx': entry_idx,
                'exit_idx': exit_idx,
                'direction': direction,
                'holding_bars': holding_bars,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'return_pct': trade_return
            })
    
    if len(trades) == 0:
        print('No trades found!')
        return None
    
    trades_df = pd.DataFrame(trades)
    
    print(f'\\nTrade analysis (position flips):')
    print(f'  Total position changes: {len(trades_df)}')
    
    # Holding period analysis
    print(f'\\nHolding period analysis:')
    print(f'  Average: {trades_df["holding_bars"].mean():.1f} bars')
    print(f'  Median: {trades_df["holding_bars"].median():.1f} bars')
    print(f'  Min: {trades_df["holding_bars"].min()} bars')
    print(f'  Max: {trades_df["holding_bars"].max()} bars')
    
    # Performance metrics
    avg_return_per_trade = trades_df['return_pct'].mean()
    volatility_per_trade = trades_df['return_pct'].std()
    total_return = trades_df['return_pct'].sum()
    win_rate = (trades_df['return_pct'] > 0).mean() * 100
    
    print(f'\\nPERFORMANCE METRICS:')
    print(f'  Average return per position: {avg_return_per_trade:.6f}%')
    print(f'  Volatility per position: {volatility_per_trade:.6f}%')
    print(f'  Total return: {total_return:.4f}%')
    print(f'  Win rate: {win_rate:.2f}%')
    print(f'  Best position: {trades_df["return_pct"].max():.4f}%')
    print(f'  Worst position: {trades_df["return_pct"].min():.4f}%')
    
    # Sharpe ratio
    sharpe_ratio = avg_return_per_trade / volatility_per_trade if volatility_per_trade > 0 else 0
    print(f'  Sharpe ratio (per position): {sharpe_ratio:.4f}')
    
    # Annualized calculations
    total_data_bars = trades_df['exit_idx'].max() - trades_df['entry_idx'].min()
    bars_per_day = 390
    trading_days_per_year = 252
    data_days = total_data_bars / bars_per_day
    
    positions_per_day = len(trades_df) / data_days
    positions_per_year = positions_per_day * trading_days_per_year
    
    annual_return = avg_return_per_trade * positions_per_year
    annual_volatility = volatility_per_trade * (positions_per_year ** 0.5)
    annual_sharpe = annual_return / annual_volatility if annual_volatility > 0 else 0
    
    print(f'\\nANNUALIZED PROJECTIONS:')
    print(f'  Data period: {data_days:.1f} days')
    print(f'  Position changes per day: {positions_per_day:.1f}')
    print(f'  Position changes per year: {positions_per_year:.0f}')
    print(f'  Annualized return: {annual_return:.2f}%')
    print(f'  Annualized volatility: {annual_volatility:.2f}%')
    print(f'  Annualized Sharpe ratio: {annual_sharpe:.4f}')
    
    # Direction analysis
    long_trades = trades_df[trades_df['direction'] == 1]
    short_trades = trades_df[trades_df['direction'] == -1]
    
    print(f'\\nDIRECTIONAL ANALYSIS:')
    if len(long_trades) > 0:
        print(f'  Long positions: {len(long_trades)} ({len(long_trades)/len(trades_df)*100:.1f}%)')
        print(f'    Avg return: {long_trades["return_pct"].mean():.4f}%')
        print(f'    Win rate: {(long_trades["return_pct"] > 0).mean()*100:.1f}%')
    
    if len(short_trades) > 0:
        print(f'  Short positions: {len(short_trades)} ({len(short_trades)/len(trades_df)*100:.1f}%)')
        print(f'    Avg return: {short_trades["return_pct"].mean():.4f}%')
        print(f'    Win rate: {(short_trades["return_pct"] > 0).mean()*100:.1f}%')
    
    # Transaction cost analysis (2x normal because always flipping position)
    print(f'\\nTRANSACTION COST SENSITIVITY (includes position flip cost):')
    cost_levels = [0, 1, 2, 5, 10]  # basis points
    for cost_bps in cost_levels:
        cost_pct = cost_bps / 100.0
        # For position flips, cost is 2x (exit current + enter new)
        net_return_per_trade = avg_return_per_trade - (2 * cost_pct)
        net_annual_return = net_return_per_trade * positions_per_year
        print(f'  {cost_bps} bps cost: {net_annual_return:.2f}% annual return')
        if net_annual_return < 0:
            print(f'    ⚠️  Strategy unprofitable at this cost level')
    
    # Sample positions
    print(f'\\nSAMPLE POSITIONS:')
    sample_trades = trades_df.head(10)[['entry_idx', 'exit_idx', 'direction', 'holding_bars', 'return_pct']]
    print(sample_trades.to_string(index=False))
    
    # Compare to RSI Tuned
    print(f'\\n📊 COMPARISON TO RSI TUNED:')
    print(f'  MA Crossover: {annual_return:.2f}% annual, {annual_sharpe:.4f} Sharpe')
    print(f'  RSI Tuned: 1.56% annual, 0.018 Sharpe')
    print(f'  MA Crossover is {"better" if annual_return > 1.56 else "worse"} in returns')
    print(f'  MA Crossover is {"better" if annual_sharpe > 0.018 else "worse"} in risk-adjusted returns')
    
    return {
        'total_positions': len(trades_df),
        'avg_return_per_position': avg_return_per_trade,
        'win_rate': win_rate,
        'annualized_return': annual_return,
        'annualized_sharpe': annual_sharpe,
        'avg_holding_bars': trades_df['holding_bars'].mean()
    }


if __name__ == "__main__":
    workspace_path = '/Users/daws/ADMF-PC/workspaces/test_ma_crossover_momentum_c94a270b'
    data_path = '/Users/daws/ADMF-PC/data/SPY_1m.parquet'
    
    results = analyze_ma_crossover_always_in(workspace_path, data_path)
    
    if results:
        print(f'\\n🎯 MA CROSSOVER MOMENTUM SUMMARY:')
        print(f'📈 Annualized return: {results["annualized_return"]:.2f}%')
        print(f'📊 Sharpe ratio: {results["annualized_sharpe"]:.4f}')
        print(f'🎲 Win rate: {results["win_rate"]:.1f}%')
        print(f'⏱️  Avg holding: {results["avg_holding_bars"]:.1f} bars')