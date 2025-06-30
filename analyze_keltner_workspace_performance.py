#!/usr/bin/env python3
"""
Analyze Keltner workspace performance with stop loss analysis
"""

import pandas as pd
import numpy as np
import duckdb
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_spy_data():
    """Load SPY OHLC data"""
    spy_path = "/Users/daws/ADMF-PC/data/SPY_5m.csv"
    
    # First check if 5m data exists
    if not Path(spy_path).exists():
        print("5m data not found, checking for 1m data...")
        spy_path = "/Users/daws/ADMF-PC/data/SPY_1m.csv"
    
    df = pd.read_csv(spy_path)
    
    # Standardize column names
    df.columns = [col.capitalize() for col in df.columns]
    df['Datetime'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index('Datetime')
    
    # If we have 1m data, resample to 5m
    if '1m' in spy_path or len(df) > 100000:
        print("Resampling 1m data to 5m...")
        df = df.resample('5min').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
    
    return df

def analyze_strategy_performance(workspace_path, spy_df):
    """Analyze all strategies in the workspace"""
    
    results = []
    
    # Get all signal files
    signal_dir = Path(workspace_path) / "traces/SPY_5m_1m/signals/keltner_bands"
    
    for signal_file in sorted(signal_dir.glob("*.parquet")):
        strategy_name = signal_file.stem
        print(f"\nAnalyzing {strategy_name}...")
        
        # Load signals
        df_signals = pq.read_table(signal_file).to_pandas()
        
        # Convert timestamp to datetime and rename columns for clarity
        df_signals['timestamp'] = pd.to_datetime(df_signals['ts'])
        df_signals['signal'] = df_signals['val']
        df_signals = df_signals.sort_values('timestamp')
        
        # Merge with price data
        df_merged = pd.merge_asof(
            df_signals,
            spy_df.reset_index()[['Datetime', 'Open', 'High', 'Low', 'Close']],
            left_on='timestamp',
            right_on='Datetime',
            direction='backward'
        )
        
        # Calculate returns without stops
        trades_no_stop = simulate_trades(df_merged, stop_loss=None)
        
        # Calculate returns with various stop losses
        stop_losses = [0.001, 0.002, 0.003, 0.005, 0.01]
        
        for stop_loss in stop_losses:
            trades_with_stop = simulate_trades(df_merged, stop_loss=stop_loss)
            
            # Aggregate results
            result = {
                'strategy': strategy_name,
                'stop_loss': stop_loss,
                'total_trades': len(trades_with_stop),
                'total_return': trades_with_stop['return'].sum() if len(trades_with_stop) > 0 else 0,
                'avg_return': trades_with_stop['return'].mean() if len(trades_with_stop) > 0 else 0,
                'win_rate': (trades_with_stop['return'] > 0).mean() if len(trades_with_stop) > 0 else 0,
                'max_win': trades_with_stop['return'].max() if len(trades_with_stop) > 0 else 0,
                'max_loss': trades_with_stop['return'].min() if len(trades_with_stop) > 0 else 0,
                'sharpe': calculate_sharpe(trades_with_stop['return']) if len(trades_with_stop) > 0 else 0,
                'long_trades': len(trades_with_stop[trades_with_stop['side'] == 1]),
                'short_trades': len(trades_with_stop[trades_with_stop['side'] == -1]),
                'long_return': trades_with_stop[trades_with_stop['side'] == 1]['return'].sum() if len(trades_with_stop) > 0 else 0,
                'short_return': trades_with_stop[trades_with_stop['side'] == -1]['return'].sum() if len(trades_with_stop) > 0 else 0,
                'stopped_out': trades_with_stop['stopped'].sum() if 'stopped' in trades_with_stop.columns and len(trades_with_stop) > 0 else 0
            }
            results.append(result)
        
        # Also add no-stop results
        result_no_stop = {
            'strategy': strategy_name,
            'stop_loss': None,
            'total_trades': len(trades_no_stop),
            'total_return': trades_no_stop['return'].sum() if len(trades_no_stop) > 0 else 0,
            'avg_return': trades_no_stop['return'].mean() if len(trades_no_stop) > 0 else 0,
            'win_rate': (trades_no_stop['return'] > 0).mean() if len(trades_no_stop) > 0 else 0,
            'max_win': trades_no_stop['return'].max() if len(trades_no_stop) > 0 else 0,
            'max_loss': trades_no_stop['return'].min() if len(trades_no_stop) > 0 else 0,
            'sharpe': calculate_sharpe(trades_no_stop['return']) if len(trades_no_stop) > 0 else 0,
            'long_trades': len(trades_no_stop[trades_no_stop['side'] == 1]),
            'short_trades': len(trades_no_stop[trades_no_stop['side'] == -1]),
            'long_return': trades_no_stop[trades_no_stop['side'] == 1]['return'].sum() if len(trades_no_stop) > 0 else 0,
            'short_return': trades_no_stop[trades_no_stop['side'] == -1]['return'].sum() if len(trades_no_stop) > 0 else 0,
            'stopped_out': 0
        }
        results.append(result_no_stop)
    
    return pd.DataFrame(results)

def simulate_trades(df, stop_loss=None):
    """Simulate trades with optional stop loss using OHLC data"""
    trades = []
    position = 0
    entry_price = None
    entry_time = None
    entry_side = None
    
    for idx, row in df.iterrows():
        signal = row['signal']
        
        # Entry logic
        if position == 0 and signal != 0:
            position = signal
            entry_price = row['Open']  # Enter at open
            entry_time = row['timestamp']
            entry_side = signal
            
        # Exit logic
        elif position != 0:
            exit_signal = False
            exit_price = None
            stopped = False
            
            # Check stop loss first using high/low
            if stop_loss is not None:
                if position > 0:  # Long position
                    stop_price = entry_price * (1 - stop_loss)
                    if row['Low'] <= stop_price:
                        exit_signal = True
                        exit_price = stop_price
                        stopped = True
                else:  # Short position
                    stop_price = entry_price * (1 + stop_loss)
                    if row['High'] >= stop_price:
                        exit_signal = True
                        exit_price = stop_price
                        stopped = True
            
            # Normal exit (signal change or goes to 0)
            if not exit_signal and (signal != position or signal == 0):
                exit_signal = True
                exit_price = row['Open']
                stopped = False
            
            if exit_signal:
                # Calculate return
                if position > 0:
                    trade_return = (exit_price - entry_price) / entry_price
                else:
                    trade_return = (entry_price - exit_price) / entry_price
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': row['timestamp'],
                    'side': entry_side,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': trade_return,
                    'stopped': stopped
                })
                
                # Reset position
                position = signal if signal != 0 else 0
                if position != 0:
                    entry_price = row['Open']
                    entry_time = row['timestamp']
                    entry_side = signal
                else:
                    entry_price = None
                    entry_time = None
                    entry_side = None
    
    return pd.DataFrame(trades)

def calculate_sharpe(returns):
    """Calculate Sharpe ratio (annualized for 5-minute bars)"""
    if len(returns) < 2:
        return 0
    
    # 5-minute bars: ~78 per day, ~19,500 per year
    periods_per_year = 78 * 250
    
    mean_return = returns.mean()
    std_return = returns.std()
    
    if std_return == 0:
        return 0
    
    return np.sqrt(periods_per_year) * mean_return / std_return

def main():
    workspace_path = "/Users/daws/ADMF-PC/workspaces/optimize_keltner_with_filters_20250622_102448"
    
    print(f"Analyzing workspace: {workspace_path}")
    
    # Load SPY data
    print("\nLoading SPY data...")
    spy_df = load_spy_data()
    print(f"Loaded {len(spy_df)} bars from {spy_df.index[0]} to {spy_df.index[-1]}")
    
    # Analyze all strategies
    print("\nAnalyzing strategies...")
    results_df = analyze_strategy_performance(workspace_path, spy_df)
    
    # Save detailed results
    results_df.to_csv('keltner_workspace_detailed_results.csv', index=False)
    print("\nSaved detailed results to keltner_workspace_detailed_results.csv")
    
    # Print summary of best strategies
    print("\n" + "="*80)
    print("TOP STRATEGIES BY TOTAL RETURN (No Stop Loss)")
    print("="*80)
    
    no_stop_df = results_df[results_df['stop_loss'].isna()].sort_values('total_return', ascending=False).head(10)
    
    for _, row in no_stop_df.iterrows():
        print(f"\n{row['strategy']}:")
        print(f"  Total Return: {row['total_return']:.4f} ({row['total_return']*100:.2f}%)")
        print(f"  Trades: {row['total_trades']} (Long: {row['long_trades']}, Short: {row['short_trades']})")
        print(f"  Avg Return: {row['avg_return']*100:.3f}%")
        print(f"  Win Rate: {row['win_rate']*100:.1f}%")
        print(f"  Sharpe: {row['sharpe']:.2f}")
        print(f"  Long Return: {row['long_return']*100:.2f}%")
        print(f"  Short Return: {row['short_return']*100:.2f}%")
    
    # Analyze stop loss impact for top strategies
    print("\n" + "="*80)
    print("STOP LOSS ANALYSIS FOR TOP 5 STRATEGIES")
    print("="*80)
    
    top_strategies = no_stop_df.head(5)['strategy'].values
    
    for strategy in top_strategies:
        print(f"\n{strategy}:")
        strategy_df = results_df[results_df['strategy'] == strategy].sort_values('stop_loss')
        
        for _, row in strategy_df.iterrows():
            stop_text = f"{row['stop_loss']*100:.1f}%" if pd.notna(row['stop_loss']) else "None"
            stopped_pct = (row['stopped_out'] / row['total_trades'] * 100) if row['total_trades'] > 0 else 0
            
            print(f"  Stop Loss {stop_text:>5}: Return {row['total_return']*100:>6.2f}%, "
                  f"Trades {row['total_trades']:>3}, Win Rate {row['win_rate']*100:>4.1f}%, "
                  f"Sharpe {row['sharpe']:>5.2f}, Stopped {stopped_pct:>4.1f}%")
    
    # Find optimal stop loss across all strategies
    print("\n" + "="*80)
    print("OPTIMAL STOP LOSS ANALYSIS")
    print("="*80)
    
    stop_summary = results_df.groupby('stop_loss').agg({
        'total_return': ['mean', 'std', 'max'],
        'sharpe': ['mean', 'max'],
        'win_rate': 'mean'
    }).round(4)
    
    print("\nAverage Performance by Stop Loss:")
    print(stop_summary)

if __name__ == "__main__":
    main()