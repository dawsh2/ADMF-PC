#\!/usr/bin/env python3
"""
Analyze Keltner strategy performance on test data
"""
import pandas as pd
import numpy as np
from pathlib import Path

def load_signals(workspace_dir):
    """Load signal traces from workspace."""
    signals_file = Path(workspace_dir) / "traces/keltner_bands/SPY_5m_compiled_strategy_0.parquet"
    if not signals_file.exists():
        print(f"❌ Signal file not found: {signals_file}")
        return None
        
    signals_df = pd.read_parquet(signals_file)
    print(f"📊 Loaded {len(signals_df)} signal changes")
    return signals_df

def load_price_data():
    """Load SPY 5m price data."""
    data_file = Path("data/SPY_5m.csv")
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Apply test split (last 20%)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    print(f"📊 Test data: {len(test_df)} bars")
    return test_df

def compute_returns(signals_df, price_df):
    """Compute returns for sparse signals."""
    results = []
    
    # Create full signal series
    signal_series = pd.Series(0, index=price_df.index)
    
    # Fill in signals from sparse data
    signals_df['ts'] = pd.to_datetime(signals_df['ts'])
    for _, row in signals_df.iterrows():
        if row['ts'] in signal_series.index:
            signal_series.loc[row['ts']] = row['val']
    
    # Forward fill signals
    signal_series = signal_series.ffill()
    
    # Compute returns
    price_df['returns'] = price_df['close'].pct_change()
    price_df['signal'] = signal_series
    price_df['signal_returns'] = price_df['signal'].shift(1) * price_df['returns']
    
    # Track positions
    position = 0
    entry_price = None
    entry_time = None
    
    for timestamp, row in price_df.iterrows():
        signal = row['signal']
        
        if position == 0 and signal != 0:
            # Enter position
            position = signal
            entry_price = row['close']
            entry_time = timestamp
            
        elif position != 0 and signal != position:
            # Exit position
            exit_price = row['close']
            returns = (exit_price - entry_price) / entry_price * position
            duration = (timestamp - entry_time).total_seconds() / 60  # minutes
            
            results.append({
                'entry_time': entry_time,
                'exit_time': timestamp,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position': position,
                'returns': returns,
                'duration_minutes': duration
            })
            
            # Reset or reverse
            if signal != 0:
                position = signal
                entry_price = row['close']
                entry_time = timestamp
            else:
                position = 0
                entry_price = None
                entry_time = None
    
    return pd.DataFrame(results)

def analyze_performance(trades_df):
    """Analyze trading performance."""
    if len(trades_df) == 0:
        print("❌ No trades executed")
        return
        
    # Basic metrics
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['returns'] > 0])
    win_rate = winning_trades / total_trades * 100
    
    avg_return = trades_df['returns'].mean()
    total_return = (1 + trades_df['returns']).prod() - 1
    
    # Convert to basis points
    avg_return_bps = avg_return * 10000
    
    # Sharpe ratio (assuming 252 trading days, 78 5-min bars per day)
    daily_returns = trades_df.groupby(trades_df['entry_time'].dt.date)['returns'].sum()
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0
    
    # Position breakdown
    long_trades = trades_df[trades_df['position'] > 0]
    short_trades = trades_df[trades_df['position'] < 0]
    
    print("\n" + "="*60)
    print("KELTNER STRATEGY PERFORMANCE ON TEST DATA")
    print("="*60)
    
    print(f"\n📊 OVERALL METRICS:")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Avg Return per Trade: {avg_return_bps:.2f} bps")
    print(f"Total Return: {total_return*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    
    print(f"\n📈 LONG TRADES: {len(long_trades)}")
    if len(long_trades) > 0:
        print(f"Win Rate: {len(long_trades[long_trades['returns'] > 0])/len(long_trades)*100:.1f}%")
        print(f"Avg Return: {long_trades['returns'].mean()*10000:.2f} bps")
        print(f"Avg Duration: {long_trades['duration_minutes'].mean():.1f} min")
    
    print(f"\n📉 SHORT TRADES: {len(short_trades)}")
    if len(short_trades) > 0:
        print(f"Win Rate: {len(short_trades[short_trades['returns'] > 0])/len(short_trades)*100:.1f}%")
        print(f"Avg Return: {short_trades['returns'].mean()*10000:.2f} bps")
        print(f"Avg Duration: {short_trades['duration_minutes'].mean():.1f} min")
    
    # Duration analysis
    print(f"\n⏱️  DURATION ANALYSIS:")
    print(f"Avg Duration: {trades_df['duration_minutes'].mean():.1f} min")
    print(f"Median Duration: {trades_df['duration_minutes'].median():.1f} min")
    print(f"Max Duration: {trades_df['duration_minutes'].max():.1f} min")
    
    # Monthly breakdown
    trades_df['month'] = trades_df['entry_time'].dt.to_period('M')
    monthly_stats = trades_df.groupby('month').agg({
        'returns': ['count', 'mean', 'sum']
    })
    
    print("\n📅 MONTHLY BREAKDOWN:")
    for month, stats in monthly_stats.iterrows():
        count = stats[('returns', 'count')]
        avg_ret = stats[('returns', 'mean')] * 10000
        total_ret = stats[('returns', 'sum')] * 100
        print(f"{month}: {count} trades, {avg_ret:.1f} bps/trade, {total_ret:.1f}% total")

def main():
    # Latest workspace
    workspace = "config/keltner/config_2826/results/20250622_154455"
    
    # Load data
    signals_df = load_signals(workspace)
    if signals_df is None:
        return
        
    price_df = load_price_data()
    
    # Compute returns
    trades_df = compute_returns(signals_df, price_df)
    
    # Analyze performance
    analyze_performance(trades_df)
    
    # Save trades for further analysis
    trades_df.to_csv('keltner_test_trades.csv', index=False)
    print(f"\n💾 Saved {len(trades_df)} trades to keltner_test_trades.csv")

if __name__ == "__main__":
    main()