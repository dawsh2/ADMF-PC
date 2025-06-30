#!/usr/bin/env python3
"""Find profitable patterns in swing pivot bounce zones signals - simplified version."""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def load_market_data():
    """Load SPY 1m data for analysis."""
    data_path = Path('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
    df = pd.read_parquet(data_path)
    
    # Basic calculations
    df['returns'] = df['close'].pct_change()
    
    # Simple moving averages
    df['sma50'] = df['close'].rolling(50).mean()
    df['sma200'] = df['close'].rolling(200).mean()
    
    # RSI calculation (simplified)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR (simplified)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # Volume
    df['volume_sma'] = df['volume'].rolling(20).mean()
    
    # Volatility
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['volatility_rank'] = df['volatility_20'].rolling(1000).rank(pct=True)
    
    # VWAP
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['pv'] = df['typical_price'] * df['volume']
    df['cumulative_pv'] = df['pv'].cumsum()
    df['cumulative_volume'] = df['volume'].cumsum()
    df['vwap'] = df['cumulative_pv'] / df['cumulative_volume']
    
    # Market regime
    df['trend'] = np.where(df['sma50'] > df['sma200'], 'up', 'down')
    df['above_vwap'] = df['close'] > df['vwap']
    
    return df

def analyze_patterns(workspace_path, num_strategies=5):
    """Analyze patterns in signal data."""
    # Load metadata
    metadata_path = Path(workspace_path) / 'metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Get strategy components
    components = {k: v for k, v in metadata.get('components', {}).items() 
                  if v.get('component_type') == 'strategy'}
    
    # Sort and take top N
    sorted_strategies = sorted(components.items(), 
                              key=lambda x: int(x[0].split('_')[-1]))[:num_strategies]
    
    # Load market data
    print("Loading market data...")
    market_df = load_market_data()
    print(f"Loaded {len(market_df)} bars of market data")
    
    all_trades = []
    
    for strategy_name, strategy_data in sorted_strategies:
        print(f"\nProcessing {strategy_name}...")
        
        # Load signal file
        signal_file = Path(workspace_path) / strategy_data['signal_file_path']
        if not signal_file.exists():
            continue
            
        signal_df = pd.read_parquet(signal_file)
        
        # Process signals into trades
        trades = []
        current_position = None
        
        for i in range(len(signal_df)):
            row = signal_df.iloc[i]
            signal = row['val']
            price = row['px']
            bar_idx = row['idx']
            
            if current_position is None and signal != 0:
                current_position = {
                    'entry_price': price,
                    'entry_bar': bar_idx,
                    'direction': signal
                }
            elif current_position is not None and (signal == 0 or signal != current_position['direction']):
                exit_price = price
                entry_price = current_position['entry_price']
                
                if current_position['direction'] > 0:
                    gross_return = (exit_price / entry_price) - 1
                else:
                    gross_return = (entry_price / exit_price) - 1
                
                net_return = gross_return - 0.0001  # 1bp round trip
                
                # Get market conditions at entry
                if bar_idx < len(market_df) and current_position['entry_bar'] < len(market_df):
                    entry_conditions = market_df.iloc[current_position['entry_bar']]
                    
                    trades.append({
                        'strategy': strategy_name,
                        'entry_bar': current_position['entry_bar'],
                        'exit_bar': bar_idx,
                        'direction': 'long' if current_position['direction'] > 0 else 'short',
                        'gross_return': gross_return,
                        'net_return': net_return,
                        'net_return_bps': net_return * 10000,
                        'bars_held': bar_idx - current_position['entry_bar'],
                        # Market conditions
                        'trend': entry_conditions.get('trend', 'unknown'),
                        'above_vwap': entry_conditions.get('above_vwap', False),
                        'rsi': entry_conditions.get('rsi', 50),
                        'volatility_rank': entry_conditions.get('volatility_rank', 0.5),
                        'volume_ratio': entry_conditions['volume'] / entry_conditions['volume_sma'] if entry_conditions['volume_sma'] > 0 else 1,
                    })
                
                if signal != 0:
                    current_position = {
                        'entry_price': price,
                        'entry_bar': bar_idx,
                        'direction': signal
                    }
                else:
                    current_position = None
        
        all_trades.extend(trades)
        print(f"  Found {len(trades)} trades")
    
    if not all_trades:
        print("No trades found!")
        return
    
    # Convert to DataFrame
    trades_df = pd.DataFrame(all_trades)
    print(f"\nTotal trades across all strategies: {len(trades_df)}")
    print(f"Overall average: {trades_df['net_return_bps'].mean():.2f} bps/trade")
    
    # Analyze patterns
    print("\n=== PATTERN ANALYSIS ===")
    
    # Direction
    print("\n--- By Direction ---")
    for direction in ['long', 'short']:
        dir_trades = trades_df[trades_df['direction'] == direction]
        if len(dir_trades) > 0:
            print(f"{direction.upper()}: {len(dir_trades)} trades, "
                  f"{dir_trades['net_return_bps'].mean():.2f} bps/trade, "
                  f"{(dir_trades['net_return'] > 0).mean()*100:.1f}% win rate")
    
    # Trend
    print("\n--- By Market Trend ---")
    for trend in ['up', 'down']:
        trend_trades = trades_df[trades_df['trend'] == trend]
        if len(trend_trades) > 0:
            print(f"\n{trend.upper()} trend: {len(trend_trades)} trades")
            for direction in ['long', 'short']:
                subset = trend_trades[trend_trades['direction'] == direction]
                if len(subset) > 0:
                    print(f"  {direction}: {len(subset)} trades, "
                          f"{subset['net_return_bps'].mean():.2f} bps/trade, "
                          f"{(subset['net_return'] > 0).mean()*100:.1f}% win")
    
    # VWAP
    print("\n--- By VWAP Position ---")
    above_vwap = trades_df[trades_df['above_vwap']]
    below_vwap = trades_df[~trades_df['above_vwap']]
    print(f"Above VWAP: {len(above_vwap)} trades, {above_vwap['net_return_bps'].mean():.2f} bps")
    print(f"Below VWAP: {len(below_vwap)} trades, {below_vwap['net_return_bps'].mean():.2f} bps")
    
    # Volatility
    print("\n--- By Volatility Rank ---")
    for threshold in [0.5, 0.7, 0.8, 0.9]:
        high_vol = trades_df[trades_df['volatility_rank'] > threshold]
        if len(high_vol) > 10:
            print(f"Vol > {int(threshold*100)}th percentile: {len(high_vol)} trades, "
                  f"{high_vol['net_return_bps'].mean():.2f} bps/trade")
    
    # Test filter combinations
    print("\n=== FILTER COMBINATIONS ===")
    
    filters = [
        ("All trades", trades_df.index >= 0),  # All
        ("Shorts only", trades_df['direction'] == 'short'),
        ("Longs only", trades_df['direction'] == 'long'),
        ("Shorts in uptrend", (trades_df['direction'] == 'short') & (trades_df['trend'] == 'up')),
        ("Longs in downtrend", (trades_df['direction'] == 'long') & (trades_df['trend'] == 'down')),
        ("High vol (>70%)", trades_df['volatility_rank'] > 0.7),
        ("High vol (>80%)", trades_df['volatility_rank'] > 0.8),
        ("Shorts + high vol", (trades_df['direction'] == 'short') & (trades_df['volatility_rank'] > 0.7)),
        ("Counter-trend shorts + vol", 
         (trades_df['direction'] == 'short') & (trades_df['trend'] == 'up') & (trades_df['volatility_rank'] > 0.7)),
        ("High RSI shorts", (trades_df['direction'] == 'short') & (trades_df['rsi'] > 65)),
        ("Low RSI longs", (trades_df['direction'] == 'long') & (trades_df['rsi'] < 35)),
        ("Above VWAP longs", (trades_df['direction'] == 'long') & trades_df['above_vwap']),
        ("Below VWAP shorts", (trades_df['direction'] == 'short') & ~trades_df['above_vwap']),
    ]
    
    print(f"\n{'Filter':<35} {'Trades':<10} {'% of Total':<12} {'Avg (bps)':<12} {'Win Rate':<10}")
    print("-" * 80)
    
    results = []
    for name, mask in filters:
        filtered = trades_df[mask]
        if len(filtered) > 0:
            avg_bps = filtered['net_return_bps'].mean()
            win_rate = (filtered['net_return'] > 0).mean() * 100
            pct_of_total = len(filtered) / len(trades_df) * 100
            
            results.append({
                'filter': name,
                'trades': len(filtered),
                'pct': pct_of_total,
                'avg_bps': avg_bps,
                'win_rate': win_rate
            })
            
            print(f"{name:<35} {len(filtered):<10} {pct_of_total:<12.1f} {avg_bps:<12.2f} {win_rate:<10.1f}")
    
    # Find best filters
    print("\n=== BEST PERFORMING FILTERS (Positive Returns) ===")
    positive_results = [r for r in results if r['avg_bps'] > 0 and r['trades'] > 20]
    positive_results.sort(key=lambda x: x['avg_bps'], reverse=True)
    
    for r in positive_results[:10]:
        print(f"{r['filter']}: {r['avg_bps']:.2f} bps/trade ({r['trades']} trades)")

def main():
    workspace_path = Path('/Users/daws/ADMF-PC/workspaces/signal_generation_320d109d')
    print("=== SWING PIVOT BOUNCE ZONES - PATTERN ANALYSIS ===")
    print(f"Analyzing workspace: {workspace_path.name}\n")
    
    analyze_patterns(workspace_path, num_strategies=10)

if __name__ == "__main__":
    main()