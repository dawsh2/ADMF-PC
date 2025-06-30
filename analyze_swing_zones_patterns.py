#!/usr/bin/env python3
"""Find profitable patterns in swing pivot bounce zones signals."""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import talib

def load_market_data():
    """Load SPY 1m data for analysis."""
    # Load the 1m SPY data
    data_path = Path('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
    if not data_path.exists():
        # Try CSV
        data_path = Path('/Users/daws/ADMF-PC/data/SPY_1m.csv')
    
    df = pd.read_parquet(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    
    # Calculate indicators for analysis
    df['returns'] = df['close'].pct_change()
    df['sma50'] = talib.SMA(df['close'], timeperiod=50)
    df['sma200'] = talib.SMA(df['close'], timeperiod=200)
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['volume_sma'] = talib.SMA(df['volume'], timeperiod=20)
    
    # Volatility metrics
    df['volatility_20'] = df['returns'].rolling(20).std() * np.sqrt(252 * 390)  # Annualized
    df['volatility_pct'] = df['volatility_20'].rolling(252*390).rank(pct=True)
    
    # VWAP calculation
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Session VWAP (resets at 9:30 AM)
    df['session'] = df.index.normalize()
    df['session_vwap'] = df.groupby('session').apply(
        lambda x: (x['close'] * x['volume']).cumsum() / x['volume'].cumsum()
    ).values
    
    # Market regime
    df['trend'] = np.where(df['sma50'] > df['sma200'], 'up', 'down')
    df['above_vwap'] = df['close'] > df['vwap']
    df['above_session_vwap'] = df['close'] > df['session_vwap']
    
    return df

def analyze_signal_patterns(workspace_path, num_strategies=10):
    """Analyze patterns in signal data."""
    # Load metadata
    metadata_path = Path(workspace_path) / 'metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Get strategy components
    components = {k: v for k, v in metadata.get('components', {}).items() 
                  if v.get('component_type') == 'strategy'}
    
    # Sort by strategy index and take top N
    sorted_strategies = sorted(components.items(), 
                              key=lambda x: int(x[0].split('_')[-1]))[:num_strategies]
    
    # Load market data
    print("Loading market data...")
    market_df = load_market_data()
    
    all_results = []
    
    for strategy_name, strategy_data in sorted_strategies:
        print(f"\nAnalyzing {strategy_name}...")
        
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
                    'entry_signal': signal,
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
                entry_conditions = market_df.iloc[current_position['entry_bar']]
                exit_conditions = market_df.iloc[bar_idx]
                
                trades.append({
                    'entry_bar': current_position['entry_bar'],
                    'exit_bar': bar_idx,
                    'direction': 'long' if current_position['direction'] > 0 else 'short',
                    'gross_return': gross_return,
                    'net_return': net_return,
                    'bars_held': bar_idx - current_position['entry_bar'],
                    # Market conditions at entry
                    'trend': entry_conditions['trend'],
                    'above_vwap': entry_conditions['above_vwap'],
                    'above_session_vwap': entry_conditions['above_session_vwap'],
                    'rsi': entry_conditions['rsi'],
                    'volatility_pct': entry_conditions['volatility_pct'],
                    'atr': entry_conditions['atr'],
                    'volume_ratio': entry_conditions['volume'] / entry_conditions['volume_sma'],
                    # Price action
                    'price_to_sma50': entry_conditions['close'] / entry_conditions['sma50'] - 1,
                    'price_to_sma200': entry_conditions['close'] / entry_conditions['sma200'] - 1,
                })
                
                if signal != 0:
                    current_position = {
                        'entry_price': price,
                        'entry_bar': bar_idx,
                        'direction': signal
                    }
                else:
                    current_position = None
        
        if not trades:
            continue
            
        trades_df = pd.DataFrame(trades)
        
        # Analyze patterns
        print(f"Total trades: {len(trades_df)}")
        print(f"Overall return: {trades_df['net_return'].mean()*10000:.2f} bps")
        
        # Direction analysis
        print("\n--- Direction Analysis ---")
        for direction in ['long', 'short']:
            dir_trades = trades_df[trades_df['direction'] == direction]
            if len(dir_trades) > 0:
                print(f"{direction.upper()}: {len(dir_trades)} trades, "
                      f"{dir_trades['net_return'].mean()*10000:.2f} bps/trade, "
                      f"{(dir_trades['net_return'] > 0).mean()*100:.1f}% win rate")
        
        # Trend analysis
        print("\n--- Trend Analysis ---")
        for trend in ['up', 'down']:
            trend_trades = trades_df[trades_df['trend'] == trend]
            if len(trend_trades) > 0:
                print(f"{trend.upper()} trend: {len(trend_trades)} trades, "
                      f"{trend_trades['net_return'].mean()*10000:.2f} bps/trade")
                
                # Direction in trend
                for direction in ['long', 'short']:
                    subset = trend_trades[trend_trades['direction'] == direction]
                    if len(subset) > 0:
                        print(f"  {direction}: {len(subset)} trades, "
                              f"{subset['net_return'].mean()*10000:.2f} bps/trade")
        
        # VWAP analysis
        print("\n--- VWAP Analysis ---")
        above_vwap = trades_df[trades_df['above_vwap']]
        below_vwap = trades_df[~trades_df['above_vwap']]
        print(f"Above VWAP: {len(above_vwap)} trades, {above_vwap['net_return'].mean()*10000:.2f} bps")
        print(f"Below VWAP: {len(below_vwap)} trades, {below_vwap['net_return'].mean()*10000:.2f} bps")
        
        # Volatility analysis
        print("\n--- Volatility Analysis ---")
        for pct in [50, 70, 80, 90]:
            high_vol = trades_df[trades_df['volatility_pct'] > pct/100]
            if len(high_vol) > 0:
                print(f"Vol > {pct}th pct: {len(high_vol)} trades, "
                      f"{high_vol['net_return'].mean()*10000:.2f} bps/trade")
        
        # Find best filters
        print("\n--- Best Filter Combinations ---")
        
        # Test combinations
        filters = [
            ("Shorts in uptrend", (trades_df['direction'] == 'short') & (trades_df['trend'] == 'up')),
            ("Longs in downtrend", (trades_df['direction'] == 'long') & (trades_df['trend'] == 'down')),
            ("High vol shorts", (trades_df['direction'] == 'short') & (trades_df['volatility_pct'] > 0.7)),
            ("Above VWAP longs", (trades_df['direction'] == 'long') & trades_df['above_vwap']),
            ("Below VWAP shorts", (trades_df['direction'] == 'short') & ~trades_df['above_vwap']),
            ("Counter-trend + high vol", 
             ((trades_df['direction'] == 'short') & (trades_df['trend'] == 'up') & (trades_df['volatility_pct'] > 0.7))),
        ]
        
        best_filters = []
        for name, mask in filters:
            filtered = trades_df[mask]
            if len(filtered) > 10:  # Need minimum trades
                avg_return = filtered['net_return'].mean() * 10000
                if avg_return > 0:
                    best_filters.append((name, len(filtered), avg_return))
        
        best_filters.sort(key=lambda x: x[2], reverse=True)
        
        for name, count, avg_bps in best_filters[:5]:
            print(f"{name}: {count} trades ({count/len(trades_df)*100:.1f}%), {avg_bps:.2f} bps/trade")
        
        # Store results
        all_results.append({
            'strategy': strategy_name,
            'trades': len(trades_df),
            'overall_bps': trades_df['net_return'].mean() * 10000,
            'trades_df': trades_df
        })
    
    return all_results

def main():
    """Run pattern analysis."""
    workspace_path = Path('/Users/daws/ADMF-PC/workspaces/signal_generation_320d109d')
    
    print("=== SWING PIVOT BOUNCE ZONES - PATTERN ANALYSIS ===")
    print(f"Analyzing workspace: {workspace_path.name}")
    
    # Analyze top strategies
    results = analyze_signal_patterns(workspace_path, num_strategies=5)
    
    # Aggregate analysis across all strategies
    if results:
        print("\n\n=== AGGREGATE ANALYSIS ACROSS ALL STRATEGIES ===")
        
        all_trades = pd.concat([r['trades_df'] for r in results])
        print(f"Total trades analyzed: {len(all_trades)}")
        print(f"Overall average: {all_trades['net_return'].mean()*10000:.2f} bps/trade")
        
        # Find universally good filters
        print("\n--- Universal Patterns ---")
        
        filters = [
            ("All shorts", all_trades['direction'] == 'short'),
            ("All longs", all_trades['direction'] == 'long'),
            ("Shorts in uptrend", (all_trades['direction'] == 'short') & (all_trades['trend'] == 'up')),
            ("Longs in downtrend", (all_trades['direction'] == 'long') & (all_trades['trend'] == 'down')),
            ("Vol > 70%", all_trades['volatility_pct'] > 0.7),
            ("Vol > 80%", all_trades['volatility_pct'] > 0.8),
            ("Counter-trend shorts + Vol>70%", 
             (all_trades['direction'] == 'short') & (all_trades['trend'] == 'up') & (all_trades['volatility_pct'] > 0.7)),
            ("High RSI shorts", (all_trades['direction'] == 'short') & (all_trades['rsi'] > 65)),
            ("Low RSI longs", (all_trades['direction'] == 'long') & (all_trades['rsi'] < 35)),
        ]
        
        print("\nFilter Performance:")
        print("-" * 80)
        print(f"{'Filter':<40} {'Trades':<10} {'% of Total':<12} {'Avg (bps)':<12} {'Win Rate':<10}")
        print("-" * 80)
        
        for name, mask in filters:
            filtered = all_trades[mask]
            if len(filtered) > 20:
                avg_bps = filtered['net_return'].mean() * 10000
                win_rate = (filtered['net_return'] > 0).mean() * 100
                pct_of_total = len(filtered) / len(all_trades) * 100
                
                print(f"{name:<40} {len(filtered):<10} {pct_of_total:<12.1f} {avg_bps:<12.2f} {win_rate:<10.1f}")

if __name__ == "__main__":
    main()