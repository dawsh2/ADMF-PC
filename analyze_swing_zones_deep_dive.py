#!/usr/bin/env python3
"""Deep dive analysis of swing pivot bounce zones - parameters and filters."""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from itertools import combinations

def load_market_data():
    """Load SPY 1m data for analysis."""
    data_path = Path('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
    df = pd.read_parquet(data_path)
    
    # Basic calculations
    df['returns'] = df['close'].pct_change()
    
    # Simple moving averages
    df['sma20'] = df['close'].rolling(20).mean()
    df['sma50'] = df['close'].rolling(50).mean()
    df['sma200'] = df['close'].rolling(200).mean()
    
    # RSI calculation (simplified)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Volume
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Volatility
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['volatility_rank'] = df['volatility_20'].rolling(1000).rank(pct=True)
    
    # VWAP and Session VWAP
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['pv'] = df['typical_price'] * df['volume']
    df['cumulative_pv'] = df['pv'].cumsum()
    df['cumulative_volume'] = df['volume'].cumsum()
    df['vwap'] = df['cumulative_pv'] / df['cumulative_volume']
    
    # Session VWAP (resets daily)
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    df['session_pv'] = df.groupby('date')['pv'].cumsum()
    df['session_volume'] = df.groupby('date')['volume'].cumsum()
    df['session_vwap'] = df['session_pv'] / df['session_volume']
    
    # Price relative to moving averages
    df['price_to_sma20'] = (df['close'] / df['sma20'] - 1) * 100
    df['price_to_sma50'] = (df['close'] / df['sma50'] - 1) * 100
    
    # Market regime
    df['trend'] = np.where(df['sma50'] > df['sma200'], 'up', 'down')
    df['trend_strength'] = np.where(df['sma20'] > df['sma50'], 'strong', 'weak')
    df['above_vwap'] = df['close'] > df['vwap']
    df['above_session_vwap'] = df['close'] > df['session_vwap']
    
    # Time features
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['minute'] = pd.to_datetime(df['timestamp']).dt.minute
    df['time_of_day'] = df['hour'] + df['minute'] / 60
    
    # Market sessions
    df['session'] = 'regular'
    df.loc[(df['hour'] == 9) & (df['minute'] < 30), 'session'] = 'premarket'
    df.loc[(df['hour'] >= 16), 'session'] = 'afterhours'
    df.loc[(df['hour'] == 9) & (df['minute'] >= 30) & (df['minute'] < 45), 'session'] = 'opening'
    df.loc[(df['hour'] == 15) & (df['minute'] >= 45), 'session'] = 'closing'
    
    return df

def extract_parameters_1m(strategy_idx):
    """Extract parameters for 1m strategies."""
    # From config: 6 * 3 * 5 * 5 * 4 = 1800 combinations
    sr_periods = [10, 15, 20, 30, 40, 50]
    min_touches = [2, 3, 4]
    entry_zones = [0.001, 0.0015, 0.002, 0.0025, 0.003]
    exit_zones = [0.001, 0.0015, 0.002, 0.0025, 0.003]
    min_ranges = [0.002, 0.003, 0.004, 0.005]
    
    # Calculate indices
    min_range_idx = strategy_idx % 4
    strategy_idx //= 4
    exit_zone_idx = strategy_idx % 5
    strategy_idx //= 5
    entry_zone_idx = strategy_idx % 5
    strategy_idx //= 5
    touch_idx = strategy_idx % 3
    strategy_idx //= 3
    sr_idx = strategy_idx % 6
    
    return {
        'sr_period': sr_periods[sr_idx],
        'min_touches': min_touches[touch_idx],
        'entry_zone': entry_zones[entry_zone_idx],
        'exit_zone': exit_zones[exit_zone_idx],
        'min_range': min_ranges[min_range_idx]
    }

def analyze_all_strategies(workspace_path):
    """Analyze all strategies and their parameters."""
    # Load metadata
    metadata_path = Path(workspace_path) / 'metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Get all strategy components
    components = {k: v for k, v in metadata.get('components', {}).items() 
                  if v.get('component_type') == 'strategy'}
    
    # Load market data
    print("Loading market data...")
    market_df = load_market_data()
    
    all_trades = []
    strategy_performance = []
    
    # Process all strategies
    for strategy_name, strategy_data in components.items():
        # Extract strategy index
        try:
            strategy_idx = int(strategy_name.split('_')[-1])
        except:
            continue
            
        # Get parameters
        params = extract_parameters_1m(strategy_idx)
        
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
                    
                    trade_data = {
                        'strategy_idx': strategy_idx,
                        'direction': 'long' if current_position['direction'] > 0 else 'short',
                        'net_return': net_return,
                        'net_return_bps': net_return * 10000,
                        'bars_held': bar_idx - current_position['entry_bar'],
                        # Parameters
                        **params,
                        # Market conditions
                        'trend': entry_conditions.get('trend', 'unknown'),
                        'trend_strength': entry_conditions.get('trend_strength', 'unknown'),
                        'above_vwap': entry_conditions.get('above_vwap', False),
                        'above_session_vwap': entry_conditions.get('above_session_vwap', False),
                        'rsi': entry_conditions.get('rsi', 50),
                        'volatility_rank': entry_conditions.get('volatility_rank', 0.5),
                        'volume_ratio': entry_conditions.get('volume_ratio', 1),
                        'hour': entry_conditions.get('hour', 12),
                        'session': entry_conditions.get('session', 'regular'),
                        'price_to_sma20': entry_conditions.get('price_to_sma20', 0),
                        'price_to_sma50': entry_conditions.get('price_to_sma50', 0),
                    }
                    
                    trades.append(trade_data)
                    all_trades.append(trade_data)
                
                if signal != 0:
                    current_position = {
                        'entry_price': price,
                        'entry_bar': bar_idx,
                        'direction': signal
                    }
                else:
                    current_position = None
        
        # Store strategy performance
        if trades:
            trades_df = pd.DataFrame(trades)
            strategy_performance.append({
                'strategy_idx': strategy_idx,
                **params,
                'total_trades': len(trades_df),
                'avg_return_bps': trades_df['net_return_bps'].mean(),
                'win_rate': (trades_df['net_return'] > 0).mean(),
                'long_trades': len(trades_df[trades_df['direction'] == 'long']),
                'short_trades': len(trades_df[trades_df['direction'] == 'short']),
                'long_return_bps': trades_df[trades_df['direction'] == 'long']['net_return_bps'].mean() if len(trades_df[trades_df['direction'] == 'long']) > 0 else 0,
                'short_return_bps': trades_df[trades_df['direction'] == 'short']['net_return_bps'].mean() if len(trades_df[trades_df['direction'] == 'short']) > 0 else 0,
            })
    
    return pd.DataFrame(all_trades), pd.DataFrame(strategy_performance)

def analyze_parameter_impact(strategy_df):
    """Analyze impact of each parameter on performance."""
    print("\n=== PARAMETER IMPACT ANALYSIS ===")
    
    parameters = ['sr_period', 'min_touches', 'entry_zone', 'exit_zone', 'min_range']
    
    for param in parameters:
        print(f"\n--- {param.upper()} Impact ---")
        param_stats = strategy_df.groupby(param).agg({
            'avg_return_bps': ['mean', 'std'],
            'total_trades': 'mean',
            'win_rate': 'mean',
            'strategy_idx': 'count'
        }).round(2)
        
        param_stats.columns = ['avg_bps', 'bps_std', 'avg_trades', 'avg_win_rate', 'count']
        param_stats = param_stats.sort_values('avg_bps', ascending=False)
        print(param_stats)
        
        # Find best value
        best_value = param_stats['avg_bps'].idxmax()
        print(f"\nBest {param}: {best_value} ({param_stats.loc[best_value, 'avg_bps']:.2f} bps)")

def analyze_filters(trades_df):
    """Analyze various filter combinations."""
    print("\n=== FILTER ANALYSIS ===")
    
    # Single filters
    filters = {
        "All trades": trades_df.index >= 0,
        "Longs only": trades_df['direction'] == 'long',
        "Shorts only": trades_df['direction'] == 'short',
        
        # Trend filters
        "Uptrend only": trades_df['trend'] == 'up',
        "Downtrend only": trades_df['trend'] == 'down',
        "Strong trend": trades_df['trend_strength'] == 'strong',
        
        # VWAP filters
        "Above VWAP": trades_df['above_vwap'] == True,
        "Below VWAP": trades_df['above_vwap'] == False,
        "Above session VWAP": trades_df['above_session_vwap'] == True,
        
        # Volatility filters
        "Vol > 60%": trades_df['volatility_rank'] > 0.6,
        "Vol > 70%": trades_df['volatility_rank'] > 0.7,
        "Vol > 80%": trades_df['volatility_rank'] > 0.8,
        "Vol > 85%": trades_df['volatility_rank'] > 0.85,
        "Vol > 90%": trades_df['volatility_rank'] > 0.9,
        "Vol > 95%": trades_df['volatility_rank'] > 0.95,
        
        # RSI filters
        "RSI > 60": trades_df['rsi'] > 60,
        "RSI < 40": trades_df['rsi'] < 40,
        "RSI extreme (>70 or <30)": (trades_df['rsi'] > 70) | (trades_df['rsi'] < 30),
        
        # Volume filters
        "High volume (>1.5x avg)": trades_df['volume_ratio'] > 1.5,
        
        # Time filters
        "Morning (9:30-11:00)": (trades_df['hour'] == 9) | ((trades_df['hour'] == 10)),
        "Afternoon (14:00-16:00)": trades_df['hour'] >= 14,
        "Opening 30min": trades_df['session'] == 'opening',
        "Closing 15min": trades_df['session'] == 'closing',
    }
    
    # Test single filters
    print("\n--- Single Filter Performance ---")
    print(f"{'Filter':<30} {'Trades':<10} {'Avg(bps)':<10} {'Win%':<8} {'Long%':<8} {'Short%':<8}")
    print("-" * 74)
    
    single_results = []
    for name, mask in filters.items():
        filtered = trades_df[mask]
        if len(filtered) > 10:
            long_pct = (filtered['direction'] == 'long').mean() * 100
            short_pct = (filtered['direction'] == 'short').mean() * 100
            result = {
                'filter': name,
                'trades': len(filtered),
                'avg_bps': filtered['net_return_bps'].mean(),
                'win_rate': (filtered['net_return'] > 0).mean() * 100,
                'long_pct': long_pct,
                'short_pct': short_pct
            }
            single_results.append(result)
            print(f"{name:<30} {len(filtered):<10} {result['avg_bps']:<10.2f} {result['win_rate']:<8.1f} {long_pct:<8.1f} {short_pct:<8.1f}")
    
    # Test combinations of profitable filters
    print("\n--- Combined Filter Performance ---")
    
    # Focus on filters that showed positive returns
    profitable_filters = {
        "Vol>70": trades_df['volatility_rank'] > 0.7,
        "Vol>80": trades_df['volatility_rank'] > 0.8,
        "Vol>85": trades_df['volatility_rank'] > 0.85,
        "Vol>90": trades_df['volatility_rank'] > 0.9,
        "Shorts": trades_df['direction'] == 'short',
        "Longs": trades_df['direction'] == 'long',
        "Downtrend": trades_df['trend'] == 'down',
        "BelowVWAP": ~trades_df['above_vwap'],
        "HighVolume": trades_df['volume_ratio'] > 1.5,
    }
    
    # Test combinations
    combo_results = []
    for r in range(2, 4):  # 2 and 3 filter combinations
        for combo in combinations(profitable_filters.keys(), r):
            mask = profitable_filters[combo[0]]
            for filter_name in combo[1:]:
                mask = mask & profitable_filters[filter_name]
            
            filtered = trades_df[mask]
            if len(filtered) > 20:  # Need minimum trades
                avg_bps = filtered['net_return_bps'].mean()
                if avg_bps > 0:  # Only show profitable
                    combo_results.append({
                        'filters': ' + '.join(combo),
                        'trades': len(filtered),
                        'avg_bps': avg_bps,
                        'win_rate': (filtered['net_return'] > 0).mean() * 100,
                        'daily_trades': len(filtered) / 306
                    })
    
    # Sort by performance
    combo_results.sort(key=lambda x: x['avg_bps'], reverse=True)
    
    print(f"\n{'Filter Combination':<40} {'Trades':<10} {'Daily':<8} {'Avg(bps)':<10} {'Win%':<8}")
    print("-" * 76)
    
    for result in combo_results[:20]:  # Top 20
        print(f"{result['filters']:<40} {result['trades']:<10} {result['daily_trades']:<8.1f} "
              f"{result['avg_bps']:<10.2f} {result['win_rate']:<8.1f}")
    
    return single_results, combo_results

def analyze_best_strategies(strategy_df, trades_df):
    """Find best parameter combinations."""
    print("\n=== BEST STRATEGY PARAMETERS ===")
    
    # Top 10 by average return
    top_strategies = strategy_df.nlargest(10, 'avg_return_bps')
    
    print("\nTop 10 Parameter Combinations:")
    print(f"{'SR':<6} {'Touch':<6} {'Entry':<8} {'Exit':<8} {'MinRng':<8} {'Trades':<8} {'Avg(bps)':<10} {'Win%':<8}")
    print("-" * 70)
    
    for _, row in top_strategies.iterrows():
        print(f"{row['sr_period']:<6} {row['min_touches']:<6} {row['entry_zone']:<8.3f} "
              f"{row['exit_zone']:<8.3f} {row['min_range']:<8.3f} {row['total_trades']:<8} "
              f"{row['avg_return_bps']:<10.2f} {row['win_rate']*100:<8.1f}")
    
    # Best high-frequency strategies (>1000 trades)
    high_freq = strategy_df[strategy_df['total_trades'] > 1000].nlargest(10, 'avg_return_bps')
    if len(high_freq) > 0:
        print("\n\nBest High-Frequency Strategies (>1000 trades):")
        for _, row in high_freq.iterrows():
            print(f"SR:{row['sr_period']} Entry:{row['entry_zone']:.3f} Exit:{row['exit_zone']:.3f} "
                  f"→ {row['total_trades']} trades, {row['avg_return_bps']:.2f} bps")

def main():
    workspace_path = Path('/Users/daws/ADMF-PC/workspaces/signal_generation_320d109d')
    
    print("=== SWING PIVOT BOUNCE ZONES - COMPREHENSIVE ANALYSIS ===")
    print(f"Analyzing workspace: {workspace_path.name}\n")
    
    # Load all trades and strategy performance
    print("Processing all strategies...")
    trades_df, strategy_df = analyze_all_strategies(workspace_path)
    
    print(f"\nTotal strategies analyzed: {len(strategy_df)}")
    print(f"Total trades: {len(trades_df)}")
    print(f"Overall average: {trades_df['net_return_bps'].mean():.2f} bps/trade")
    
    # Parameter impact
    analyze_parameter_impact(strategy_df)
    
    # Filter analysis
    single_results, combo_results = analyze_filters(trades_df)
    
    # Best strategies
    analyze_best_strategies(strategy_df, trades_df)
    
    # Long vs Short breakdown
    print("\n\n=== LONG vs SHORT PERFORMANCE ===")
    for direction in ['long', 'short']:
        dir_trades = trades_df[trades_df['direction'] == direction]
        print(f"\n{direction.upper()} trades: {len(dir_trades)}")
        
        # By volatility
        print(f"\nVolatility impact on {direction}s:")
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
            vol_trades = dir_trades[dir_trades['volatility_rank'] > threshold]
            if len(vol_trades) > 10:
                print(f"  Vol>{int(threshold*100)}%: {len(vol_trades)} trades, "
                      f"{vol_trades['net_return_bps'].mean():.2f} bps, "
                      f"{(vol_trades['net_return'] > 0).mean()*100:.1f}% win")
        
        # By trend
        print(f"\nTrend impact on {direction}s:")
        for trend in ['up', 'down']:
            trend_trades = dir_trades[dir_trades['trend'] == trend]
            if len(trend_trades) > 0:
                print(f"  {trend} trend: {len(trend_trades)} trades, "
                      f"{trend_trades['net_return_bps'].mean():.2f} bps")
    
    # Save results
    trades_df.to_csv('swing_zones_all_trades.csv', index=False)
    strategy_df.to_csv('swing_zones_strategy_performance.csv', index=False)
    
    print("\n\nResults saved to CSV files for further analysis.")

if __name__ == "__main__":
    main()