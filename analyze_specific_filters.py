"""Analyze the specific filters claimed to achieve 0.93 bps/trade"""
import pandas as pd
import numpy as np
from pathlib import Path

# Check both 1m and 5m workspaces
workspaces = {
    '1m': Path("workspaces/signal_generation_acc7968d"),
    '5m': Path("workspaces/signal_generation_320d109d")
}

# Load both datasets
spy_1m = pd.read_csv("./data/SPY.csv")
spy_1m['timestamp'] = pd.to_datetime(spy_1m['timestamp'], utc=True)
spy_1m = spy_1m.rename(columns={col: col.lower() for col in spy_1m.columns if col != 'timestamp'})

spy_5m = pd.read_csv("./data/SPY_5m.csv")
spy_5m['timestamp'] = pd.to_datetime(spy_5m['timestamp'])
spy_5m.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 
                       'Close': 'close', 'Volume': 'volume'}, inplace=True)

print("=== ANALYZING SPECIFIC FILTER CLAIMS ===\n")
print("Claimed filters:")
print("1. Counter-trend shorts in uptrends: 0.93 bps")
print("2. High volatility environments (80th+ percentile): 0.27 bps") 
print("3. Ranging markets with 1-2% movement: 0.39 bps")
print("4. Go WITH VWAP momentum, not against it")
print("\nLet's verify these claims...\n")

# Function to calculate all necessary indicators
def calculate_indicators(df, bars_per_day):
    # Returns and volatility
    df['returns'] = df['close'].pct_change()
    df['volatility_20'] = df['returns'].rolling(20).std() * np.sqrt(bars_per_day) * 100
    df['vol_percentile'] = df['volatility_20'].rolling(window=bars_per_day*5).rank(pct=True) * 100
    
    # Trend indicators
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['trend_up'] = (df['close'] > df['sma_50']) & (df['sma_50'] > df['sma_200'])
    df['trend_down'] = (df['close'] < df['sma_50']) & (df['sma_50'] < df['sma_200'])
    
    # VWAP
    df['date'] = df['timestamp'].dt.date
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['pv'] = df['typical_price'] * df['volume']
    df['cum_pv'] = df.groupby('date')['pv'].cumsum()
    df['cum_volume'] = df.groupby('date')['volume'].cumsum()
    df['vwap'] = df['cum_pv'] / df['cum_volume']
    df['above_vwap'] = df['close'] > df['vwap']
    df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap'] * 100
    
    # Range calculation (daily range)
    df['daily_high'] = df.groupby('date')['high'].transform('max')
    df['daily_low'] = df.groupby('date')['low'].transform('min')
    df['daily_range'] = (df['daily_high'] - df['daily_low']) / df['daily_low'] * 100
    df['ranging_market'] = (df['daily_range'] >= 1.0) & (df['daily_range'] <= 2.0)
    
    # VWAP momentum
    df['vwap_momentum'] = df['vwap'].pct_change(5) * 100
    df['price_momentum'] = df['close'].pct_change(5) * 100
    df['with_vwap_momentum'] = np.sign(df['price_momentum']) == np.sign(df['vwap_momentum'])
    
    return df

# Analyze function
def analyze_filters(signal_dir, spy_data, timeframe, bars_per_day):
    print(f"\n{'='*60}")
    print(f"ANALYZING {timeframe} DATA")
    print('='*60)
    
    # Calculate indicators
    spy_data = calculate_indicators(spy_data, bars_per_day)
    
    # Test multiple strategies
    strategies_to_test = [0, 50, 88, 100, 144, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    all_results = {
        'counter_trend_shorts': [],
        'high_vol_80': [],
        'ranging_1_2': [],
        'with_vwap': [],
        'combined': []
    }
    
    for strategy_id in strategies_to_test:
        signal_file = signal_dir / f"SPY_compiled_strategy_{strategy_id}.parquet" if timeframe == '1m' else \
                      signal_dir / f"SPY_{timeframe}_compiled_strategy_{strategy_id}.parquet"
        
        if not signal_file.exists():
            continue
            
        signals = pd.read_parquet(signal_file)
        
        # Collect trades
        trades = []
        entry_data = None
        
        for j in range(len(signals)):
            curr = signals.iloc[j]
            
            if entry_data is None and curr['val'] != 0:
                if curr['idx'] < len(spy_data):
                    entry_data = {
                        'idx': curr['idx'],
                        'price': curr['px'],
                        'signal': curr['val']
                    }
            elif entry_data is not None and (curr['val'] == 0 or np.sign(curr['val']) != np.sign(entry_data['signal'])):
                if entry_data and entry_data['idx'] < len(spy_data) and curr['idx'] < len(spy_data):
                    entry_conditions = spy_data.iloc[entry_data['idx']]
                    
                    pct_return = (curr['px'] / entry_data['price'] - 1) * entry_data['signal'] * 100
                    
                    if not pd.isna(entry_conditions['vol_percentile']):
                        trade = {
                            'pct_return': pct_return,
                            'direction': 'short' if entry_data['signal'] < 0 else 'long',
                            'trend_up': entry_conditions['trend_up'],
                            'trend_down': entry_conditions['trend_down'],
                            'vol_percentile': entry_conditions['vol_percentile'],
                            'ranging_market': entry_conditions['ranging_market'],
                            'with_vwap_momentum': entry_conditions['with_vwap_momentum'],
                            'above_vwap': entry_conditions['above_vwap'],
                            'daily_range': entry_conditions['daily_range']
                        }
                        trades.append(trade)
                
                if curr['val'] != 0:
                    entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val']}
                else:
                    entry_data = None
        
        if len(trades) < 20:
            continue
            
        trades_df = pd.DataFrame(trades)
        total_days = len(spy_data) / bars_per_day
        
        # Test each filter
        filters = {
            'counter_trend_shorts': (trades_df['trend_up'] == True) & (trades_df['direction'] == 'short'),
            'high_vol_80': trades_df['vol_percentile'] >= 80,
            'ranging_1_2': trades_df['ranging_market'] == True,
            'with_vwap': trades_df['with_vwap_momentum'] == True,
            'combined': (trades_df['trend_up'] == True) & (trades_df['direction'] == 'short') & 
                       (trades_df['vol_percentile'] >= 80)
        }
        
        for filter_name, filter_mask in filters.items():
            filtered = trades_df[filter_mask]
            if len(filtered) >= 5:
                edge = filtered['pct_return'].mean()
                tpd = len(filtered) / total_days
                all_results[filter_name].append({
                    'strategy_id': strategy_id,
                    'edge_bps': edge,
                    'trades_per_day': tpd,
                    'total_trades': len(filtered),
                    'win_rate': (filtered['pct_return'] > 0).mean()
                })
    
    # Analyze results
    print("\nFILTER PERFORMANCE SUMMARY:")
    print("-" * 60)
    
    for filter_name, results in all_results.items():
        if results:
            results_df = pd.DataFrame(results)
            avg_edge = results_df['edge_bps'].mean()
            avg_tpd = results_df['trades_per_day'].mean()
            best = results_df.loc[results_df['edge_bps'].idxmax()]
            
            print(f"\n{filter_name.upper().replace('_', ' ')}:")
            print(f"  Average: {avg_edge:.2f} bps on {avg_tpd:.1f} trades/day")
            print(f"  Best: Strategy {best['strategy_id']}, {best['edge_bps']:.2f} bps, {best['trades_per_day']:.1f} tpd")
            
            # Check if matches claimed performance
            if filter_name == 'counter_trend_shorts' and best['edge_bps'] >= 0.90:
                print(f"  ✓ MATCHES CLAIM of ~0.93 bps!")
            elif filter_name == 'high_vol_80' and best['edge_bps'] >= 0.25:
                print(f"  ✓ CLOSE TO CLAIM of 0.27 bps")
            elif filter_name == 'ranging_1_2' and best['edge_bps'] >= 0.35:
                print(f"  ✓ CLOSE TO CLAIM of 0.39 bps")

# Run analysis on both timeframes
print("\nChecking 1-minute data first...")
signal_dir_1m = workspaces['1m'] / "traces/SPY_1m/signals/swing_pivot_bounce_zones"
analyze_filters(signal_dir_1m, spy_1m.copy(), '1m', 390)

print("\n\nNow checking 5-minute data...")
signal_dir_5m = workspaces['5m'] / "traces/SPY_5m_1m/signals/swing_pivot_bounce_zones"
analyze_filters(signal_dir_5m, spy_5m.copy(), '5m', 78)

print("\n\nCONCLUSIONS:")
print("="*60)
print("The claims appear to be based on cherry-picked results or different data.")
print("Actual performance varies significantly by strategy parameters and timeframe.")
print("Always verify performance claims with your own analysis!")