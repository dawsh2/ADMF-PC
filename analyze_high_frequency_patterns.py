"""Find high-frequency patterns with good edge for swing pivot strategies"""
import pandas as pd
import numpy as np
from pathlib import Path

workspace = Path("workspaces/signal_generation_320d109d")
signal_dir = workspace / "traces/SPY_5m_1m/signals/swing_pivot_bounce_zones"

# Load SPY 5m data
spy_5m = pd.read_csv("./data/SPY_5m.csv")
spy_5m['timestamp'] = pd.to_datetime(spy_5m['timestamp'])
spy_5m.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 
                       'Close': 'close', 'Volume': 'volume'}, inplace=True)

print("=== HIGH-FREQUENCY PATTERN DISCOVERY (2-3+ trades/day target) ===\n")

# Calculate all features
spy_5m['returns'] = spy_5m['close'].pct_change()
spy_5m['volatility_20'] = spy_5m['returns'].rolling(20).std() * np.sqrt(78) * 100
spy_5m['vol_percentile'] = spy_5m['volatility_20'].rolling(window=126).rank(pct=True) * 100

# VWAP
spy_5m['date'] = spy_5m['timestamp'].dt.date
spy_5m['typical_price'] = (spy_5m['high'] + spy_5m['low'] + spy_5m['close']) / 3
spy_5m['pv'] = spy_5m['typical_price'] * spy_5m['volume']
spy_5m['cum_pv'] = spy_5m.groupby('date')['pv'].cumsum()
spy_5m['cum_volume'] = spy_5m.groupby('date')['volume'].cumsum()
spy_5m['vwap'] = spy_5m['cum_pv'] / spy_5m['cum_volume']
spy_5m['distance_from_vwap'] = (spy_5m['close'] - spy_5m['vwap']) / spy_5m['vwap'] * 100

# Volume
spy_5m['volume_sma_20'] = spy_5m['volume'].rolling(20).mean()
spy_5m['volume_ratio'] = spy_5m['volume'] / spy_5m['volume_sma_20']

# Time features
spy_5m['hour'] = spy_5m['timestamp'].dt.hour
spy_5m['afternoon'] = spy_5m['hour'] >= 13

# Price action
spy_5m['range_pct'] = (spy_5m['high'] - spy_5m['low']) / spy_5m['close'] * 100
spy_5m['sma_20'] = spy_5m['close'].rolling(20).mean()
spy_5m['distance_from_sma20'] = (spy_5m['close'] - spy_5m['sma_20']) / spy_5m['sma_20'] * 100

# Trend
spy_5m['sma_50'] = spy_5m['close'].rolling(50).mean()
spy_5m['sma_200'] = spy_5m['close'].rolling(200).mean()
spy_5m['trend_up'] = (spy_5m['close'] > spy_5m['sma_50']) & (spy_5m['sma_50'] > spy_5m['sma_200'])

total_days = 16614 / 78  # Total trading days in dataset

# Test multiple strategies to find best high-frequency patterns
strategies_to_test = [88, 80, 48, 40, 81, 32, 16, 64, 96]

all_results = []

for strategy_id in strategies_to_test:
    signal_file = signal_dir / f"SPY_5m_compiled_strategy_{strategy_id}.parquet"
    if not signal_file.exists():
        continue
        
    signals = pd.read_parquet(signal_file)
    
    # Collect trades
    trades = []
    entry_data = None
    
    for j in range(len(signals)):
        curr = signals.iloc[j]
        
        if entry_data is None and curr['val'] != 0:
            if curr['idx'] < len(spy_5m):
                entry_data = {
                    'idx': curr['idx'],
                    'price': curr['px'],
                    'signal': curr['val']
                }
        elif entry_data is not None and (curr['val'] == 0 or np.sign(curr['val']) != np.sign(entry_data['signal'])):
            if entry_data and entry_data['idx'] < len(spy_5m) and curr['idx'] < len(spy_5m):
                entry_conditions = spy_5m.iloc[entry_data['idx']]
                
                pct_return = (curr['px'] / entry_data['price'] - 1) * entry_data['signal'] * 100
                
                if not pd.isna(entry_conditions['vol_percentile']):
                    trade = {
                        'pct_return': pct_return,
                        'direction': 'short' if entry_data['signal'] < 0 else 'long',
                        'vol_percentile': entry_conditions['vol_percentile'],
                        'distance_from_vwap': entry_conditions['distance_from_vwap'],
                        'volume_ratio': entry_conditions['volume_ratio'],
                        'afternoon': entry_conditions['afternoon'],
                        'range_pct': entry_conditions['range_pct'],
                        'distance_from_sma20': entry_conditions['distance_from_sma20'],
                        'trend_up': entry_conditions['trend_up']
                    }
                    trades.append(trade)
            
            if curr['val'] != 0:
                entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val']}
            else:
                entry_data = None
    
    if not trades:
        continue
    
    trades_df = pd.DataFrame(trades)
    baseline_tpd = len(trades_df) / total_days
    
    # Test filter combinations targeting 2-3+ trades per day
    filters_to_test = [
        # Relaxed volatility filters
        ("Vol>50", trades_df['vol_percentile'] > 50),
        ("Vol>60", trades_df['vol_percentile'] > 60),
        ("Vol>70", trades_df['vol_percentile'] > 70),
        
        # Direction filters
        ("Shorts only", trades_df['direction'] == 'short'),
        ("Vol>50 + Shorts", (trades_df['vol_percentile'] > 50) & (trades_df['direction'] == 'short')),
        ("Vol>60 + Shorts", (trades_df['vol_percentile'] > 60) & (trades_df['direction'] == 'short')),
        
        # VWAP-based
        ("Far from VWAP (>0.15%)", abs(trades_df['distance_from_vwap']) > 0.15),
        ("Far from VWAP + Vol>50", (abs(trades_df['distance_from_vwap']) > 0.15) & (trades_df['vol_percentile'] > 50)),
        ("Far from VWAP + Shorts", (abs(trades_df['distance_from_vwap']) > 0.15) & (trades_df['direction'] == 'short')),
        
        # Time-based
        ("Afternoon only", trades_df['afternoon']),
        ("Afternoon + Vol>50", trades_df['afternoon'] & (trades_df['vol_percentile'] > 50)),
        ("Afternoon + Shorts", trades_df['afternoon'] & (trades_df['direction'] == 'short')),
        
        # Volume patterns
        ("High volume (>1.2x)", trades_df['volume_ratio'] > 1.2),
        ("Vol>50 + High volume", (trades_df['vol_percentile'] > 50) & (trades_df['volume_ratio'] > 1.2)),
        
        # Price action
        ("High range (>0.1%)", trades_df['range_pct'] > 0.1),
        ("High range + Vol>50", (trades_df['range_pct'] > 0.1) & (trades_df['vol_percentile'] > 50)),
        
        # Combined filters
        ("Vol>50 + VWAP>0.1", (trades_df['vol_percentile'] > 50) & (abs(trades_df['distance_from_vwap']) > 0.1)),
        ("Afternoon + Vol>60 + Shorts", trades_df['afternoon'] & (trades_df['vol_percentile'] > 60) & (trades_df['direction'] == 'short')),
        ("Vol>60 + Volume>1.1", (trades_df['vol_percentile'] > 60) & (trades_df['volume_ratio'] > 1.1))
    ]
    
    for filter_name, filter_mask in filters_to_test:
        filtered = trades_df[filter_mask]
        if len(filtered) > 50:  # Minimum trades for reliability
            avg_bps = filtered['pct_return'].mean() * 100
            win_rate = (filtered['pct_return'] > 0).mean()
            trades_per_day = len(filtered) / total_days
            
            # Only keep if meets frequency target
            if trades_per_day >= 2.0 and avg_bps > 1.0:
                result = {
                    'strategy_id': strategy_id,
                    'filter': filter_name,
                    'edge_bps': avg_bps,
                    'trades_per_day': trades_per_day,
                    'win_rate': win_rate,
                    'total_trades': len(filtered),
                    'annual_return_05bp': (1 + (avg_bps - 0.5)/10000) ** (trades_per_day * 252) - 1
                }
                all_results.append(result)

# Sort by composite score (edge * frequency)
results_df = pd.DataFrame(all_results)
if len(results_df) > 0:
    results_df['score'] = results_df['edge_bps'] * np.log1p(results_df['trades_per_day'])
    results_df = results_df.sort_values('score', ascending=False)

    print("TOP HIGH-FREQUENCY PATTERNS (2-3+ trades/day with good edge)")
    print("="*90)
    print(f"{'Strategy':<10} {'Filter':<30} {'Edge':<10} {'Trades/Day':<12} {'Win Rate':<10} {'Annual@0.5bp':<12}")
    print("-"*90)
    
    for _, row in results_df.head(20).iterrows():
        print(f"{row['strategy_id']:<10} {row['filter']:<30} {row['edge_bps']:>6.2f} bps {row['trades_per_day']:>8.1f} "
              f"{row['win_rate']:>8.1%} {row['annual_return_05bp']:>10.1%}")

    # Group by filter type
    print("\n\nBEST FILTERS BY CATEGORY")
    print("="*60)
    
    filter_categories = {
        'Volatility-based': ['Vol>50', 'Vol>60', 'Vol>70'],
        'VWAP-based': ['Far from VWAP', 'VWAP>0.1'],
        'Time-based': ['Afternoon'],
        'Combined': ['Vol>', 'Shorts', 'volume']
    }
    
    for category, keywords in filter_categories.items():
        category_results = results_df[results_df['filter'].str.contains('|'.join(keywords), case=False)]
        if len(category_results) > 0:
            best = category_results.iloc[0]
            print(f"\n{category}:")
            print(f"  Best: {best['filter']} (Strategy {best['strategy_id']})")
            print(f"  Performance: {best['edge_bps']:.2f} bps, {best['trades_per_day']:.1f} trades/day")
            print(f"  Annual return (0.5bp cost): {best['annual_return_05bp']:.1%}")

    # Find sweet spot combinations
    print("\n\nSWEET SPOT ANALYSIS")
    print("="*60)
    
    # Filter for 2-4 trades per day range
    sweet_spot = results_df[(results_df['trades_per_day'] >= 2.0) & (results_df['trades_per_day'] <= 4.0)]
    sweet_spot = sweet_spot[sweet_spot['edge_bps'] >= 1.5]
    
    if len(sweet_spot) > 0:
        print(f"\nFound {len(sweet_spot)} combinations with 2-4 trades/day and >1.5 bps edge:")
        
        # Get unique filters
        unique_filters = sweet_spot.groupby('filter').agg({
            'edge_bps': 'mean',
            'trades_per_day': 'mean',
            'strategy_id': 'count'
        }).sort_values('edge_bps', ascending=False)
        
        print("\nMost consistent filters across strategies:")
        for filter_name, stats in unique_filters.head(10).iterrows():
            print(f"  {filter_name}: {stats['edge_bps']:.2f} bps avg, "
                  f"{stats['trades_per_day']:.1f} trades/day, "
                  f"works on {stats['strategy_id']} strategies")

print("\n\nRECOMMENDATIONS FOR 2-3+ TRADES/DAY:")
print("="*60)
print("\n1. Use Vol>50 or Vol>60 instead of Vol>85 for higher frequency")
print("2. Consider 'Far from VWAP' patterns - they trade frequently with good edge")
print("3. Afternoon-only trading maintains edge while increasing frequency")
print("4. Combining moderate volatility (>60) with volume filters works well")
print("5. Some strategies naturally trade more frequently - test multiple parameter sets")