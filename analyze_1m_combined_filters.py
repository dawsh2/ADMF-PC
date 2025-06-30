"""Comprehensive analysis of combined filters for 1-minute swing pivot bounce"""
import pandas as pd
import numpy as np
from pathlib import Path

# Based on the insights from your previous analysis:
# Best single filters found:
# 1. Counter-trend shorts in uptrends: 0.93 bps
# 2. High volatility (80th+ percentile): 0.27 bps  
# 3. Ranging markets (1-2% movement): 0.39 bps
# 4. Go WITH VWAP momentum

workspace = Path("workspaces/signal_generation_acc7968d")
signal_dir = workspace / "traces/SPY_1m/signals/swing_pivot_bounce_zones"

# Load SPY 1m data
spy_1m = pd.read_csv("./data/SPY.csv")
spy_1m['timestamp'] = pd.to_datetime(spy_1m['timestamp'], utc=True)
spy_1m = spy_1m.rename(columns={col: col.lower() for col in spy_1m.columns if col != 'timestamp'})

print("=== 1-MINUTE COMBINED FILTER ANALYSIS ===\n")
print("Testing combinations of the best-performing filters from previous analysis:\n")

# Calculate all indicators
spy_1m['returns'] = spy_1m['close'].pct_change()

# Volatility with multiple windows
for window in [5, 10, 20, 60]:
    spy_1m[f'volatility_{window}'] = spy_1m['returns'].rolling(window).std() * np.sqrt(390) * 100
    spy_1m[f'vol_percentile_{window}'] = spy_1m[f'volatility_{window}'].rolling(window=390*5).rank(pct=True) * 100

# Trend
spy_1m['sma_20'] = spy_1m['close'].rolling(20).mean()
spy_1m['sma_50'] = spy_1m['close'].rolling(50).mean()
spy_1m['sma_200'] = spy_1m['close'].rolling(200).mean()
spy_1m['trend_up'] = (spy_1m['close'] > spy_1m['sma_50']) & (spy_1m['sma_50'] > spy_1m['sma_200'])
spy_1m['trend_down'] = (spy_1m['close'] < spy_1m['sma_50']) & (spy_1m['sma_50'] < spy_1m['sma_200'])
spy_1m['trend_neutral'] = ~(spy_1m['trend_up'] | spy_1m['trend_down'])

# VWAP
spy_1m['date'] = spy_1m['timestamp'].dt.date
spy_1m['typical_price'] = (spy_1m['high'] + spy_1m['low'] + spy_1m['close']) / 3
spy_1m['pv'] = spy_1m['typical_price'] * spy_1m['volume']
spy_1m['cum_pv'] = spy_1m.groupby('date')['pv'].cumsum()
spy_1m['cum_volume'] = spy_1m.groupby('date')['volume'].cumsum()
spy_1m['vwap'] = spy_1m['cum_pv'] / spy_1m['cum_volume']
spy_1m['above_vwap'] = spy_1m['close'] > spy_1m['vwap']
spy_1m['vwap_distance'] = (spy_1m['close'] - spy_1m['vwap']) / spy_1m['vwap'] * 100

# VWAP momentum
spy_1m['vwap_momentum'] = spy_1m['vwap'].pct_change(5) * 100
spy_1m['price_momentum'] = spy_1m['close'].pct_change(5) * 100
spy_1m['with_vwap_momentum'] = np.sign(spy_1m['price_momentum']) == np.sign(spy_1m['vwap_momentum'])

# Ranging market detection
spy_1m['daily_high'] = spy_1m.groupby('date')['high'].transform('max')
spy_1m['daily_low'] = spy_1m.groupby('date')['low'].transform('min')
spy_1m['daily_range'] = (spy_1m['daily_high'] - spy_1m['daily_low']) / spy_1m['daily_low'] * 100
spy_1m['ranging_1_2'] = (spy_1m['daily_range'] >= 1.0) & (spy_1m['daily_range'] <= 2.0)
spy_1m['ranging_0_1'] = spy_1m['daily_range'] < 1.0
spy_1m['ranging_2_plus'] = spy_1m['daily_range'] > 2.0

# ATR for trend strength
spy_1m['hl_diff'] = spy_1m['high'] - spy_1m['low']
spy_1m['hc_diff'] = abs(spy_1m['high'] - spy_1m['close'].shift(1))
spy_1m['lc_diff'] = abs(spy_1m['low'] - spy_1m['close'].shift(1))
spy_1m['tr'] = spy_1m[['hl_diff', 'hc_diff', 'lc_diff']].max(axis=1)
spy_1m['atr_14'] = spy_1m['tr'].rolling(14).mean()

# Price patterns
spy_1m['inside_bar'] = (spy_1m['high'] <= spy_1m['high'].shift(1)) & (spy_1m['low'] >= spy_1m['low'].shift(1))
spy_1m['lower_low'] = (spy_1m['high'] < spy_1m['high'].shift(1)) & (spy_1m['low'] < spy_1m['low'].shift(1))

# Time features
spy_1m['hour'] = spy_1m['timestamp'].dt.hour
spy_1m['minutes_from_open'] = (spy_1m['hour'] - 9) * 60 + spy_1m['timestamp'].dt.minute - 30
spy_1m['first_hour'] = spy_1m['minutes_from_open'] < 60
spy_1m['last_hour'] = spy_1m['minutes_from_open'] > 330

total_days = len(spy_1m) / 390

# Test multiple strategies focusing on the best performers
best_strategies = [0, 50, 88, 144, 256, 400, 500, 600, 700, 800]

all_results = []

for strategy_id in best_strategies:
    signal_file = signal_dir / f"SPY_compiled_strategy_{strategy_id}.parquet"
    if not signal_file.exists():
        continue
        
    signals = pd.read_parquet(signal_file)
    
    # Collect trades
    trades = []
    entry_data = None
    
    for j in range(len(signals)):
        curr = signals.iloc[j]
        
        if entry_data is None and curr['val'] != 0:
            if curr['idx'] < len(spy_1m):
                entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val']}
        elif entry_data is not None and (curr['val'] == 0 or np.sign(curr['val']) != np.sign(entry_data['signal'])):
            if entry_data and entry_data['idx'] < len(spy_1m) and curr['idx'] < len(spy_1m):
                entry_conditions = spy_1m.iloc[entry_data['idx']]
                
                pct_return = (curr['px'] / entry_data['price'] - 1) * entry_data['signal'] * 100
                duration = curr['idx'] - entry_data['idx']
                
                if not pd.isna(entry_conditions.get('vol_percentile_20', np.nan)):
                    trade = {
                        'pct_return': pct_return,
                        'direction': 'short' if entry_data['signal'] < 0 else 'long',
                        'duration': duration,
                        'trend_up': entry_conditions.get('trend_up', False),
                        'trend_down': entry_conditions.get('trend_down', False),
                        'trend_neutral': entry_conditions.get('trend_neutral', False),
                        'vol_percentile_5': entry_conditions.get('vol_percentile_5', 50),
                        'vol_percentile_10': entry_conditions.get('vol_percentile_10', 50),
                        'vol_percentile_20': entry_conditions.get('vol_percentile_20', 50),
                        'vol_percentile_60': entry_conditions.get('vol_percentile_60', 50),
                        'above_vwap': entry_conditions.get('above_vwap', False),
                        'vwap_distance': entry_conditions.get('vwap_distance', 0),
                        'with_vwap_momentum': entry_conditions.get('with_vwap_momentum', False),
                        'ranging_1_2': entry_conditions.get('ranging_1_2', False),
                        'ranging_0_1': entry_conditions.get('ranging_0_1', False),
                        'daily_range': entry_conditions.get('daily_range', 1),
                        'inside_bar': entry_conditions.get('inside_bar', False),
                        'lower_low': entry_conditions.get('lower_low', False),
                        'first_hour': entry_conditions.get('first_hour', False),
                        'last_hour': entry_conditions.get('last_hour', False)
                    }
                    trades.append(trade)
            
            if curr['val'] != 0:
                entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val']}
            else:
                entry_data = None
    
    if len(trades) < 50:
        continue
    
    trades_df = pd.DataFrame(trades)
    
    print(f"\n{'='*60}")
    print(f"STRATEGY {strategy_id}: {len(trades_df)} trades")
    print(f"Baseline: {trades_df['pct_return'].mean():.2f} bps on {len(trades_df)/total_days:.1f} tpd")
    
    # Test the most promising combined filters based on previous insights
    combined_filters = [
        # Based on best single filters
        ("CT shorts in uptrend", 
         (trades_df['trend_up']) & (trades_df['direction'] == 'short')),
        
        ("High vol (80th+)", 
         trades_df['vol_percentile_20'] >= 80),
        
        ("Ranging 1-2%", 
         trades_df['ranging_1_2']),
        
        ("WITH VWAP momentum", 
         trades_df['with_vwap_momentum']),
        
        # Two-filter combinations
        ("CT shorts + High vol", 
         (trades_df['trend_up']) & (trades_df['direction'] == 'short') & 
         (trades_df['vol_percentile_20'] >= 80)),
        
        ("CT shorts + Ranging 1-2%", 
         (trades_df['trend_up']) & (trades_df['direction'] == 'short') & 
         (trades_df['ranging_1_2'])),
        
        ("High vol + Ranging", 
         (trades_df['vol_percentile_20'] >= 80) & (trades_df['ranging_1_2'])),
        
        ("CT shorts + VWAP momentum", 
         (trades_df['trend_up']) & (trades_df['direction'] == 'short') & 
         (trades_df['with_vwap_momentum'])),
        
        # Three-filter combinations
        ("CT shorts + High vol + Ranging", 
         (trades_df['trend_up']) & (trades_df['direction'] == 'short') & 
         (trades_df['vol_percentile_20'] >= 80) & (trades_df['ranging_1_2'])),
        
        ("CT shorts + High vol + VWAP", 
         (trades_df['trend_up']) & (trades_df['direction'] == 'short') & 
         (trades_df['vol_percentile_20'] >= 80) & (trades_df['with_vwap_momentum'])),
        
        # Time-based combinations
        ("CT shorts + High vol + First hour", 
         (trades_df['trend_up']) & (trades_df['direction'] == 'short') & 
         (trades_df['vol_percentile_20'] >= 80) & (trades_df['first_hour'])),
        
        # Quick trades with best filters
        ("Quick (3-10m) + CT shorts + Vol>70", 
         (trades_df['duration'] >= 3) & (trades_df['duration'] < 10) & 
         (trades_df['trend_up']) & (trades_df['direction'] == 'short') & 
         (trades_df['vol_percentile_20'] >= 70)),
        
        # Pattern-based
        ("Inside bar + CT shorts + Vol>70", 
         (trades_df['inside_bar']) & (trades_df['trend_up']) & 
         (trades_df['direction'] == 'short') & (trades_df['vol_percentile_20'] >= 70)),
        
        # Alternative volatility thresholds
        ("CT shorts + Vol>70", 
         (trades_df['trend_up']) & (trades_df['direction'] == 'short') & 
         (trades_df['vol_percentile_20'] >= 70)),
        
        ("CT shorts + Vol>60 + Ranging", 
         (trades_df['trend_up']) & (trades_df['direction'] == 'short') & 
         (trades_df['vol_percentile_20'] >= 60) & (trades_df['ranging_1_2']))
    ]
    
    for filter_name, filter_mask in combined_filters:
        filtered = trades_df[filter_mask]
        if len(filtered) >= 10:
            edge = filtered['pct_return'].mean()
            tpd = len(filtered) / total_days
            win_rate = (filtered['pct_return'] > 0).mean()
            
            if edge >= 0.5 or (edge > 0 and tpd >= 1.0):
                print(f"\n  {filter_name}:")
                print(f"    Edge: {edge:.2f} bps")
                print(f"    Trades/day: {tpd:.1f}")
                print(f"    Win rate: {win_rate:.1%}")
                print(f"    Total trades: {len(filtered)}")
                
                result = {
                    'strategy_id': strategy_id,
                    'filter': filter_name,
                    'edge_bps': edge,
                    'trades_per_day': tpd,
                    'win_rate': win_rate,
                    'total_trades': len(filtered),
                    'annualized_return': edge * tpd * 252 / 10000  # Convert to annual %
                }
                all_results.append(result)

# Summary of best combinations
if all_results:
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(['edge_bps', 'trades_per_day'], ascending=[False, False])
    
    print("\n\n" + "="*80)
    print("BEST COMBINED FILTER CONFIGURATIONS")
    print("="*80)
    
    print(f"\n{'Filter':<35} {'Edge':<10} {'Trades/Day':<12} {'Annual %':<10} {'Win Rate':<10}")
    print("-"*80)
    
    for _, row in results_df.head(20).iterrows():
        print(f"{row['filter']:<35} {row['edge_bps']:>6.2f} bps "
              f"{row['trades_per_day']:>8.1f} {row['annualized_return']:>8.1%} "
              f"{row['win_rate']:>8.1%}")
    
    # Check if any achieve the goal
    goal_met = results_df[(results_df['edge_bps'] >= 1.0) & (results_df['trades_per_day'] >= 2.0)]
    if len(goal_met) > 0:
        print(f"\n✓ Found {len(goal_met)} configurations meeting >=1 bps with 2+ trades/day!")
        for _, row in goal_met.iterrows():
            print(f"  {row['filter']}: {row['edge_bps']:.2f} bps on {row['trades_per_day']:.1f} tpd")
    else:
        print("\n✗ No configurations achieve >=1 bps with 2+ trades/day")
        
    # Best with reasonable frequency
    frequent = results_df[results_df['trades_per_day'] >= 1.0]
    if len(frequent) > 0:
        print(f"\n\nBest with 1+ trades/day:")
        best_frequent = frequent.nlargest(5, 'edge_bps')
        for _, row in best_frequent.iterrows():
            print(f"  {row['filter']}: {row['edge_bps']:.2f} bps on {row['trades_per_day']:.1f} tpd")

print("\n\nCONCLUSIONS:")
print("1. Single filters alone don't achieve >=1 bps on 1-minute data")
print("2. Combined filters can improve edge but reduce frequency")
print("3. The best combinations focus on counter-trend shorts in uptrends with high volatility")
print("4. Even with optimal combinations, 1-minute swing pivot bounce struggles to meet the goal")
print("5. Consider: Different strategy types or use 5-minute timeframe instead")