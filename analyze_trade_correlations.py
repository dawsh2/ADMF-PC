"""Analyze correlations between trade outcomes and market conditions"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load market data
market_df = pd.read_parquet('./data/SPY_1m.parquet')
market_df['timestamp'] = pd.to_datetime(market_df['timestamp'])
market_df = market_df.set_index('timestamp').sort_index()

# Load signals
workspace = Path("workspaces/signal_generation_7ecda4b8")
signal_file = workspace / "traces/SPY_1m/signals/bollinger_rsi_simple_signals/SPY_compiled_strategy_0.parquet"
signals_df = pd.read_parquet(signal_file)
signals_df['ts'] = pd.to_datetime(signals_df['ts'])

print("=== Trade Outcome Correlation Analysis ===\n")

# First, let's calculate various market features
print("Calculating market features...")

# Add technical indicators to market data
# 1. Volume features
market_df['volume_ma_20'] = market_df['volume'].rolling(20).mean()
market_df['volume_ratio'] = market_df['volume'] / market_df['volume_ma_20']

# 2. Price momentum
market_df['returns_1m'] = market_df['close'].pct_change()
market_df['returns_5m'] = market_df['close'].pct_change(5)
market_df['returns_10m'] = market_df['close'].pct_change(10)
market_df['returns_30m'] = market_df['close'].pct_change(30)

# 3. Volatility
market_df['high_low_range'] = (market_df['high'] - market_df['low']) / market_df['close']
market_df['volatility_20'] = market_df['returns_1m'].rolling(20).std()

# 4. Market microstructure
market_df['spread'] = market_df['high'] - market_df['low']
market_df['price_position'] = (market_df['close'] - market_df['low']) / (market_df['high'] - market_df['low'])

# 5. Time features
market_df['hour'] = market_df.index.hour
market_df['minute'] = market_df.index.minute
market_df['day_of_week'] = market_df.index.dayofweek
market_df['minutes_from_open'] = (market_df.index.hour - 9.5) * 60 + market_df.index.minute
market_df['minutes_to_close'] = 390 - market_df['minutes_from_open']

# 6. Trend indicators
market_df['sma_20'] = market_df['close'].rolling(20).mean()
market_df['sma_50'] = market_df['close'].rolling(50).mean()
market_df['price_vs_sma20'] = (market_df['close'] - market_df['sma_20']) / market_df['sma_20']
market_df['trend_strength'] = (market_df['sma_20'] - market_df['sma_50']) / market_df['sma_50']

# Convert signals to trades with entry conditions
trades = []
current_position = 0
entry_time = None
entry_idx = None

for i in range(len(signals_df)):
    row = signals_df.iloc[i]
    new_signal = row['val']
    
    if current_position != 0 and new_signal != current_position:
        if entry_time is not None:
            exit_time = row['ts']
            exit_price = row['px']
            entry_price = signals_df.iloc[entry_idx]['px']
            pnl_pct = (exit_price / entry_price - 1) * current_position * 100
            
            # Get market conditions at entry
            try:
                entry_conditions = market_df.loc[entry_time]
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'direction': 'long' if current_position > 0 else 'short',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'win': 1 if pnl_pct > 0 else 0,
                    # Entry conditions
                    'volume_ratio': entry_conditions['volume_ratio'],
                    'returns_5m': entry_conditions['returns_5m'],
                    'returns_10m': entry_conditions['returns_10m'],
                    'returns_30m': entry_conditions['returns_30m'],
                    'volatility_20': entry_conditions['volatility_20'],
                    'high_low_range': entry_conditions['high_low_range'],
                    'price_position': entry_conditions['price_position'],
                    'hour': entry_conditions['hour'],
                    'minutes_from_open': entry_conditions['minutes_from_open'],
                    'minutes_to_close': entry_conditions['minutes_to_close'],
                    'day_of_week': entry_conditions['day_of_week'],
                    'price_vs_sma20': entry_conditions['price_vs_sma20'],
                    'trend_strength': entry_conditions['trend_strength']
                })
            except:
                pass  # Skip if we can't find market data
    
    if new_signal != 0 and new_signal != current_position:
        entry_time = row['ts']
        entry_idx = i
    
    current_position = new_signal

trades_df = pd.DataFrame(trades)
trades_df = trades_df.dropna()  # Remove trades with missing data

print(f"Analyzing {len(trades_df)} trades with complete data\n")

# Analyze correlations
print("=== Feature Correlations with Trade Outcomes ===\n")

# 1. Basic statistics by win/loss
winners = trades_df[trades_df['win'] == 1]
losers = trades_df[trades_df['win'] == 0]

features = ['volume_ratio', 'returns_5m', 'returns_10m', 'returns_30m', 
            'volatility_20', 'high_low_range', 'price_position',
            'price_vs_sma20', 'trend_strength']

print(f"{'Feature':<20} {'Winners Mean':<15} {'Losers Mean':<15} {'Difference':<15}")
print("-" * 70)

for feature in features:
    if feature in winners.columns:
        win_mean = winners[feature].mean()
        lose_mean = losers[feature].mean()
        diff = win_mean - lose_mean
        print(f"{feature:<20} {win_mean:>13.4f}  {lose_mean:>13.4f}  {diff:>13.4f}")

# 2. Time-based analysis
print("\n=== Win Rate by Time of Day ===")
time_bins = [(9.5, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16)]
for start, end in time_bins:
    mask = (trades_df['hour'] + trades_df['minutes_from_open']/60 >= start) & \
           (trades_df['hour'] + trades_df['minutes_from_open']/60 < end)
    subset = trades_df[mask]
    if len(subset) > 0:
        wr = subset['win'].mean()
        print(f"{start:02.0f}:30-{end:02.0f}:30: {len(subset):>4} trades, {wr*100:>5.1f}% win rate")

# 3. Direction-specific analysis
print("\n=== Feature Analysis by Direction ===")
for direction in ['long', 'short']:
    dir_trades = trades_df[trades_df['direction'] == direction]
    dir_winners = dir_trades[dir_trades['win'] == 1]
    dir_losers = dir_trades[dir_trades['win'] == 0]
    
    print(f"\n{direction.upper()} trades ({len(dir_trades)} total, {dir_trades['win'].mean()*100:.1f}% win rate):")
    
    # Find most discriminative features
    feature_diffs = {}
    for feature in features:
        if feature in dir_winners.columns and len(dir_winners) > 0 and len(dir_losers) > 0:
            win_mean = dir_winners[feature].mean()
            lose_mean = dir_losers[feature].mean()
            diff = abs(win_mean - lose_mean)
            feature_diffs[feature] = {
                'diff': diff,
                'win_mean': win_mean,
                'lose_mean': lose_mean
            }
    
    # Sort by difference
    sorted_features = sorted(feature_diffs.items(), key=lambda x: x[1]['diff'], reverse=True)
    
    print("Top 3 discriminative features:")
    for feat, data in sorted_features[:3]:
        print(f"  {feat}: Winners={data['win_mean']:.4f}, Losers={data['lose_mean']:.4f}")

# 4. Find potential filters
print("\n=== Potential Filters to Improve Win Rate ===")

# Test various filter thresholds
filters_to_test = [
    ('volume_ratio', 'above', [0.5, 0.8, 1.0, 1.2]),
    ('volatility_20', 'below', [0.001, 0.002, 0.003]),
    ('returns_30m', 'long_above_short_below', [-0.002, 0, 0.002]),
    ('minutes_from_open', 'above', [30, 60, 120]),
    ('minutes_to_close', 'above', [30, 60, 120])
]

original_wr = trades_df['win'].mean()
original_count = len(trades_df)

print(f"\nOriginal: {original_count} trades, {original_wr*100:.1f}% win rate\n")

for feature, filter_type, thresholds in filters_to_test:
    print(f"\n{feature} ({filter_type}):")
    for threshold in thresholds:
        if filter_type == 'above':
            filtered = trades_df[trades_df[feature] > threshold]
        elif filter_type == 'below':
            filtered = trades_df[trades_df[feature] < threshold]
        elif filter_type == 'long_above_short_below':
            # Complex filter for momentum
            long_filtered = trades_df[(trades_df['direction'] == 'long') & 
                                    (trades_df[feature] > threshold)]
            short_filtered = trades_df[(trades_df['direction'] == 'short') & 
                                     (trades_df[feature] < threshold)]
            filtered = pd.concat([long_filtered, short_filtered])
        
        if len(filtered) > 50:  # Only show if meaningful sample
            new_wr = filtered['win'].mean()
            improvement = (new_wr - original_wr) / original_wr * 100
            
            print(f"  Threshold {threshold:>6.3f}: {len(filtered):>4} trades ({len(filtered)/original_count*100:>4.1f}%), "
                  f"{new_wr*100:>5.1f}% WR ({improvement:>+5.1f}%)")

# 5. Combined filters
print("\n=== Combined Filter Analysis ===")

# Test a few promising combinations
combo1 = trades_df[(trades_df['volume_ratio'] > 0.8) & 
                   (trades_df['volatility_20'] < 0.002)]
if len(combo1) > 20:
    print(f"Volume > 0.8 AND Volatility < 0.002: {len(combo1)} trades, {combo1['win'].mean()*100:.1f}% WR")

combo2 = trades_df[(trades_df['minutes_from_open'] > 60) & 
                   (trades_df['minutes_to_close'] > 60)]
if len(combo2) > 20:
    print(f"Avoid first/last hour: {len(combo2)} trades, {combo2['win'].mean()*100:.1f}% WR")

# Save detailed data for further analysis
trades_df.to_csv('trades_with_features.csv', index=False)
print(f"\nSaved {len(trades_df)} trades with features to 'trades_with_features.csv'")