#!/usr/bin/env python3
"""
Analyze which base strategy types are working vs not working.
"""

import duckdb

# Connect to the analytics database
db_path = 'workspaces/20250614_211925_indicator_grid_v3_SPY/analytics.duckdb'
conn = duckdb.connect(db_path, read_only=True)

print("=== Base Strategy Type Analysis ===\n")

# Get total count by strategy type
print("Strategy types and their expansions:")
strategy_type_counts = conn.execute("""
    SELECT strategy_type, COUNT(*) as total_instances
    FROM strategies
    GROUP BY strategy_type
    ORDER BY total_instances DESC
""").fetchall()

total_strategies = sum(count for _, count in strategy_type_counts)
print(f"Total strategy instances: {total_strategies}")
print(f"Total base strategy types: {len(strategy_type_counts)}\n")

for strategy_type, count in strategy_type_counts:
    print(f"  {strategy_type}: {count} instances")

# Since there are no signals in event_archives, let's check if there are any
# clues about which strategies attempted to run
print(f"\n=== Analysis ===")
print("Since event_archives is empty, all 882 strategies failed to generate signals.")
print("This suggests systematic issues rather than individual strategy problems.")

print(f"\nBase strategy types that need debugging ({len(strategy_type_counts)} total):")

# Group by likely categories based on naming
trend_strategies = []
oscillator_strategies = []
crossover_strategies = []
volume_strategies = []
volatility_strategies = []
other_strategies = []

for strategy_type, count in strategy_type_counts:
    if any(word in strategy_type for word in ['crossover', 'sma', 'ema', 'dema', 'tema', 'macd']):
        crossover_strategies.append((strategy_type, count))
    elif any(word in strategy_type for word in ['rsi', 'cci', 'stochastic', 'williams', 'ultimate', 'oscillator']):
        oscillator_strategies.append((strategy_type, count))
    elif any(word in strategy_type for word in ['adx', 'aroon', 'supertrend', 'parabolic', 'linear']):
        trend_strategies.append((strategy_type, count))
    elif any(word in strategy_type for word in ['volume', 'mfi', 'obv', 'vwap', 'chaikin', 'accumulation']):
        volume_strategies.append((strategy_type, count))
    elif any(word in strategy_type for word in ['bollinger', 'keltner', 'donchian', 'atr', 'breakout']):
        volatility_strategies.append((strategy_type, count))
    else:
        other_strategies.append((strategy_type, count))

print(f"\n** CROSSOVER STRATEGIES ({len(crossover_strategies)} types, {sum(c for _, c in crossover_strategies)} instances) **")
for strategy_type, count in crossover_strategies:
    print(f"  ✗ {strategy_type}: {count}")

print(f"\n** OSCILLATOR STRATEGIES ({len(oscillator_strategies)} types, {sum(c for _, c in oscillator_strategies)} instances) **")
for strategy_type, count in oscillator_strategies:
    print(f"  ✗ {strategy_type}: {count}")

print(f"\n** TREND STRATEGIES ({len(trend_strategies)} types, {sum(c for _, c in trend_strategies)} instances) **")
for strategy_type, count in trend_strategies:
    print(f"  ✗ {strategy_type}: {count}")

print(f"\n** VOLUME STRATEGIES ({len(volume_strategies)} types, {sum(c for _, c in volume_strategies)} instances) **")
for strategy_type, count in volume_strategies:
    print(f"  ✗ {strategy_type}: {count}")

print(f"\n** VOLATILITY STRATEGIES ({len(volatility_strategies)} types, {sum(c for _, c in volatility_strategies)} instances) **")
for strategy_type, count in volatility_strategies:
    print(f"  ✗ {strategy_type}: {count}")

print(f"\n** OTHER STRATEGIES ({len(other_strategies)} types, {sum(c for _, c in other_strategies)} instances) **")
for strategy_type, count in other_strategies:
    print(f"  ✗ {strategy_type}: {count}")

# Recommendations
print(f"\n=== SYSTEMATIC DEBUGGING NEEDED ===")
print("Since ALL strategies failed, this suggests:")
print("1. FeatureHub configuration issue - features not being computed")
print("2. Feature naming mismatch - strategies looking for wrong feature names") 
print("3. Insufficient warmup period - complex indicators need more bars")
print("4. Signal generation logic issue - strategies returning None")

print(f"\nPriority order for debugging:")
print(f"1. CROSSOVER (simplest): {len(crossover_strategies)} types - should work with basic SMA/EMA")
print(f"2. OSCILLATOR (medium): {len(oscillator_strategies)} types - need RSI, Stochastic, etc.")
print(f"3. VOLATILITY (medium): {len(volatility_strategies)} types - need ATR, Bollinger bands")
print(f"4. VOLUME (complex): {len(volume_strategies)} types - need volume-based features")
print(f"5. TREND (complex): {len(trend_strategies)} types - need ADX, Aroon, etc.")
print(f"6. OTHER (various): {len(other_strategies)} types - mixed complexity")

conn.close()