#!/usr/bin/env python3
"""
Analyze Bollinger Bands strategies with high trade frequency (>=400 trades).
"""

import pandas as pd

# Load the results from previous analysis
df = pd.read_csv('bollinger_performance_analysis.csv')

print("High Frequency Bollinger Bands Analysis (>=400 trades)")
print("=" * 60)

# Filter for strategies with at least 400 trades
high_freq = df[df['num_trades'] >= 400].copy()

print(f"\nFound {len(high_freq)} strategies with >=400 trades out of {len(df)} total")

# Sort by total return
high_freq_sorted = high_freq.sort_values('total_return', ascending=False)

print(f"\n{'Rank':<5} {'Period':<7} {'StdDev':<7} {'Return':<10} {'Sharpe':<8} {'WinRate':<8} {'Trades':<8}")
print("-" * 70)

for idx, row in high_freq_sorted.iterrows():
    # Find rank in overall results
    overall_rank = df.sort_values('total_return', ascending=False).index.get_loc(idx) + 1
    print(f"{overall_rank:<5} {row['period']:<7.0f} {row['std_dev']:<7.1f} "
          f"{row['total_return']:>9.2%} {row['sharpe']:>7.2f} "
          f"{row['win_rate']:>7.1%} {row['num_trades']:>7.0f}")

# Group by period to see patterns
print("\n\nAnalysis by Period (for high-frequency strategies):")
print("-" * 50)

period_stats = high_freq.groupby('period').agg({
    'total_return': ['mean', 'max', 'count'],
    'num_trades': 'mean'
}).round(3)

print("\nPeriods with high-frequency strategies:")
for period in sorted(high_freq['period'].unique()):
    strategies = high_freq[high_freq['period'] == period]
    best = strategies.loc[strategies['total_return'].idxmax()]
    print(f"\nPeriod {period}:")
    print(f"  Strategies: {len(strategies)}")
    print(f"  Best return: {best['total_return']:.2%} (std_dev={best['std_dev']})")
    print(f"  Avg trades: {strategies['num_trades'].mean():.0f}")

# Find optimal std_dev for each period
print("\n\nOptimal std_dev by period (high-frequency only):")
print("-" * 50)

pivot = high_freq.pivot_table(
    values='total_return', 
    index='period', 
    columns='std_dev',
    aggfunc='first'
)

for period in sorted(high_freq['period'].unique())[:10]:  # Show first 10
    row = pivot.loc[period]
    best_std = row.idxmax()
    best_return = row.max()
    print(f"Period {period}: Best std_dev={best_std} with {best_return:.2%} return")

# Specific analysis for period=11
period_11 = high_freq[high_freq['period'] == 11]
if not period_11.empty:
    print(f"\n\nDetailed analysis for Period=11:")
    print("-" * 50)
    period_11_sorted = period_11.sort_values('total_return', ascending=False)
    
    print(f"{'StdDev':<7} {'Return':<10} {'Sharpe':<8} {'WinRate':<8} {'Trades':<8}")
    for _, row in period_11_sorted.iterrows():
        print(f"{row['std_dev']:<7.1f} {row['total_return']:>9.2%} "
              f"{row['sharpe']:>7.2f} {row['win_rate']:>7.1%} {row['num_trades']:>7.0f}")

# Check if exit_threshold might be affecting results
print("\n\nNote: These results use default exit_threshold=0.001 (0.1%)")
print("Your successful run might have used a different exit threshold.")
print("\nFor high-frequency strategies (many trades), the exit threshold is critical:")
print("- Too tight (0.001): Many small losses, low win rate")
print("- Optimal (0.002-0.003): Better win rate while maintaining frequency")
print("- Too loose (0.005+): Fewer trades, might miss opportunities")