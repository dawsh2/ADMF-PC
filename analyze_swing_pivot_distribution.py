#!/usr/bin/env python3
"""
Detailed distribution analysis of swing pivot bounce zones strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the results
df = pd.read_csv("swing_pivot_analysis_results.csv")

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Swing Pivot Bounce Zones Strategy Analysis', fontsize=16)

# 1. Return distribution histogram
ax1 = axes[0, 0]
returns_pct = df['total_return'] * 100
ax1.hist(returns_pct, bins=50, edgecolor='black', alpha=0.7)
ax1.axvline(returns_pct.mean(), color='red', linestyle='--', label=f'Mean: {returns_pct.mean():.2f}%')
ax1.axvline(returns_pct.median(), color='green', linestyle='--', label=f'Median: {returns_pct.median():.2f}%')
ax1.set_xlabel('Total Return (%)')
ax1.set_ylabel('Number of Strategies')
ax1.set_title('Distribution of Strategy Returns')
ax1.legend()

# 2. Win rate distribution
ax2 = axes[0, 1]
win_rates = df['win_rate'] * 100
ax2.hist(win_rates, bins=30, edgecolor='black', alpha=0.7, color='orange')
ax2.axvline(win_rates.mean(), color='red', linestyle='--', label=f'Mean: {win_rates.mean():.1f}%')
ax2.set_xlabel('Win Rate (%)')
ax2.set_ylabel('Number of Strategies')
ax2.set_title('Distribution of Win Rates')
ax2.legend()

# 3. Number of trades distribution
ax3 = axes[0, 2]
ax3.hist(df['num_trades'], bins=30, edgecolor='black', alpha=0.7, color='green')
ax3.axvline(df['num_trades'].mean(), color='red', linestyle='--', label=f'Mean: {df["num_trades"].mean():.0f}')
ax3.set_xlabel('Number of Trades')
ax3.set_ylabel('Number of Strategies')
ax3.set_title('Distribution of Trade Counts')
ax3.legend()

# 4. Return vs Win Rate scatter
ax4 = axes[1, 0]
ax4.scatter(df['win_rate'] * 100, df['total_return'] * 100, alpha=0.5)
ax4.set_xlabel('Win Rate (%)')
ax4.set_ylabel('Total Return (%)')
ax4.set_title('Return vs Win Rate')
ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax4.axvline(x=50, color='black', linestyle='-', alpha=0.3)

# 5. Return vs Number of Trades
ax5 = axes[1, 1]
ax5.scatter(df['num_trades'], df['total_return'] * 100, alpha=0.5, color='purple')
ax5.set_xlabel('Number of Trades')
ax5.set_ylabel('Total Return (%)')
ax5.set_title('Return vs Trade Count')
ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# 6. Trade duration distribution
ax6 = axes[1, 2]
durations = df['avg_trade_duration']
ax6.hist(durations, bins=30, edgecolor='black', alpha=0.7, color='brown')
ax6.axvline(durations.mean(), color='red', linestyle='--', label=f'Mean: {durations.mean():.1f} bars')
ax6.set_xlabel('Average Trade Duration (bars)')
ax6.set_ylabel('Number of Strategies')
ax6.set_title('Distribution of Trade Durations')
ax6.legend()

plt.tight_layout()
plt.savefig('swing_pivot_distribution_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# Create performance quintile analysis
print("\n=== Performance Quintile Analysis ===")
df['return_quintile'] = pd.qcut(df['total_return'], q=5, labels=['Q1 (Worst)', 'Q2', 'Q3', 'Q4', 'Q5 (Best)'])

quintile_stats = df.groupby('return_quintile').agg({
    'total_return': ['mean', 'min', 'max'],
    'win_rate': 'mean',
    'num_trades': 'mean',
    'avg_trade_duration': 'mean',
    'avg_win': 'mean',
    'avg_loss': 'mean'
}).round(4)

print(quintile_stats)

# Calculate Sharpe ratio approximation (assuming daily returns)
# This is simplified - assumes 252 trading days and uses total return
daily_return_approx = df['total_return'] / 252  # Rough approximation
sharpe_approx = (daily_return_approx.mean() / daily_return_approx.std()) * np.sqrt(252)
print(f"\nApproximate Sharpe Ratio: {sharpe_approx:.3f}")

# Risk-Return Profile
print("\n=== Risk-Return Profile ===")
print(f"Mean Return: {df['total_return'].mean()*100:.2f}%")
print(f"Std Dev of Returns: {df['total_return'].std()*100:.2f}%")
print(f"Skewness: {stats.skew(df['total_return']):.3f}")
print(f"Kurtosis: {stats.kurtosis(df['total_return']):.3f}")
print(f"95th Percentile Return: {df['total_return'].quantile(0.95)*100:.2f}%")
print(f"5th Percentile Return: {df['total_return'].quantile(0.05)*100:.2f}%")

# Strategy clustering based on characteristics
print("\n=== Strategy Characteristics by Performance ===")
top_performers = df[df['total_return'] > df['total_return'].quantile(0.9)]
bottom_performers = df[df['total_return'] < df['total_return'].quantile(0.1)]

print("\nTop 10% Performers:")
print(f"  Average Win Rate: {top_performers['win_rate'].mean()*100:.1f}%")
print(f"  Average Trade Count: {top_performers['num_trades'].mean():.0f}")
print(f"  Average Trade Duration: {top_performers['avg_trade_duration'].mean():.1f} bars")
print(f"  Average Win Size: {top_performers['avg_win'].mean()*100:.2f}%")
print(f"  Average Loss Size: {top_performers['avg_loss'].mean()*100:.2f}%")

print("\nBottom 10% Performers:")
print(f"  Average Win Rate: {bottom_performers['win_rate'].mean()*100:.1f}%")
print(f"  Average Trade Count: {bottom_performers['num_trades'].mean():.0f}")
print(f"  Average Trade Duration: {bottom_performers['avg_trade_duration'].mean():.1f} bars")
print(f"  Average Win Size: {bottom_performers['avg_win'].mean()*100:.2f}%")
print(f"  Average Loss Size: {bottom_performers['avg_loss'].mean()*100:.2f}%")

# Save detailed statistics
with open('swing_pivot_detailed_stats.txt', 'w') as f:
    f.write("=== Swing Pivot Bounce Zones Detailed Statistics ===\n\n")
    f.write(f"Total Strategies Analyzed: {len(df)}\n")
    f.write(f"Total Trades Executed: {df['num_trades'].sum()}\n\n")
    
    f.write("Return Statistics:\n")
    f.write(f"  Mean: {df['total_return'].mean()*100:.2f}%\n")
    f.write(f"  Median: {df['total_return'].median()*100:.2f}%\n")
    f.write(f"  Std Dev: {df['total_return'].std()*100:.2f}%\n")
    f.write(f"  Min: {df['total_return'].min()*100:.2f}%\n")
    f.write(f"  Max: {df['total_return'].max()*100:.2f}%\n")
    f.write(f"  Positive Returns: {(df['total_return'] > 0).sum()} ({(df['total_return'] > 0).mean()*100:.1f}%)\n\n")
    
    f.write("Trade Statistics:\n")
    f.write(f"  Avg Trades per Strategy: {df['num_trades'].mean():.1f}\n")
    f.write(f"  Avg Win Rate: {df['win_rate'].mean()*100:.1f}%\n")
    f.write(f"  Avg Trade Duration: {df['avg_trade_duration'].mean():.1f} bars\n")
    f.write(f"  Avg Winning Trade: {df['avg_win'].mean()*100:.2f}%\n")
    f.write(f"  Avg Losing Trade: {df['avg_loss'].mean()*100:.2f}%\n\n")
    
    f.write("Performance Quintiles:\n")
    f.write(str(quintile_stats))

print("\nAnalysis complete. Files generated:")
print("- swing_pivot_distribution_analysis.png")
print("- swing_pivot_detailed_stats.txt")