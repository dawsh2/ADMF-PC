#!/usr/bin/env python3
"""
Statistical analysis of swing pivot bounce zones strategies (no plotting).
"""

import pandas as pd
import numpy as np
from scipy import stats

# Load the results
df = pd.read_csv("swing_pivot_analysis_results.csv")

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
print(f"Max Drawdown Estimate: {(df['total_return'].min())*100:.2f}%")

# Strategy clustering based on characteristics
print("\n=== Strategy Characteristics by Performance ===")
top_performers = df[df['total_return'] > df['total_return'].quantile(0.9)]
bottom_performers = df[df['total_return'] < df['total_return'].quantile(0.1)]

print(f"\nTop 10% Performers ({len(top_performers)} strategies):")
print(f"  Average Return: {top_performers['total_return'].mean()*100:.2f}%")
print(f"  Average Win Rate: {top_performers['win_rate'].mean()*100:.1f}%")
print(f"  Average Trade Count: {top_performers['num_trades'].mean():.0f}")
print(f"  Average Trade Duration: {top_performers['avg_trade_duration'].mean():.1f} bars")
print(f"  Average Win Size: {top_performers['avg_win'].mean()*100:.2f}%")
print(f"  Average Loss Size: {top_performers['avg_loss'].mean()*100:.2f}%")
print(f"  Win/Loss Ratio: {abs(top_performers['avg_win'].mean() / top_performers['avg_loss'].mean()):.2f}")

print(f"\nBottom 10% Performers ({len(bottom_performers)} strategies):")
print(f"  Average Return: {bottom_performers['total_return'].mean()*100:.2f}%")
print(f"  Average Win Rate: {bottom_performers['win_rate'].mean()*100:.1f}%")
print(f"  Average Trade Count: {bottom_performers['num_trades'].mean():.0f}")
print(f"  Average Trade Duration: {bottom_performers['avg_trade_duration'].mean():.1f} bars")
print(f"  Average Win Size: {bottom_performers['avg_win'].mean()*100:.2f}%")
print(f"  Average Loss Size: {bottom_performers['avg_loss'].mean()*100:.2f}%")
print(f"  Win/Loss Ratio: {abs(bottom_performers['avg_win'].mean() / bottom_performers['avg_loss'].mean()):.2f}")

# Return distribution analysis
print("\n=== Return Distribution Analysis ===")
print(f"Strategies with >1% return: {(df['total_return'] > 0.01).sum()} ({(df['total_return'] > 0.01).mean()*100:.1f}%)")
print(f"Strategies with >0.5% return: {(df['total_return'] > 0.005).sum()} ({(df['total_return'] > 0.005).mean()*100:.1f}%)")
print(f"Strategies with >0% return: {(df['total_return'] > 0).sum()} ({(df['total_return'] > 0).mean()*100:.1f}%)")
print(f"Strategies with <-1% return: {(df['total_return'] < -0.01).sum()} ({(df['total_return'] < -0.01).mean()*100:.1f}%)")

# Trade frequency analysis
print("\n=== Trade Frequency Analysis ===")
print(f"Strategies with >200 trades: {(df['num_trades'] > 200).sum()}")
print(f"Strategies with 100-200 trades: {((df['num_trades'] >= 100) & (df['num_trades'] <= 200)).sum()}")
print(f"Strategies with 50-100 trades: {((df['num_trades'] >= 50) & (df['num_trades'] < 100)).sum()}")
print(f"Strategies with <50 trades: {(df['num_trades'] < 50).sum()}")

# Win rate distribution
print("\n=== Win Rate Analysis ===")
print(f"Strategies with >50% win rate: {(df['win_rate'] > 0.5).sum()} ({(df['win_rate'] > 0.5).mean()*100:.1f}%)")
print(f"Strategies with 45-50% win rate: {((df['win_rate'] >= 0.45) & (df['win_rate'] <= 0.5)).sum()}")
print(f"Strategies with <45% win rate: {(df['win_rate'] < 0.45).sum()}")

# Correlation analysis
print("\n=== Correlation Analysis ===")
correlations = df[['total_return', 'win_rate', 'num_trades', 'avg_trade_duration', 'avg_win', 'avg_loss']].corr()
print("Correlations with total_return:")
print(correlations['total_return'].sort_values(ascending=False))

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
    f.write(f"  Positive Returns: {(df['total_return'] > 0).sum()} ({(df['total_return'] > 0).mean()*100:.1f}%)\n")
    f.write(f"  Sharpe Ratio (approx): {sharpe_approx:.3f}\n\n")
    
    f.write("Trade Statistics:\n")
    f.write(f"  Avg Trades per Strategy: {df['num_trades'].mean():.1f}\n")
    f.write(f"  Avg Win Rate: {df['win_rate'].mean()*100:.1f}%\n")
    f.write(f"  Avg Trade Duration: {df['avg_trade_duration'].mean():.1f} bars\n")
    f.write(f"  Avg Winning Trade: {df['avg_win'].mean()*100:.2f}%\n")
    f.write(f"  Avg Losing Trade: {df['avg_loss'].mean()*100:.2f}%\n\n")
    
    f.write("Performance Quintiles:\n")
    f.write(str(quintile_stats))
    f.write("\n\nCorrelation Matrix:\n")
    f.write(str(correlations))

# Find strategies with specific characteristics
print("\n=== Notable Strategy Patterns ===")

# High win rate but negative returns
high_wr_neg_return = df[(df['win_rate'] > 0.5) & (df['total_return'] < 0)]
print(f"Strategies with >50% win rate but negative returns: {len(high_wr_neg_return)}")

# Low win rate but positive returns  
low_wr_pos_return = df[(df['win_rate'] < 0.45) & (df['total_return'] > 0)]
print(f"Strategies with <45% win rate but positive returns: {len(low_wr_pos_return)}")

# Very active traders
very_active = df[df['num_trades'] > 200]
print(f"Very active strategies (>200 trades): {len(very_active)}, avg return: {very_active['total_return'].mean()*100:.2f}%")

# Quick trades
quick_trades = df[df['avg_trade_duration'] < 1.5]
print(f"Quick trade strategies (<1.5 bars): {len(quick_trades)}, avg return: {quick_trades['total_return'].mean()*100:.2f}%")

print("\nAnalysis complete. Results saved to swing_pivot_detailed_stats.txt")