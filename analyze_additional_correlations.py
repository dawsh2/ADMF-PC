"""Analyze additional correlations with trade outcomes"""
import pandas as pd
import numpy as np

# Load the trades with features
trades_df = pd.read_csv('trades_with_features.csv')
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

print("=== Additional Correlation Analysis ===\n")

# 1. Analyze consecutive wins/losses
print("=== Consecutive Trade Analysis ===")
trades_df = trades_df.sort_values('entry_time').reset_index(drop=True)

# Add previous trade outcome
trades_df['prev_win'] = trades_df['win'].shift(1)
trades_df['prev_pnl'] = trades_df['pnl_pct'].shift(1)

# Win rate after wins vs losses
after_win = trades_df[trades_df['prev_win'] == 1].dropna()
after_loss = trades_df[trades_df['prev_win'] == 0].dropna()

print(f"Win rate after a winning trade: {after_win['win'].mean()*100:.1f}% ({len(after_win)} trades)")
print(f"Win rate after a losing trade: {after_loss['win'].mean()*100:.1f}% ({len(after_loss)} trades)")

# Streaks analysis
trades_df['win_streak'] = 0
trades_df['loss_streak'] = 0

for i in range(1, len(trades_df)):
    if trades_df.loc[i-1, 'win'] == 1:
        if i > 1 and trades_df.loc[i-2, 'win'] == 1:
            trades_df.loc[i, 'win_streak'] = trades_df.loc[i-1, 'win_streak'] + 1
        else:
            trades_df.loc[i, 'win_streak'] = 1
    else:
        if i > 1 and trades_df.loc[i-2, 'win'] == 0:
            trades_df.loc[i, 'loss_streak'] = trades_df.loc[i-1, 'loss_streak'] + 1
        else:
            trades_df.loc[i, 'loss_streak'] = 1

# Performance by streak length
print("\nWin rate by current win streak:")
for streak in range(0, 5):
    subset = trades_df[trades_df['win_streak'] == streak]
    if len(subset) > 10:
        print(f"  After {streak} wins: {subset['win'].mean()*100:.1f}% ({len(subset)} trades)")

print("\nWin rate by current loss streak:")
for streak in range(0, 4):
    subset = trades_df[trades_df['loss_streak'] == streak]
    if len(subset) > 10:
        print(f"  After {streak} losses: {subset['win'].mean()*100:.1f}% ({len(subset)} trades)")

# 2. Market structure analysis
print("\n=== Market Structure Analysis ===")

# Add more market structure features
trades_df['price_range_5m'] = trades_df['returns_5m'].abs()
trades_df['price_range_30m'] = trades_df['returns_30m'].abs()

# Trend consistency
trades_df['trend_aligned'] = ((trades_df['direction'] == 'long') & (trades_df['returns_30m'] < 0)) | \
                             ((trades_df['direction'] == 'short') & (trades_df['returns_30m'] > 0))

trend_aligned = trades_df[trades_df['trend_aligned']]
trend_against = trades_df[~trades_df['trend_aligned']]

print(f"Trend-aligned trades (mean reversion): {len(trend_aligned)}, WR: {trend_aligned['win'].mean()*100:.1f}%")
print(f"Trend-against trades: {len(trend_against)}, WR: {trend_against['win'].mean()*100:.1f}%")

# 3. Intraday patterns
print("\n=== Intraday Patterns ===")

# First/last 30 minutes
first_30 = trades_df[trades_df['minutes_from_open'] <= 30]
last_30 = trades_df[trades_df['minutes_to_close'] <= 30]
middle = trades_df[(trades_df['minutes_from_open'] > 30) & (trades_df['minutes_to_close'] > 30)]

print(f"First 30 min: {len(first_30)} trades, {first_30['win'].mean()*100:.1f}% WR, {first_30['pnl_pct'].mean():.3f}% avg")
print(f"Last 30 min: {len(last_30)} trades, {last_30['win'].mean()*100:.1f}% WR, {last_30['pnl_pct'].mean():.3f}% avg")
print(f"Middle of day: {len(middle)} trades, {middle['win'].mean()*100:.1f}% WR, {middle['pnl_pct'].mean():.3f}% avg")

# 4. Volume patterns
print("\n=== Volume Pattern Analysis ===")

# Volume acceleration
trades_df['volume_increasing'] = trades_df['volume_ratio'] > 1.0

vol_inc = trades_df[trades_df['volume_increasing']]
vol_dec = trades_df[~trades_df['volume_increasing']]

print(f"Volume increasing: {len(vol_inc)} trades, {vol_inc['win'].mean()*100:.1f}% WR")
print(f"Volume decreasing: {len(vol_dec)} trades, {vol_dec['win'].mean()*100:.1f}% WR")

# Extreme volume
extreme_vol = trades_df[trades_df['volume_ratio'] > 2.0]
normal_vol = trades_df[(trades_df['volume_ratio'] > 0.5) & (trades_df['volume_ratio'] <= 2.0)]
low_vol = trades_df[trades_df['volume_ratio'] <= 0.5]

print(f"\nVolume categories:")
print(f"  Low (<0.5x): {len(low_vol)} trades, {low_vol['win'].mean()*100:.1f}% WR, {low_vol['pnl_pct'].mean():.3f}% avg")
print(f"  Normal (0.5-2x): {len(normal_vol)} trades, {normal_vol['win'].mean()*100:.1f}% WR, {normal_vol['pnl_pct'].mean():.3f}% avg")
print(f"  Extreme (>2x): {len(extreme_vol)} trades, {extreme_vol['win'].mean()*100:.1f}% WR, {extreme_vol['pnl_pct'].mean():.3f}% avg")

# 5. Combined filter exploration
print("\n=== Advanced Filter Combinations ===")

# Test various combinations
filters = [
    ("Volume > 1.0 + Trend Aligned", 
     (trades_df['volume_ratio'] > 1.0) & trades_df['trend_aligned']),
    
    ("Volume > 1.2 + Middle of Day",
     (trades_df['volume_ratio'] > 1.2) & (trades_df['minutes_from_open'] > 30) & (trades_df['minutes_to_close'] > 30)),
    
    ("Volume > 1.0 + After Loss",
     (trades_df['volume_ratio'] > 1.0) & (trades_df['prev_win'] == 0)),
    
    ("Trend Aligned + Not First 30min",
     trades_df['trend_aligned'] & (trades_df['minutes_from_open'] > 30)),
    
    ("Volume 0.8-2.0 + Trend Aligned",
     (trades_df['volume_ratio'] > 0.8) & (trades_df['volume_ratio'] < 2.0) & trades_df['trend_aligned']),
]

# Calculate annualization
date_range_days = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days
base_tpd = len(trades_df) / date_range_days
exec_cost = 0.0001

for name, mask in filters:
    filtered = trades_df[mask]
    if len(filtered) > 50:  # Meaningful sample
        wr = filtered['win'].mean()
        avg_ret = filtered['pnl_pct'].mean()
        
        # Annual calculation
        filter_ratio = len(filtered) / len(trades_df)
        tpd = base_tpd * filter_ratio
        tpy = tpd * 252
        net_ret = (avg_ret / 100) - (2 * exec_cost)
        
        if net_ret > -1:
            annual = (1 + net_ret) ** tpy - 1
            annual_str = f"{annual*100:.1f}%"
        else:
            annual_str = "LOSS"
        
        print(f"\n{name}:")
        print(f"  Trades: {len(filtered)} ({len(filtered)/len(trades_df)*100:.1f}%)")
        print(f"  Win Rate: {wr*100:.1f}%")
        print(f"  Avg Return: {avg_ret:.3f}%")
        print(f"  Annual Net: {annual_str}")

# 6. Feature importance ranking
print("\n=== Feature Importance for Win/Loss ===")

features_to_rank = ['volume_ratio', 'volatility_20', 'price_range_5m', 
                   'price_range_30m', 'high_low_range', 'price_position']

importance = {}
for feature in features_to_rank:
    # Calculate difference in means
    win_mean = trades_df[trades_df['win']==1][feature].mean()
    loss_mean = trades_df[trades_df['win']==0][feature].mean()
    
    # Normalize by overall std
    std = trades_df[feature].std()
    if std > 0:
        importance[feature] = abs(win_mean - loss_mean) / std

# Sort by importance
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

print("\nFeature importance (normalized difference):")
for feat, imp in sorted_importance:
    print(f"  {feat}: {imp:.3f}")

# Save enhanced dataset
trades_df.to_csv('trades_enhanced_analysis.csv', index=False)
print(f"\nSaved {len(trades_df)} trades with enhanced features to 'trades_enhanced_analysis.csv'")