"""Analyze the impact of volume filtering on strategy profitability"""
import pandas as pd
import numpy as np

# Load the saved trades with features
trades_df = pd.read_csv('trades_with_features.csv')
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

print("=== Volume Filter Impact on Profitability ===\n")

# Test various volume thresholds
volume_thresholds = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]

# Calculate base metrics
date_range_days = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days
original_trades_per_day = len(trades_df) / date_range_days
exec_cost = 0.0001  # 1 basis point

print(f"Original Performance:")
print(f"  Trades: {len(trades_df)}")
print(f"  Win Rate: {trades_df['win'].mean()*100:.1f}%")
print(f"  Avg Return: {trades_df['pnl_pct'].mean():.3f}%")
print(f"  Trades/Day: {original_trades_per_day:.2f}")

# Calculate original annual return
avg_return_decimal = trades_df['pnl_pct'].mean() / 100
net_return_per_trade = avg_return_decimal - (2 * exec_cost)
trades_per_year_orig = original_trades_per_day * 252
annual_net_orig = (1 + net_return_per_trade) ** trades_per_year_orig - 1
print(f"  Annual Return (net): {annual_net_orig*100:.1f}%\n")

print(f"{'Volume Threshold':<18} {'Trades':<8} {'Win Rate':<10} {'Avg Return':<12} {'Trades/Day':<12} {'Annual Net':<12} {'Sharpe':<8}")
print("-" * 95)

best_config = {'annual_net': -999}

for threshold in volume_thresholds:
    filtered = trades_df[trades_df['volume_ratio'] > threshold]
    
    if len(filtered) < 20:  # Skip if too few trades
        continue
    
    trades = len(filtered)
    win_rate = filtered['win'].mean()
    avg_return = filtered['pnl_pct'].mean()
    
    # Estimate trades per day
    # Assuming filtering reduces opportunities proportionally
    filter_ratio = len(filtered) / len(trades_df)
    est_trades_per_day = original_trades_per_day * filter_ratio
    trades_per_year = est_trades_per_day * 252
    
    # Annual return
    net_return_per_trade = (avg_return / 100) - (2 * exec_cost)
    if net_return_per_trade > -1 and trades_per_year > 0:
        annual_net = (1 + net_return_per_trade) ** trades_per_year - 1
        annual_net_val = annual_net * 100
        annual_net_str = f"{annual_net_val:.1f}%"
    else:
        annual_net_val = -100
        annual_net_str = "LOSS"
    
    # Calculate Sharpe
    filtered['date'] = filtered['exit_time'].dt.date
    filtered['pnl_decimal_net'] = (filtered['pnl_pct'] / 100) - (2 * exec_cost)
    
    daily_returns = filtered.groupby('date')['pnl_decimal_net'].sum()
    # Adjust for reduced trading frequency
    daily_returns = daily_returns * filter_ratio
    
    date_range = pd.date_range(start=daily_returns.index.min(), 
                              end=daily_returns.index.max(), 
                              freq='D')
    daily_returns = daily_returns.reindex(date_range.date, fill_value=0)
    
    daily_mean = daily_returns.mean()
    daily_std = daily_returns.std()
    
    if daily_std > 0:
        sharpe = (daily_mean * 252) / (daily_std * np.sqrt(252))
    else:
        sharpe = 0
    
    if annual_net_val > best_config['annual_net']:
        best_config = {
            'threshold': threshold,
            'trades': trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'trades_per_day': est_trades_per_day,
            'annual_net': annual_net_val,
            'sharpe': sharpe
        }
    
    print(f"Volume > {threshold:<6.1f}    {trades:<8} {win_rate*100:>8.1f}%   {avg_return:>10.3f}%  {est_trades_per_day:>10.2f}  {annual_net_str:>11}  {sharpe:>6.2f}")

print(f"\n*** Best Volume Filter: > {best_config['threshold']:.1f} ***")
print(f"Trades: {best_config['trades']} ({best_config['trades']/len(trades_df)*100:.1f}% of original)")
print(f"Win Rate: {best_config['win_rate']*100:.1f}% (from {trades_df['win'].mean()*100:.1f}%)")
print(f"Average Return: {best_config['avg_return']:.3f}% (from {trades_df['pnl_pct'].mean():.3f}%)")
print(f"Annual Return (net): {best_config['annual_net']:.1f}%")
print(f"Sharpe Ratio: {best_config['sharpe']:.2f}")

# Analyze what makes high volume trades better
print(f"\n=== Why High Volume Trades Perform Better ===")

high_vol = trades_df[trades_df['volume_ratio'] > best_config['threshold']]
low_vol = trades_df[trades_df['volume_ratio'] <= best_config['threshold']]

print(f"\nHigh Volume Trades (>{best_config['threshold']:.1f}x avg):")
print(f"  Count: {len(high_vol)}")
print(f"  Long/Short: {(high_vol['direction']=='long').sum()}/{(high_vol['direction']=='short').sum()}")
print(f"  Avg holding period: {high_vol['pnl_pct'].index.size} bars")  # This needs fixing
print(f"  Return distribution:")
print(f"    25th percentile: {high_vol['pnl_pct'].quantile(0.25):.3f}%")
print(f"    Median: {high_vol['pnl_pct'].median():.3f}%")
print(f"    75th percentile: {high_vol['pnl_pct'].quantile(0.75):.3f}%")

print(f"\nLow Volume Trades (<={best_config['threshold']:.1f}x avg):")
print(f"  Count: {len(low_vol)}")
print(f"  Long/Short: {(low_vol['direction']=='long').sum()}/{(low_vol['direction']=='short').sum()}")
print(f"  Return distribution:")
print(f"    25th percentile: {low_vol['pnl_pct'].quantile(0.25):.3f}%")
print(f"    Median: {low_vol['pnl_pct'].median():.3f}%")
print(f"    75th percentile: {low_vol['pnl_pct'].quantile(0.75):.3f}%")

# Test combining volume filter with other promising filters
print(f"\n=== Combined Filters ===")

# Volume + specific hours
mid_day = trades_df[(trades_df['hour'] >= 10) & (trades_df['hour'] <= 14)]
vol_midday = trades_df[(trades_df['volume_ratio'] > 1.2) & 
                       (trades_df['hour'] >= 10) & 
                       (trades_df['hour'] <= 14)]

if len(vol_midday) > 20:
    wr = vol_midday['win'].mean()
    avg_ret = vol_midday['pnl_pct'].mean()
    print(f"\nVolume >1.2 + Mid-day (10am-2pm):")
    print(f"  Trades: {len(vol_midday)} ({len(vol_midday)/len(trades_df)*100:.1f}%)")
    print(f"  Win Rate: {wr*100:.1f}%")
    print(f"  Avg Return: {avg_ret:.3f}%")
    
    # Calculate annual return
    filter_ratio = len(vol_midday) / len(trades_df)
    tpd = original_trades_per_day * filter_ratio
    tpy = tpd * 252
    net_ret = (avg_ret / 100) - (2 * exec_cost)
    if net_ret > -1:
        annual = (1 + net_ret) ** tpy - 1
        print(f"  Annual Return (net): {annual*100:.1f}%")

# Volume + momentum alignment
momentum_aligned_long = trades_df[(trades_df['direction'] == 'long') & 
                                  (trades_df['returns_30m'] < 0)]
momentum_aligned_short = trades_df[(trades_df['direction'] == 'short') & 
                                   (trades_df['returns_30m'] > 0)]
momentum_aligned = pd.concat([momentum_aligned_long, momentum_aligned_short])

vol_momentum = momentum_aligned[momentum_aligned['volume_ratio'] > 1.2]

if len(vol_momentum) > 20:
    wr = vol_momentum['win'].mean()
    avg_ret = vol_momentum['pnl_pct'].mean()
    print(f"\nVolume >1.2 + Momentum Aligned:")
    print(f"  Trades: {len(vol_momentum)} ({len(vol_momentum)/len(trades_df)*100:.1f}%)")
    print(f"  Win Rate: {wr*100:.1f}%")
    print(f"  Avg Return: {avg_ret:.3f}%")
    
    # Calculate annual return
    filter_ratio = len(vol_momentum) / len(trades_df)
    tpd = original_trades_per_day * filter_ratio
    tpy = tpd * 252
    net_ret = (avg_ret / 100) - (2 * exec_cost)
    if net_ret > -1:
        annual = (1 + net_ret) ** tpy - 1
        print(f"  Annual Return (net): {annual*100:.1f}%")