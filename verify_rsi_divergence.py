#!/usr/bin/env python3
"""
Verify the RSI divergence strategy results
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load price data
prices = pd.read_csv("./data/SPY_1m.csv")

# Calculate RSI
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

prices['rsi'] = calculate_rsi(prices['Close'])

# Bollinger Bands
prices['bb_middle'] = prices['Close'].rolling(20).mean()
prices['bb_std'] = prices['Close'].rolling(20).std()
prices['bb_upper'] = prices['bb_middle'] + 2 * prices['bb_std']
prices['bb_lower'] = prices['bb_middle'] - 2 * prices['bb_std']

print("="*60)
print("RSI DIVERGENCE STRATEGY VERIFICATION")
print("="*60)

# Track all trades for verification
all_trades = []
trade_details = []

# Look for bullish divergence
for i in range(50, len(prices) - 50):
    if prices.iloc[i]['Close'] < prices.iloc[i]['bb_lower']:
        # Look back for previous low below band
        for j in range(max(i - 20, 50), i):
            if prices.iloc[j]['Close'] < prices.iloc[j]['bb_lower']:
                # Check divergence conditions
                price_makes_lower_low = prices.iloc[i]['Low'] < prices.iloc[j]['Low']
                rsi_makes_higher_low = prices.iloc[i]['rsi'] > prices.iloc[j]['rsi'] + 5
                
                if price_makes_lower_low and rsi_makes_higher_low:
                    # Wait for confirmation
                    for k in range(i + 1, min(i + 10, len(prices))):
                        if prices.iloc[k]['Close'] > prices.iloc[k]['bb_lower']:
                            entry_idx = k
                            entry_price = prices.iloc[k]['Close']
                            entry_time = prices.iloc[k]['timestamp']
                            
                            # Find exit
                            for m in range(entry_idx + 1, min(entry_idx + 50, len(prices))):
                                if prices.iloc[m]['Close'] >= prices.iloc[m]['bb_middle']:
                                    exit_idx = m
                                    exit_price = prices.iloc[m]['Close']
                                    exit_time = prices.iloc[m]['timestamp']
                                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                                    
                                    trade = {
                                        'entry_idx': entry_idx,
                                        'exit_idx': exit_idx,
                                        'duration': exit_idx - entry_idx,
                                        'entry_time': entry_time,
                                        'exit_time': exit_time,
                                        'entry_price': entry_price,
                                        'exit_price': exit_price,
                                        'pnl_pct': pnl_pct,
                                        'pnl_dollars': (exit_price - entry_price) * 100,  # Per 100 shares
                                        'rsi_at_low1': prices.iloc[j]['rsi'],
                                        'rsi_at_low2': prices.iloc[i]['rsi'],
                                        'rsi_diff': prices.iloc[i]['rsi'] - prices.iloc[j]['rsi'],
                                        'price_low1': prices.iloc[j]['Low'],
                                        'price_low2': prices.iloc[i]['Low'],
                                        'price_diff_pct': (prices.iloc[i]['Low'] - prices.iloc[j]['Low']) / prices.iloc[j]['Low'] * 100
                                    }
                                    
                                    all_trades.append(trade)
                                    
                                    # Keep detailed info for first few trades
                                    if len(trade_details) < 5:
                                        trade_details.append(trade)
                                    
                                    break
                            break
                    break

trades_df = pd.DataFrame(all_trades)

print(f"\nTotal trades found: {len(trades_df)}")

if len(trades_df) > 0:
    # Overall statistics
    print(f"\nOVERALL STATISTICS:")
    print(f"Win rate: {(trades_df['pnl_pct'] > 0).mean():.1%}")
    print(f"Average PnL per trade: {trades_df['pnl_pct'].mean():.3f}%")
    print(f"Median PnL per trade: {trades_df['pnl_pct'].median():.3f}%")
    print(f"Total gross return: {trades_df['pnl_pct'].sum():.2f}%")
    print(f"Transaction cost (1bp): {len(trades_df) * 0.01:.2f}%")
    print(f"Net return: {trades_df['pnl_pct'].sum() - len(trades_df) * 0.01:.2f}%")
    
    # Duration analysis
    print(f"\nDURATION ANALYSIS:")
    print(f"Average duration: {trades_df['duration'].mean():.1f} bars")
    print(f"Median duration: {trades_df['duration'].median():.0f} bars")
    
    print("\nBy duration bucket:")
    for d_range in [(1, 5), (6, 10), (11, 20), (21, 50)]:
        d_trades = trades_df[trades_df['duration'].between(d_range[0], d_range[1])]
        if len(d_trades) > 0:
            net = d_trades['pnl_pct'].sum() - len(d_trades) * 0.01
            win_rate = (d_trades['pnl_pct'] > 0).mean()
            print(f"  {d_range[0]:2d}-{d_range[1]:2d} bars: {len(d_trades):3d} trades, "
                  f"{win_rate:.1%} win, {net:6.2f}% net")
    
    # Divergence characteristics
    print(f"\nDIVERGENCE CHARACTERISTICS:")
    print(f"Average RSI difference: {trades_df['rsi_diff'].mean():.1f} points")
    print(f"Average price difference: {trades_df['price_diff_pct'].mean():.2f}%")
    
    # Sample trades
    print(f"\nSAMPLE TRADES (First 5):")
    print("-" * 80)
    for i, trade in enumerate(trade_details[:5]):
        print(f"\nTrade {i+1}:")
        print(f"  Entry: {trade['entry_time']} @ ${trade['entry_price']:.2f}")
        print(f"  Exit: {trade['exit_time']} @ ${trade['exit_price']:.2f}")
        print(f"  Duration: {trade['duration']} bars")
        print(f"  PnL: {trade['pnl_pct']:.3f}% (${trade['pnl_dollars']:.2f} per 100 shares)")
        print(f"  RSI divergence: Price {trade['price_diff_pct']:.2f}% lower, RSI {trade['rsi_diff']:.1f} points higher")
    
    # Risk analysis
    print(f"\nRISK ANALYSIS:")
    print(f"Largest win: {trades_df['pnl_pct'].max():.2f}%")
    print(f"Largest loss: {trades_df['pnl_pct'].min():.2f}%")
    print(f"Win/Loss ratio: {abs(trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() / trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].mean()):.2f}")
    
    # Monthly breakdown (approximate)
    trades_per_month = len(trades_df) / (len(prices) / (390 * 20))  # Approximate trading days
    print(f"\nApproximate trades per month: {trades_per_month:.1f}")
    
    # Cost sensitivity
    print(f"\nCOST SENSITIVITY:")
    for cost_bp in [0.5, 1.0, 2.0, 3.0, 5.0]:
        net = trades_df['pnl_pct'].sum() - len(trades_df) * cost_bp / 100
        print(f"  {cost_bp} bps: {net:.2f}% net return")

# Verify we're finding the same trades
print(f"\n\nVERIFICATION:")
print(f"This analysis found: {len(trades_df)} trades")
print(f"Previous analysis found: 494 trades")
print(f"Match: {'YES' if abs(len(trades_df) - 494) < 10 else 'NO'}")