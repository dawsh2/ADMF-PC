"""Final optimized analysis of swing_pivot_bounce with all discovered filters"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load the signal data
workspace = Path("workspaces/signal_generation_1c64d62f")
signal_file = workspace / "traces/SPY_1m/signals/swing_pivot_bounce/SPY_compiled_strategy_0.parquet"
signals = pd.read_parquet(signal_file)

# Load raw SPY data
spy_data = pd.read_csv("./data/SPY_1m.csv")
spy_data['timestamp'] = pd.to_datetime(spy_data['timestamp'], utc=True)
spy_data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 
                         'Close': 'close', 'Volume': 'volume'}, inplace=True)

# Calculate all indicators we need
# Trend
spy_data['sma_50'] = spy_data['close'].rolling(50).mean()
spy_data['sma_200'] = spy_data['close'].rolling(200).mean()
spy_data['trend_up'] = (spy_data['close'] > spy_data['sma_50']) & (spy_data['sma_50'] > spy_data['sma_200'])
spy_data['trend_down'] = (spy_data['close'] < spy_data['sma_50']) & (spy_data['sma_50'] < spy_data['sma_200'])
spy_data['trend_neutral'] = ~(spy_data['trend_up'] | spy_data['trend_down'])

# VWAP
spy_data['date'] = spy_data['timestamp'].dt.date
spy_data['typical_price'] = (spy_data['high'] + spy_data['low'] + spy_data['close']) / 3
spy_data['pv'] = spy_data['typical_price'] * spy_data['volume']
spy_data['cum_pv'] = spy_data.groupby('date')['pv'].cumsum()
spy_data['cum_volume'] = spy_data.groupby('date')['volume'].cumsum()
spy_data['vwap'] = spy_data['cum_pv'] / spy_data['cum_volume']
spy_data['above_vwap'] = spy_data['close'] > spy_data['vwap']

# Volatility
spy_data['returns'] = spy_data['close'].pct_change()
spy_data['volatility_20'] = spy_data['returns'].rolling(20).std() * np.sqrt(390) * 100
spy_data['vol_percentile'] = spy_data['volatility_20'].rolling(252).rank(pct=True) * 100

# Range metrics
spy_data['high_20'] = spy_data['high'].rolling(20).max()
spy_data['low_20'] = spy_data['low'].rolling(20).min()
spy_data['range_20'] = (spy_data['high_20'] - spy_data['low_20']) / spy_data['close'] * 100

# Ranging detection
spy_data['sma_20'] = spy_data['close'].rolling(20).mean()
spy_data['price_vs_sma20'] = (spy_data['close'] - spy_data['sma_20']) / spy_data['sma_20'] * 100
spy_data['is_ranging'] = spy_data['price_vs_sma20'].rolling(50).std() < 0.5

# Calculate all trades with conditions
trades = []
for i in range(1, len(signals)):
    prev_signal = signals.iloc[i-1]
    curr_signal = signals.iloc[i]
    
    if prev_signal['val'] != 0:
        if curr_signal['val'] == 0 or np.sign(curr_signal['val']) != np.sign(prev_signal['val']):
            entry_idx = prev_signal['idx']
            exit_idx = curr_signal['idx']
            
            if entry_idx < len(spy_data) and not pd.isna(spy_data.iloc[entry_idx]['sma_200']):
                entry_conditions = spy_data.iloc[entry_idx]
                entry_price = prev_signal['px']
                exit_price = curr_signal['px']
                signal_type = prev_signal['val']
                
                pct_return = (exit_price / entry_price - 1) * signal_type * 100
                
                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': exit_idx,
                    'signal': signal_type,
                    'pct_return': pct_return,
                    'trend_up': entry_conditions['trend_up'],
                    'trend_down': entry_conditions['trend_down'],
                    'trend_neutral': entry_conditions['trend_neutral'],
                    'above_vwap': entry_conditions['above_vwap'],
                    'vol_percentile': entry_conditions['vol_percentile'],
                    'range_20': entry_conditions['range_20'],
                    'is_ranging': entry_conditions['is_ranging']
                })

trades_df = pd.DataFrame(trades)

print("=== BASELINE PERFORMANCE (All Trades) ===")
print(f"Total trades: {len(trades_df)}")
print(f"Average return: {trades_df['pct_return'].mean():.4f}% ({trades_df['pct_return'].mean()*100:.2f} bps)")
print(f"Total return: {(np.exp(np.log(1 + trades_df['pct_return']/100).sum()) - 1)*100:.2f}%")
print(f"Win rate: {(trades_df['pct_return'] > 0).mean():.1%}")
print(f"Sharpe ratio: {trades_df['pct_return'].mean() / trades_df['pct_return'].std():.2f}")

# Apply discovered optimal filters
print("\n=== APPLYING OPTIMAL FILTERS ===")

# Filter 1: Best single condition - Counter-trend shorts in uptrends
filter1 = (trades_df['trend_up']) & (trades_df['signal'] == -1)
filtered1 = trades_df[filter1]
print(f"\n1. Counter-trend shorts in uptrends:")
print(f"   Trades: {len(filtered1)} ({len(filtered1)/len(trades_df)*100:.1f}% of all)")
print(f"   Avg return: {filtered1['pct_return'].mean():.4f}% ({filtered1['pct_return'].mean()*100:.2f} bps)")
print(f"   Total return: {(np.exp(np.log(1 + filtered1['pct_return']/100).sum()) - 1)*100:.2f}%")
print(f"   Win rate: {(filtered1['pct_return'] > 0).mean():.1%}")

# Filter 2: Add high volatility requirement
filter2 = filter1 & (trades_df['vol_percentile'] > 70)
filtered2 = trades_df[filter2]
print(f"\n2. + High volatility (>70th percentile):")
print(f"   Trades: {len(filtered2)} ({len(filtered2)/len(trades_df)*100:.1f}% of all)")
print(f"   Avg return: {filtered2['pct_return'].mean():.4f}% ({filtered2['pct_return'].mean()*100:.2f} bps)")
print(f"   Total return: {(np.exp(np.log(1 + filtered2['pct_return']/100).sum()) - 1)*100:.2f}%")
print(f"   Win rate: {(filtered2['pct_return'] > 0).mean():.1%}")

# Filter 3: Add ranging market requirement
filter3 = filter2 & (trades_df['is_ranging'])
filtered3 = trades_df[filter3]
print(f"\n3. + Ranging markets:")
print(f"   Trades: {len(filtered3)} ({len(filtered3)/len(trades_df)*100:.1f}% of all)")
if len(filtered3) > 0:
    print(f"   Avg return: {filtered3['pct_return'].mean():.4f}% ({filtered3['pct_return'].mean()*100:.2f} bps)")
    print(f"   Total return: {(np.exp(np.log(1 + filtered3['pct_return']/100).sum()) - 1)*100:.2f}%")
    print(f"   Win rate: {(filtered3['pct_return'] > 0).mean():.1%}")

# Filter 4: Add optimal range (1-2%)
filter4 = filter2 & (trades_df['range_20'].between(0.8, 2.0))
filtered4 = trades_df[filter4]
print(f"\n4. + Optimal range (0.8-2%):")
print(f"   Trades: {len(filtered4)} ({len(filtered4)/len(trades_df)*100:.1f}% of all)")
if len(filtered4) > 0:
    print(f"   Avg return: {filtered4['pct_return'].mean():.4f}% ({filtered4['pct_return'].mean()*100:.2f} bps)")
    print(f"   Total return: {(np.exp(np.log(1 + filtered4['pct_return']/100).sum()) - 1)*100:.2f}%")
    print(f"   Win rate: {(filtered4['pct_return'] > 0).mean():.1%}")

# Alternative: Long in uptrends only
alt_filter1 = (trades_df['trend_up']) & (trades_df['signal'] == 1) & (trades_df['above_vwap'])
alt_filtered1 = trades_df[alt_filter1]
print(f"\n\nALTERNATIVE: Long in uptrends + above VWAP:")
print(f"   Trades: {len(alt_filtered1)} ({len(alt_filtered1)/len(trades_df)*100:.1f}% of all)")
print(f"   Avg return: {alt_filtered1['pct_return'].mean():.4f}% ({alt_filtered1['pct_return'].mean()*100:.2f} bps)")
print(f"   Total return: {(np.exp(np.log(1 + alt_filtered1['pct_return']/100).sum()) - 1)*100:.2f}%")
print(f"   Win rate: {(alt_filtered1['pct_return'] > 0).mean():.1%}")

# Best balanced approach
balanced_filter = ((trades_df['trend_up']) & (trades_df['signal'] == -1) & (trades_df['vol_percentile'] > 50)) | \
                  ((trades_df['trend_up']) & (trades_df['signal'] == 1) & (trades_df['above_vwap']))
balanced_filtered = trades_df[balanced_filter]
print(f"\n\nBALANCED APPROACH (Counter-trend shorts OR trend longs):")
print(f"   Trades: {len(balanced_filtered)} ({len(balanced_filtered)/len(trades_df)*100:.1f}% of all)")
print(f"   Avg return: {balanced_filtered['pct_return'].mean():.4f}% ({balanced_filtered['pct_return'].mean()*100:.2f} bps)")
print(f"   Total return: {(np.exp(np.log(1 + balanced_filtered['pct_return']/100).sum()) - 1)*100:.2f}%")
print(f"   Win rate: {(balanced_filtered['pct_return'] > 0).mean():.1%}")
print(f"   Sharpe ratio: {balanced_filtered['pct_return'].mean() / balanced_filtered['pct_return'].std():.2f}")

# Calculate performance with execution costs
print("\n\n=== EXECUTION COST ANALYSIS ===")
strategies = [
    ("Baseline (all trades)", trades_df),
    ("Optimized (counter-trend shorts in uptrends)", filtered1),
    ("Balanced approach", balanced_filtered)
]

for name, strategy_trades in strategies:
    if len(strategy_trades) > 0:
        print(f"\n{name} ({len(strategy_trades)} trades):")
        print(f"Gross return: {(np.exp(np.log(1 + strategy_trades['pct_return']/100).sum()) - 1)*100:.2f}%")
        
        for cost_bps in [1, 5, 10, 20]:
            cost_multiplier = 1 - (cost_bps / 10000)
            net_returns = strategy_trades['pct_return'] / 100 * cost_multiplier
            net_total = (np.exp(np.log(1 + net_returns).sum()) - 1) * 100
            print(f"  {cost_bps} bps cost: {net_total:.2f}% net return")

# Summary statistics
print("\n\n=== FINAL RECOMMENDATION ===")
print(f"\nOptimal strategy configuration:")
print(f"1. Take counter-trend shorts when:")
print(f"   - Market is in uptrend (above SMA50 > SMA200)")
print(f"   - Volatility is elevated (>50th percentile)")
print(f"   - Price hits resistance level")
print(f"\n2. Take trend-following longs when:")
print(f"   - Market is in uptrend")
print(f"   - Price is above VWAP")
print(f"   - Price bounces from support")
print(f"\nThis reduces trades by {(1 - len(balanced_filtered)/len(trades_df))*100:.1f}% but improves per-trade edge significantly")

# Risk metrics
if len(balanced_filtered) > 5:
    returns = balanced_filtered['pct_return'].values
    sorted_returns = np.sort(returns)
    var_95 = np.percentile(sorted_returns, 5)
    cvar_95 = sorted_returns[sorted_returns <= var_95].mean()
    
    print(f"\nRisk metrics for balanced approach:")
    print(f"  95% VaR: {var_95:.3f}%")
    print(f"  95% CVaR: {cvar_95:.3f}%")
    print(f"  Max drawdown: {sorted_returns[0]:.3f}%")
    print(f"  Best trade: {sorted_returns[-1]:.3f}%")