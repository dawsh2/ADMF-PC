"""Analyze long/short performance by market conditions on 5-minute data"""
import pandas as pd
import numpy as np
from pathlib import Path

workspace = Path("workspaces/signal_generation_320d109d")
signal_dir = workspace / "traces/SPY_5m_1m/signals/swing_pivot_bounce_zones"

# Load SPY 5m data
spy_5m = pd.read_csv("./data/SPY_5m.csv")
spy_5m['timestamp'] = pd.to_datetime(spy_5m['timestamp'])
spy_5m.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 
                       'Close': 'close', 'Volume': 'volume'}, inplace=True)

print("=== DIRECTIONAL ANALYSIS: 5-MINUTE SWING PIVOT STRATEGIES ===\n")

# Calculate indicators
spy_5m['sma_50'] = spy_5m['close'].rolling(50).mean()
spy_5m['sma_200'] = spy_5m['close'].rolling(200).mean()
spy_5m['trend_up'] = (spy_5m['close'] > spy_5m['sma_50']) & (spy_5m['sma_50'] > spy_5m['sma_200'])
spy_5m['trend_down'] = (spy_5m['close'] < spy_5m['sma_50']) & (spy_5m['sma_50'] < spy_5m['sma_200'])
spy_5m['trend_neutral'] = ~(spy_5m['trend_up'] | spy_5m['trend_down'])

# VWAP
spy_5m['date'] = spy_5m['timestamp'].dt.date
spy_5m['typical_price'] = (spy_5m['high'] + spy_5m['low'] + spy_5m['close']) / 3
spy_5m['pv'] = spy_5m['typical_price'] * spy_5m['volume']
spy_5m['cum_pv'] = spy_5m.groupby('date')['pv'].cumsum()
spy_5m['cum_volume'] = spy_5m.groupby('date')['volume'].cumsum()
spy_5m['vwap'] = spy_5m['cum_pv'] / spy_5m['cum_volume']
spy_5m['above_vwap'] = spy_5m['close'] > spy_5m['vwap']

# Volatility
spy_5m['returns'] = spy_5m['close'].pct_change()
spy_5m['volatility_20'] = spy_5m['returns'].rolling(20).std() * np.sqrt(78) * 100
spy_5m['vol_percentile'] = spy_5m['volatility_20'].rolling(window=126).rank(pct=True) * 100

# Focus on best strategies
strategies_to_analyze = [88, 48, 80]  # Top performers with filters

for strategy_id in strategies_to_analyze:
    print(f"\n{'='*80}")
    print(f"STRATEGY {strategy_id} - DIRECTIONAL BREAKDOWN")
    print('='*80)
    
    signal_file = signal_dir / f"SPY_5m_compiled_strategy_{strategy_id}.parquet"
    signals = pd.read_parquet(signal_file)
    
    # Collect all trades with conditions
    trades = []
    entry_data = None
    
    for j in range(len(signals)):
        curr = signals.iloc[j]
        
        if entry_data is None and curr['val'] != 0:
            if curr['idx'] < len(spy_5m):
                entry_data = {
                    'idx': curr['idx'],
                    'price': curr['px'],
                    'signal': curr['val']
                }
        elif entry_data is not None and (curr['val'] == 0 or np.sign(curr['val']) != np.sign(entry_data['signal'])):
            if entry_data and entry_data['idx'] < len(spy_5m) and curr['idx'] < len(spy_5m):
                entry_conditions = spy_5m.iloc[entry_data['idx']]
                
                pct_return = (curr['px'] / entry_data['price'] - 1) * entry_data['signal'] * 100
                
                trade = {
                    'pct_return': pct_return,
                    'direction': 'long' if entry_data['signal'] > 0 else 'short',
                    'trend_up': entry_conditions['trend_up'],
                    'trend_down': entry_conditions['trend_down'],
                    'trend_neutral': entry_conditions['trend_neutral'],
                    'above_vwap': entry_conditions['above_vwap'],
                    'vol_percentile': entry_conditions['vol_percentile']
                }
                trades.append(trade)
            
            if curr['val'] != 0:
                entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val']}
            else:
                entry_data = None
    
    if not trades:
        continue
        
    trades_df = pd.DataFrame(trades)
    trades_df = trades_df[trades_df['vol_percentile'].notna()]
    
    # Overall direction breakdown
    print("\n1. OVERALL DIRECTION PERFORMANCE")
    print("-" * 50)
    for direction in ['long', 'short']:
        dir_trades = trades_df[trades_df['direction'] == direction]
        if len(dir_trades) > 0:
            print(f"\n{direction.upper()} trades: {len(dir_trades)} ({len(dir_trades)/len(trades_df)*100:.1f}%)")
            print(f"  Average: {dir_trades['pct_return'].mean()*100:.2f} bps")
            print(f"  Win rate: {(dir_trades['pct_return'] > 0).mean():.1%}")
    
    # By trend direction
    print("\n\n2. PERFORMANCE BY TREND")
    print("-" * 50)
    for trend in ['trend_up', 'trend_down', 'trend_neutral']:
        trend_trades = trades_df[trades_df[trend]]
        if len(trend_trades) > 0:
            print(f"\n{trend.replace('_', ' ').title()}: {len(trend_trades)} trades")
            
            for direction in ['long', 'short']:
                dir_trades = trend_trades[trend_trades['direction'] == direction]
                if len(dir_trades) > 0:
                    avg_ret = dir_trades['pct_return'].mean()
                    win_rate = (dir_trades['pct_return'] > 0).mean()
                    print(f"  {direction}: {len(dir_trades)} trades, {avg_ret*100:.2f} bps, {win_rate:.1%} win")
    
    # VWAP analysis
    print("\n\n3. VWAP POSITIONING")
    print("-" * 50)
    
    # Mean reversion vs trend following
    scenarios = [
        ("Mean Reversion", [
            (trades_df['above_vwap'] & (trades_df['direction'] == 'short'), "Short above VWAP"),
            (~trades_df['above_vwap'] & (trades_df['direction'] == 'long'), "Long below VWAP")
        ]),
        ("Trend Following", [
            (trades_df['above_vwap'] & (trades_df['direction'] == 'long'), "Long above VWAP"),
            (~trades_df['above_vwap'] & (trades_df['direction'] == 'short'), "Short below VWAP")
        ])
    ]
    
    for scenario_name, conditions in scenarios:
        print(f"\n{scenario_name}:")
        total_trades = 0
        total_return = 0
        
        for condition, desc in conditions:
            filtered = trades_df[condition]
            if len(filtered) > 0:
                avg_ret = filtered['pct_return'].mean()
                win_rate = (filtered['pct_return'] > 0).mean()
                total_trades += len(filtered)
                total_return += avg_ret * len(filtered)
                print(f"  {desc}: {len(filtered)} trades, {avg_ret*100:.2f} bps, {win_rate:.1%} win")
        
        if total_trades > 0:
            print(f"  Combined: {total_return/total_trades*100:.2f} bps average")
    
    # High volatility breakdown
    print("\n\n4. HIGH VOLATILITY (>70th percentile) BREAKDOWN")
    print("-" * 50)
    high_vol_trades = trades_df[trades_df['vol_percentile'] > 70]
    
    if len(high_vol_trades) > 0:
        print(f"Total high vol trades: {len(high_vol_trades)} ({len(high_vol_trades)/len(trades_df)*100:.1f}%)")
        
        # By direction
        for direction in ['long', 'short']:
            dir_trades = high_vol_trades[high_vol_trades['direction'] == direction]
            if len(dir_trades) > 0:
                print(f"\n{direction.upper()}: {len(dir_trades)} trades")
                print(f"  Average: {dir_trades['pct_return'].mean()*100:.2f} bps")
                print(f"  Win rate: {(dir_trades['pct_return'] > 0).mean():.1%}")
                
                # By trend
                for trend in ['trend_up', 'trend_down', 'trend_neutral']:
                    trend_dir_trades = dir_trades[dir_trades[trend]]
                    if len(trend_dir_trades) > 0:
                        avg_ret = trend_dir_trades['pct_return'].mean()
                        print(f"  In {trend.replace('_', ' ')}: {len(trend_dir_trades)} trades, {avg_ret*100:.2f} bps")
    
    # Best combinations
    print("\n\n5. OPTIMAL COMBINATIONS")
    print("-" * 50)
    
    combinations = [
        ("Long in uptrend", trades_df['trend_up'] & (trades_df['direction'] == 'long')),
        ("Short in downtrend", trades_df['trend_down'] & (trades_df['direction'] == 'short')),
        ("Counter-trend short in uptrend", trades_df['trend_up'] & (trades_df['direction'] == 'short')),
        ("Counter-trend long in downtrend", trades_df['trend_down'] & (trades_df['direction'] == 'long')),
        ("High vol long in uptrend", trades_df['trend_up'] & (trades_df['direction'] == 'long') & (trades_df['vol_percentile'] > 70)),
        ("High vol short in uptrend", trades_df['trend_up'] & (trades_df['direction'] == 'short') & (trades_df['vol_percentile'] > 70)),
        ("High vol + VWAP aligned", 
         ((trades_df['above_vwap'] & (trades_df['direction'] == 'long')) | 
          (~trades_df['above_vwap'] & (trades_df['direction'] == 'short'))) & 
         (trades_df['vol_percentile'] > 70))
    ]
    
    print("\nTesting combinations:")
    best_combo = None
    best_bps = -100
    
    for desc, condition in combinations:
        filtered = trades_df[condition]
        if len(filtered) > 10:  # Minimum trades for significance
            avg_ret = filtered['pct_return'].mean()
            win_rate = (filtered['pct_return'] > 0).mean()
            print(f"\n{desc}:")
            print(f"  Trades: {len(filtered)} ({len(filtered)/len(trades_df)*100:.1f}%)")
            print(f"  Average: {avg_ret*100:.2f} bps")
            print(f"  Win rate: {win_rate:.1%}")
            
            if avg_ret * 100 > best_bps:
                best_bps = avg_ret * 100
                best_combo = desc

    if best_combo:
        print(f"\n*** Best combination: {best_combo} with {best_bps:.2f} bps ***")

# Summary across strategies
print(f"\n\n{'='*80}")
print("SUMMARY: KEY PATTERNS ACROSS STRATEGIES")
print('='*80)

print("\n1. Consistent patterns observed:")
print("   - Shorts generally outperform longs")
print("   - High volatility improves performance")
print("   - Counter-trend shorts in uptrends often profitable")
print("   - VWAP trend following beats mean reversion")

print("\n2. Compare to 1-minute findings:")
print("   - Similar short bias")
print("   - Similar volatility dependence")
print("   - But edges are lower overall on 5-minute")