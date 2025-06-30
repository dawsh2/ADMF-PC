"""Apply 1-minute filters to 5-minute swing pivot strategies - fixed"""
import pandas as pd
import numpy as np
from pathlib import Path

workspace = Path("workspaces/signal_generation_320d109d")
signal_dir = workspace / "traces/SPY_5m_1m/signals/swing_pivot_bounce_zones"

# Load SPY 5m data to calculate indicators
spy_5m = pd.read_csv("./data/SPY_5m.csv")
spy_5m['timestamp'] = pd.to_datetime(spy_5m['timestamp'])
spy_5m.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 
                       'Close': 'close', 'Volume': 'volume'}, inplace=True)

print("=== APPLYING 1-MINUTE FILTERS TO 5-MINUTE STRATEGIES ===\n")

# Calculate indicators on 5-minute data
print("Calculating indicators on 5-minute data...")

# Trend indicators
spy_5m['sma_50'] = spy_5m['close'].rolling(50).mean()
spy_5m['sma_200'] = spy_5m['close'].rolling(200).mean()
spy_5m['trend_up'] = (spy_5m['close'] > spy_5m['sma_50']) & (spy_5m['sma_50'] > spy_5m['sma_200'])
spy_5m['trend_down'] = (spy_5m['close'] < spy_5m['sma_50']) & (spy_5m['sma_50'] < spy_5m['sma_200'])

# Volatility
spy_5m['returns'] = spy_5m['close'].pct_change()
spy_5m['volatility_20'] = spy_5m['returns'].rolling(20).std() * np.sqrt(78) * 100  # 78 5-min bars per day
spy_5m['vol_percentile'] = spy_5m['volatility_20'].rolling(window=int(252*78/390)).rank(pct=True) * 100  # ~126 bars

# VWAP (daily reset)
spy_5m['date'] = spy_5m['timestamp'].dt.date
spy_5m['typical_price'] = (spy_5m['high'] + spy_5m['low'] + spy_5m['close']) / 3
spy_5m['pv'] = spy_5m['typical_price'] * spy_5m['volume']
spy_5m['cum_pv'] = spy_5m.groupby('date')['pv'].cumsum()
spy_5m['cum_volume'] = spy_5m.groupby('date')['volume'].cumsum()
spy_5m['vwap'] = spy_5m['cum_pv'] / spy_5m['cum_volume']
spy_5m['above_vwap'] = spy_5m['close'] > spy_5m['vwap']

# Analyze best strategies with filters
best_strategies = [88, 80, 48, 40, 81]  # Top 5 from previous analysis

print(f"\nAnalyzing top {len(best_strategies)} strategies with filters...")

results_summary = []

for strategy_id in best_strategies:
    print(f"\n{'='*70}")
    print(f"STRATEGY {strategy_id}")
    print('='*70)
    
    signal_file = signal_dir / f"SPY_5m_compiled_strategy_{strategy_id}.parquet"
    signals = pd.read_parquet(signal_file)
    
    # Analyze all trades first
    all_trades = []
    entry_data = None
    
    for j in range(len(signals)):
        curr = signals.iloc[j]
        
        if entry_data is None and curr['val'] != 0:
            # Entry
            if curr['idx'] < len(spy_5m):
                entry_data = {
                    'idx': curr['idx'],
                    'price': curr['px'],
                    'signal': curr['val']
                }
        elif entry_data is not None and (curr['val'] == 0 or np.sign(curr['val']) != np.sign(entry_data['signal'])):
            # Exit
            if entry_data['idx'] < len(spy_5m) and curr['idx'] < len(spy_5m):
                entry_conditions = spy_5m.iloc[entry_data['idx']]
                
                pct_return = (curr['px'] / entry_data['price'] - 1) * entry_data['signal'] * 100
                
                trade = {
                    'pct_return': pct_return,
                    'signal': entry_data['signal'],
                    'trend_up': entry_conditions['trend_up'] if not pd.isna(entry_conditions.get('trend_up')) else False,
                    'trend_down': entry_conditions['trend_down'] if not pd.isna(entry_conditions.get('trend_down')) else False,
                    'above_vwap': entry_conditions['above_vwap'] if not pd.isna(entry_conditions.get('above_vwap')) else False,
                    'vol_percentile': entry_conditions['vol_percentile'] if not pd.isna(entry_conditions.get('vol_percentile')) else 50
                }
                all_trades.append(trade)
            
            # Reset or flip
            if curr['val'] != 0:
                entry_data = {
                    'idx': curr['idx'],
                    'price': curr['px'],
                    'signal': curr['val']
                }
            else:
                entry_data = None
    
    if not all_trades:
        print("No valid trades found")
        continue
        
    trades_df = pd.DataFrame(all_trades)
    
    # Remove trades with missing data
    valid_trades = trades_df[trades_df['vol_percentile'].notna()]
    
    baseline_return = valid_trades['pct_return'].mean()
    baseline_win_rate = (valid_trades['pct_return'] > 0).mean()
    
    print(f"\nBaseline (all {len(valid_trades)} trades):")
    print(f"- Avg return: {baseline_return:.4f}% ({baseline_return*100:.2f} bps)")
    print(f"- Win rate: {baseline_win_rate:.1%}")
    
    strategy_results = {
        'strategy_id': strategy_id,
        'baseline_trades': len(valid_trades),
        'baseline_bps': baseline_return * 100
    }
    
    # Apply filters
    filters = [
        ("Counter-trend shorts in uptrends", 
         (valid_trades['trend_up']) & (valid_trades['signal'] == -1)),
        
        ("CT shorts + High vol (>70th)", 
         (valid_trades['trend_up']) & (valid_trades['signal'] == -1) & (valid_trades['vol_percentile'] > 70)),
         
        ("CT shorts + Very high vol (>80th)", 
         (valid_trades['trend_up']) & (valid_trades['signal'] == -1) & (valid_trades['vol_percentile'] > 80)),
         
        ("Any high vol trade (>70th)", 
         valid_trades['vol_percentile'] > 70),
         
        ("High vol shorts only", 
         (valid_trades['signal'] == -1) & (valid_trades['vol_percentile'] > 70))
    ]
    
    best_filter = None
    best_annual_05bp = -100
    
    for filter_name, filter_mask in filters:
        filtered = valid_trades[filter_mask]
        if len(filtered) > 0:
            avg_return = filtered['pct_return'].mean()
            win_rate = (filtered['pct_return'] > 0).mean()
            
            print(f"\n{filter_name}:")
            print(f"  - Trades: {len(filtered)} ({len(filtered)/len(valid_trades)*100:.1f}% of total)")
            print(f"  - Avg return: {avg_return:.4f}% ({avg_return*100:.2f} bps)")
            print(f"  - Win rate: {win_rate:.1%}")
            
            # Annual projection
            bars_per_year = 78 * 252
            base_trades_per_year = len(valid_trades) * (bars_per_year / 16614)
            filtered_trades_per_year = base_trades_per_year * (len(filtered) / len(valid_trades))
            
            print(f"  - Trades/year: {filtered_trades_per_year:.0f}")
            
            # Calculate returns with costs
            annual_returns = {}
            for cost_bps in [0, 0.5, 1.0]:
                net_edge = avg_return * 100 - cost_bps
                if net_edge > 0:
                    annual_return = (1 + net_edge/10000) ** filtered_trades_per_year - 1
                    annual_returns[cost_bps] = annual_return * 100
                    print(f"  - {cost_bps} bps cost: {annual_return*100:.1f}% annual")
                else:
                    annual_returns[cost_bps] = -100
                    print(f"  - {cost_bps} bps cost: NEGATIVE")
            
            # Track best filter
            if annual_returns.get(0.5, -100) > best_annual_05bp:
                best_annual_05bp = annual_returns.get(0.5, -100)
                best_filter = filter_name
                strategy_results[f'best_filter'] = filter_name
                strategy_results[f'best_filter_bps'] = avg_return * 100
                strategy_results[f'best_filter_annual_05bp'] = best_annual_05bp
    
    results_summary.append(strategy_results)

# Summary comparison
print(f"\n\n{'='*70}")
print("FILTER EFFECTIVENESS SUMMARY")
print('='*70)

print("\nBest filtered results for each strategy (at 0.5bp cost):")
for result in results_summary:
    if 'best_filter' in result:
        print(f"\nStrategy {result['strategy_id']}:")
        print(f"  Baseline: {result['baseline_bps']:.2f} bps")
        print(f"  Best filter: {result['best_filter']}")
        print(f"  Filtered edge: {result['best_filter_bps']:.2f} bps")
        print(f"  Annual return (0.5bp): {result['best_filter_annual_05bp']:.1f}%")

print("\n\nCOMPARISON TO 1-MINUTE:")
print("-" * 50)
print("1-minute baseline: 0.04 bps → 1.26 bps with high vol filter (31.5x improvement)")
print("1-minute annual at 0.5bp: 14.5%")
print("\n5-minute results show if similar improvements are possible...")