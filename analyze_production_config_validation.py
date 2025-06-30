"""Validate the production configuration against historical performance"""
import pandas as pd
import numpy as np
from pathlib import Path

# Best performing workspace for 5-minute data
workspace = Path("workspaces/signal_generation_320d109d")
signal_dir = workspace / "traces/SPY_5m_1m/signals/swing_pivot_bounce_zones"

# Load SPY 5m data
spy_5m = pd.read_csv("./data/SPY_5m.csv")
spy_5m['timestamp'] = pd.to_datetime(spy_5m['timestamp'])
spy_5m.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 
                       'Close': 'close', 'Volume': 'volume'}, inplace=True)

print("=== PRODUCTION CONFIGURATION VALIDATION ===\n")
print("Configuration: Vol>85 + Shorts Only")
print("Expected: 2.0-2.2 bps edge, 0.6-1.0 trades per day\n")

# Calculate indicators
spy_5m['returns'] = spy_5m['close'].pct_change()
spy_5m['volatility_20'] = spy_5m['returns'].rolling(20).std() * np.sqrt(78) * 100
spy_5m['vol_percentile'] = spy_5m['volatility_20'].rolling(window=126).rank(pct=True) * 100

# Test on best performing strategies
strategies_to_validate = [88, 80, 48]  # Top 3 strategies

validation_results = []

for strategy_id in strategies_to_validate:
    print(f"\n{'='*60}")
    print(f"VALIDATING STRATEGY {strategy_id}")
    print('='*60)
    
    signal_file = signal_dir / f"SPY_5m_compiled_strategy_{strategy_id}.parquet"
    signals = pd.read_parquet(signal_file)
    
    # Collect trades
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
                
                if not pd.isna(entry_conditions['vol_percentile']):
                    trade = {
                        'pct_return': pct_return,
                        'direction': 'short' if entry_data['signal'] < 0 else 'long',
                        'vol_percentile': entry_conditions['vol_percentile'],
                        'timestamp': entry_conditions['timestamp']
                    }
                    trades.append(trade)
            
            if curr['val'] != 0:
                entry_data = {'idx': curr['idx'], 'price': curr['px'], 'signal': curr['val']}
            else:
                entry_data = None
    
    if not trades:
        continue
    
    trades_df = pd.DataFrame(trades)
    
    # Apply production filter: Vol>85 + Shorts
    production_filter = (trades_df['vol_percentile'] > 85) & (trades_df['direction'] == 'short')
    filtered_trades = trades_df[production_filter]
    
    if len(filtered_trades) > 0:
        # Calculate metrics
        avg_return_bps = filtered_trades['pct_return'].mean() * 100
        win_rate = (filtered_trades['pct_return'] > 0).mean()
        
        # Daily frequency
        total_days = 16614 / 78  # Total 5-min bars / bars per day
        trades_per_day = len(filtered_trades) / total_days
        
        # Calculate by month for consistency check
        filtered_trades['month'] = pd.to_datetime(filtered_trades['timestamp']).dt.to_period('M')
        monthly_stats = filtered_trades.groupby('month').agg({
            'pct_return': ['mean', 'count']
        })
        
        print(f"\nProduction Filter Results:")
        print(f"- Total trades: {len(filtered_trades)} (from {len(trades_df)} baseline)")
        print(f"- Average edge: {avg_return_bps:.2f} bps")
        print(f"- Win rate: {win_rate:.1%}")
        print(f"- Trades per day: {trades_per_day:.2f}")
        
        print(f"\nMonthly Consistency:")
        for month, stats in monthly_stats.iterrows():
            monthly_edge = stats[('pct_return', 'mean')] * 100
            monthly_count = stats[('pct_return', 'count')]
            print(f"  {month}: {monthly_edge:.2f} bps on {monthly_count} trades")
        
        # Calculate annualized returns with costs
        print(f"\nAnnualized Returns:")
        for cost_bps in [0, 0.5, 1.0]:
            net_edge_bps = avg_return_bps - cost_bps
            if net_edge_bps > 0:
                trades_per_year = trades_per_day * 252
                annual_return = (1 + net_edge_bps/10000) ** trades_per_year - 1
                print(f"  {cost_bps} bps cost: {annual_return*100:.1f}%")
            else:
                print(f"  {cost_bps} bps cost: NEGATIVE")
        
        # Drawdown analysis
        filtered_trades = filtered_trades.sort_values('timestamp')
        filtered_trades['cumulative_return'] = (1 + filtered_trades['pct_return']/100).cumprod()
        filtered_trades['running_max'] = filtered_trades['cumulative_return'].cummax()
        filtered_trades['drawdown'] = (filtered_trades['cumulative_return'] / filtered_trades['running_max'] - 1) * 100
        max_drawdown = filtered_trades['drawdown'].min()
        
        print(f"\nRisk Metrics:")
        print(f"- Max drawdown: {max_drawdown:.2f}%")
        print(f"- Sharpe ratio (approx): {(avg_return_bps * trades_per_day * 252) / (filtered_trades['pct_return'].std() * np.sqrt(trades_per_day * 252)):.2f}")
        
        validation_results.append({
            'strategy_id': strategy_id,
            'edge_bps': avg_return_bps,
            'trades_per_day': trades_per_day,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown
        })

# Summary
print(f"\n\n{'='*60}")
print("VALIDATION SUMMARY")
print('='*60)

if validation_results:
    avg_edge = np.mean([r['edge_bps'] for r in validation_results])
    avg_tpd = np.mean([r['trades_per_day'] for r in validation_results])
    
    print(f"\nAverage across validated strategies:")
    print(f"- Edge: {avg_edge:.2f} bps")
    print(f"- Trades/day: {avg_tpd:.2f}")
    print(f"- Within expected range: {'YES' if 1.5 <= avg_edge <= 2.5 and 0.5 <= avg_tpd <= 1.2 else 'NO'}")
    
    print(f"\nProduction Configuration Status:")
    if avg_edge >= 1.5 and avg_tpd >= 0.5:
        print("✅ VALIDATED - Configuration meets minimum requirements")
        print("   - Edge above 1.5 bps threshold")
        print("   - Trade frequency above 0.5/day threshold")
    else:
        print("⚠️  WARNING - Configuration below thresholds")
        if avg_edge < 1.5:
            print("   - Edge below 1.5 bps minimum")
        if avg_tpd < 0.5:
            print("   - Trade frequency below 0.5/day minimum")

print("\nRecommendations:")
print("1. Monitor actual performance closely in paper trading")
print("2. Ensure execution quality to preserve the small edge")
print("3. Consider position sizing carefully given low frequency")
print("4. Have clear stop rules if edge deteriorates")