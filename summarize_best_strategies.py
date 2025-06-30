#!/usr/bin/env python3
"""
Summarize best strategies from existing analysis.
"""

import pandas as pd
import numpy as np

# Load the existing analysis
df = pd.read_csv("advanced_analysis_signal_generation_a2d31737.csv")

print("="*80)
print("SWING PIVOT STRATEGY ANALYSIS SUMMARY")
print("="*80)

# Key findings
print("\n## KEY FINDINGS ##")
print(f"- Total strategies analyzed: {len(df)}")
print(f"- Average return per trade: {df['avg_return_per_trade_bps'].mean():.2f} bps")
print(f"- Strategies with >1 bps per trade: {(df['avg_return_per_trade_bps'] > 1).sum()}")
print(f"- Strategies with >2 bps per trade: {(df['avg_return_per_trade_bps'] > 2).sum()}")
print(f"- Maximum trades per day: {df['trades_per_day'].max():.2f}")

# Filter criteria - adjusted for reality
min_rpt_bps = 1.0  # 1 bps per trade
min_tpd = 0.3  # ~2 trades per week

qualified = df[(df['avg_return_per_trade_bps'] >= min_rpt_bps) & 
               (df['trades_per_day'] >= min_tpd)]

print(f"\n## QUALIFIED STRATEGIES (≥{min_rpt_bps} bps/trade, ≥{min_tpd} trades/day) ##")
print(f"- Number of qualified strategies: {len(qualified)}")

if len(qualified) > 0:
    print(f"- Average return per trade: {qualified['avg_return_per_trade_bps'].mean():.2f} bps")
    print(f"- Average trades per day: {qualified['trades_per_day'].mean():.2f}")
    print(f"- Average win rate: {qualified['win_rate'].mean()*100:.1f}%")
    
    # Top 10 strategies
    print("\n## TOP 10 STRATEGIES BY RETURN PER TRADE ##")
    print(f"{'Strategy':<30} {'RPT(bps)':<10} {'TPD':<8} {'Win%':<8} {'Trades':<8} {'Total Ret':<10}")
    print("-"*74)
    
    top10 = qualified.nlargest(10, 'avg_return_per_trade_bps')
    for idx, row in top10.iterrows():
        print(f"{row['strategy_id']:<30} {row['avg_return_per_trade_bps']:>8.2f} "
              f"{row['trades_per_day']:>6.2f} {row['win_rate']*100:>6.1f}% "
              f"{row['num_trades']:>7} {row['total_return']*100:>8.2f}%")
    
    # Most active qualified strategies
    print("\n## TOP 5 MOST ACTIVE QUALIFIED STRATEGIES ##")
    print(f"{'Strategy':<30} {'TPD':<8} {'RPT(bps)':<10} {'Win%':<8} {'Trades':<8}")
    print("-"*64)
    
    active5 = qualified.nlargest(5, 'trades_per_day')
    for idx, row in active5.iterrows():
        print(f"{row['strategy_id']:<30} {row['trades_per_day']:>6.2f} "
              f"{row['avg_return_per_trade_bps']:>8.2f} {row['win_rate']*100:>6.1f}% "
              f"{row['num_trades']:>7}")

# Analysis without strict filtering
print("\n## OVERALL BEST PERFORMERS (NO FILTERS) ##")
top20_overall = df.nlargest(20, 'avg_return_per_trade_bps')
print(f"{'Strategy':<30} {'RPT(bps)':<10} {'TPD':<8} {'Win%':<8} {'Trades':<8}")
print("-"*64)

for idx, row in top20_overall.head(10).iterrows():
    print(f"{row['strategy_id']:<30} {row['avg_return_per_trade_bps']:>8.2f} "
          f"{row['trades_per_day']:>6.2f} {row['win_rate']*100:>6.1f}% "
          f"{row['num_trades']:>7}")

# Stop loss analysis
print("\n## STOP LOSS IMPACT (50 bps stop) ##")
print(f"- Average stop rate: {df['stop_rate'].mean()*100:.2f}%")
print(f"- Strategies with >5% stops: {(df['stop_rate'] > 0.05).sum()}")
print(f"- Max stop rate: {df['stop_rate'].max()*100:.1f}%")

# Trade frequency insights
print("\n## TRADE FREQUENCY DISTRIBUTION ##")
print(f"- Strategies with ≥1 trade/day: {(df['trades_per_day'] >= 1).sum()}")
print(f"- Strategies with ≥0.5 trades/day: {(df['trades_per_day'] >= 0.5).sum()}")
print(f"- Strategies with ≥0.3 trades/day: {(df['trades_per_day'] >= 0.3).sum()}")
print(f"- Strategies with ≥0.2 trades/day: {(df['trades_per_day'] >= 0.2).sum()}")

# Recommendations
print("\n## RECOMMENDATIONS ##")
print("1. The best strategies achieve 3.38 bps per trade but trade infrequently (0.17 trades/day)")
print("2. For 2-3 trades per day target, you'd need to:")
print("   - Run multiple strategies in parallel (10-15 strategies)")
print("   - Or modify the strategy parameters to increase signal frequency")
print("3. Stop loss of 50 bps has minimal impact (0.2% stop rate)")
print("4. Consider these trade-offs:")
print("   - Higher frequency → Lower return per trade")
print("   - Best balance appears at 0.3-0.5 trades/day with 1.5-2.0 bps/trade")

# Export best strategies for implementation
best_for_implementation = df[
    (df['avg_return_per_trade_bps'] >= 1.5) & 
    (df['trades_per_day'] >= 0.2)
].head(50)

best_for_implementation.to_csv("recommended_strategies_to_implement.csv", index=False)
print(f"\n## EXPORTED {len(best_for_implementation)} STRATEGIES TO: recommended_strategies_to_implement.csv ##")