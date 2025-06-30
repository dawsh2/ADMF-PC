#!/usr/bin/env python3
"""
Analyze trade quality and identify filtering opportunities
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_trade_quality(parquet_file, strategy_name):
    """Analyze individual trade quality and identify problem areas."""
    
    df = pd.read_parquet(parquet_file)
    
    # Extract all trades
    trades = []
    entry_price = None
    entry_signal = None
    entry_idx = None
    
    for _, row in df.iterrows():
        signal = row['val']
        price = row['px']
        bar_idx = row['idx']
        
        if entry_price is None and signal != 0:
            entry_price = price
            entry_signal = signal
            entry_idx = bar_idx
        elif entry_price is not None and (signal == 0 or signal != entry_signal):
            if entry_signal > 0:
                pnl_pct = (price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - price) / entry_price * 100
            
            trades.append({
                'entry_idx': entry_idx,
                'exit_idx': bar_idx,
                'duration': bar_idx - entry_idx,
                'entry_price': entry_price,
                'exit_price': price,
                'pnl_pct': pnl_pct,
                'signal_type': 'long' if entry_signal > 0 else 'short'
            })
            
            if signal != 0:
                entry_price = price
                entry_signal = signal
                entry_idx = bar_idx
            else:
                entry_price = None
    
    if not trades:
        return None
    
    trades_df = pd.DataFrame(trades)
    
    print(f"\n{'='*60}")
    print(f"Analysis for {strategy_name}")
    print(f"{'='*60}")
    
    # Duration analysis
    print(f"\nTrade Duration Analysis:")
    print(f"  Mean duration: {trades_df['duration'].mean():.1f} bars")
    print(f"  Median duration: {trades_df['duration'].median():.1f} bars")
    print(f"  < 5 bars: {(trades_df['duration'] < 5).sum()} trades ({(trades_df['duration'] < 5).mean():.1%})")
    print(f"  < 10 bars: {(trades_df['duration'] < 10).sum()} trades ({(trades_df['duration'] < 10).mean():.1%})")
    
    # Quick flip analysis
    quick_trades = trades_df[trades_df['duration'] < 5]
    if len(quick_trades) > 0:
        print(f"\nQuick trades (<5 bars) performance:")
        print(f"  Count: {len(quick_trades)}")
        print(f"  Avg PnL: {quick_trades['pnl_pct'].mean():.3f}%")
        print(f"  Total PnL: {quick_trades['pnl_pct'].sum():.2f}%")
        print(f"  Cost @ 1bp: {len(quick_trades) * 0.01:.2f}%")
        print(f"  Net: {quick_trades['pnl_pct'].sum() - len(quick_trades) * 0.01:.2f}%")
    
    # Profitability by duration
    duration_buckets = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 1000)]
    print(f"\nProfitability by Duration:")
    for low, high in duration_buckets:
        bucket_trades = trades_df[(trades_df['duration'] >= low) & (trades_df['duration'] < high)]
        if len(bucket_trades) > 0:
            avg_pnl = bucket_trades['pnl_pct'].mean()
            win_rate = (bucket_trades['pnl_pct'] > 0).mean()
            count = len(bucket_trades)
            print(f"  {low:3d}-{high:3d} bars: {count:4d} trades, {avg_pnl:6.3f}% avg, {win_rate:.1%} win rate")
    
    # Consecutive losses
    trades_df['is_loss'] = trades_df['pnl_pct'] < 0
    trades_df['loss_streak'] = trades_df['is_loss'].groupby((~trades_df['is_loss']).cumsum()).cumsum()
    max_loss_streak = trades_df['loss_streak'].max()
    print(f"\nMax consecutive losses: {max_loss_streak}")
    
    # Small PnL trades (noise)
    noise_threshold = 0.05  # 5 bps
    noise_trades = trades_df[abs(trades_df['pnl_pct']) < noise_threshold]
    print(f"\nNoise trades (|PnL| < {noise_threshold}%):")
    print(f"  Count: {len(noise_trades)} ({len(noise_trades)/len(trades_df):.1%})")
    print(f"  Would save {len(noise_trades) * 0.01:.2f}% in costs by filtering")
    
    return trades_df

def main():
    workspace_path = Path("/Users/daws/ADMF-PC/workspaces/signal_generation_45c186a6")
    
    # Analyze specific high-frequency strategies
    test_strategies = [
        ("pivot_bounces", "SPY_compiled_strategy_77", 3193),  # 12.11% return
        ("bollinger_bands", "SPY_compiled_strategy_6", 6437),  # 8.82% return
        ("donchian_bands", "SPY_compiled_strategy_18", 1619),  # 2.93% return
        ("vwap_deviation", "SPY_compiled_strategy_39", 146),   # 9.81% return
    ]
    
    all_trades = {}
    for strat_type, strat_file, expected_trades in test_strategies:
        file_path = workspace_path / "traces" / "SPY_1m" / "signals" / strat_type / f"{strat_file}.parquet"
        if file_path.exists():
            trades_df = analyze_trade_quality(file_path, f"{strat_type} - {strat_file}")
            if trades_df is not None:
                all_trades[strat_type] = trades_df
    
    # Filtering recommendations
    print(f"\n{'='*60}")
    print("FILTERING RECOMMENDATIONS")
    print(f"{'='*60}")
    
    print("\n1. DURATION FILTER:")
    print("   - Filter trades with duration < 10 bars")
    print("   - This eliminates quick reversals and noise trades")
    
    print("\n2. MINIMUM MOVE FILTER:")
    print("   - Require minimum price move before signaling")
    print("   - E.g., price must move 0.1% from signal generation")
    
    print("\n3. SIGNAL PERSISTENCE:")
    print("   - Require signal to persist for N bars before acting")
    print("   - Reduces whipsaws in choppy markets")
    
    print("\n4. VOLUME/VOLATILITY FILTER:")
    print("   - Only trade when volume > X or volatility > Y")
    print("   - Avoid low-activity periods where spreads widen")
    
    print("\n5. TIME-OF-DAY FILTER:")
    print("   - Avoid first/last 30 minutes")
    print("   - Focus on liquid mid-day trading")

if __name__ == "__main__":
    main()