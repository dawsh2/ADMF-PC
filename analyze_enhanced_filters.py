#!/usr/bin/env python3
"""Analyze enhanced Keltner optimization with filters."""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from glob import glob

def get_strategy_description(strategy_id):
    """Map strategy ID to filter description based on optimize_keltner_enhanced_5m.yaml."""
    
    descriptions = [
        # 1. Base winner (1 strategy)
        "Base: Period=50, Mult=0.60",
        
        # 2. Fine-tune multipliers (6 strategies)
        "Mult=0.58", "Mult=0.59", "Mult=0.60", "Mult=0.61", "Mult=0.62", "Mult=0.63",
        
        # 3. VWAP stretch filter (6 strategies)
        "VWAP>0.25%", "VWAP>0.30%", "VWAP>0.35%", "VWAP>0.40%", "VWAP>0.45%", "VWAP>0.50%",
        
        # 4. RSI extremes (9 strategies)
        "RSI<25/70", "RSI<25/75", "RSI<25/80",
        "RSI<30/70", "RSI<30/75", "RSI<30/80", 
        "RSI<35/70", "RSI<35/75", "RSI<35/80",
        
        # 5. Volume spike (4 strategies)
        "Vol>1.5x", "Vol>2.0x", "Vol>2.5x", "Vol>3.0x",
        
        # 6. Volatility regime (3 strategies)
        "VolPct>60%", "VolPct>70%", "VolPct>80%",
        
        # 7. Combined VWAP + RSI (4 strategies)
        "VWAP>0.3%+RSI<30/70", "VWAP>0.3%+RSI<35/75",
        "VWAP>0.4%+RSI<30/70", "VWAP>0.4%+RSI<35/75",
        
        # 8. VWAP + Volume (4 strategies)
        "VWAP>0.3%+Vol>1.5x", "VWAP>0.3%+Vol>2.0x",
        "VWAP>0.35%+Vol>1.5x", "VWAP>0.35%+Vol>2.0x",
        
        # 9. Trend alignment (3 strategies)
        "Trend SMA50", "Trend SMA100", "Trend SMA200",
        
        # 10. Counter-trend RSI (1 strategy)
        "Counter-trend RSI",
        
        # 11. Triple filter (1 strategy)
        "VWAP+Vol+VolPct",
        
        # 12. Momentum confirmation (3 strategies)
        "ROC>0.1%", "ROC>0.2%", "ROC>0.3%",
        
        # 13. Bollinger squeeze (3 strategies)
        "BB Width<1%", "BB Width<1.5%", "BB Width<2%",
        
        # 14. ATR filter (4 strategies)
        "ATR>0.5", "ATR>0.75", "ATR>1.0", "ATR>1.25"
    ]
    
    if strategy_id < len(descriptions):
        return descriptions[strategy_id]
    else:
        return f"Strategy {strategy_id}"

def analyze_enhanced_workspace(workspace_path: str):
    """Analyze enhanced optimization results."""
    
    workspace = Path(workspace_path)
    signal_pattern = str(workspace / "traces/SPY_*/signals/keltner_bands/*.parquet")
    signal_files = sorted(glob(signal_pattern))
    
    print(f"Analyzing enhanced optimization: {workspace}")
    print(f"Found {len(signal_files)} strategies\n")
    
    results = []
    
    for signal_file in signal_files:
        try:
            signals_df = pd.read_parquet(signal_file)
            if signals_df.empty:
                continue
            
            # Extract strategy ID
            strategy_name = Path(signal_file).stem
            strategy_id = int(strategy_name.split('_')[-1])
            
            # Count trades
            total_signals = len(signals_df[signals_df['val'] != 0])
            if total_signals == 0:
                continue
            
            # Calculate returns
            trade_returns = []
            entry_price = None
            entry_signal = None
            
            for _, row in signals_df.iterrows():
                signal = row['val']
                price = row['px']
                
                if signal != 0 and entry_price is None:
                    entry_price = price
                    entry_signal = signal
                elif entry_price is not None and (signal == 0 or signal == -entry_signal):
                    log_return = np.log(price / entry_price) * entry_signal * 0.9998
                    trade_returns.append(log_return)
                    
                    if signal != 0:
                        entry_price = price
                        entry_signal = signal
                    else:
                        entry_price = None
                        entry_signal = None
            
            if not trade_returns:
                continue
            
            # Calculate metrics
            trade_returns_bps = [r * 10000 for r in trade_returns]
            edge_bps = np.mean(trade_returns_bps)
            win_rate = len([r for r in trade_returns_bps if r > 0]) / len(trade_returns_bps) * 100
            
            # Time span
            first_ts = pd.to_datetime(signals_df['ts'].iloc[0])
            last_ts = pd.to_datetime(signals_df['ts'].iloc[-1])
            trading_days = (last_ts - first_ts).days or 1
            trades_per_day = len(trade_returns) / trading_days * 252 / 365
            
            results.append({
                'strategy_id': strategy_id,
                'description': get_strategy_description(strategy_id),
                'edge_bps': edge_bps,
                'win_rate': win_rate,
                'trades_per_day': trades_per_day,
                'annual_trades': trades_per_day * 252,
                'total_trades': len(trade_returns)
            })
            
        except Exception as e:
            print(f"Error processing {strategy_name}: {e}")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Print results by category
    print("=== ENHANCED KELTNER OPTIMIZATION RESULTS ===\n")
    
    # 1. Base and multiplier tuning
    print("1. BASE STRATEGY & MULTIPLIER TUNING:")
    base_df = df[df['strategy_id'] < 7].sort_values('edge_bps', ascending=False)
    for _, row in base_df.iterrows():
        if row['annual_trades'] > 100:
            print(f"  {row['description']:15s}: {row['edge_bps']:6.2f} bps, "
                  f"{row['annual_trades']:4.0f} trades/yr, {row['win_rate']:5.1f}% win")
    
    # 2. VWAP filters
    print("\n2. VWAP DISTANCE FILTERS:")
    vwap_df = df[(df['strategy_id'] >= 7) & (df['strategy_id'] < 13)].sort_values('edge_bps', ascending=False)
    for _, row in vwap_df.iterrows():
        if row['annual_trades'] > 50:
            print(f"  {row['description']:15s}: {row['edge_bps']:6.2f} bps, "
                  f"{row['annual_trades']:4.0f} trades/yr, {row['win_rate']:5.1f}% win")
    
    # 3. RSI filters
    print("\n3. RSI EXTREME FILTERS:")
    rsi_df = df[(df['strategy_id'] >= 13) & (df['strategy_id'] < 22)].sort_values('edge_bps', ascending=False)
    for _, row in rsi_df.head(3).iterrows():
        if row['annual_trades'] > 50:
            print(f"  {row['description']:15s}: {row['edge_bps']:6.2f} bps, "
                  f"{row['annual_trades']:4.0f} trades/yr, {row['win_rate']:5.1f}% win")
    
    # Find strategies with >1 bps edge
    print("\n=== STRATEGIES EXCEEDING 1 BPS EDGE ===")
    high_edge = df[df['edge_bps'] > 1.0].sort_values('annual_trades', ascending=False)
    if not high_edge.empty:
        for _, row in high_edge.iterrows():
            print(f"{row['description']:20s}: {row['edge_bps']:6.2f} bps, "
                  f"{row['annual_trades']:4.0f} trades/yr, {row['win_rate']:5.1f}% win")
    else:
        print("No strategies exceeded 1 bps edge")
    
    # Best overall
    print("\n=== BEST OVERALL (Edge × Frequency) ===")
    df['expected_return'] = df['edge_bps'] * df['annual_trades'] / 10000
    best_overall = df.sort_values('expected_return', ascending=False).head(10)
    
    for _, row in best_overall.iterrows():
        print(f"{row['description']:20s}: {row['edge_bps']:6.2f} bps × "
              f"{row['annual_trades']:4.0f} trades = {row['expected_return']:6.2%} annual")
    
    # Summary statistics
    print(f"\n=== SUMMARY ===")
    print(f"Total strategies tested: {len(df)}")
    print(f"Strategies with positive edge: {len(df[df['edge_bps'] > 0])}")
    print(f"Strategies with >1 bps edge: {len(df[df['edge_bps'] > 1])}")
    print(f"Best edge: {df['edge_bps'].max():.2f} bps ({df.loc[df['edge_bps'].idxmax(), 'description']})")
    print(f"Best expected return: {df['expected_return'].max():.2%} ({df.loc[df['expected_return'].idxmax(), 'description']})")

if __name__ == "__main__":
    import sys
    workspace = sys.argv[1] if len(sys.argv) > 1 else "workspaces/signal_generation_15c51c13"
    analyze_enhanced_workspace(workspace)