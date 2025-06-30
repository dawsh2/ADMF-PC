#!/usr/bin/env python3
"""
Compare signal generation results between 1m and 15m timeframes.

This helps determine if 15m timeframe produces better trading opportunities
that can offset higher execution costs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

def analyze_signals(signal_file):
    """Analyze signals from a signal generation run."""
    
    with open(signal_file, 'r') as f:
        data = json.load(f)
    
    signals = data.get('signals', {})
    if not signals:
        return None
    
    # Get the strategy key (should be only one for single strategy runs)
    strategy_key = list(signals.keys())[0]
    signal_list = signals[strategy_key]
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(signal_list)
    if df.empty:
        return None
    
    # Filter out FLAT signals for trade analysis
    trades = df[df['value'] != 0].copy()
    
    # Calculate stats
    stats = {
        'total_signals': len(df),
        'total_trades': len(trades),
        'long_trades': len(trades[trades['value'] > 0]),
        'short_trades': len(trades[trades['value'] < 0]),
        'signal_rate': len(trades) / len(df) if len(df) > 0 else 0,
    }
    
    return stats, trades

def main():
    """Compare 1m vs 15m timeframe results."""
    
    # Paths to workspace directories
    workspaces_dir = Path('/Users/daws/ADMF-PC/workspaces')
    
    print("=== Timeframe Comparison Analysis ===\n")
    
    # Find recent signal generation runs
    recent_runs = []
    for workspace in workspaces_dir.glob('signal_generation_*'):
        signals_file = workspace / 'signals.json'
        config_file = workspace / 'config.json'
        
        if signals_file.exists() and config_file.exists():
            # Load config to check timeframe
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Check if it's our Bollinger RSI strategy
            strategy_config = config.get('strategy', {})
            if 'bollinger_rsi_simple_signals' in strategy_config:
                symbols = config.get('symbols', [])
                timeframe = '15m' if 'SPY_15m' in symbols else '1m'
                
                recent_runs.append({
                    'workspace': workspace,
                    'timeframe': timeframe,
                    'config': config,
                    'signals_file': signals_file
                })
    
    # Analyze by timeframe
    results = {'1m': [], '15m': []}
    
    for run in recent_runs:
        stats, trades = analyze_signals(run['signals_file'])
        if stats:
            results[run['timeframe']].append({
                'stats': stats,
                'trades': trades,
                'workspace': run['workspace'].name
            })
    
    # Compare results
    for timeframe in ['1m', '15m']:
        if results[timeframe]:
            print(f"\n{timeframe} Timeframe Results:")
            print("-" * 40)
            
            total_signals = sum(r['stats']['total_signals'] for r in results[timeframe])
            total_trades = sum(r['stats']['total_trades'] for r in results[timeframe])
            total_longs = sum(r['stats']['long_trades'] for r in results[timeframe])
            total_shorts = sum(r['stats']['short_trades'] for r in results[timeframe])
            
            print(f"Runs analyzed: {len(results[timeframe])}")
            print(f"Total signals: {total_signals}")
            print(f"Total trades: {total_trades}")
            print(f"  - Longs: {total_longs}")
            print(f"  - Shorts: {total_shorts}")
            print(f"Signal rate: {total_trades/total_signals:.1%}" if total_signals > 0 else "N/A")
            
            # Show latest run details
            if results[timeframe]:
                latest = results[timeframe][-1]
                print(f"\nLatest run: {latest['workspace']}")
                
    # Key insights for timeframe selection
    print("\n\n=== Key Insights for Timeframe Selection ===")
    print("-" * 50)
    
    print("\n15-Minute Advantages:")
    print("- Larger price moves per bar (more profit potential)")
    print("- Lower execution frequency (reduced slippage impact)")
    print("- More reliable technical patterns")
    print("- Better risk/reward ratios")
    
    print("\n15-Minute Disadvantages:")
    print("- Fewer trading opportunities")
    print("- Slower reaction to market changes")
    print("- Larger potential drawdowns")
    
    print("\n\nRecommendation:")
    print("Use 15m timeframe if execution costs are significant (> 0.05% per trade)")
    print("as the larger moves can better offset transaction costs.")

if __name__ == "__main__":
    main()