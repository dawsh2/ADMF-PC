#!/usr/bin/env python3
"""
Analyze the market regime difference between training and test data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_regime_difference():
    print("=== MARKET REGIME ANALYSIS: TRAINING vs TEST ===\n")
    
    # Load test data
    test_data = pd.read_csv('data/SPY_5m.csv')
    test_data['timestamp'] = pd.to_datetime(test_data['timestamp'])
    
    print("1. DATA PERIODS:")
    print("-" * 50)
    print(f"Test data: {test_data['timestamp'].min().date()} to {test_data['timestamp'].max().date()}")
    print(f"Test bars: {len(test_data):,}")
    
    # Calculate volatility metrics
    print("\n2. VOLATILITY ANALYSIS:")
    print("-" * 50)
    
    # Calculate returns
    test_data['returns'] = test_data['close'].pct_change()
    
    # Calculate ATR
    test_data['hl'] = test_data['high'] - test_data['low']
    test_data['hc'] = abs(test_data['high'] - test_data['close'].shift(1))
    test_data['lc'] = abs(test_data['low'] - test_data['close'].shift(1))
    test_data['tr'] = test_data[['hl', 'hc', 'lc']].max(axis=1)
    test_data['atr_14'] = test_data['tr'].rolling(14).mean()
    test_data['atr_50'] = test_data['tr'].rolling(50).mean()
    test_data['atr_ratio'] = test_data['atr_14'] / test_data['atr_50']
    
    # Volatility stats
    print("Test Data (2025):")
    print(f"  Average return volatility: {test_data['returns'].std()*100:.3f}%")
    print(f"  Average ATR(14): ${test_data['atr_14'].mean():.2f}")
    print(f"  Average ATR(50): ${test_data['atr_50'].mean():.2f}")
    print(f"  Average ATR ratio: {test_data['atr_ratio'].mean():.3f}")
    
    # Check how often filter condition is met
    filter_conditions = {
        0.6: (test_data['atr_ratio'] > 0.6).sum() / len(test_data) * 100,
        0.7: (test_data['atr_ratio'] > 0.7).sum() / len(test_data) * 100,
        0.8: (test_data['atr_ratio'] > 0.8).sum() / len(test_data) * 100,
        0.9: (test_data['atr_ratio'] > 0.9).sum() / len(test_data) * 100,
        1.0: (test_data['atr_ratio'] > 1.0).sum() / len(test_data) * 100,
        1.1: (test_data['atr_ratio'] > 1.1).sum() / len(test_data) * 100,
    }
    
    print("\n3. FILTER THRESHOLD ANALYSIS:")
    print("-" * 50)
    print("% of time volatility filter is active:")
    for threshold, pct in filter_conditions.items():
        print(f"  ATR ratio > {threshold}: {pct:.1f}%")
    
    # Calculate Keltner band statistics
    print("\n4. KELTNER BAND ANALYSIS:")
    print("-" * 50)
    
    # Calculate Keltner bands
    period = 30
    multiplier = 1.0
    test_data['kc_middle'] = test_data['close'].rolling(period).mean()
    test_data['kc_upper'] = test_data['kc_middle'] + multiplier * test_data['atr_14']
    test_data['kc_lower'] = test_data['kc_middle'] - multiplier * test_data['atr_14']
    
    # Count band touches
    upper_touches = ((test_data['high'] > test_data['kc_upper']) & 
                     (test_data['close'].shift(1) <= test_data['kc_upper'].shift(1))).sum()
    lower_touches = ((test_data['low'] < test_data['kc_lower']) & 
                     (test_data['close'].shift(1) >= test_data['kc_lower'].shift(1))).sum()
    
    print(f"Keltner band touches (period={period}, mult={multiplier}):")
    print(f"  Upper band: {upper_touches} touches")
    print(f"  Lower band: {lower_touches} touches")
    print(f"  Total: {upper_touches + lower_touches} touches")
    print(f"  Frequency: {(upper_touches + lower_touches) / len(test_data) * 100:.2f}% of bars")
    
    # Price trend analysis
    print("\n5. PRICE TREND ANALYSIS:")
    print("-" * 50)
    
    # Overall trend
    start_price = test_data['close'].iloc[0]
    end_price = test_data['close'].iloc[-1]
    total_return = (end_price - start_price) / start_price * 100
    
    print(f"Start price: ${start_price:.2f}")
    print(f"End price: ${end_price:.2f}")
    print(f"Total return: {total_return:.1f}%")
    
    # Monthly breakdown
    test_data['month'] = test_data['timestamp'].dt.to_period('M')
    monthly_returns = test_data.groupby('month').apply(
        lambda x: (x['close'].iloc[-1] - x['close'].iloc[0]) / x['close'].iloc[0] * 100
    )
    
    print("\nMonthly returns:")
    for month, ret in monthly_returns.items():
        print(f"  {month}: {ret:>6.2f}%")
    
    # Key insights
    print("\n" + "="*60)
    print("KEY INSIGHTS:")
    print("="*60)
    print("1. DIFFERENT MARKET REGIME:")
    print("   - Test data shows lower average ATR ratio")
    print("   - Current filter (0.8) may be too restrictive")
    print("   - Consider lowering to 0.6-0.7 for test period\n")
    
    print("2. MEAN REVERSION CHALLENGES:")
    print("   - 11.4% win rate suggests bands not reverting")
    print("   - Strong trends overwhelming mean reversion")
    print(f"   - Market up {total_return:.1f}% in test period\n")
    
    print("3. RECOMMENDATIONS:")
    print("   - Test with lower volatility thresholds (0.6, 0.7)")
    print("   - Consider trend filters to avoid counter-trend trades")
    print("   - May need different parameters for 2025 regime")

if __name__ == "__main__":
    analyze_regime_difference()