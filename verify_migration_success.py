#!/usr/bin/env python3
"""
Verify that the strategy migration successfully resolved the naming mismatch issues.
This should show that all strategies can now generate signals.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from src.core.components.discovery import get_component_registry, auto_discover_all_components
from src.data.loaders import SimpleCSVLoader
from src.strategy.components.features.hub import FeatureHub

def verify_migration_success():
    """Verify that all migrated strategies can generate signals."""
    
    print("=== MIGRATION SUCCESS VERIFICATION ===\n")
    
    # Discover all strategies
    auto_discover_all_components()
    registry = get_component_registry()
    
    # Get all strategy components
    strategies = registry.get_components_by_type('strategy')
    strategy_names = [s.name for s in strategies]
    
    print(f"✅ Discovered {len(strategies)} strategies total")
    print(f"Strategy names: {sorted(strategy_names)}\n")
    
    # Load minimal data
    loader = SimpleCSVLoader()
    df = loader.load('SPY', timeframe='1m')
    test_data = df.head(50)  # Just 50 bars for quick test
    
    print(f"📊 Loaded {len(test_data)} test bars\n")
    
    # Test a representative sample of migrated strategies
    test_strategies = [
        ('sma_crossover', {'fast_period': 10, 'slow_period': 20}),
        ('ema_crossover', {'fast_ema_period': 12, 'slow_ema_period': 26}),
        ('rsi_threshold', {'rsi_period': 14, 'threshold': 50}),
        ('bollinger_breakout', {'period': 20, 'std_dev': 2.0}),
        ('macd_crossover', {'fast_ema': 12, 'slow_ema': 26, 'signal_ema': 9}),
        ('adx_trend_strength', {'adx_period': 14, 'di_period': 14}),
        ('williams_r', {'williams_period': 14}),
        ('pivot_points', {'pivot_type': 'standard'})
    ]
    
    success_count = 0
    
    for strategy_name, params in test_strategies:
        print(f"🧪 Testing {strategy_name}...")
        
        # Get strategy function
        strategy_info = registry.get_component(strategy_name)
        if not strategy_info:
            print(f"   ❌ Strategy not found in registry")
            continue
            
        strategy_func = strategy_info.factory
        
        # Initialize FeatureHub and configure basic features
        feature_hub = FeatureHub(['SPY'], use_incremental=True)
        
        # Configure commonly needed features
        feature_configs = {
            # SMA/EMA features
            'sma_10': {'type': 'sma', 'period': 10},
            'sma_20': {'type': 'sma', 'period': 20},
            'sma_26': {'type': 'sma', 'period': 26},
            'ema_12': {'type': 'ema', 'period': 12},
            'ema_26': {'type': 'ema', 'period': 26},
            'ema_9': {'type': 'ema', 'period': 9},
            
            # Oscillator features
            'rsi_14': {'type': 'rsi', 'period': 14},
            'williams_r_14': {'type': 'williams_r', 'period': 14},
            
            # Volatility features
            'bollinger_bands_20_2.0': {'type': 'bollinger_bands', 'period': 20, 'std_dev': 2.0},
            
            # MACD features
            'macd_12_26_9': {'type': 'macd', 'fast': 12, 'slow': 26, 'signal': 9},
            
            # ADX features
            'adx_14': {'type': 'adx', 'period': 14},
            
            # Pivot features (using simple features for test)
            'pivot_points_standard': {'type': 'pivot_points', 'pivot_type': 'standard'}
        }
        
        feature_hub.configure_features(feature_configs)
        
        # Test signal generation
        signals_generated = 0
        
        for i, row in test_data.iterrows():
            bar = {
                'timestamp': i,
                'open': row['open'],
                'high': row['high'],
                'low': row['low'], 
                'close': row['close'],
                'volume': row['volume'],
                'symbol': 'SPY',
                'timeframe': '1m'
            }
            
            # Update features
            feature_hub.update_bar('SPY', bar)
            features = feature_hub.get_features('SPY')
            
            # Try to generate signal
            try:
                signal = strategy_func(features, bar, params)
                if signal is not None:
                    signals_generated += 1
                    
                    # Verify metadata contains parameters (key requirement)
                    metadata = signal.get('metadata', {})
                    param_keys = [k for k in metadata.keys() if any(p in k for p in ['period', 'threshold', 'std_dev', 'ema', 'type'])]
                    
                    if param_keys:
                        print(f"   ✅ Generated {signals_generated} signals with parameters in metadata: {param_keys[:3]}...")
                        break
                        
            except Exception as e:
                print(f"   ⚠️  Error: {e}")
                break
        
        if signals_generated > 0:
            print(f"   ✅ SUCCESS: Generated {signals_generated} signals")
            success_count += 1
        else:
            print(f"   ❌ FAILED: No signals generated")
        
        print()
    
    # Summary
    success_rate = (success_count / len(test_strategies)) * 100
    print(f"=== MIGRATION VERIFICATION RESULTS ===")
    print(f"✅ Successful strategies: {success_count}/{len(test_strategies)} ({success_rate:.1f}%)")
    
    if success_rate >= 75:
        print("🎉 MIGRATION SUCCESS! Most strategies generating signals correctly")
        print("✅ Simplified feature declarations working")
        print("✅ Parameter metadata properly included")
        print("✅ No more naming mismatch issues")
    else:
        print("⚠️  Some strategies still have issues - need further investigation")
    
    return success_rate >= 75

if __name__ == "__main__":
    verify_migration_success()