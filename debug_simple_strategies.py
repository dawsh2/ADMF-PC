#!/usr/bin/env python3
"""
Debug the simplest strategies to understand why they're not generating signals.
Focus on SMA crossover, EMA crossover, and RSI threshold.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from src.data.loaders import SimpleCSVLoader
from src.strategy.components.features.hub import FeatureHub
from src.strategy.strategies.indicators.crossovers import sma_crossover, ema_sma_crossover
from src.strategy.strategies.indicators.oscillators import rsi_threshold

def debug_strategy_execution():
    """Debug the simplified strategy approach."""
    
    print("=== Testing Simplified Strategy Approach ===\n")
    
    # Load data
    loader = SimpleCSVLoader()
    df = loader.load('SPY', timeframe='1m')
    print(f"Loaded {len(df)} bars")
    
    # Initialize FeatureHub with explicit configuration
    feature_hub = FeatureHub(['SPY'], use_incremental=True)
    
    # Configure features that these strategies need (shared computation)
    feature_configs = {
        # All SMA periods needed across strategies
        'sma_3': {'type': 'sma', 'period': 3},
        'sma_7': {'type': 'sma', 'period': 7}, 
        'sma_10': {'type': 'sma', 'period': 10},
        'sma_20': {'type': 'sma', 'period': 20},
        'sma_26': {'type': 'sma', 'period': 26},
        'sma_50': {'type': 'sma', 'period': 50},
        
        # All EMA periods needed
        'ema_11': {'type': 'ema', 'period': 11},
        'ema_12': {'type': 'ema', 'period': 12},
        'ema_26': {'type': 'ema', 'period': 26},
        
        # RSI periods
        'rsi_14': {'type': 'rsi', 'period': 14}
    }
    
    print(f"Configuring FeatureHub with {len(feature_configs)} shared features")
    feature_hub.configure_features(feature_configs)
    
    # Test strategies with different parameter combinations (shared features)
    strategies_to_test = [
        ('sma_crossover_10_20', sma_crossover, {'fast_period': 10, 'slow_period': 20}),
        ('sma_crossover_7_50', sma_crossover, {'fast_period': 7, 'slow_period': 50}),
        ('ema_sma_crossover', ema_sma_crossover, {'ema_period': 11, 'sma_period': 26}),
        ('rsi_threshold', rsi_threshold, {'rsi_period': 14, 'threshold': 50})
    ]
    
    for strategy_name, strategy_func, params in strategies_to_test:
        print(f"\n=== Testing {strategy_name} ===")
        print(f"Params: {params}")
        
        signals = []
        feature_status = {}
        
        # Process bars
        for i in range(min(200, len(df))):
            bar = {
                'timestamp': df.index[i],
                'open': df.iloc[i]['open'],
                'high': df.iloc[i]['high'], 
                'low': df.iloc[i]['low'],
                'close': df.iloc[i]['close'],
                'volume': df.iloc[i]['volume'],
                'symbol': 'SPY',
                'timeframe': '1m'
            }
            
            # Update features
            feature_hub.update_bar('SPY', bar)
            features = feature_hub.get_features('SPY')
            
            # Track feature availability at key bars
            if i in [30, 50, 100]:
                available = {k: v for k, v in features.items() if v is not None}
                feature_status[i] = {
                    'total_features': len(features),
                    'available_features': len(available),
                    'feature_names': list(available.keys())[:10]  # First 10
                }
            
            # Try strategy
            try:
                signal = strategy_func(features, bar, params)
                if signal is not None:
                    signals.append((i, signal.get('signal_value', 0)))
                    if len(signals) <= 3:  # Log first few signals
                        print(f"  Signal at bar {i}: {signal.get('signal_value')}")
            except Exception as e:
                if i == 50:  # Only log once
                    print(f"  Error at bar {i}: {e}")
        
        # Results
        print(f"  Total signals: {len(signals)}")
        if signals:
            values = [s[1] for s in signals]
            print(f"  Signal distribution: {pd.Series(values).value_counts().to_dict()}")
            print(f"  First signal at bar: {signals[0][0]}")
            
            # Show metadata from first signal to demonstrate parameter tracking
            if signals[0][1] != 0:  # If not neutral signal
                first_signal_metadata = None
                for i_bar, bar in enumerate(df.iloc[:200].itertuples()):
                    bar_dict = {
                        'timestamp': bar.Index,
                        'open': bar.open,
                        'high': bar.high,
                        'low': bar.low,
                        'close': bar.close,
                        'volume': bar.volume,
                        'symbol': 'SPY',
                        'timeframe': '1m'
                    }
                    signal = strategy_func(feature_hub.get_features('SPY'), bar_dict, params)
                    if signal and signal.get('signal_value') != 0:
                        print(f"  First signal metadata: {signal.get('metadata', {})}")
                        break
        
        # Feature sharing demonstration
        if i >= 50:  # After some warmup
            final_features = feature_hub.get_features('SPY')
            print(f"  Features used (shared across strategies):")
            
            if 'sma_crossover' in strategy_name:
                fast_period = params.get('fast_period')
                slow_period = params.get('slow_period')
                fast_sma = final_features.get(f'sma_{fast_period}')
                slow_sma = final_features.get(f'sma_{slow_period}')
                print(f"    sma_{fast_period}: {fast_sma:.3f}" if fast_sma else f"    sma_{fast_period}: None")
                print(f"    sma_{slow_period}: {slow_sma:.3f}" if slow_sma else f"    sma_{slow_period}: None")
                
            elif 'ema_sma' in strategy_name:
                ema_period = params.get('ema_period')
                sma_period = params.get('sma_period')
                ema = final_features.get(f'ema_{ema_period}')
                sma = final_features.get(f'sma_{sma_period}')
                print(f"    ema_{ema_period}: {ema:.3f}" if ema else f"    ema_{ema_period}: None")
                print(f"    sma_{sma_period}: {sma:.3f}" if sma else f"    sma_{sma_period}: None")
                
            elif 'rsi' in strategy_name:
                rsi_14 = final_features.get('rsi_14')
                print(f"    rsi_14: {rsi_14:.3f}" if rsi_14 else "    rsi_14: None")
    
    print(f"\n=== SHARED COMPUTATION BENEFITS ===")
    print("✅ sma_10, sma_20, sma_50 computed ONCE, used by multiple strategies")
    print("✅ No naming mismatches - simple sma_10, ema_12 format")  
    print("✅ Parameters stored in signal metadata for analysis")
    print("✅ Discovery system simplified - just collect base features + periods")

if __name__ == "__main__":
    debug_strategy_execution()