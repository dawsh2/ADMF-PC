#!/usr/bin/env python3
"""Test the full pipeline for a single strategy to debug the issue."""

import logging
from src.core.containers.components.feature_hub_component import create_feature_hub_component
from src.strategy.state import StrategyState
from src.strategy.strategies.indicators.volatility import bollinger_breakout

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("=== TESTING FULL PIPELINE ===\n")

# 1. Create FeatureHub with bollinger bands config
feature_configs = {
    'bollinger_bands_11_1.5': {'feature': 'bollinger_bands', 'period': 11, 'std_dev': 1.5},
    'bollinger_bands_20_2.0': {'feature': 'bollinger_bands', 'period': 20, 'std_dev': 2.0},
}

container_config = {
    'name': 'feature_hub',
    'type': 'feature_hub',
    'symbols': ['SPY'],
    'features': feature_configs
}

fh_component = create_feature_hub_component(container_config)
feature_hub = fh_component._feature_hub

print(f"1. Created FeatureHub with {len(feature_hub.feature_configs)} features")

# 2. Create StrategyState (ComponentState)
from src.strategy.state import ComponentState
strategy_state = ComponentState(symbols=['SPY'])

# Manually set the feature hub
strategy_state._feature_hub = feature_hub

# Add the strategy
strategies = [
    {
        'name': 'bollinger_breakout_test1',
        'type': 'bollinger_breakout',
        'params': {'period': 11, 'std_dev': 1.5}
    },
    {
        'name': 'bollinger_breakout_test2', 
        'type': 'bollinger_breakout',
        'params': {'period': 20, 'std_dev': 2.0}
    }
]

for strat in strategies:
    strategy_state.add_strategy(
        strategy_id=strat['name'],
        strategy_func=bollinger_breakout,
        parameters=strat['params']
    )

print(f"2. Added {len(strategies)} strategies")

# 3. Feed bars and check signals
symbol = 'SPY'
signals_generated = []

print("\n3. Feeding bars...")
for i in range(30):
    bar = {
        'timestamp': 1234567890 + i * 60,
        'open': 100 + i * 0.1,
        'high': 101 + i * 0.1,
        'low': 99 + i * 0.1,
        'close': 100.5 + i * 0.1 + (0.5 if i == 25 else 0),  # Spike at bar 25
        'volume': 1000000,
        'symbol': symbol,
        'timeframe': '1m'
    }
    
    # Update FeatureHub
    bar_dict = {
        'open': bar['open'],
        'high': bar['high'],
        'low': bar['low'],
        'close': bar['close'],
        'volume': bar['volume']
    }
    feature_hub.update_bar(symbol, bar_dict)
    
    # Get features
    features = feature_hub.get_features(symbol)
    
    # Show feature state at key bars
    if i == 19 or i == 25 or i == 29:
        print(f"\n  Bar {i+1}:")
        print(f"    Price: {bar['close']:.2f}")
        bb_features = {k: v for k, v in features.items() if 'bollinger' in k}
        if bb_features:
            print(f"    Bollinger features available: {len(bb_features)}")
            for k, v in sorted(bb_features.items())[:3]:
                print(f"      {k}: {v:.4f}")
        else:
            print(f"    No Bollinger features yet")
    
    # Process strategies
    if features:  # Only process if we have features
        signals = strategy_state.process_bar(symbol, bar)
        if signals:
            for signal in signals:
                signals_generated.append((i+1, signal))
                print(f"\n  📊 SIGNAL at bar {i+1}: {signal['signal_value']} from {signal['strategy_id']}")
                print(f"     Price: {bar['close']:.2f}")
                meta = signal.get('metadata', {})
                print(f"     Upper: {meta.get('upper_band', 'N/A')}, Lower: {meta.get('lower_band', 'N/A')}")

print(f"\n\n=== SUMMARY ===")
print(f"Total signals generated: {len(signals_generated)}")
if not signals_generated:
    print("\n❌ NO SIGNALS GENERATED!")
    print("\nDebugging info:")
    print(f"  - FeatureHub has data: {feature_hub.has_sufficient_data(symbol)}")
    print(f"  - Final features available: {len(feature_hub.get_features(symbol))}")
    
    # Show final feature state
    final_features = feature_hub.get_features(symbol)
    print("\n  Final feature values:")
    for k, v in sorted(final_features.items()):
        print(f"    {k}: {v}")
else:
    print("\n✅ Signals generated successfully")