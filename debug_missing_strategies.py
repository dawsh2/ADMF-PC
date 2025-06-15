#!/usr/bin/env python3
"""
Debug why specific strategies aren't executing
"""
import os
import sys

# Add path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.containers.factory import ContainerFactory
from src.core.events.bus import EventBus
from src.data.handlers import CSVDataHandler
from src.strategy.state import StrategyState
from src.strategy.components.features.hub import FeatureHub
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Test a specific missing strategy
print("Testing pivot_points strategy execution...")

# Create minimal setup
event_bus = EventBus()
container_factory = ContainerFactory(event_bus)

# Create strategy container with pivot_points
strategy_config = {
    'strategies': [{
        'type': 'pivot_points',
        'name': 'pivot_points_test',
        'params': {'pivot_type': 'standard'}
    }],
    'features': {
        'pivot_points': {
            'params': ['pivot_type'],
            'defaults': {'pivot_type': 'standard'}
        }
    }
}

try:
    # Create strategy state
    strategy_state = StrategyState(
        strategies=strategy_config['strategies'],
        feature_configs=strategy_config['features']
    )
    
    print(f"Strategy state created with {len(strategy_state.strategies)} strategies")
    
    # Check if strategy was loaded
    for strat_id, strat_info in strategy_state.strategies.items():
        print(f"  - {strat_id}: {strat_info}")
        
    # Test feature hub
    feature_hub = FeatureHub()
    feature_hub.configure_features({
        'pivot_points_standard': {
            'feature': 'pivot_points',
            'pivot_type': 'standard'
        }
    })
    
    # Test bar processing
    test_bar = {
        'open': 100.0,
        'high': 101.0,
        'low': 99.0,
        'close': 100.5,
        'volume': 1000000,
        'timestamp': '2023-01-01 10:00:00',
        'symbol': 'SPY',
        'timeframe': '1m'
    }
    
    feature_hub.update_bar('SPY', test_bar)
    features = feature_hub.get_features('SPY')
    print(f"\nFeatures computed: {list(features.keys())}")
    
    # Process bar through strategy
    signals = strategy_state.process_bar(test_bar, features)
    print(f"\nSignals generated: {signals}")
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()

# Now test why it's not included in the run
print("\n\nChecking strategy discovery...")
from src.core.components.discovery import discover_strategies

discovered = discover_strategies()
print(f"Discovered {len(discovered)} strategies")

missing_strategies = [
    'pivot_points', 'fibonacci_retracement', 'stochastic_rsi', 
    'ultimate_oscillator', 'vortex_crossover', 'macd_crossover'
]

for strat in missing_strategies:
    if strat in discovered:
        print(f"✓ {strat} is discovered")
    else:
        print(f"✗ {strat} NOT discovered")