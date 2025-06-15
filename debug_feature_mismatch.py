#!/usr/bin/env python3
"""
Debug the feature mismatch issue in grid search.

This script will:
1. Set up a minimal test case with bollinger_breakout strategy
2. Trace through the entire flow from topology building to signal generation
3. Log all feature names at each step
"""

import logging
import yaml
from src.core.coordinator.topology import TopologyBuilder
from src.core.coordinator.topology_runner import TopologyRunner
from src.data.loaders import SimpleCSVLoader

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Focus on key components
logging.getLogger('src.core.coordinator.topology').setLevel(logging.INFO)
logging.getLogger('src.strategy.state').setLevel(logging.DEBUG)
logging.getLogger('src.strategy.components.features').setLevel(logging.DEBUG)

print("=== DEBUGGING FEATURE MISMATCH ===\n")

# 1. Create a minimal test configuration
test_config = {
    'strategies': [
        {
            'type': 'bollinger_breakout',
            'name': 'bollinger_test',
            'params': {
                'period': 20,
                'std_dev': 2.0
            }
        }
    ],
    'classifiers': [],
    'data': {
        'symbols': ['SPY'],
        'start_date': '2023-01-01',
        'end_date': '2023-01-10'
    }
}

# 2. Build topology
print("Building topology...")
topology_builder = TopologyBuilder()

topology_def = {
    'mode': 'signal_generation',
    'config': test_config,
    'metadata': {'test': True}
}

topology = topology_builder.build_topology(topology_def)

print(f"\nTopology built with {len(topology['containers'])} containers")

# Check what features were inferred
if 'feature_hub' in topology['containers']:
    fh_config = topology['containers']['feature_hub']['config']
    features = fh_config.get('features', {})
    print(f"\nFeature Hub configured with {len(features)} features:")
    for feat_id, feat_config in features.items():
        print(f"  {feat_id}: {feat_config}")

# Check what strategies were configured
if 'strategy_state' in topology['containers']:
    ss_config = topology['containers']['strategy_state']['config']
    strategies = ss_config.get('strategies', [])
    print(f"\nStrategy State configured with {len(strategies)} strategies:")
    for strat in strategies:
        print(f"  {strat.get('name')}: params={strat.get('params')}")

# 3. Run the topology with a few bars
print("\n\nRunning topology with test data...")

# Load some test data
loader = SimpleCSVLoader()
df = loader.load('SPY', timeframe='1m')
print(f"Loaded {len(df)} bars of SPY data")

# Create runner
runner = TopologyRunner(mode='live')

# Limit to first 50 bars for testing
test_bars = df.iloc[:50]

print(f"\nProcessing {len(test_bars)} bars...")

# Track signals
signals_generated = []

# Process bars manually to see what happens
from src.core.events.types import Event, EventType
from datetime import datetime

# Start the runner
runner.run(topology, mode='live')

# Get the root container
root_container = runner.container_network.root_container

# Process each bar
for i, (timestamp, row) in enumerate(test_bars.iterrows()):
    bar = {
        'timestamp': timestamp,
        'open': row['open'],
        'high': row['high'],
        'low': row['low'],
        'close': row['close'],
        'volume': row['volume'],
        'symbol': 'SPY',
        'timeframe': '1m'
    }
    
    # Create BAR event
    event = Event(
        event_type=EventType.BAR.value,
        timestamp=datetime.now(),
        source='test',
        payload={
            'symbol': 'SPY',
            'bar': type('Bar', (), bar)()  # Create simple bar object
        }
    )
    
    # Publish to root container
    root_container.event_bus.publish(event)
    
    # Check for signals every 10 bars
    if (i + 1) % 10 == 0:
        print(f"\nAfter bar {i+1}:")
        
        # Check feature hub state
        fh_container = root_container.get_child('feature_hub')
        if fh_container:
            fh_component = fh_container.get_component('feature_hub')
            if fh_component:
                feature_hub = fh_component.get_feature_hub()
                features = feature_hub.get_features('SPY')
                print(f"  Features available: {len(features)}")
                # Show bollinger features
                bb_features = {k: v for k, v in features.items() if 'bollinger' in k}
                if bb_features:
                    print(f"  Bollinger features: {list(bb_features.keys())}")
                else:
                    print(f"  No bollinger features found!")
                    print(f"  Sample features: {list(features.keys())[:5]}")

print(f"\n\nTotal signals generated: {len(signals_generated)}")

# Stop the runner
runner.stop()

print("\n=== DEBUGGING COMPLETE ===")