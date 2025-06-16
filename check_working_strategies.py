#!/usr/bin/env python3
"""
Check which strategies are generating signals vs not working.
Runs minimal simulation to identify strategies that need fixes.
"""

import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.events.bus import EventBus
from src.core.events.types import Event, EventType
from src.core.containers.factory import ContainerFactory
from src.core.coordinator.coordinator import Coordinator
from src.data.handlers import CSVDataHandler


def create_minimal_market_data(num_bars=10):
    """Create minimal market data for testing"""
    dates = pd.date_range(
        start=datetime(2024, 1, 1),
        periods=num_bars,
        freq='D'
    )
    
    # Create trending data to trigger various signal types
    base_price = 100.0
    trend = np.linspace(0, 10, num_bars)  # Upward trend
    noise = np.random.normal(0, 0.5, num_bars)
    
    prices = base_price + trend + noise
    
    return pd.DataFrame({
        'timestamp': dates,
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(1000000, 2000000, num_bars),
        'symbol': 'TEST'
    })


def track_signal_generation(config_path):
    """Run simulation and track which strategies generate signals"""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Track results
    strategy_signals = defaultdict(int)
    strategy_instances = defaultdict(set)
    strategy_errors = defaultdict(list)
    
    # Create event bus
    event_bus = EventBus()
    
    # Track all signal events
    signals_by_strategy = defaultdict(list)
    
    def on_signal(event):
        if hasattr(event, 'data') and event.data:
            signal = event.data
            strategy_type = signal.get('metadata', {}).get('strategy_type', 'unknown')
            strategy_name = signal.get('metadata', {}).get('strategy_name', 'unknown')
            strategy_signals[strategy_type] += 1
            strategy_instances[strategy_type].add(strategy_name)
            signals_by_strategy[strategy_type].append({
                'name': strategy_name,
                'direction': signal.get('direction', 'unknown'),
                'strength': signal.get('strength', 0.0)
            })
    
    event_bus.subscribe(EventType.SIGNAL, on_signal)
    
    # Track errors
    def on_error(event):
        if hasattr(event, 'error'):
            strategy_type = getattr(event, 'strategy_type', 'unknown')
            strategy_errors[strategy_type].append(str(event.error))
    
    # Subscribe to error event if it exists
    # event_bus.subscribe('strategy_error', on_error)
    
    # Create minimal data
    print("Creating test data...")
    df = create_minimal_market_data(num_bars=50)  # More bars for better signal generation
    
    # Setup components
    print("Setting up components...")
    container_factory = ContainerFactory()
    
    # Create topology from config
    topology_config = config.get('topology', {})
    if 'file' in topology_config:
        # Load topology from file
        topology_path = Path(topology_config['file'])
        if not topology_path.is_absolute():
            topology_path = Path(config_path).parent / topology_path
        
        with open(topology_path, 'r') as f:
            topology_data = yaml.safe_load(f)
            topology_config = topology_data.get('topology', topology_config)
    
    # Create coordinator
    coordinator = TopologyCoordinator(
        container_factory=container_factory,
        event_bus=event_bus,
        topology_config=topology_config
    )
    
    # Initialize
    print("Initializing topology...")
    coordinator.initialize()
    
    # Get expected strategy types from config
    expected_types = set()
    if 'strategies' in topology_config:
        for strategy_def in topology_config['strategies']:
            if isinstance(strategy_def, dict) and 'type' in strategy_def:
                expected_types.add(strategy_def['type'])
    
    # Process bars
    print(f"Processing {len(df)} bars...")
    for i, (_, row) in enumerate(df.iterrows()):
        bar_data = {
            'symbol': row['symbol'],
            'timestamp': row['timestamp'],
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row['volume']
        }
        
        market_event = Event(
            event_type=EventType.BAR,
            timestamp=row['timestamp'],
            data=bar_data
        )
        
        event_bus.publish(market_event)
        
        # Give strategies time to process
        event_bus.process_pending()
        
        if i % 10 == 0:
            print(f"  Processed {i+1} bars, signals so far: {sum(strategy_signals.values())}")
    
    # Final processing
    event_bus.process_pending()
    
    return {
        'strategy_signals': dict(strategy_signals),
        'strategy_instances': {k: list(v) for k, v in strategy_instances.items()},
        'strategy_errors': dict(strategy_errors),
        'expected_types': expected_types,
        'signals_detail': dict(signals_by_strategy)
    }


def main():
    """Main analysis"""
    config_path = "config/expansive_grid_search.yaml"
    
    print("=" * 80)
    print("STRATEGY SIGNAL GENERATION CHECK")
    print("=" * 80)
    
    results = track_signal_generation(config_path)
    
    # Analyze results
    working_strategies = set(results['strategy_signals'].keys())
    expected_strategies = results['expected_types']
    not_working = expected_strategies - working_strategies
    
    print(f"\nExpected strategy types: {len(expected_strategies)}")
    print(f"Working strategies: {len(working_strategies)}")
    print(f"Not working: {len(not_working)}")
    
    print("\n" + "=" * 80)
    print("WORKING STRATEGIES (generating signals):")
    print("=" * 80)
    
    for strategy_type in sorted(working_strategies):
        signal_count = results['strategy_signals'][strategy_type]
        instances = results['strategy_instances'][strategy_type]
        print(f"\n{strategy_type}:")
        print(f"  - Total signals: {signal_count}")
        print(f"  - Instances: {len(instances)}")
        if len(instances) <= 5:
            for instance in sorted(instances):
                print(f"    * {instance}")
        else:
            print(f"    * {len(instances)} instances (too many to list)")
        
        # Show sample signals
        signals = results['signals_detail'].get(strategy_type, [])
        if signals:
            print(f"  - Sample signal: {signals[0]['direction']} @ {signals[0]['strength']:.3f}")
    
    print("\n" + "=" * 80)
    print("NOT WORKING STRATEGIES (no signals):")
    print("=" * 80)
    
    for strategy_type in sorted(not_working):
        print(f"\n{strategy_type}:")
        if strategy_type in results['strategy_errors']:
            print(f"  - Errors: {results['strategy_errors'][strategy_type][:2]}")  # First 2 errors
        else:
            print(f"  - No errors recorded (strategy may not be initializing)")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(f"Total expected types: {len(expected_strategies)}")
    print(f"Working: {len(working_strategies)} ({len(working_strategies)/len(expected_strategies)*100:.1f}%)")
    print(f"Not working: {len(not_working)} ({len(not_working)/len(expected_strategies)*100:.1f}%)")
    print(f"Total signals generated: {sum(results['strategy_signals'].values())}")
    
    # List all expected types for reference
    print("\n" + "=" * 80)
    print("ALL EXPECTED STRATEGY TYPES:")
    print("=" * 80)
    for i, strategy_type in enumerate(sorted(expected_strategies), 1):
        status = "✓" if strategy_type in working_strategies else "✗"
        print(f"{i:2d}. [{status}] {strategy_type}")
    
    return results


if __name__ == "__main__":
    main()