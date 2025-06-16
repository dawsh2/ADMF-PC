#!/usr/bin/env python3
"""
Simplified check for which strategies are generating signals.
Directly tests strategy components without full topology.
"""

import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
import sys
import importlib

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.strategy.strategies.registry import get_strategy_registry
from src.core.events.bus import EventBus
from src.core.events.types import Event, EventType
from src.strategy.state import StrategyState


def create_test_data(num_bars=100):
    """Create test market data with trends"""
    dates = pd.date_range(
        start=datetime(2024, 1, 1),
        periods=num_bars,
        freq='D'
    )
    
    # Create various market conditions
    base_price = 100.0
    
    # First 30 bars: uptrend
    trend1 = np.linspace(0, 10, 30)
    # Next 30 bars: downtrend  
    trend2 = np.linspace(10, 0, 30)
    # Next 20 bars: sideways
    trend3 = np.ones(20) * 5
    # Last 20 bars: volatile
    trend4 = np.sin(np.linspace(0, 4*np.pi, 20)) * 5 + 5
    
    trend = np.concatenate([trend1, trend2, trend3, trend4])[:num_bars]
    noise = np.random.normal(0, 0.5, num_bars)
    
    prices = base_price + trend + noise
    
    return pd.DataFrame({
        'timestamp': dates,
        'open': prices * 0.995,
        'high': prices * 1.005,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.randint(1000000, 2000000, num_bars)
    })


def test_strategy_directly(strategy_type, strategy_params=None):
    """Test a strategy type directly"""
    try:
        # Get registry
        registry = get_strategy_registry()
        
        # Get strategy class
        strategy_class = registry.get(strategy_type)
        if not strategy_class:
            return None, f"Strategy type '{strategy_type}' not found in registry"
        
        # Create event bus
        event_bus = EventBus()
        
        # Track signals
        signals = []
        def on_signal(event):
            if hasattr(event, 'data'):
                signals.append(event.data)
        
        event_bus.subscribe(EventType.SIGNAL, on_signal)
        
        # Create strategy instance
        strategy = strategy_class(
            strategy_id=f"{strategy_type}_test",
            event_bus=event_bus,
            config=strategy_params or {}
        )
        
        # Initialize strategy
        strategy.initialize()
        
        # Create test data
        df = create_test_data()
        
        # Process each bar
        for _, row in df.iterrows():
            bar_event = Event(
                event_type=EventType.BAR,
                timestamp=row['timestamp'],
                data={
                    'symbol': 'TEST',
                    'timestamp': row['timestamp'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                }
            )
            
            # Process bar
            strategy.on_bar(bar_event)
            
            # Process any pending events
            event_bus.process_pending()
        
        return signals, None
        
    except Exception as e:
        return None, str(e)


def analyze_topology_strategies():
    """Analyze strategies from topology config"""
    config_path = "config/expansive_grid_search.yaml"
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get topology config
    topology_config = config.get('topology', {})
    if 'file' in topology_config:
        topology_path = Path(topology_config['file'])
        if not topology_path.is_absolute():
            topology_path = Path(config_path).parent / topology_path
        
        with open(topology_path, 'r') as f:
            topology_data = yaml.safe_load(f)
            topology_config = topology_data.get('topology', topology_config)
    
    # Extract strategy definitions
    strategy_defs = []
    if 'strategies' in topology_config:
        for strategy_def in topology_config['strategies']:
            if isinstance(strategy_def, dict) and 'type' in strategy_def:
                strategy_defs.append(strategy_def)
    
    return strategy_defs


def main():
    """Main analysis"""
    print("=" * 80)
    print("STRATEGY SIGNAL GENERATION TEST (SIMPLIFIED)")
    print("=" * 80)
    
    # Get strategy definitions from config
    strategy_defs = analyze_topology_strategies()
    
    print(f"\nFound {len(strategy_defs)} strategy definitions in config")
    
    # Get unique strategy types
    strategy_types = {}
    for strategy_def in strategy_defs:
        stype = strategy_def['type']
        if stype not in strategy_types:
            strategy_types[stype] = []
        strategy_types[stype].append(strategy_def)
    
    print(f"Unique strategy types: {len(strategy_types)}")
    
    # Test each strategy type
    working = []
    not_working = []
    errors = {}
    
    print("\nTesting strategies...")
    print("-" * 80)
    
    for i, (strategy_type, defs) in enumerate(sorted(strategy_types.items()), 1):
        print(f"{i:2d}. Testing {strategy_type}...", end='', flush=True)
        
        # Test with first definition's params
        params = defs[0].get('params', {})
        signals, error = test_strategy_directly(strategy_type, params)
        
        if error:
            print(f" ✗ ERROR: {error}")
            not_working.append(strategy_type)
            errors[strategy_type] = error
        elif signals:
            print(f" ✓ Generated {len(signals)} signals")
            working.append(strategy_type)
        else:
            print(f" ✗ No signals generated")
            not_working.append(strategy_type)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(f"Total strategy types: {len(strategy_types)}")
    print(f"Working: {len(working)} ({len(working)/len(strategy_types)*100:.1f}%)")
    print(f"Not working: {len(not_working)} ({len(not_working)/len(strategy_types)*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("WORKING STRATEGIES:")
    print("=" * 80)
    for strategy_type in sorted(working):
        print(f"  ✓ {strategy_type}")
    
    print("\n" + "=" * 80)
    print("NOT WORKING STRATEGIES:")
    print("=" * 80)
    for strategy_type in sorted(not_working):
        error_msg = errors.get(strategy_type, "No signals generated")
        print(f"  ✗ {strategy_type}")
        if error_msg and len(error_msg) < 100:
            print(f"    Error: {error_msg}")
        elif error_msg:
            print(f"    Error: {error_msg[:100]}...")
    
    # Check registry
    print("\n" + "=" * 80)
    print("STRATEGY REGISTRY CHECK:")
    print("=" * 80)
    try:
        registry = get_strategy_registry()
        registered = list(registry.keys())
        print(f"Registered strategies: {len(registered)}")
        
        # Find missing from registry
        missing_from_registry = set(strategy_types.keys()) - set(registered)
        if missing_from_registry:
            print("\nMissing from registry:")
            for missing in sorted(missing_from_registry):
                print(f"  - {missing}")
    except Exception as e:
        print(f"Error accessing registry: {e}")
    
    return working, not_working, errors


if __name__ == "__main__":
    main()