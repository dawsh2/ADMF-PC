#!/usr/bin/env python3
"""
Example demonstrating signal storage for replay with full metadata.

This shows how SignalObserver stores comprehensive signal data including
bar context, features, and classifier states for perfect signal replay.
"""

from src.core.events.tracing.observers import SignalObserver
from src.core.events.types import Event, EventType
from datetime import datetime
import uuid
import json

def example_signal_storage_for_replay():
    """Demonstrate comprehensive signal storage with full metadata."""
    
    # Create observer focused on storage and replay
    observer = SignalObserver(
        max_signals_total=1000,
        max_signals_per_strategy=500,
        retention_policy="all"  # Keep all signals for replay
    )
    
    print("=== Signal Storage for Replay Example ===\n")
    
    # Simulate a comprehensive signal event (as per storage doc)
    correlation_id = f"trade_{uuid.uuid4().hex[:8]}"
    signal_event = Event(
        event_type=EventType.SIGNAL.value,
        payload={
            'strategy_id': 'momentum_1',
            'strategy_name': 'Momentum Strategy',
            'signal_value': 1.0,
            'symbol': 'AAPL',
            'timeframe': '1m',
            'direction': 'long',
            'confidence': 0.85,
            'entry_price': 150.0,
            
            # Full context for replay (as specified in storage doc)
            'bars': {
                'AAPL': {
                    'open': 149.5, 'high': 150.2, 'low': 149.0, 'close': 150.0,
                    'volume': 1000000, 'timestamp': '2024-01-15T10:30:00'
                }
            },
            'features': {
                'sma_20': 148.5,
                'rsi': 65.2,
                'momentum': 0.75,
                'volume_ratio': 1.2
            },
            'classifier_states': {
                'trend': 'TRENDING',
                'volatility': 'LOW',
                'regime': 'BULL'
            },
            'parameters': {
                'fast_period': 10,
                'slow_period': 20,
                'threshold': 0.02
            }
        },
        source_id='momentum_strategy',
        container_id='feature_container_1',
        correlation_id=correlation_id
    )
    
    print("1. Storing signal with full metadata for replay")
    observer.on_publish(signal_event)
    
    print("   Signal stored with comprehensive context:")
    stored_signals = observer.get_signals_for_replay('momentum_1')
    signal = stored_signals[0]
    
    print(f"   - Strategy: {signal['strategy_name']}")
    print(f"   - Signal value: {signal['signal_value']}")
    print(f"   - Bar data: {bool(signal['bar_data'])}")
    print(f"   - Features: {list(signal['features'].keys())}")
    print(f"   - Classifier states: {list(signal['classifier_states'].keys())}")
    print(f"   - Full payload preserved: {bool(signal['full_payload'])}")
    
    # Add a few more signals to demonstrate storage
    for i in range(3):
        correlation_id = f"trade_{uuid.uuid4().hex[:8]}"
        additional_signal = Event(
            event_type=EventType.SIGNAL.value,
            payload={
                'strategy_id': 'momentum_1',
                'strategy_name': 'Momentum Strategy',
                'signal_value': 0.5 + i * 0.2,
                'symbol': 'AAPL',
                'bars': {'AAPL': {'close': 150.0 + i}},
                'features': {'sma_20': 148.5 + i},
                'classifier_states': {'trend': 'TRENDING'}
            },
            source_id='momentum_strategy',
            container_id='feature_container_1',
            correlation_id=correlation_id
        )
        observer.on_publish(additional_signal)
    
    print(f"\n2. Storage statistics:")
    summary = observer.get_summary()
    storage_stats = summary['signal_storage']
    print(f"   - Total signals stored: {storage_stats['total_signals_stored']}")
    print(f"   - Storage utilization: {storage_stats['storage_utilization']['percentage']}")
    print(f"   - Signals by strategy: {storage_stats['signals_by_strategy']}")
    
    print(f"\n3. Export signals for replay:")
    export_path = "/tmp/momentum_signals_for_replay.json"
    observer.export_signals_for_replay(export_path, 'momentum_1')
    
    # Show exported format
    with open(export_path) as f:
        exported = json.load(f)
    
    print(f"   - Exported {exported['metadata']['signal_count']} signals")
    print(f"   - Export includes full metadata for perfect replay")
    print(f"   - First signal contains: {list(exported['signals'][0].keys())}")
    
    print(f"\n4. Demonstrate replay data access:")
    replay_signals = observer.get_signals_for_replay('momentum_1')
    print(f"   - Retrieved {len(replay_signals)} signals for replay")
    
    # Show how signal can be perfectly reconstructed
    first_signal = replay_signals[0]
    print(f"   - Signal reconstruction possible:")
    print(f"     * Bar data: {first_signal['bar_data']['AAPL']['close']}")
    print(f"     * Features: {first_signal['features']['sma_20']}")
    print(f"     * Classifier state: {first_signal['classifier_states']['trend']}")
    print(f"     * Strategy params: {first_signal['strategy_parameters']}")
    
    print("\n✅ Signal storage for replay working correctly!")
    print("   All signals preserved with full context for perfect reconstruction")

def example_storage_limits():
    """Show storage limit behavior."""
    
    print("\n=== Storage Limits Example ===\n")
    
    # Create observer with small limits for demonstration
    observer = SignalObserver(
        max_signals_total=3,
        max_signals_per_strategy=2
    )
    
    print("Creating observer with small limits:")
    print(f"- max_signals_total: {observer.max_signals_total}")
    print(f"- max_signals_per_strategy: {observer.max_signals_per_strategy}")
    
    # Add signals beyond limits
    for i in range(5):
        signal_event = Event(
            event_type=EventType.SIGNAL.value,
            payload={
                'strategy_id': 'test_strategy',
                'signal_value': i,
                'symbol': 'TEST'
            },
            source_id='test',
            correlation_id=f"test_{i}"
        )
        observer.on_publish(signal_event)
        
        count = observer.get_signal_count()
        print(f"After signal {i}: {count['total_signals']} total signals stored")
    
    print(f"\nFinal storage state:")
    summary = observer.get_summary()
    print(f"- Storage utilization: {summary['signal_storage']['storage_utilization']['percentage']}")
    print("- Oldest signals automatically evicted to maintain limits")

if __name__ == "__main__":
    example_signal_storage_for_replay()
    example_storage_limits()