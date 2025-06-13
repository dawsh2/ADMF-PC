#!/usr/bin/env python3
"""
Test script to verify that SparsePortfolioTracer captures both strategy and classifier signals.
"""

import logging
from datetime import datetime
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import required components
from src.core.containers.container import Container, ContainerConfig
from src.core.events.types import Event, EventType
from src.strategy.types import SignalType

def test_classifier_signal_tracing():
    """Test that both strategy and classifier signals are traced."""
    
    print("\n=== Testing Classifier Signal Tracing ===\n")
    
    # Create container configuration with sparse tracing enabled
    config = ContainerConfig(
        name="test_portfolio",
        container_type="portfolio",
        config={
            'managed_strategies': ['momentum_strategy', 'mean_reversion_strategy'],
            'managed_classifiers': ['trend_classifier', 'volatility_classifier'],
            'execution': {
                'enable_event_tracing': True,
                'trace_settings': {
                    'use_sparse_storage': True,
                    'storage_backend': 'hierarchical',
                    'container_settings': {
                        '*': {'enabled': True}
                    }
                }
            },
            'metadata': {
                'workflow_id': 'test_classifier_trace'
            }
        }
    )
    
    # Create portfolio container
    container = Container(config)
    container.initialize()
    container.start()
    
    print("1. Portfolio container created with sparse tracing")
    print(f"   - Managing strategies: {config.config['managed_strategies']}")
    print(f"   - Managing classifiers: {config.config['managed_classifiers']}")
    
    # Simulate some signals
    signals = [
        # Strategy signals
        {
            'timestamp': datetime.now(),
            'strategy_id': 'AAPL_momentum_strategy',
            'symbol': 'AAPL',
            'direction': 'long',
            'signal_type': 'entry',
            'metadata': {'component_type': 'strategy'}
        },
        {
            'timestamp': datetime.now(),
            'strategy_id': 'AAPL_mean_reversion_strategy',
            'symbol': 'AAPL',
            'direction': 'short',
            'signal_type': 'entry',
            'metadata': {'component_type': 'strategy'}
        },
        # Classifier signals (regime changes)
        {
            'timestamp': datetime.now(),
            'strategy_id': 'AAPL_trend_classifier',
            'symbol': 'AAPL',
            'direction': 'trending_up',  # Categorical value
            'signal_type': SignalType.CLASSIFICATION.value,
            'metadata': {
                'component_type': 'classifier',
                'regime': 'trending_up',
                'confidence': 0.85,
                'classification': 'trending_up',
                'regime_change': True
            }
        },
        {
            'timestamp': datetime.now(),
            'strategy_id': 'AAPL_volatility_classifier',
            'symbol': 'AAPL',
            'direction': 'high_volatility',  # Categorical value
            'signal_type': SignalType.CLASSIFICATION.value,
            'metadata': {
                'component_type': 'classifier',
                'regime': 'high_volatility',
                'confidence': 0.92,
                'classification': 'high_volatility',
                'regime_change': True
            }
        },
        # Some repeated signals (should not be stored due to sparse storage)
        {
            'timestamp': datetime.now(),
            'strategy_id': 'AAPL_momentum_strategy',
            'symbol': 'AAPL',
            'direction': 'long',  # Same as before
            'signal_type': 'entry',
            'metadata': {'component_type': 'strategy'}
        },
        {
            'timestamp': datetime.now(),
            'strategy_id': 'AAPL_trend_classifier',
            'symbol': 'AAPL',
            'direction': 'trending_up',  # Same as before
            'signal_type': SignalType.CLASSIFICATION.value,
            'metadata': {
                'component_type': 'classifier',
                'regime': 'trending_up',
                'confidence': 0.87,  # Slightly different confidence but same regime
                'classification': 'trending_up'
            }
        },
        # Change signals
        {
            'timestamp': datetime.now(),
            'strategy_id': 'AAPL_momentum_strategy',
            'symbol': 'AAPL',
            'direction': 'flat',  # Changed
            'signal_type': 'exit',
            'metadata': {'component_type': 'strategy'}
        },
        {
            'timestamp': datetime.now(),
            'strategy_id': 'AAPL_volatility_classifier',
            'symbol': 'AAPL',
            'direction': 'low_volatility',  # Changed
            'signal_type': SignalType.CLASSIFICATION.value,
            'metadata': {
                'component_type': 'classifier',
                'regime': 'low_volatility',
                'confidence': 0.78,
                'classification': 'low_volatility',
                'regime_change': True,
                'previous_regime': 'high_volatility'
            }
        }
    ]
    
    print("\n2. Publishing test signals...")
    for i, signal_data in enumerate(signals):
        event = Event(
            event_type=EventType.SIGNAL.value,
            timestamp=signal_data['timestamp'],
            payload=signal_data,
            source_id='test_source'
        )
        container.event_bus.publish(event)
        print(f"   - Published signal {i+1}: {signal_data['strategy_id']} -> {signal_data['direction']}")
    
    # Get tracer statistics
    if hasattr(container, '_portfolio_tracer'):
        stats = container._portfolio_tracer.get_statistics()
        print("\n3. Sparse storage statistics:")
        print(f"   - Total signals seen: {stats['total_signals_seen']}")
        print(f"   - Signal changes stored: {stats['signal_changes_stored']}")
        print(f"   - Compression ratio: {stats['compression_ratio']:.1f}x")
        print(f"   - Managed strategies: {stats['managed_strategies']}")
        print(f"   - Managed classifiers: {stats['managed_classifiers']}")
        
        # Flush to disk
        print("\n4. Flushing sparse storage to disk...")
        container._portfolio_tracer.flush()
        
        # Check output files
        base_dir = Path('./workspaces/test_classifier_trace')
        if base_dir.exists():
            print("\n5. Output files created:")
            for file in base_dir.rglob('*.json'):
                print(f"   - {file}")
                
                # Read and display content
                with open(file, 'r') as f:
                    data = json.load(f)
                    if 'signal_changes' in data:
                        print(f"     Signal changes: {len(data['signal_changes'])}")
                        for change in data['signal_changes'][:3]:  # Show first 3
                            print(f"     - Bar {change['idx']}: {change['strat']} -> {change['val']}")
                    if 'signal_statistics' in data:
                        stats = data['signal_statistics']
                        if 'position_breakdown' in stats:
                            print(f"     Position breakdown: {stats['position_breakdown']}")
                        if 'regime_breakdown' in stats:
                            print(f"     Regime breakdown: {stats['regime_breakdown']}")
    
    # Cleanup
    container.stop()
    container.cleanup()
    
    print("\n=== Test Complete ===\n")

if __name__ == "__main__":
    test_classifier_signal_tracing()