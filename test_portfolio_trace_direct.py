#!/usr/bin/env python3
"""Direct test of portfolio container receiving and tracing events."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.containers.factory import ContainerFactory
from src.core.events.types import Event, EventType

def test_portfolio_traces_received_events():
    """Test that portfolio properly traces events received via receive_event."""
    
    # Create factory
    factory = ContainerFactory()
    
    # Create portfolio with hierarchical tracing
    portfolio_config = {
        'execution': {
            'enable_event_tracing': True,
            'trace_settings': {
                'storage_backend': 'hierarchical',
                'batch_size': 1,  # Immediate flush
                'max_events': 100
            }
        }
    }
    
    portfolio = factory.create_portfolio_container(
        name='test_portfolio',
        strategies=['test_strategy'],
        config=portfolio_config
    )
    
    # Initialize and start
    portfolio.initialize()
    portfolio.start()
    
    print(f"Portfolio container type: {portfolio.container_type}")
    print(f"Portfolio has tracer: {hasattr(portfolio.event_bus, '_tracer')}")
    
    # Create a test signal
    signal = Event(
        event_type=EventType.SIGNAL,
        payload={
            'symbol': 'SPY',
            'direction': 'long',
            'strength': 0.5,
            'strategy_id': 'test_strategy'
        },
        source_id='test_source',
        container_id='test_strategy_container'
    )
    
    # Send signal to portfolio via receive_event (simulating parent forwarding)
    print(f"\nSending SIGNAL to portfolio via receive_event...")
    portfolio.receive_event(signal)
    
    # Check tracer's storage
    if hasattr(portfolio.event_bus, '_tracer'):
        tracer = portfolio.event_bus._tracer
        print(f"Tracer storage type: {type(tracer.storage)}")
        
        # Check hierarchical storage
        if hasattr(tracer.storage, 'event_buffers'):
            print(f"Event buffers: {list(tracer.storage.event_buffers.keys())}")
            for container_id, events in tracer.storage.event_buffers.items():
                print(f"  {container_id}: {len(events)} events")
                
        # Force flush
        print("\nFlushing storage...")
        if hasattr(tracer.storage, 'flush_all'):
            tracer.storage.flush_all()
        
        # Check files
        if hasattr(tracer.storage, 'base_dir'):
            base_dir = Path(tracer.storage.base_dir)
            print(f"Storage base dir: {base_dir}")
            if base_dir.exists():
                for item in base_dir.rglob("*.jsonl"):
                    print(f"  Found: {item}")
    
    # Clean up
    portfolio.stop()
    portfolio.cleanup()
    
    print("\nTest completed")

if __name__ == '__main__':
    test_portfolio_traces_received_events()