#!/usr/bin/env python3
"""Test that portfolio containers properly trace received SIGNAL events."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.containers.factory import ContainerFactory
from src.core.events.types import Event, EventType
from src.portfolio.state import PortfolioState

def test_portfolio_receive_event_tracing():
    """Test portfolio container traces received signals."""
    
    # Create factory
    factory = ContainerFactory()
    
    # Create portfolio with tracing enabled
    portfolio_config = {
        'execution': {
            'enable_event_tracing': True,
            'trace_settings': {
                'storage_backend': 'memory',
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
    
    # Send signal to portfolio via receive_event
    print(f"Sending signal to portfolio...")
    print(f"Portfolio has tracer: {hasattr(portfolio.event_bus, '_tracer')}")
    print(f"Portfolio type: {portfolio.container_type}")
    
    portfolio.receive_event(signal)
    
    # Check if event was traced
    if hasattr(portfolio.event_bus, '_tracer') and portfolio.event_bus._tracer:
        print(f"Tracer type: {type(portfolio.event_bus._tracer)}")
        print(f"Tracer has events: {hasattr(portfolio.event_bus._tracer, '_events')}")
        if hasattr(portfolio.event_bus._tracer, '_events'):
            print(f"Number of traced events: {len(portfolio.event_bus._tracer._events)}")
    
    # Clean up
    portfolio.stop()
    portfolio.cleanup()
    
    print("Test completed")

if __name__ == '__main__':
    test_portfolio_receive_event_tracing()