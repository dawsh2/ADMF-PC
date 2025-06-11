#!/usr/bin/env python3
"""
Manual signal test - directly publish SIGNAL events to test tracing
"""

import time
from datetime import datetime
from src.core.events.types import Event, EventType

def test_signal_events():
    """Test script to manually publish signal events"""
    
    # Create a simple signal event
    signal_event = Event(
        event_type=EventType.SIGNAL.value,
        payload={
            'symbol': 'SPY',
            'direction': 'long',
            'strength': 0.8,
            'strategy_id': 'test_momentum',
            'price': 521.50,
            'reason': 'Manual test signal',
            'indicators': {
                'rsi': 35.0,
                'sma': 520.0
            }
        },
        source_id='manual_test',
        container_id='test_container',
        timestamp=datetime.now(),
        metadata={'manual_test': True}
    )
    
    print("Created signal event:")
    print(f"  Type: {signal_event.event_type}")
    print(f"  Symbol: {signal_event.payload['symbol']}")
    print(f"  Direction: {signal_event.payload['direction']}")
    print(f"  Strength: {signal_event.payload['strength']}")
    print(f"  Event ID: {signal_event.event_id}")
    
    return signal_event

if __name__ == '__main__':
    signal = test_signal_events()
    print("\nSignal event ready for testing!")