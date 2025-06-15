#\!/usr/bin/env python3
"""Trace which strategies were in the 18 ready ones."""

import sys
sys.path.insert(0, '/Users/daws/ADMF-PC')

# Let's test strategies that should have matching feature names
# These are likely the 18 that became ready

# DEMA strategies should work - features match
test_cases = [
    ('dema_crossover', {'fast_dema_period': 3, 'slow_dema_period': 15}, {'dema_3': 100.0, 'dema_15': 99.0}),
    ('dema_crossover', {'fast_dema_period': 7, 'slow_dema_period': 23}, {'dema_7': 100.0, 'dema_23': 99.0}),
]

# Import the strategy
from src.strategy.strategies.indicators.crossovers import dema_crossover

bar = {
    'timestamp': '2023-01-01T09:30:00',
    'symbol': 'SPY',
    'timeframe': '1m',
    'open': 100.0,
    'high': 100.5,
    'low': 99.5,
    'close': 100.2,
    'volume': 1000000
}

print("Testing strategies that should have been ready:\n")

for strategy_name, params, features in test_cases:
    print(f"Testing {strategy_name} with params {params}")
    result = dema_crossover(features, bar, params)
    print(f"Result: {result}")
    if result:
        print(f"Signal value: {result['signal_value']}")
    print()

# Now let's check if the signal would be published
print("\nChecking signal event structure:")
if result:
    # This is what should be published as an event
    from src.core.events.event import Event
    from src.core.events.event_types import EventType
    
    signal_event = Event(
        event_type=EventType.SIGNAL,
        data=result
    )
    print(f"Event that should be published: {signal_event}")
