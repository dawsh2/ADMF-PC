#!/usr/bin/env python3
"""Debug strategy execution in ComponentState."""

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

import pandas as pd
from src.strategy.state import ComponentState
from src.strategy.components.features.hub import FeatureHub
from src.core.events.types import Event, EventType

# Create a mock container for event publishing
class MockContainer:
    def __init__(self):
        self.events = []
        
    def publish_event(self, event: Event):
        self.events.append(event)
        print(f"Event published: {event.event_type} - {event.payload}")
        
    def resolve_component(self, name: str):
        if name == "feature_hub":
            return FeatureHub()
        return None

# Create ComponentState
state = ComponentState(symbols=['SPY'], verbose_signals=True)
state._container = MockContainer()

# Set up feature hub
feature_hub = FeatureHub()
state._feature_hub = feature_hub

# Register a test strategy
test_strategy_id = "bollinger_bands_20_2.0"
test_params = {
    'period': 20,
    'std_dev': 2.0,
    '_strategy_type': 'bollinger_bands'
}

# Import the actual strategy function
from src.strategy.strategies.indicators.volatility import bollinger_bands

# Register component
state._components[test_strategy_id] = {
    'function': bollinger_bands,
    'parameters': test_params,
    'component_type': 'strategy',
    'last_output': None,
    'filter': None
}

print(f"Registered strategy: {test_strategy_id}")
print(f"Components: {list(state._components.keys())}")

# Load some test data
df = pd.read_csv('data/SPY_5m.csv')
print(f"\nLoaded {len(df)} bars")

# Process bars to find signals
signals_found = []

# Create a simple bar object
class SimpleBar:
    def __init__(self, open, high, low, close, volume):
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

for i in range(100):  # Process first 100 bars
    bar_row = df.iloc[i]
    
    # Create bar object
    bar = SimpleBar(
        open=bar_row['open'],
        high=bar_row['high'],
        low=bar_row['low'],
        close=bar_row['close'],
        volume=bar_row['volume']
    )
    
    # Create event payload with bar object
    payload = {
        'symbol': 'SPY',
        'timeframe': '5m',
        'timestamp': bar_row['timestamp'],
        'bar': bar
    }
    
    # Create BAR event
    event = Event(
        event_type=EventType.BAR.value,
        payload=payload,
        source_id="test",
        timestamp=bar_row['timestamp']
    )
    
    print(f"\nBar {i}: close={bar.close:.2f}")
    
    # Process the bar
    state.on_bar(event)
    
    # Check if any signals were generated
    if state._container.events:
        for evt in state._container.events:
            if evt.event_type == EventType.SIGNAL.value:
                signals_found.append((i, evt.payload))
                print(f"  -> SIGNAL GENERATED: {evt.payload}")
        state._container.events.clear()
    
    # Also check feature hub state
    if i >= 20:  # After warmup
        features = feature_hub.get_features('SPY')
        if features:
            bb_features = {k: v for k, v in features.items() if 'bollinger' in k}
            if bb_features:
                print(f"  Features: {bb_features}")

print(f"\n=== Summary ===")
print(f"Total signals found: {len(signals_found)}")
for bar_num, signal in signals_found:
    print(f"  Bar {bar_num}: {signal}")