# Integration Guide for Time Synchronization Component

## Overview
This guide outlines the minimal changes needed to integrate the TimeAlignmentBuffer synchronization component into your existing event-driven backtest architecture.

## Architecture Overview

Your event-driven backtest follows this topology:

```
Root Container
│
├── Symbol_Timeframe Containers (publish BARs)
│
├── Feature Container 
│   ├── Receives all BARs
│   ├── Time synchronization via buffer
│   ├── Calculates features
│   ├── Calls Classifier(s) → gets market regime/state
│   ├── Calls relevant strategies based on classification
│   └── Publishes enriched signals
│
└── Portfolio Containers (subscribe by strategy_id/classifier combo)
```

## Signal Object Structure

The Feature Container publishes enriched signals with this structure:

```python
{
    event_type: "SIGNAL",
    payload: {
        # Identity
        strategy_id: "momentum_fast_v1",
        classifier_id: "trend_classifier",
        classification: "strong_uptrend",
        
        # Signal details
        symbol: "AAPL",
        direction: "BUY",
        strength: 0.85,
        
        # Context data
        bar_data: {
            "AAPL_1m": {
                "open": 150.25,
                "high": 150.75,
                "low": 150.10,
                "close": 150.50,
                "volume": 1000000,
                "timestamp": "2024-01-01 09:35:00"
            },
            "SPY_1m": {...}  # If strategy uses multiple symbols
        },
        features: {
            "SMA_20": 150.5,
            "RSI": 65.2,
            "BB_upper": 151.0,
            "BB_lower": 149.0
        },
        
        # Timing
        timestamp: "2024-01-01 09:35:00",
        bar_close_times: {
            "AAPL_1m": "2024-01-01 09:35:00",
            "SPY_1m": "2024-01-01 09:35:00"
        }
    }
}
```

This rich signal structure allows Portfolio Containers to:
- Filter by strategy_id or classifier_id
- Make decisions with full context (bar data, features, classification)
- Verify timing alignment
- Implement complex portfolio logic without additional data lookups

## Required Modifications

### 1. Events Module - Minor Additions

Add missing event types to your existing `events/types.py`:

```python
# In events/types.py - add to EventType enum
class EventType(Enum):
    # ... existing types ...
    FEATURES = "FEATURES"  # If not already there
    CLASSIFICATION = "CLASSIFICATION"  # If not already there
```

### 2. New Component File

Create a new file for the synchronization component:

**File Location:** `src/core/containers/components/time_synchronization.py`

This file will contain:
- `TimeAlignmentBuffer` class
- `StrategyDataRequirement` dataclass
- Helper setup functions

### 3. Symbol_Timeframe Containers - Enhanced BAR Events

Modify your data streaming containers to publish bars with enhanced format:

```python
# In your DataStreamer or similar component
bar_event = create_market_event(
    event_type=EventType.BAR,
    symbol=self.symbol,
    data={
        'open': bar.open,
        'high': bar.high,
        'low': bar.low,
        'close': bar.close,
        'volume': bar.volume,
        'bar_close_time': bar.timestamp,  # When bar period ended
        'timeframe': self.timeframe,      # '1m', '5m', etc.
        'is_complete': True               # False for live/partial bars
    },
    source_id=self.container_id,
    container_id=self.container.container_id
)
```

### 4. Feature Container Setup

Replace direct BAR subscriptions with TimeAlignmentBuffer:

```python
# In your feature container initialization
from .components.time_synchronization import TimeAlignmentBuffer, StrategyDataRequirement

def setup_feature_container(container, config):
    # Define strategy requirements
    strategy_requirements = []
    for strat_config in config.get('strategies', []):
        req = StrategyDataRequirement(
            strategy_id=strat_config['id'],
            strategy_function=strat_config['function'],
            classifier_id=strat_config.get('classifier_id'),
            required_data=strat_config['required_data'],  # [('SPY', '1m'), ('QQQ', '1m')]
            alignment_mode='wait_for_all'
        )
        strategy_requirements.append(req)
    
    # Create and add the synchronization component
    time_buffer = TimeAlignmentBuffer(strategy_requirements=strategy_requirements)
    container.add_component('time_buffer', time_buffer)
    
    # Add other components (classifier, feature calculator)
    container.add_component('classifier', YourClassifier())
    container.add_component('feature_calculator', YourFeatureCalculator())
```

## What DOESN'T Need Modification

These components work without any changes:

1. **Core Container class** (`container.py`) - No modifications needed
2. **Event Bus** (`events/bus.py`) - Works as-is
3. **Portfolio Containers** - Already compatible with enriched signal structure
4. **Protocol definitions** (`protocols.py`) - No changes needed
5. **Routing infrastructure** - If you're using it

## Integration Checklist

- [ ] Add FEATURES and CLASSIFICATION to EventType enum (if missing)
- [ ] Create `time_synchronization.py` with TimeAlignmentBuffer component
- [ ] Update Symbol_Timeframe containers to publish enhanced BAR events
- [ ] Modify Feature Container setup to use TimeAlignmentBuffer
- [ ] Update strategy configuration to include `required_data` specifications
- [ ] Test with single symbol/timeframe first
- [ ] Test with multiple symbols/timeframes

## Configuration Example

Here's how your topology configuration would look:

```python
config = {
    'symbols_timeframes': [
        {'symbol': 'SPY', 'timeframe': '1m'},
        {'symbol': 'SPY', 'timeframe': '5m'},
        {'symbol': 'QQQ', 'timeframe': '1m'},
        {'symbol': 'NVDA', 'timeframe': '5m'}
    ],
    'strategies': [
        {
            'id': 'momentum_multi',
            'function': momentum_strategy_func,
            'classifier_id': 'trend_classifier',
            'required_data': [('SPY', '1m'), ('QQQ', '1m'), ('NVDA', '5m')]
        },
        {
            'id': 'spy_multi_timeframe',
            'function': multi_tf_strategy_func,
            'required_data': [('SPY', '1m'), ('SPY', '5m')]
        }
    ],
    'portfolios': [
        {
            'id': 'portfolio_1',
            'strategy_assignments': ['momentum_multi', 'spy_multi_timeframe']
        }
    ]
}
```

## Benefits of This Approach

1. **Minimal Changes**: Only 4 touch points in your existing code
2. **Clean Separation**: Synchronization logic is completely contained in one component
3. **Backward Compatible**: Existing single-symbol strategies work without modification
4. **Testable**: Each component can be tested in isolation
5. **Configurable**: Easy to add new strategies with different data requirements

## Portfolio Container Signal Subscription

Portfolio containers can subscribe to signals using multiple strategies:

```python
# In Portfolio Container setup
def setup_portfolio_container(container, config):
    # Get strategy assignments from config
    strategy_assignments = config.get('strategy_assignments', [])
    classifier_filters = config.get('classifier_filters', [])
    
    # Create signal processor with filters
    signal_processor = SignalProcessor(
        portfolio_id=config['portfolio_id'],
        strategy_whitelist=strategy_assignments,
        classifier_whitelist=classifier_filters
    )
    
    # Subscribe to SIGNAL events
    container.event_bus.subscribe(EventType.SIGNAL.value, signal_processor.on_signal)
    
    # Signal processor filters signals
    def on_signal(self, event: Event):
        payload = event.payload
        
        # Filter by strategy_id
        if payload['strategy_id'] not in self.strategy_whitelist:
            return
            
        # Optional: Filter by classifier
        if self.classifier_whitelist:
            if payload.get('classifier_id') not in self.classifier_whitelist:
                return
                
        # Optional: Filter by classification state
        if payload.get('classification') == 'no_trade':
            return
            
        # Process the signal with full context
        self.process_signal_with_context(payload)
```

## Testing Strategy

1. **Start Simple**: Test with single symbol, single timeframe
2. **Add Complexity**: Gradually add multiple timeframes for same symbol
3. **Full Test**: Finally test with multiple symbols and multiple timeframes
4. **Edge Cases**: Test with missing data, delayed bars, etc.
