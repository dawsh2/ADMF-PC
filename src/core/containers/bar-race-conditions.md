# Bar Synchronization and Race Condition Prevention

## Overview

When processing data from multiple symbol/timeframe sources, race conditions can occur if bars arrive at different times or speeds. This document describes how to prevent these issues using the Protocol + Composition architecture.

## The Problem

In a multi-symbol trading system, we face several synchronization challenges:

1. **Asynchronous Bar Arrival**: SPY_1m and QQQ_1m bars may arrive at different times
2. **Missing Data**: Some symbols may have gaps while others don't
3. **Feature Consistency**: Features must be calculated on synchronized data
4. **Strategy Timing**: Strategies need complete data before making decisions

Without proper synchronization, strategies could:
- Make decisions on stale data
- Miss correlations between symbols
- Generate inconsistent signals
- Create race conditions in portfolio updates

## The Solution: BarSynchronizer Component

Following the Protocol + Composition pattern, we solve this with a dedicated component:

```python
@dataclass
class BarSynchronizer:
    """Component that synchronizes bars from multiple sources."""
    expected_sources: Set[str]  # e.g., {'SPY_1m', 'QQQ_1m', 'SPY_5m'}
    timeout_ms: int = 100
    require_all: bool = True  # If False, proceed with available data after timeout
    
    def initialize(self, container: 'Container') -> None:
        self.container = container
        self.bar_buffer = {}
        self.current_timestamp = None
        self.last_emit_time = None
        
    def start(self) -> None:
        # Subscribe to BAR events
        self.container.event_bus.subscribe(EventType.BAR, self.on_bar)
        
    def on_bar(self, event: Event) -> None:
        """Collect bars until synchronized."""
        source = f"{event.payload['symbol']}_{event.payload['timeframe']}"
        timestamp = event.payload['timestamp']
        
        # New timestamp? Process previous and reset
        if timestamp != self.current_timestamp and self.bar_buffer:
            self._check_and_emit()
            self.bar_buffer.clear()
            
        self.current_timestamp = timestamp
        self.bar_buffer[source] = event.payload
        
        # Check if we have all required sources
        if self._is_ready():
            self._emit_synchronized_bars()
            self.bar_buffer.clear()
    
    def _is_ready(self) -> bool:
        """Check if we're ready to emit bars."""
        if self.require_all:
            return set(self.bar_buffer.keys()) >= self.expected_sources
        else:
            # Emit if we have any data and timeout has passed
            return (len(self.bar_buffer) > 0 and 
                    self._timeout_exceeded())
    
    def _emit_synchronized_bars(self):
        """Emit synchronized bars event."""
        # Include metadata about synchronization
        self.container.event_bus.publish(Event(
            event_type=EventType.SYNCHRONIZED_BARS,
            payload={
                'bars': self.bar_buffer.copy(),
                'timestamp': self.current_timestamp,
                'sources_present': list(self.bar_buffer.keys()),
                'sources_missing': list(self.expected_sources - set(self.bar_buffer.keys()))
            }
        ))
        self.last_emit_time = time.time()
```

## Integration with Feature Calculation

The BarSynchronizer works seamlessly with other components:

```python
@dataclass
class MultiSymbolFeatureCalculator:
    """Calculate features on synchronized multi-symbol data."""
    
    def start(self) -> None:
        # Subscribe to synchronized bars, not raw bars
        self.container.event_bus.subscribe(EventType.SYNCHRONIZED_BARS, self.on_synchronized_bars)
    
    def on_synchronized_bars(self, event: Event) -> None:
        """Calculate features only on synchronized data."""
        bars = event.payload['bars']
        
        # Calculate single-symbol features
        features = {}
        for source, bar_data in bars.items():
            features[source] = self.calculate_features(bar_data)
        
        # Calculate cross-symbol features (correlations, spreads, etc.)
        if 'SPY_1m' in bars and 'QQQ_1m' in bars:
            features['correlation'] = self.calculate_correlation(
                bars['SPY_1m'], bars['QQQ_1m']
            )
        
        # Emit features event
        self.container.event_bus.publish(Event(
            event_type=EventType.FEATURES,
            payload={
                'features': features,
                'bars': bars,  # Include original bars for strategies
                'timestamp': event.payload['timestamp']
            }
        ))
```

## Container Composition Pattern

Create a feature aggregation container using only composition:

```python
def create_feature_aggregator_pattern():
    """Pattern for synchronized feature aggregation."""
    return {
        'feature_aggregator': {
            'role': ContainerRole.PROCESSOR,
            'components': {
                'synchronizer': {
                    'type': 'BarSynchronizer',
                    'config': {
                        'expected_sources': ['SPY_1m', 'SPY_5m', 'QQQ_1m'],
                        'timeout_ms': 100,
                        'require_all': True
                    }
                },
                'feature_calc': {
                    'type': 'MultiSymbolFeatureCalculator',
                    'config': {
                        'indicators': ['sma', 'rsi', 'correlation']
                    }
                },
                'strategy_runner': {
                    'type': 'StrategyRunner',
                    'config': {
                        'strategies': ['momentum', 'pairs_trading'],
                        'parallel': True
                    }
                }
            }
        }
    }
```

## Benefits of This Approach

1. **No Race Conditions**: Bars are synchronized before processing
2. **Flexible Synchronization**: Can require all sources or proceed with partial data
3. **Clean Separation**: Synchronization logic is isolated in a component
4. **Testable**: Each component can be tested independently
5. **Configurable**: Timeout and requirements are configuration-driven
6. **Follows Architecture**: Pure Protocol + Composition, no inheritance

## Handling Edge Cases

### Missing Data
```python
def _handle_missing_data(self, bars: Dict, missing: List[str]) -> Dict:
    """Handle missing data points."""
    # Option 1: Use last known values
    for source in missing:
        if source in self.last_known_values:
            bars[source] = self.last_known_values[source]
    
    # Option 2: Skip this timestamp
    # Option 3: Interpolate
    return bars
```

### Different Frequencies
When mixing timeframes (1m, 5m, daily), align to the highest frequency:
```python
def _align_timeframes(self, bars: Dict) -> Dict:
    """Align different timeframes to common timestamp."""
    # 5m bars only update every 5 minutes
    # Use most recent 5m bar for intermediate 1m timestamps
    pass
```

## Testing Synchronization

```python
def test_bar_synchronization():
    """Test that bars are properly synchronized."""
    container = Container()
    synchronizer = BarSynchronizer(
        expected_sources={'SPY_1m', 'QQQ_1m'},
        require_all=True
    )
    container.add_component('sync', synchronizer)
    
    # Emit bars out of order
    container.event_bus.publish(create_bar_event('QQQ_1m', timestamp=1000))
    assert len(synchronizer.bar_buffer) == 1
    
    # Complete the set
    container.event_bus.publish(create_bar_event('SPY_1m', timestamp=1000))
    assert len(synchronizer.bar_buffer) == 0  # Buffer cleared after emit
```

## Conclusion

By using a dedicated BarSynchronizer component, we:
- Eliminate race conditions in multi-symbol processing
- Maintain the Protocol + Composition architecture
- Keep synchronization logic isolated and testable
- Enable flexible handling of missing or delayed data

This pattern ensures that features and strategies always operate on consistent, synchronized data across all symbols and timeframes.