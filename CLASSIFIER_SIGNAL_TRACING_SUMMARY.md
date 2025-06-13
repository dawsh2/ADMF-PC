# Classifier Signal Tracing Implementation Summary

## Overview
Extended the SparsePortfolioTracer to capture both strategy and classifier signals, enabling sparse storage of regime classification changes alongside traditional trading signals.

## Changes Made

### 1. Container Configuration (`src/core/containers/container.py`)

#### Portfolio Container Setup
- Added `managed_classifiers` parameter to portfolio container configuration
- Modified `_setup_portfolio_tracing()` to pass both strategies and classifiers to the tracer

```python
# Get managed classifiers from config (if any)
managed_classifiers = self.config.config.get('managed_classifiers', [])

self._portfolio_tracer = PortfolioTracer(
    container_id=self.container_id,
    workflow_id=workflow_id,
    managed_strategies=managed_strategies,
    managed_classifiers=managed_classifiers,  # New parameter
    storage_config=storage_config,
    portfolio_container=self
)
```

#### Strategy Container Setup
- Extended `_setup_strategy_signal_tracing()` to create tracers for both strategies and classifiers
- Retrieves classifier IDs from ComponentState using new `get_classifier_ids()` method
- Creates separate tracers for each classifier with `classifier_` prefix

```python
# Get all classifier IDs from the strategy state (ComponentState)
classifier_ids = strategy_state.get_classifier_ids() if hasattr(strategy_state, 'get_classifier_ids') else []

# Create tracers for classifiers
for classifier_id in classifier_ids:
    tracer = SparsePortfolioTracer(
        container_id=f"classifier_{classifier_id}",
        workflow_id=workflow_id,
        managed_strategies=[],
        managed_classifiers=[classifier_id],
        storage_config=storage_config,
        portfolio_container=self
    )
```

### 2. SparsePortfolioTracer (`src/core/events/observers/sparse_portfolio_tracer.py`)

#### Constructor Changes
- Added `managed_classifiers` parameter with default empty list
- Stores both managed strategies and classifiers as sets

```python
def __init__(self, 
             container_id: str,
             workflow_id: str,
             managed_strategies: List[str],
             managed_classifiers: Optional[List[str]] = None,  # New parameter
             storage_config: Optional[Dict[str, Any]] = None,
             portfolio_container: Optional[Any] = None):
```

#### Signal Processing
- Extended `on_event()` to check both strategies and classifiers when determining if a signal should be traced
- Handles categorical classifier outputs (regime names) alongside numeric strategy signals

```python
# Check if this signal is from a managed strategy or classifier
is_managed = False

# Check strategies
for strategy_name in self.managed_strategies:
    if strategy_name in strategy_id:
        is_managed = True
        break

# Check classifiers if not found in strategies
if not is_managed:
    for classifier_name in self.managed_classifiers:
        if classifier_name in strategy_id:
            is_managed = True
            break
```

#### Parameter Storage
- Extended `flush()` method to extract and store classifier parameters alongside strategy parameters
- Classifier metadata is prefixed with "classifier_" for clear identification

### 3. Temporal Sparse Storage (`src/core/events/storage/temporal_sparse_storage.py`)

#### Signal Value Handling
- Modified `SignalChange` dataclass to accept `Any` type for signal_value (not just int)
- Updated `process_signal()` to handle categorical values for classifiers

```python
# Convert direction to value for standard strategy signals
# Keep categorical values for classifier signals
if direction == 'long':
    signal_value = 1
elif direction == 'short':
    signal_value = -1
elif direction == 'flat':
    signal_value = 0
else:
    # Categorical value (e.g., regime classification)
    signal_value = direction
```

#### Statistics Calculation
- Enhanced statistics to track both numeric positions and categorical regimes
- Added `regime_breakdown` to complement `position_breakdown`

```python
if isinstance(change.signal_value, int):
    # Numeric signals (strategies)
    # ... existing position tracking ...
else:
    # Categorical signals (classifiers)
    regime = str(change.signal_value)
    if 'regime_breakdown' not in stats:
        stats['regime_breakdown'] = {}
    stats['regime_breakdown'][regime] = stats['regime_breakdown'].get(regime, 0) + 1
```

### 4. ComponentState (`src/strategy/state.py`)

#### New Method
- Added `get_classifier_ids()` method to retrieve all classifier component IDs

```python
def get_classifier_ids(self) -> List[str]:
    """Get all classifier component IDs."""
    return [comp_id for comp_id, comp_info in self._components.items() 
            if comp_info['component_type'] == 'classifier']
```

## Usage Example

Configure a portfolio container with both strategies and classifiers:

```python
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
                'storage_backend': 'hierarchical'
            }
        }
    }
)
```

## Output Format

The sparse storage JSON now includes both types of signals:

```json
{
  "changes": [
    {
      "val": 1,  // Numeric strategy signal (long)
      "strat": "AAPL_momentum_strategy"
    },
    {
      "val": "trending_up",  // Categorical classifier signal
      "strat": "AAPL_trend_classifier"
    }
  ],
  "signal_statistics": {
    "position_breakdown": {
      "long": 1, "short": 1, "flat": 1
    },
    "regime_breakdown": {
      "trending_up": 1,
      "high_volatility": 1,
      "low_volatility": 1
    }
  }
}
```

## Benefits

1. **Unified Tracing**: Both strategies and classifiers use the same sparse storage infrastructure
2. **Efficiency**: Only regime changes are stored, not repeated classifications
3. **Flexibility**: Supports any categorical regime values without predefined mappings
4. **Backward Compatible**: Existing strategy-only configurations continue to work unchanged
5. **Clear Separation**: Classifier data is clearly distinguished in statistics and metadata

## Testing

The implementation was verified with `test_classifier_signal_tracing.py`, which demonstrates:
- Publishing both strategy and classifier signals
- Sparse storage compression (only changes stored)
- Proper categorization in output statistics
- Correct file generation with both signal types