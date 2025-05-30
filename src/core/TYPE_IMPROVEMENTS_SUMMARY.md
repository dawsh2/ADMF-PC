# Core Module Type Improvements Summary

This document summarizes the type annotation improvements made to reduce `Any` usage in the core module.

## Overview

We've created a comprehensive type system in `src/core/types/` to replace the extensive use of `Dict[str, Any]` throughout the codebase. This improves:

1. **Type Safety**: Catch errors at development time rather than runtime
2. **IDE Support**: Better autocomplete and type hints
3. **Documentation**: Types serve as inline documentation
4. **Maintainability**: Clear contracts between components

## New Type Definitions

### 1. Configuration Types (`types/common.py`)

Instead of `Dict[str, Any]` for configurations, we now have:

```python
# Before
config: Dict[str, Any]

# After
config: DataConfig  # Specific typed dictionary with known fields
config: OptimizationConfig
config: BacktestConfig
```

Key TypedDict definitions:
- `DataConfig`: Data source configuration with symbols, dates, frequency
- `InfrastructureConfig`: Logging, monitoring, metrics settings
- `OptimizationConfig`: Optimizer settings, objectives, constraints
- `BacktestConfig`: Trading simulation parameters

### 2. Event Payloads (`types/common.py`)

Replaced generic event payloads with typed versions:

```python
# Before
def create_market_event(data: Dict[str, Any]) -> Event

# After  
def create_market_event(data: MarketDataPayload) -> Event
```

Key payload types:
- `MarketDataPayload`: OHLCV data with typed fields
- `SignalPayload`: Trading signals with type, strength, metadata
- `OrderPayload`: Order details with all required fields
- `PositionPayload`: Position information with P&L

### 3. Enhanced Protocols (`types/protocols.py`)

Created specific protocols to replace generic interfaces:

```python
# Before
class SignalGenerator(Protocol):
    def generate_signal(self, data: Any) -> Optional[Dict[str, Any]]

# After
class SignalGenerator(Protocol):
    def generate_signal(self, data: MarketDataPayload) -> Optional[SignalPayload]
```

Key protocol improvements:
- `MetricsProvider`: Returns `ComponentMetrics` instead of `Dict[str, Any]`
- `OrderExecutor`: Uses `OrderRequest`/`OrderResponse` types
- `PortfolioManager`: Returns `PortfolioState` with typed positions
- `DataProvider[T]`: Generic protocol for typed data access

### 4. Generic Types

Introduced type variables for better generic support:

```python
T = TypeVar('T')
ConfigT = TypeVar('ConfigT', bound='BaseConfig')
ResultT = TypeVar('ResultT', bound='BaseResult')

# Usage in protocols
class ConfigurableComponent(Protocol[ConfigT]):
    def configure(self, config: ConfigT) -> None
    
class ResultProducer(Protocol[ResultT]):
    def get_result(self) -> ResultT
```

## Implementation Examples

### Updated File Examples

1. **`coordinator/types.py`**:
   - Now imports and uses `DataConfig`, `OptimizationConfig`, etc.
   - Comments added for remaining `Dict[str, Any]` that need specific types

2. **`events/types.py`**:
   - Event factory functions now accept typed payloads
   - `create_market_event()` takes `MarketDataPayload`
   - `create_signal_event()` takes `SignalPayload`

3. **`components/protocols.py`**:
   - Component protocols use specific types for parameters
   - `SignalGenerator` returns typed signals
   - `OrderExecutor` uses typed order structures

4. **`containers/universal.py`**:
   - `ComponentSpec` uses `ComponentParams`
   - `get_stats()` returns `ContainerStats`
   - Registration metadata is typed

### New Example Component

Created `examples/typed_component_example.py` showing best practices:

```python
class TypedMomentumStrategy(Component, Lifecycle, SignalGenerator, Monitorable):
    def __init__(self, params: ComponentParams):
        # Typed parameter extraction
        
    def generate_signal(self, data: MarketDataPayload) -> Optional[SignalPayload]:
        # Returns typed signal
        
    def get_metrics(self) -> ComponentMetrics:
        # Returns typed metrics
        
    def get_state(self) -> StrategyState:
        # Returns typed state
```

## Migration Guide

To update existing components:

1. **Import new types**:
   ```python
   from ..types import (
       MarketDataPayload, SignalPayload, ComponentParams,
       ComponentMetrics, ValidationResult
   )
   ```

2. **Replace generic dictionaries**:
   ```python
   # Before
   def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
   
   # After
   def process_data(self, data: MarketDataPayload) -> SignalPayload:
   ```

3. **Use TypedDict for new structures**:
   ```python
   class MyCustomData(TypedDict):
       field1: str
       field2: float
       optional_field: NotRequired[int]
   ```

## Remaining Work

Areas that still need type improvements:

1. **Health Status**: Currently returns `Dict[str, Any]`, should use `HealthStatus` protocol
2. **Shared Resources**: In `ExecutionContext`, needs a `SharedResources` TypedDict
3. **Phase Data**: In `PhaseResult`, needs specific typed dictionaries per phase
4. **Custom Metrics**: Some components have domain-specific metrics needing types

## Benefits

1. **Compile-time Safety**: MyPy and IDEs can catch type errors early
2. **Better Refactoring**: Type changes propagate automatically
3. **Self-documenting**: Types show expected structure without reading docs
4. **Reduced Bugs**: Eliminates KeyError and type mismatch runtime errors
5. **Enhanced Developer Experience**: Autocomplete works properly

## Best Practices

1. **Use TypedDict for data structures** instead of plain dictionaries
2. **Create domain-specific protocols** for component interfaces  
3. **Use generics** for reusable patterns
4. **Add NotRequired** for optional fields in TypedDict
5. **Document remaining Any usage** with TODO comments
6. **Prefer specific types** over Union types when possible