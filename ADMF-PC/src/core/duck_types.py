"""
Minimal duck-typing friendly type hints for ADMF-PC.

This module provides type aliases that preserve maximum flexibility
while offering IDE support. All types here work with duck typing!

Key principles:
1. Use Dict[str, Any] for maximum flexibility
2. Protocols only for essential behavior contracts  
3. Type aliases for clarity without restriction
4. No enforcement - just documentation
"""

from typing import Dict, Any, Protocol, runtime_checkable, Optional

# =============================================================================
# Type Aliases - These are just names for Dict[str, Any]
# They provide clarity without restricting what objects can be used
# =============================================================================

# Configuration types - ANY dict-like object works
ComponentConfig = Dict[str, Any]
ComponentContext = Dict[str, Any]
ComponentParams = Dict[str, Any]

# Event and data types - flexible structure
EventPayload = Dict[str, Any]
MarketData = Dict[str, Any]
Signal = Dict[str, Any]
Order = Dict[str, Any]
Fill = Dict[str, Any]

# State types - any dict-like representation
PortfolioState = Dict[str, Any]
StrategyState = Dict[str, Any]
ComponentState = Dict[str, Any]

# Results and metrics - flexible output
ValidationResult = Dict[str, Any]
OptimizationResult = Dict[str, Any]
BacktestResult = Dict[str, Any]
ComponentMetrics = Dict[str, Any]

# Parameter spaces - any parameter description
ParameterSpace = Dict[str, Any]
ParameterConfig = Dict[str, Any]


# =============================================================================
# Minimal Protocols - Only define what's absolutely necessary
# These use duck typing - any object with the right methods/properties works!
# =============================================================================

@runtime_checkable
class ComponentLike(Protocol):
    """Minimal component interface - just needs an ID."""
    @property
    def component_id(self) -> str: ...


@runtime_checkable
class DictLike(Protocol):
    """Anything that behaves like a dict."""
    def __getitem__(self, key: str) -> Any: ...
    def get(self, key: str, default: Any = None) -> Any: ...


@runtime_checkable
class ConfigLike(Protocol):
    """Minimal config interface."""
    def get(self, key: str, default: Any = None) -> Any: ...


@runtime_checkable
class EventLike(Protocol):
    """Minimal event interface."""
    @property
    def event_type(self) -> str: ...
    @property  
    def payload(self) -> Any: ...


# =============================================================================
# Usage Examples - Shows how flexible this system is
# =============================================================================

"""
# All of these work with our type system:

# 1. Regular dict
config1: ComponentConfig = {"strategy": "momentum", "lookback": 20}

# 2. Custom config class
class MyConfig:
    def __getitem__(self, key): return getattr(self, key, None)
    def get(self, key, default=None): return getattr(self, key, default)
    
config2: ConfigLike = MyConfig()  # Works!

# 3. SimpleNamespace
from types import SimpleNamespace
config3: ComponentConfig = SimpleNamespace(strategy="momentum", lookback=20).__dict__

# 4. Any component with component_id
class MyStrategy:
    @property
    def component_id(self): return "my_strategy"
    
component: ComponentLike = MyStrategy()  # Works!

# 5. Event from anywhere
event1: EventPayload = {"symbol": "AAPL", "price": 150.0}
event2: EventLike = SimpleNamespace(event_type="BAR", payload={"data": "here"})

# The beauty: Everything works with duck typing!
"""