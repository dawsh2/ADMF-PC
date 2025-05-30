"""
Minimal type hints for ADMF-PC that preserve duck typing.

This module provides the absolute minimum type hints needed for clarity
while maintaining the system's flexibility and composability.

The key insight: Dict[str, Any] is CORRECT for a duck-typing system!
It means "any dict-like object" which is exactly what we want.
"""

from typing import Dict, Any, Protocol, runtime_checkable, Optional

# =============================================================================
# These type aliases are purely for documentation/clarity
# They don't restrict what can be used - Dict[str, Any] accepts anything dict-like
# =============================================================================

# Core types - keep it simple!
Config = Dict[str, Any]      # Any configuration dict
Context = Dict[str, Any]     # Any context dict  
Payload = Dict[str, Any]     # Any event payload
Parameters = Dict[str, Any]  # Any parameters dict
Result = Dict[str, Any]      # Any result dict

# The only protocols we really need are for core behaviors
@runtime_checkable
class Component(Protocol):
    """Minimal component - just needs an ID."""
    @property
    def component_id(self) -> str: ...


@runtime_checkable
class HasLifecycle(Protocol):
    """Component with lifecycle methods."""
    def initialize(self, context: Dict[str, Any]) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...


# That's it! The beauty of duck typing is we don't need more.
# Components can have ANY additional methods/properties they want.
# The framework uses hasattr() to check capabilities at runtime.