"""
Null strategy for testing and development.

This strategy does nothing - it doesn't generate any signals.
Useful for testing topology lifecycle without actual trading logic.
"""

from typing import Optional, Dict, Any
from ..protocols import StrategyProtocol
from ..types import Signal
from ...data.models import Bar


class NullStrategy:
    """
    A strategy that does nothing.
    
    Used for:
    - Testing topology lifecycle
    - Development without trading logic
    - Placeholder in empty configurations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize null strategy."""
        self.name = config.get('name', 'null_strategy') if config else 'null_strategy'
        self.enabled = config.get('enabled', True) if config else True
        self.config = config or {}
        
    def on_bar(self, bar: Bar) -> Optional[Signal]:
        """Process bar - always returns None (no signal)."""
        return None
        
    def initialize(self):
        """Initialize strategy - no-op."""
        pass
        
    def start(self):
        """Start strategy - no-op."""
        pass
        
    def stop(self):
        """Stop strategy - no-op."""
        pass
        
    def cleanup(self):
        """Clean up strategy - no-op."""
        pass
        
    def get_state(self) -> Dict[str, Any]:
        """Get strategy state."""
        return {
            'name': self.name,
            'type': 'null',
            'enabled': self.enabled,
            'signals_generated': 0
        }