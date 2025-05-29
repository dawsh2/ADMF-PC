"""
IndicatorHub - Centralized indicator computation.

The IndicatorHub computes all indicators once per bar and broadcasts
the results via the event bus. This ensures efficient computation
and consistent values across all strategies.
"""

from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import logging

from ...core.events import Event, EventType
from ..protocols import Indicator


logger = logging.getLogger(__name__)


class IndicatorHub:
    """
    Centralized hub for indicator computation.
    
    Features:
    - Computes each indicator only once per bar
    - Broadcasts indicator values via event bus
    - Supports dynamic indicator registration
    - Provides caching for current values
    """
    
    def __init__(self, name: str = "indicator_hub"):
        self.name = name
        self.indicators: Dict[str, Indicator] = {}
        self.current_values: Dict[str, float] = {}
        self.last_update: Optional[datetime] = None
        
        # Track which indicators are ready
        self.ready_indicators: Set[str] = set()
        
        # Capabilities
        self._lifecycle = None
        self._events = None
    
    def register_indicator(self, name: str, indicator: Indicator) -> None:
        """Register an indicator for computation."""
        if name in self.indicators:
            logger.warning(f"Indicator {name} already registered, replacing")
        
        self.indicators[name] = indicator
        logger.info(f"Registered indicator: {name}")
    
    def register_multiple(self, indicators: Dict[str, Indicator]) -> None:
        """Register multiple indicators at once."""
        for name, indicator in indicators.items():
            self.register_indicator(name, indicator)
    
    def setup_subscriptions(self) -> None:
        """Set up event subscriptions."""
        if self._events:
            self._events.subscribe(EventType.BAR, self.on_bar)
    
    def on_bar(self, event: Event) -> None:
        """Process new market bar and compute all indicators."""
        bar_data = event.payload
        
        # Extract price and timestamp
        price = bar_data.get('close', bar_data.get('price'))
        timestamp = bar_data.get('timestamp', datetime.now())
        
        # Skip if already processed this timestamp
        if self.last_update == timestamp:
            return
        
        self.last_update = timestamp
        
        # Compute all indicators
        new_values = {}
        newly_ready = []
        
        for name, indicator in self.indicators.items():
            try:
                value = indicator.calculate(price, timestamp)
                
                if value is not None:
                    new_values[name] = value
                    
                    # Check if indicator just became ready
                    if indicator.ready and name not in self.ready_indicators:
                        self.ready_indicators.add(name)
                        newly_ready.append(name)
                
            except Exception as e:
                logger.error(f"Error computing indicator {name}: {e}")
        
        # Update current values
        self.current_values.update(new_values)
        
        # Log newly ready indicators
        if newly_ready:
            logger.info(f"Indicators now ready: {newly_ready}")
        
        # Broadcast indicator update event
        self._broadcast_indicator_update(bar_data, new_values)
    
    def _broadcast_indicator_update(self, bar_data: Dict[str, Any], 
                                   values: Dict[str, float]) -> None:
        """Broadcast indicator values to all subscribers."""
        if not self._events or not self._events.event_bus:
            return
        
        # Create indicator event
        indicator_data = {
            'timestamp': bar_data.get('timestamp'),
            'symbol': bar_data.get('symbol'),
            'bar_data': bar_data,
            'indicators': values,
            'all_indicators': self.current_values.copy(),
            'ready_indicators': list(self.ready_indicators)
        }
        
        # Publish as INDICATOR_UPDATE event
        event = Event('INDICATOR_UPDATE', indicator_data)
        self._events.event_bus.publish(event)
    
    def get_value(self, indicator_name: str) -> Optional[float]:
        """Get current value of an indicator."""
        return self.current_values.get(indicator_name)
    
    def get_all_values(self) -> Dict[str, float]:
        """Get all current indicator values."""
        return self.current_values.copy()
    
    def is_ready(self, indicator_name: str) -> bool:
        """Check if an indicator is ready (has enough data)."""
        return indicator_name in self.ready_indicators
    
    def get_ready_indicators(self) -> List[str]:
        """Get list of indicators that are ready."""
        return list(self.ready_indicators)
    
    def reset(self) -> None:
        """Reset all indicators and state."""
        self.current_values.clear()
        self.ready_indicators.clear()
        self.last_update = None
        
        # Reset all indicators
        for indicator in self.indicators.values():
            if hasattr(indicator, 'reset'):
                indicator.reset()
        
        logger.info("IndicatorHub reset complete")
    
    def get_indicator_info(self) -> Dict[str, Any]:
        """Get information about registered indicators."""
        return {
            'registered_count': len(self.indicators),
            'ready_count': len(self.ready_indicators),
            'indicators': {
                name: {
                    'ready': name in self.ready_indicators,
                    'current_value': self.current_values.get(name),
                    'type': type(indicator).__name__
                }
                for name, indicator in self.indicators.items()
            }
        }