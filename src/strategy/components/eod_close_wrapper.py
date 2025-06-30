"""
End-of-Day Position Closing Wrapper

Wraps any strategy to force position closes at end of day.
This prevents overnight holding and weekend gap risk.

Configurable options:
- close_time: Time to close positions (default: 15:50 for 3:50 PM)
- no_entry_after: Time after which no new entries allowed (default: 15:30)
"""

import logging
from typing import Dict, Any, Optional
from datetime import time
import pandas as pd

from ..protocols import StatelessStrategy

logger = logging.getLogger(__name__)


class EODCloseWrapper:
    """
    Wraps a strategy to add end-of-day position closing.
    
    This wrapper:
    1. Blocks new entries after a cutoff time
    2. Forces exit signals at EOD close time
    3. Passes through all signals during normal hours
    """
    
    def __init__(self, 
                 strategy: StatelessStrategy,
                 close_time: time = time(15, 50),  # 3:50 PM
                 no_entry_after: time = time(15, 30),  # 3:30 PM
                 enabled: bool = True):
        """
        Initialize the EOD wrapper.
        
        Args:
            strategy: The underlying strategy to wrap
            close_time: Time to force close positions
            no_entry_after: Time after which no new entries
            enabled: Whether EOD closing is enabled
        """
        self.strategy = strategy
        self.close_time = close_time
        self.no_entry_after = no_entry_after
        self.enabled = enabled
        self._last_signal = None
        
    def generate_signal(self, features: Dict[str, Any], bar: Dict[str, Any], 
                       params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate signal with EOD position management.
        
        Args:
            features: Feature values
            bar: Current bar data
            params: Strategy parameters
            
        Returns:
            Modified signal dictionary
        """
        # Get the underlying strategy signal
        signal = self.strategy.generate_signal(features, bar, params)
        
        if not self.enabled or not signal:
            return signal
            
        # Extract time information
        current_time = self._get_bar_time(bar)
        if not current_time:
            # Can't determine time, pass through original signal
            logger.warning("Cannot determine bar time for EOD check")
            return signal
            
        signal_direction = signal.get('direction', 'flat')
        
        # Check if we're in a position
        in_position = self._last_signal and self._last_signal != 'flat'
        
        # Force close at EOD
        if current_time >= self.close_time and in_position:
            logger.info(f"EOD close triggered at {current_time}")
            signal['direction'] = 'flat'
            signal['metadata'] = signal.get('metadata', {})
            signal['metadata']['eod_close'] = True
            signal['metadata']['original_direction'] = signal_direction
            self._last_signal = 'flat'
            return signal
            
        # Block new entries after cutoff
        if current_time >= self.no_entry_after and signal_direction != 'flat':
            if not in_position:  # Only block new entries, not exits
                logger.debug(f"Entry blocked after {self.no_entry_after}: {signal_direction}")
                signal['direction'] = 'flat'
                signal['metadata'] = signal.get('metadata', {})
                signal['metadata']['entry_blocked'] = True
                signal['metadata']['original_direction'] = signal_direction
                return signal
        
        # Update last signal for position tracking
        self._last_signal = signal_direction
        
        return signal
    
    def _get_bar_time(self, bar: Dict[str, Any]) -> Optional[time]:
        """Extract time from bar data."""
        if 'timestamp' in bar:
            try:
                ts = pd.to_datetime(bar['timestamp'])
                return ts.time()
            except Exception as e:
                logger.error(f"Failed to parse timestamp: {e}")
                return None
                
        # Alternative: use bar_of_day if available
        if 'bar_of_day' in bar:
            # Assuming 5-minute bars starting at 9:30 AM
            bar_num = bar['bar_of_day']
            minutes_since_open = bar_num * 5
            hours = 9 + (30 + minutes_since_open) // 60
            minutes = (30 + minutes_since_open) % 60
            return time(hours, minutes)
            
        return None
    
    @property
    def required_features(self):
        """Pass through required features from wrapped strategy."""
        return self.strategy.required_features


def add_eod_close_filter(filter_expr: str, 
                        close_time_minutes: int = 390,  # 6.5 hours = 390 minutes after 9:30
                        no_entry_minutes: int = 360) -> str:  # 6 hours = 360 minutes after 9:30
    """
    Add EOD close logic to an existing filter expression.
    
    This modifies the filter to:
    1. Force flat signals after close time
    2. Block entries after no-entry time
    
    Args:
        filter_expr: Original filter expression
        close_time_minutes: Minutes after market open to close (390 = 3:30 PM)
        no_entry_minutes: Minutes after market open to stop entries (360 = 3:30 PM)
        
    Returns:
        Modified filter expression
    """
    # Calculate bar numbers (assuming 5-minute bars)
    close_bar = close_time_minutes // 5
    no_entry_bar = no_entry_minutes // 5
    
    # Build the EOD logic
    eod_logic = f"(bar_of_day >= {close_bar} and signal != 0)"  # Force exit
    entry_block = f"(bar_of_day >= {no_entry_bar} and signal != 0 and signal != flat_signal)"  # Block new entries
    
    # Combine with original filter
    if filter_expr and filter_expr.strip():
        # Wrap original in parentheses for safety
        modified = f"({filter_expr}) and not {eod_logic} and not {entry_block}"
    else:
        # No existing filter, just add EOD logic
        modified = f"not {eod_logic} and not {entry_block}"
        
    return modified


# Example usage in configuration
"""
# Option 1: Use as a filter expression
strategy:
  keltner_bands:
    period: 20
    multiplier: 0.5
    filter: |
      signal == 0 or (
        (bar_of_day < 72)  # No signals after 3:30 PM (72 * 5 = 360 minutes)
      )

# Option 2: Use time-based filter
strategy:
  keltner_bands:
    period: 20
    multiplier: 0.5
    filter: |
      signal == 0 or (
        (bar_of_day < 72) and  # No new entries after 3:30 PM
        (bar_of_day < 78 or signal == 0)  # Force exit at 3:50 PM
      )
"""