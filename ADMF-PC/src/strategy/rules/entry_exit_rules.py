"""
Entry and exit rules for positions.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging

from ..protocols import Rule


logger = logging.getLogger(__name__)


class EntryRule(ABC):
    """Base class for entry rules."""
    
    def __init__(self, name: str):
        self.name = name
        self.entries_triggered = 0
    
    @abstractmethod
    def should_enter(self, signal: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Determine if position should be entered."""
        pass
    
    def reset(self) -> None:
        """Reset rule state."""
        self.entries_triggered = 0


class ExitRule(ABC):
    """Base class for exit rules."""
    
    def __init__(self, name: str):
        self.name = name
        self.exits_triggered = 0
    
    @abstractmethod
    def should_exit(self, position: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Determine if position should be exited. Returns exit info if yes."""
        pass
    
    def reset(self) -> None:
        """Reset rule state."""
        self.exits_triggered = 0


class StopLossRule(ExitRule):
    """
    Exit when price moves against position by specified amount.
    """
    
    def __init__(self, 
                 name: str = "stop_loss",
                 stop_type: str = "fixed",  # fixed, percent, atr
                 stop_distance: float = 0.02,  # 2% default
                 use_high_low: bool = False):
        super().__init__(name)
        self.stop_type = stop_type
        self.stop_distance = stop_distance
        self.use_high_low = use_high_low
    
    def should_exit(self, position: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if stop loss is hit."""
        entry_price = position.get('entry_price', 0)
        current_price = context.get('close', context.get('price', 0))
        position_side = position.get('side', 'long')
        
        if entry_price <= 0:
            return None
        
        # Use high/low for more accurate stop checking
        if self.use_high_low:
            if position_side == 'long':
                current_price = context.get('low', current_price)
            else:
                current_price = context.get('high', current_price)
        
        # Calculate stop level
        if self.stop_type == "fixed":
            stop_distance = self.stop_distance
        elif self.stop_type == "percent":
            stop_distance = entry_price * self.stop_distance
        elif self.stop_type == "atr":
            atr = context.get('atr', entry_price * 0.02)
            stop_distance = atr * self.stop_distance
        else:
            stop_distance = self.stop_distance
        
        # Check stop condition
        if position_side == 'long':
            stop_level = entry_price - stop_distance
            if current_price <= stop_level:
                self.exits_triggered += 1
                return {
                    'reason': 'stop_loss',
                    'exit_price': stop_level,
                    'loss': (stop_level - entry_price) / entry_price
                }
        else:  # short
            stop_level = entry_price + stop_distance
            if current_price >= stop_level:
                self.exits_triggered += 1
                return {
                    'reason': 'stop_loss',
                    'exit_price': stop_level,
                    'loss': (entry_price - stop_level) / entry_price
                }
        
        return None


class TakeProfitRule(ExitRule):
    """
    Exit when price reaches profit target.
    """
    
    def __init__(self,
                 name: str = "take_profit",
                 target_type: str = "fixed",  # fixed, percent, atr, risk_reward
                 target_distance: float = 0.04,  # 4% default
                 risk_reward_ratio: float = 2.0):
        super().__init__(name)
        self.target_type = target_type
        self.target_distance = target_distance
        self.risk_reward_ratio = risk_reward_ratio
    
    def should_exit(self, position: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if profit target is hit."""
        entry_price = position.get('entry_price', 0)
        current_price = context.get('close', context.get('price', 0))
        position_side = position.get('side', 'long')
        
        if entry_price <= 0:
            return None
        
        # Calculate target level
        if self.target_type == "fixed":
            target_distance = self.target_distance
        elif self.target_type == "percent":
            target_distance = entry_price * self.target_distance
        elif self.target_type == "atr":
            atr = context.get('atr', entry_price * 0.02)
            target_distance = atr * self.target_distance
        elif self.target_type == "risk_reward":
            # Use stop loss distance * risk reward ratio
            stop_distance = position.get('stop_distance', entry_price * 0.02)
            target_distance = stop_distance * self.risk_reward_ratio
        else:
            target_distance = self.target_distance
        
        # Check target condition
        if position_side == 'long':
            target_level = entry_price + target_distance
            if current_price >= target_level:
                self.exits_triggered += 1
                return {
                    'reason': 'take_profit',
                    'exit_price': target_level,
                    'profit': (target_level - entry_price) / entry_price
                }
        else:  # short
            target_level = entry_price - target_distance
            if current_price <= target_level:
                self.exits_triggered += 1
                return {
                    'reason': 'take_profit',
                    'exit_price': target_level,
                    'profit': (entry_price - target_level) / entry_price
                }
        
        return None


class TrailingStopRule(ExitRule):
    """
    Exit with trailing stop that follows favorable price movement.
    """
    
    def __init__(self,
                 name: str = "trailing_stop",
                 trail_type: str = "percent",  # fixed, percent, atr
                 trail_distance: float = 0.02,  # 2% default
                 activation_profit: float = 0.01):  # Activate after 1% profit
        super().__init__(name)
        self.trail_type = trail_type
        self.trail_distance = trail_distance
        self.activation_profit = activation_profit
        
        # State tracking
        self.highest_price: Dict[str, float] = {}  # position_id -> highest
        self.lowest_price: Dict[str, float] = {}   # position_id -> lowest
        self.is_activated: Dict[str, bool] = {}    # position_id -> activated
    
    def should_exit(self, position: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if trailing stop is hit."""
        position_id = position.get('id', 'default')
        entry_price = position.get('entry_price', 0)
        current_price = context.get('close', context.get('price', 0))
        position_side = position.get('side', 'long')
        
        if entry_price <= 0:
            return None
        
        # Initialize tracking for new position
        if position_id not in self.highest_price:
            self.highest_price[position_id] = current_price
            self.lowest_price[position_id] = current_price
            self.is_activated[position_id] = False
        
        # Update extremes
        if position_side == 'long':
            self.highest_price[position_id] = max(self.highest_price[position_id], current_price)
            extreme_price = self.highest_price[position_id]
            
            # Check activation
            if not self.is_activated[position_id]:
                profit = (extreme_price - entry_price) / entry_price
                if profit >= self.activation_profit:
                    self.is_activated[position_id] = True
        else:  # short
            self.lowest_price[position_id] = min(self.lowest_price[position_id], current_price)
            extreme_price = self.lowest_price[position_id]
            
            # Check activation
            if not self.is_activated[position_id]:
                profit = (entry_price - extreme_price) / entry_price
                if profit >= self.activation_profit:
                    self.is_activated[position_id] = True
        
        # Only trail if activated
        if not self.is_activated[position_id]:
            return None
        
        # Calculate trail distance
        if self.trail_type == "fixed":
            trail_distance = self.trail_distance
        elif self.trail_type == "percent":
            trail_distance = extreme_price * self.trail_distance
        elif self.trail_type == "atr":
            atr = context.get('atr', extreme_price * 0.02)
            trail_distance = atr * self.trail_distance
        else:
            trail_distance = self.trail_distance
        
        # Check trail condition
        if position_side == 'long':
            trail_level = extreme_price - trail_distance
            if current_price <= trail_level:
                self.exits_triggered += 1
                self._cleanup_position(position_id)
                return {
                    'reason': 'trailing_stop',
                    'exit_price': trail_level,
                    'profit': (trail_level - entry_price) / entry_price,
                    'max_profit': (extreme_price - entry_price) / entry_price
                }
        else:  # short
            trail_level = extreme_price + trail_distance
            if current_price >= trail_level:
                self.exits_triggered += 1
                self._cleanup_position(position_id)
                return {
                    'reason': 'trailing_stop',
                    'exit_price': trail_level,
                    'profit': (entry_price - trail_level) / entry_price,
                    'max_profit': (entry_price - extreme_price) / entry_price
                }
        
        return None
    
    def _cleanup_position(self, position_id: str) -> None:
        """Clean up tracking for closed position."""
        self.highest_price.pop(position_id, None)
        self.lowest_price.pop(position_id, None)
        self.is_activated.pop(position_id, None)
    
    def reset(self) -> None:
        """Reset rule state."""
        super().reset()
        self.highest_price.clear()
        self.lowest_price.clear()
        self.is_activated.clear()


class TimeBasedExitRule(ExitRule):
    """
    Exit after specified time period.
    """
    
    def __init__(self,
                 name: str = "time_exit",
                 max_duration: timedelta = timedelta(days=5),
                 exit_time: Optional[str] = None):  # e.g., "15:30" for EOD exit
        super().__init__(name)
        self.max_duration = max_duration
        self.exit_time = exit_time
    
    def should_exit(self, position: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if time-based exit conditions are met."""
        entry_time = position.get('entry_time')
        current_time = context.get('timestamp', datetime.now())
        
        if not entry_time:
            return None
        
        # Check max duration
        if current_time - entry_time >= self.max_duration:
            self.exits_triggered += 1
            return {
                'reason': 'max_duration',
                'exit_price': context.get('close', context.get('price')),
                'duration': str(current_time - entry_time)
            }
        
        # Check specific exit time (e.g., end of day)
        if self.exit_time:
            exit_hour, exit_minute = map(int, self.exit_time.split(':'))
            if (current_time.hour == exit_hour and 
                current_time.minute >= exit_minute):
                self.exits_triggered += 1
                return {
                    'reason': 'scheduled_exit',
                    'exit_price': context.get('close', context.get('price')),
                    'exit_time': self.exit_time
                }
        
        return None