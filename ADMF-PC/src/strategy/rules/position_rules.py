"""
Position sizing rules.
"""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import numpy as np
import logging

from ..protocols import Rule


logger = logging.getLogger(__name__)


class PositionSizingRule(ABC):
    """Base class for position sizing rules."""
    
    def __init__(self, name: str):
        self.name = name
        self.positions_sized = 0
    
    @abstractmethod
    def calculate_size(self, signal: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate position size based on signal and context."""
        pass
    
    def reset(self) -> None:
        """Reset rule state."""
        self.positions_sized = 0


class FixedSizeRule(PositionSizingRule):
    """
    Fixed position size.
    """
    
    def __init__(self, 
                 name: str = "fixed_size",
                 size: float = 100,
                 unit: str = "shares"):  # shares, contracts, units
        super().__init__(name)
        self.size = size
        self.unit = unit
    
    def calculate_size(self, signal: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Return fixed size."""
        self.positions_sized += 1
        return self.size


class PercentEquityRule(PositionSizingRule):
    """
    Size position as percentage of equity.
    """
    
    def __init__(self,
                 name: str = "percent_equity",
                 percent: float = 0.02,  # 2% default
                 max_percent: float = 0.05,  # 5% max
                 use_available_equity: bool = True):
        super().__init__(name)
        self.percent = percent
        self.max_percent = max_percent
        self.use_available_equity = use_available_equity
    
    def calculate_size(self, signal: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate size based on equity percentage."""
        if self.use_available_equity:
            equity = context.get('available_equity', context.get('equity', 100000))
        else:
            equity = context.get('equity', 100000)
        
        price = context.get('close', context.get('price', 1))
        
        # Base position value
        position_value = equity * self.percent
        
        # Apply max constraint
        max_value = equity * self.max_percent
        position_value = min(position_value, max_value)
        
        # Convert to shares/units
        size = position_value / price if price > 0 else 0
        
        self.positions_sized += 1
        return size


class VolatilityBasedRule(PositionSizingRule):
    """
    Size position based on volatility (risk parity approach).
    """
    
    def __init__(self,
                 name: str = "volatility_based",
                 target_risk: float = 0.001,  # 0.1% portfolio risk per position
                 volatility_measure: str = "atr",  # atr, std, realized
                 lookback: int = 20):
        super().__init__(name)
        self.target_risk = target_risk
        self.volatility_measure = volatility_measure
        self.lookback = lookback
    
    def calculate_size(self, signal: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate size based on volatility."""
        equity = context.get('equity', 100000)
        price = context.get('close', context.get('price', 1))
        
        # Get volatility measure
        if self.volatility_measure == "atr":
            volatility = context.get('atr', price * 0.02)
        elif self.volatility_measure == "std":
            volatility = context.get(f'volatility_{self.lookback}', price * 0.02)
        else:
            volatility = context.get('realized_volatility', price * 0.02)
        
        # Calculate position size for target risk
        # Size = (Equity * TargetRisk) / Volatility
        if volatility > 0:
            position_value = (equity * self.target_risk) / (volatility / price)
            size = position_value / price
        else:
            size = 0
        
        self.positions_sized += 1
        return size


class KellyRule(PositionSizingRule):
    """
    Kelly Criterion position sizing.
    
    Size = (p * b - q) / b
    where:
    - p = probability of win
    - q = probability of loss (1 - p)
    - b = win/loss ratio
    """
    
    def __init__(self,
                 name: str = "kelly",
                 kelly_fraction: float = 0.25,  # Use 25% of full Kelly
                 min_confidence: float = 0.55,  # Minimum win probability
                 default_win_loss_ratio: float = 1.5):
        super().__init__(name)
        self.kelly_fraction = kelly_fraction
        self.min_confidence = min_confidence
        self.default_win_loss_ratio = default_win_loss_ratio
        
        # Track performance for dynamic Kelly
        self.wins = 0
        self.losses = 0
        self.total_win_amount = 0
        self.total_loss_amount = 0
    
    def calculate_size(self, signal: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate Kelly size."""
        equity = context.get('equity', 100000)
        price = context.get('close', context.get('price', 1))
        
        # Get win probability from signal confidence
        p = signal.get('confidence', 0.5)
        
        # Skip if confidence too low
        if p < self.min_confidence:
            return 0
        
        q = 1 - p
        
        # Get win/loss ratio
        if self.wins > 10 and self.losses > 10:
            # Use historical data if available
            avg_win = self.total_win_amount / self.wins
            avg_loss = self.total_loss_amount / self.losses
            b = avg_win / avg_loss if avg_loss > 0 else self.default_win_loss_ratio
        else:
            # Use default or signal-provided ratio
            b = signal.get('win_loss_ratio', self.default_win_loss_ratio)
        
        # Calculate Kelly percentage
        kelly_pct = (p * b - q) / b
        
        # Apply Kelly fraction (for safety)
        kelly_pct *= self.kelly_fraction
        
        # Ensure positive and reasonable size
        kelly_pct = max(0, min(kelly_pct, 0.25))  # Cap at 25% of equity
        
        # Convert to position size
        position_value = equity * kelly_pct
        size = position_value / price if price > 0 else 0
        
        self.positions_sized += 1
        return size
    
    def update_performance(self, trade_result: Dict[str, Any]) -> None:
        """Update performance statistics for dynamic Kelly."""
        pnl = trade_result.get('pnl', 0)
        
        if pnl > 0:
            self.wins += 1
            self.total_win_amount += pnl
        elif pnl < 0:
            self.losses += 1
            self.total_loss_amount += abs(pnl)
    
    def reset(self) -> None:
        """Reset rule state."""
        super().reset()
        self.wins = 0
        self.losses = 0
        self.total_win_amount = 0
        self.total_loss_amount = 0