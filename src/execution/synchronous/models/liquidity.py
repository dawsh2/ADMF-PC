"""Liquidity models for backtest execution.

Models for simulating market liquidity constraints.
"""

from typing import Dict, Any

from ...types import Order
from ...sync_protocols import LiquidityModel


class UnlimitedLiquidityModel:
    """Unlimited liquidity model - all orders can be filled."""
    
    def can_fill_order(
        self, 
        order: Order, 
        market_data: Dict[str, Any]
    ) -> tuple[bool, float]:
        """All orders can be filled completely."""
        return True, 1.0


class VolumeBasedLiquidityModel:
    """Liquidity model based on market volume."""
    
    def __init__(
        self,
        max_participation_rate: float = 0.1,  # 10% of volume
        min_volume_threshold: float = 1000.0
    ):
        """Initialize volume-based liquidity model.
        
        Args:
            max_participation_rate: Maximum participation rate in volume
            min_volume_threshold: Minimum volume required for any fill
        """
        self.max_participation_rate = max_participation_rate
        self.min_volume_threshold = min_volume_threshold
    
    def can_fill_order(
        self, 
        order: Order, 
        market_data: Dict[str, Any]
    ) -> tuple[bool, float]:
        """Check if order can be filled based on volume."""
        volume = market_data.get('volume', 0)
        
        # Check minimum volume threshold
        if volume < self.min_volume_threshold:
            return False, 0.0
        
        # Calculate maximum fillable quantity
        max_quantity = volume * self.max_participation_rate
        order_quantity = float(order.quantity)
        
        if order_quantity <= max_quantity:
            return True, 1.0  # Can fill completely
        else:
            fill_ratio = max_quantity / order_quantity
            return True, fill_ratio  # Partial fill


class TimeBasedLiquidityModel:
    """Liquidity model with time-of-day variations."""
    
    def __init__(
        self,
        market_open_liquidity: float = 1.0,
        midday_liquidity: float = 0.8,
        market_close_liquidity: float = 0.9
    ):
        """Initialize time-based liquidity model.
        
        Args:
            market_open_liquidity: Liquidity factor at market open
            midday_liquidity: Liquidity factor during midday
            market_close_liquidity: Liquidity factor at market close
        """
        self.market_open_liquidity = market_open_liquidity
        self.midday_liquidity = midday_liquidity
        self.market_close_liquidity = market_close_liquidity
    
    def can_fill_order(
        self, 
        order: Order, 
        market_data: Dict[str, Any]
    ) -> tuple[bool, float]:
        """Check if order can be filled based on time of day."""
        # For simplicity, assume midday liquidity
        # In real implementation, would check timestamp
        liquidity_factor = self.midday_liquidity
        
        # Simple liquidity check
        if liquidity_factor > 0.5:
            return True, liquidity_factor
        else:
            return False, 0.0