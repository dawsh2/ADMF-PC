"""Slippage models for backtest execution.

Synchronous slippage calculation models for market simulation.
"""

from typing import Dict, Any
from decimal import Decimal
from dataclasses import dataclass

from ...types import Order, OrderSide
from ...sync_protocols import SlippageModel


@dataclass
class MarketConditions:
    """Market conditions for slippage calculation."""
    price: Decimal
    volume: Decimal
    bid: Decimal
    ask: Decimal
    volatility: Decimal = Decimal("0.02")
    liquidity_factor: Decimal = Decimal("1.0")
    
    @property
    def spread(self) -> Decimal:
        """Calculate bid-ask spread."""
        return self.ask - self.bid
    
    @property
    def mid_price(self) -> Decimal:
        """Calculate mid price."""
        return (self.bid + self.ask) / Decimal("2")


class PercentageSlippageModel:
    """Simple percentage-based slippage model."""
    
    def __init__(
        self,
        base_slippage_pct: float = 0.001,
        volatility_multiplier: float = 2.0,
        volume_impact_factor: float = 0.1
    ):
        """Initialize percentage slippage model.
        
        Args:
            base_slippage_pct: Base slippage percentage (0.001 = 0.1%)
            volatility_multiplier: Multiplier for volatility impact
            volume_impact_factor: Factor for volume-based impact
        """
        self.base_slippage_pct = Decimal(str(base_slippage_pct))
        self.volatility_multiplier = Decimal(str(volatility_multiplier))
        self.volume_impact_factor = Decimal(str(volume_impact_factor))
    
    def calculate_slippage(
        self,
        order: Order,
        market_price: float,
        volume: float = 0
    ) -> float:
        """Calculate percentage-based slippage."""
        price_decimal = Decimal(str(market_price))
        
        # Base slippage
        base_slippage = price_decimal * self.base_slippage_pct
        
        # Volume impact
        if volume > 0:
            order_volume_ratio = order.quantity / Decimal(str(volume))
            volume_impact = (
                price_decimal * order_volume_ratio * self.volume_impact_factor
            )
        else:
            volume_impact = Decimal("0")
        
        # Total slippage (direction based on order side)
        total_slippage = base_slippage + volume_impact
        
        if order.side == OrderSide.SELL:
            total_slippage = -total_slippage
        
        return float(total_slippage)


class FixedSlippageModel:
    """Fixed slippage model for simple scenarios."""
    
    def __init__(self, slippage_amount: float = 0.01):
        """Initialize fixed slippage model.
        
        Args:
            slippage_amount: Fixed slippage amount in currency units
        """
        self.slippage_amount = slippage_amount
    
    def calculate_slippage(
        self,
        order: Order,
        market_price: float,
        volume: float = 0
    ) -> float:
        """Calculate fixed slippage."""
        if order.side == OrderSide.SELL:
            return -self.slippage_amount
        return self.slippage_amount


class ZeroSlippageModel:
    """Zero slippage model for ideal execution scenarios."""
    
    def calculate_slippage(
        self,
        order: Order,
        market_price: float,
        volume: float = 0
    ) -> float:
        """Always returns zero slippage."""
        return 0.0