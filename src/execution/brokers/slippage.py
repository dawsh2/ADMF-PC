"""Reusable slippage models for simulated brokers.

This module contains various slippage calculation models that can be composed
with different broker implementations following Protocol + Composition.
"""

from typing import Dict, Any, Protocol
from decimal import Decimal
from dataclasses import dataclass

from ..protocols import Order, OrderSide
from ...core.components.discovery import execution_model


class SlippageModel(Protocol):
    """Protocol for slippage calculation models."""
    
    def calculate_slippage(
        self,
        order: Order,
        market_price: Decimal,
        market_data: Dict[str, Any]
    ) -> Decimal:
        """Calculate slippage for order."""
        ...


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


@execution_model(
    model_type='slippage',
    params={
        'base_slippage_pct': 0.001,
        'volatility_multiplier': 2.0,
        'volume_impact_factor': 0.1
    }
)
class PercentageSlippageModel:
    """Simple percentage-based slippage model."""
    
    def __init__(
        self,
        base_slippage_pct: Decimal = Decimal("0.001"),
        volatility_multiplier: Decimal = Decimal("2.0"),
        volume_impact_factor: Decimal = Decimal("0.1")
    ):
        """Initialize percentage slippage model.
        
        Args:
            base_slippage_pct: Base slippage percentage (0.001 = 0.1%)
            volatility_multiplier: Multiplier for volatility impact
            volume_impact_factor: Factor for volume-based impact
        """
        self.base_slippage_pct = base_slippage_pct
        self.volatility_multiplier = volatility_multiplier
        self.volume_impact_factor = volume_impact_factor
    
    def calculate_slippage(
        self,
        order: Order,
        market_price: Decimal,
        market_data: Dict[str, Any]
    ) -> Decimal:
        """Calculate percentage-based slippage with market impact."""
        conditions = market_data.get('conditions')
        if not conditions:
            # Fallback to simple calculation
            base = market_price * self.base_slippage_pct
            return base if order.side == OrderSide.BUY else -base
        
        # Base slippage
        base_slippage = market_price * self.base_slippage_pct
        
        # Volatility adjustment
        volatility_impact = (
            market_price * conditions.volatility * self.volatility_multiplier
        )
        
        # Volume impact
        if conditions.volume > 0:
            order_volume_ratio = Decimal(str(order.quantity)) / conditions.volume
            volume_impact = (
                market_price * order_volume_ratio * self.volume_impact_factor
            )
        else:
            volume_impact = Decimal("0")
        
        # Total slippage (direction based on order side)
        total_slippage = base_slippage + volatility_impact + volume_impact
        
        if order.side == OrderSide.SELL:
            total_slippage = -total_slippage
        
        return total_slippage


@execution_model(
    model_type='slippage',
    name='volume_impact',
    params={
        'permanent_impact_factor': 0.0001,
        'temporary_impact_factor': 0.0002,
        'liquidity_threshold': 0.01
    },
    description='Advanced slippage model based on market impact theory'
)
class VolumeImpactSlippageModel:
    """Advanced slippage model based on market impact theory."""
    
    def __init__(
        self,
        permanent_impact_factor: Decimal = Decimal("0.0001"),
        temporary_impact_factor: Decimal = Decimal("0.0002"),
        liquidity_threshold: Decimal = Decimal("0.01")
    ):
        """Initialize volume impact slippage model.
        
        Args:
            permanent_impact_factor: Factor for permanent market impact
            temporary_impact_factor: Factor for temporary market impact  
            liquidity_threshold: Threshold for low liquidity penalty
        """
        self.permanent_impact_factor = permanent_impact_factor
        self.temporary_impact_factor = temporary_impact_factor
        self.liquidity_threshold = liquidity_threshold
    
    def calculate_slippage(
        self,
        order: Order,
        market_price: Decimal,
        market_data: Dict[str, Any]
    ) -> Decimal:
        """Calculate slippage based on market impact theory."""
        conditions = market_data.get('conditions')
        if not conditions:
            base = market_price * Decimal("0.001")  # Default 0.1%
            return base if order.side == OrderSide.BUY else -base
        
        # Calculate order participation rate
        if conditions.volume > 0:
            participation_rate = Decimal(str(order.quantity)) / conditions.volume
        else:
            participation_rate = Decimal("1.0")  # Assume high impact for zero volume
        
        # Permanent impact (square root of participation rate)
        permanent_impact = (
            market_price * self.permanent_impact_factor * 
            (participation_rate ** Decimal("0.5"))
        )
        
        # Temporary impact (linear with participation rate)
        temporary_impact = (
            market_price * self.temporary_impact_factor * participation_rate
        )
        
        # Liquidity penalty for high participation rates
        if participation_rate > self.liquidity_threshold:
            liquidity_penalty = (
                market_price * (participation_rate - self.liquidity_threshold) * 
                Decimal("0.01")
            )
        else:
            liquidity_penalty = Decimal("0")
        
        # Total slippage
        total_slippage = permanent_impact + temporary_impact + liquidity_penalty
        
        # Direction based on order side
        if order.side == OrderSide.SELL:
            total_slippage = -total_slippage
        
        return total_slippage


@execution_model(
    model_type='slippage',
    name='fixed',
    params={'slippage_amount': 0.01}
)
class FixedSlippageModel:
    """Fixed slippage model for simple scenarios."""
    
    def __init__(self, slippage_amount: Decimal = Decimal("0.01")):
        """Initialize fixed slippage model.
        
        Args:
            slippage_amount: Fixed slippage amount in currency units
        """
        self.slippage_amount = slippage_amount
    
    def calculate_slippage(
        self,
        order: Order,
        market_price: Decimal,
        market_data: Dict[str, Any]
    ) -> Decimal:
        """Calculate fixed slippage."""
        if order.side == OrderSide.SELL:
            return -self.slippage_amount
        return self.slippage_amount


@execution_model(
    model_type='slippage',
    name='zero',
    description='Zero slippage for ideal execution scenarios'
)
class ZeroSlippageModel:
    """Zero slippage model for ideal execution scenarios."""
    
    def calculate_slippage(
        self,
        order: Order,
        market_price: Decimal,
        market_data: Dict[str, Any]
    ) -> Decimal:
        """Always returns zero slippage."""
        return Decimal("0.0")