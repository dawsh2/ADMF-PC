"""Reusable liquidity and fill probability models for simulated brokers.

This module contains models for determining order fill probability, partial fills,
and liquidity constraints following Protocol + Composition.
"""

import random
from typing import Dict, Any, Protocol, Optional, Tuple
from decimal import Decimal

from ..protocols import Order, OrderType, OrderSide
from .slippage import MarketConditions


class LiquidityModel(Protocol):
    """Protocol for liquidity calculation models."""
    
    def should_fill_order(
        self,
        order: Order,
        conditions: MarketConditions
    ) -> bool:
        """Determine if order should be filled."""
        ...
    
    def calculate_fill_quantity(
        self,
        order: Order,
        conditions: MarketConditions
    ) -> Decimal:
        """Calculate fill quantity based on liquidity."""
        ...


class BasicLiquidityModel:
    """Basic liquidity model with simple fill probability."""
    
    def __init__(
        self,
        fill_probability: Decimal = Decimal("0.95"),
        partial_fill_enabled: bool = True,
        min_fill_ratio: Decimal = Decimal("0.1"),
        max_participation_rate: Decimal = Decimal("0.2")
    ):
        """Initialize basic liquidity model.
        
        Args:
            fill_probability: Probability of order being filled
            partial_fill_enabled: Whether partial fills are allowed
            min_fill_ratio: Minimum fill ratio for partial fills
            max_participation_rate: Maximum participation rate in market volume
        """
        self.fill_probability = fill_probability
        self.partial_fill_enabled = partial_fill_enabled
        self.min_fill_ratio = min_fill_ratio
        self.max_participation_rate = max_participation_rate
    
    def should_fill_order(
        self,
        order: Order,
        conditions: MarketConditions
    ) -> bool:
        """Determine if order should be filled based on probability and order type."""
        # Basic probability check
        if random.random() > float(self.fill_probability):
            return False
        
        # Order type specific checks
        if order.order_type == OrderType.MARKET:
            return True
        
        elif order.order_type == OrderType.LIMIT:
            limit_price = Decimal(str(order.price))
            if order.side == OrderSide.BUY:
                return conditions.ask <= limit_price
            else:  # SELL
                return conditions.bid >= limit_price
        
        elif order.order_type == OrderType.STOP:
            stop_price = Decimal(str(order.stop_price))
            if order.side == OrderSide.BUY:
                return conditions.price >= stop_price
            else:  # SELL
                return conditions.price <= stop_price
        
        elif order.order_type == OrderType.STOP_LIMIT:
            stop_price = Decimal(str(order.stop_price))
            limit_price = Decimal(str(order.price))
            
            # Check if stop triggered and limit can be filled
            if order.side == OrderSide.BUY:
                stop_triggered = conditions.price >= stop_price
                if stop_triggered:
                    return conditions.ask <= limit_price
            else:  # SELL
                stop_triggered = conditions.price <= stop_price
                if stop_triggered:
                    return conditions.bid >= limit_price
            
            return False
        
        return False
    
    def calculate_fill_quantity(
        self,
        order: Order,
        conditions: MarketConditions
    ) -> Decimal:
        """Calculate fill quantity based on market liquidity."""
        order_quantity = Decimal(str(order.quantity))
        
        # Check participation rate limit
        if conditions.volume > 0:
            max_quantity = conditions.volume * self.max_participation_rate
            order_quantity = min(order_quantity, max_quantity)
        
        # Handle partial fills
        if self.partial_fill_enabled and random.random() < 0.2:  # 20% chance
            fill_ratio = Decimal(str(random.uniform(float(self.min_fill_ratio), 1.0)))
            return order_quantity * fill_ratio
        
        return order_quantity


class AdvancedLiquidityModel:
    """Advanced liquidity model with volatility and time-of-day effects."""
    
    def __init__(
        self,
        base_fill_probability: Decimal = Decimal("0.95"),
        volatility_impact_factor: Decimal = Decimal("0.5"),
        volume_impact_threshold: Decimal = Decimal("0.05"),
        time_of_day_effects: bool = True
    ):
        """Initialize advanced liquidity model.
        
        Args:
            base_fill_probability: Base probability of order being filled
            volatility_impact_factor: How much volatility affects fill probability
            volume_impact_threshold: Volume threshold for reduced fill probability
            time_of_day_effects: Whether to simulate time-of-day liquidity effects
        """
        self.base_fill_probability = base_fill_probability
        self.volatility_impact_factor = volatility_impact_factor
        self.volume_impact_threshold = volume_impact_threshold
        self.time_of_day_effects = time_of_day_effects
    
    def should_fill_order(
        self,
        order: Order,
        conditions: MarketConditions
    ) -> bool:
        """Advanced fill probability calculation."""
        # Start with base probability
        fill_prob = float(self.base_fill_probability)
        
        # Adjust for volatility (higher volatility = lower fill probability)
        if conditions.volatility > Decimal("0.02"):  # Above 2% volatility
            volatility_penalty = float(conditions.volatility * self.volatility_impact_factor)
            fill_prob = max(0.1, fill_prob - volatility_penalty)
        
        # Adjust for order size relative to volume
        if conditions.volume > 0:
            participation_rate = float(Decimal(str(order.quantity)) / conditions.volume)
            if participation_rate > float(self.volume_impact_threshold):
                size_penalty = (participation_rate - float(self.volume_impact_threshold)) * 0.5
                fill_prob = max(0.1, fill_prob - size_penalty)
        
        # Check adjusted probability
        if random.random() > fill_prob:
            return False
        
        # Order type specific logic (same as BasicLiquidityModel)
        return self._check_order_type_conditions(order, conditions)
    
    def calculate_fill_quantity(
        self,
        order: Order,
        conditions: MarketConditions
    ) -> Decimal:
        """Advanced fill quantity calculation with market impact."""
        order_quantity = Decimal(str(order.quantity))
        
        # Calculate maximum fillable quantity based on liquidity
        if conditions.volume > 0:
            # More sophisticated participation rate calculation
            liquidity_factor = conditions.liquidity_factor
            effective_volume = conditions.volume * liquidity_factor
            max_participation = min(Decimal("0.3"), effective_volume / Decimal("1000"))  # Dynamic limit
            max_quantity = effective_volume * max_participation
            order_quantity = min(order_quantity, max_quantity)
        
        # Advanced partial fill logic
        if random.random() < 0.15:  # 15% chance of partial fill
            # Partial fill size depends on volatility and volume
            volatility_factor = min(Decimal("1.0"), conditions.volatility * Decimal("10"))
            min_fill = Decimal("0.3") - (volatility_factor * Decimal("0.2"))  # Lower min in volatile markets
            max_fill = Decimal("0.9") + (volatility_factor * Decimal("0.1"))   # Higher max in volatile markets
            
            fill_ratio = Decimal(str(random.uniform(float(min_fill), float(max_fill))))
            return order_quantity * fill_ratio
        
        return order_quantity
    
    def _check_order_type_conditions(self, order: Order, conditions: MarketConditions) -> bool:
        """Check order type specific conditions."""
        if order.order_type == OrderType.MARKET:
            return True
        
        elif order.order_type == OrderType.LIMIT:
            limit_price = Decimal(str(order.price))
            if order.side == OrderSide.BUY:
                return conditions.ask <= limit_price
            else:
                return conditions.bid >= limit_price
        
        elif order.order_type == OrderType.STOP:
            stop_price = Decimal(str(order.stop_price))
            if order.side == OrderSide.BUY:
                return conditions.price >= stop_price
            else:
                return conditions.price <= stop_price
        
        elif order.order_type == OrderType.STOP_LIMIT:
            stop_price = Decimal(str(order.stop_price))
            limit_price = Decimal(str(order.price))
            
            if order.side == OrderSide.BUY:
                if conditions.price >= stop_price:
                    return conditions.ask <= limit_price
            else:
                if conditions.price <= stop_price:
                    return conditions.bid >= limit_price
            
            return False
        
        return False


class PerfectLiquidityModel:
    """Perfect liquidity model - all orders fill completely."""
    
    def should_fill_order(
        self,
        order: Order,
        conditions: MarketConditions
    ) -> bool:
        """Always returns True - perfect liquidity."""
        # Still respect order type logic for realism
        if order.order_type == OrderType.LIMIT:
            limit_price = Decimal(str(order.price))
            if order.side == OrderSide.BUY:
                return conditions.ask <= limit_price
            else:
                return conditions.bid >= limit_price
        
        return True
    
    def calculate_fill_quantity(
        self,
        order: Order,
        conditions: MarketConditions
    ) -> Decimal:
        """Always returns full order quantity."""
        return Decimal(str(order.quantity))


# Pre-configured liquidity models for common scenarios
def create_liquid_market_model() -> LiquidityModel:
    """Create model for highly liquid markets (e.g., SPY, QQQ)."""
    return BasicLiquidityModel(
        fill_probability=Decimal("0.98"),
        partial_fill_enabled=False,  # Liquid markets fill completely
        max_participation_rate=Decimal("0.1")
    )


def create_illiquid_market_model() -> LiquidityModel:
    """Create model for illiquid markets (e.g., small cap stocks)."""
    return AdvancedLiquidityModel(
        base_fill_probability=Decimal("0.85"),
        volatility_impact_factor=Decimal("1.0"),  # High volatility impact
        volume_impact_threshold=Decimal("0.02"),  # Low threshold
        time_of_day_effects=True
    )


def create_crypto_market_model() -> LiquidityModel:
    """Create model for crypto markets with high volatility."""
    return AdvancedLiquidityModel(
        base_fill_probability=Decimal("0.90"),
        volatility_impact_factor=Decimal("0.8"),
        volume_impact_threshold=Decimal("0.03"),
        time_of_day_effects=False  # 24/7 trading
    )