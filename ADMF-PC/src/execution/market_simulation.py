"""Market simulation for slippage and commission models."""

from typing import Optional, Protocol, Dict, Any
from dataclasses import dataclass
import random
from datetime import datetime
import uuid

from .protocols import Order, Fill, OrderSide, OrderType, FillType
from ..core.logging.structured import get_logger


logger = get_logger(__name__)


class SlippageModel(Protocol):
    """Slippage model interface."""
    
    def calculate_slippage(
        self,
        order: Order,
        market_price: float,
        volume: float,
        spread: float = 0.01
    ) -> float:
        """Calculate slippage for order."""
        ...


class CommissionModel(Protocol):
    """Commission model interface."""
    
    def calculate_commission(
        self,
        order: Order,
        fill_price: float,
        fill_quantity: float
    ) -> float:
        """Calculate commission for order."""
        ...


@dataclass
class FixedSlippageModel:
    """Fixed slippage model."""
    slippage_percent: float = 0.001  # 0.1% default
    
    def calculate_slippage(
        self,
        order: Order,
        market_price: float,
        volume: float,
        spread: float = 0.01
    ) -> float:
        """Calculate fixed percentage slippage."""
        base_slippage = market_price * self.slippage_percent
        
        # Add impact for large orders
        order_impact = 0.0
        if volume > 0:
            order_ratio = order.quantity / volume
            if order_ratio > 0.01:  # More than 1% of volume
                order_impact = market_price * order_ratio * 0.001
        
        # Direction based on order side
        if order.side == OrderSide.BUY:
            return base_slippage + order_impact
        else:
            return -(base_slippage + order_impact)


@dataclass
class VolumeSlippageModel:
    """Volume-based slippage model."""
    base_impact: float = 0.0001  # Base market impact
    
    def calculate_slippage(
        self,
        order: Order,
        market_price: float,
        volume: float,
        spread: float = 0.01
    ) -> float:
        """Calculate volume-based slippage."""
        if volume <= 0:
            # No volume data, use simple model
            return market_price * 0.001 * (1 if order.side == OrderSide.BUY else -1)
        
        # Calculate market impact based on order size
        order_ratio = order.quantity / volume
        impact = self.base_impact * (1 + order_ratio ** 0.5)
        
        # Add spread component
        spread_cost = spread / 2
        
        # Total slippage
        slippage = market_price * impact + spread_cost
        
        # Direction based on order side
        if order.side == OrderSide.BUY:
            return slippage
        else:
            return -slippage


@dataclass
class FixedCommissionModel:
    """Fixed commission per trade."""
    commission_per_trade: float = 1.0
    
    def calculate_commission(
        self,
        order: Order,
        fill_price: float,
        fill_quantity: float
    ) -> float:
        """Calculate fixed commission."""
        return self.commission_per_trade


@dataclass
class PerShareCommissionModel:
    """Per-share commission model."""
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    max_commission: float = 5.0
    
    def calculate_commission(
        self,
        order: Order,
        fill_price: float,
        fill_quantity: float
    ) -> float:
        """Calculate per-share commission."""
        commission = fill_quantity * self.commission_per_share
        return max(self.min_commission, min(commission, self.max_commission))


@dataclass
class PercentCommissionModel:
    """Percentage-based commission model."""
    commission_percent: float = 0.001  # 0.1%
    min_commission: float = 1.0
    
    def calculate_commission(
        self,
        order: Order,
        fill_price: float,
        fill_quantity: float
    ) -> float:
        """Calculate percentage commission."""
        commission = fill_quantity * fill_price * self.commission_percent
        return max(self.min_commission, commission)


class MarketSimulator:
    """Simulates market conditions for order execution."""
    
    def __init__(
        self,
        slippage_model: Optional[SlippageModel] = None,
        commission_model: Optional[CommissionModel] = None,
        fill_probability: float = 0.95,
        partial_fill_enabled: bool = True
    ):
        """Initialize market simulator."""
        self.slippage_model = slippage_model or FixedSlippageModel()
        self.commission_model = commission_model or PerShareCommissionModel()
        self.fill_probability = fill_probability
        self.partial_fill_enabled = partial_fill_enabled
        
        logger.info("Initialized MarketSimulator")
    
    async def simulate_fill(
        self,
        order: Order,
        market_price: float,
        volume: float,
        spread: float = 0.01
    ) -> Optional[Fill]:
        """Simulate order fill with market conditions."""
        # Check fill probability
        if random.random() > self.fill_probability:
            logger.info(f"Order {order.order_id} not filled (probability)")
            return None
        
        # Determine fill price based on order type
        fill_price = self._calculate_fill_price(order, market_price, spread)
        if fill_price is None:
            logger.info(f"Order {order.order_id} not filled (price conditions)")
            return None
        
        # Calculate slippage
        slippage = self.slippage_model.calculate_slippage(
            order, market_price, volume, spread
        )
        fill_price += slippage
        
        # Determine fill quantity
        fill_quantity = self._calculate_fill_quantity(order, volume)
        if fill_quantity <= 0:
            logger.info(f"Order {order.order_id} not filled (no liquidity)")
            return None
        
        # Calculate commission
        commission = self.commission_model.calculate_commission(
            order, fill_price, fill_quantity
        )
        
        # Create fill
        fill = Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=fill_price,
            commission=commission,
            slippage=slippage,
            fill_type=FillType.FULL if fill_quantity >= order.quantity else FillType.PARTIAL,
            executed_at=datetime.now(),
            metadata={
                "market_price": market_price,
                "volume": volume,
                "spread": spread
            }
        )
        
        logger.info(
            f"Simulated fill: {order.side.name} {fill_quantity} {order.symbol} "
            f"@ {fill_price:.2f} (market: {market_price:.2f}, "
            f"slippage: {slippage:.4f}, commission: {commission:.2f})"
        )
        
        return fill
    
    def _calculate_fill_price(
        self,
        order: Order,
        market_price: float,
        spread: float
    ) -> Optional[float]:
        """Calculate fill price based on order type."""
        if order.order_type == OrderType.MARKET:
            # Market orders fill at market price
            return market_price
        
        elif order.order_type == OrderType.LIMIT:
            # Limit orders fill at limit price or better
            if order.side == OrderSide.BUY:
                if market_price <= order.price:
                    return min(market_price, order.price)
            else:  # SELL
                if market_price >= order.price:
                    return max(market_price, order.price)
            return None
        
        elif order.order_type == OrderType.STOP:
            # Stop orders trigger at stop price
            if order.side == OrderSide.BUY:
                if market_price >= order.stop_price:
                    return market_price
            else:  # SELL
                if market_price <= order.stop_price:
                    return market_price
            return None
        
        elif order.order_type == OrderType.STOP_LIMIT:
            # Stop limit orders trigger at stop price, fill at limit
            if order.side == OrderSide.BUY:
                if market_price >= order.stop_price and market_price <= order.price:
                    return min(market_price, order.price)
            else:  # SELL
                if market_price <= order.stop_price and market_price >= order.price:
                    return max(market_price, order.price)
            return None
        
        return None
    
    def _calculate_fill_quantity(self, order: Order, volume: float) -> float:
        """Calculate fill quantity based on available liquidity."""
        if not self.partial_fill_enabled:
            return order.quantity
        
        # Simulate available liquidity
        if volume <= 0:
            # No volume data, assume full fill
            return order.quantity
        
        # Can fill up to certain percentage of volume
        max_fill = min(order.quantity, volume * 0.1)  # Max 10% of volume
        
        # Random partial fill
        if random.random() < 0.2:  # 20% chance of partial fill
            fill_percent = random.uniform(0.5, 0.95)
            return max_fill * fill_percent
        
        return min(order.quantity, max_fill)
    
    def calculate_slippage(
        self,
        order: Order,
        market_price: float,
        volume: float
    ) -> float:
        """Calculate slippage for order."""
        return self.slippage_model.calculate_slippage(order, market_price, volume)
    
    def calculate_commission(self, order: Order, fill_price: float) -> float:
        """Calculate commission for order."""
        return self.commission_model.calculate_commission(
            order, fill_price, order.quantity
        )