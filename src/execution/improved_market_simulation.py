"""
Improved market simulator with proper dependency injection and enhanced models.

This simulator follows the core system's patterns and provides configurable
slippage and commission models through dependency injection.
"""

import asyncio
from typing import Dict, Optional, Any, List, Tuple, Protocol
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
import uuid
import random
import logging

from ..core.components.protocols import Component, Lifecycle
from .protocols import (
    Order, Fill, OrderSide, OrderType, FillType, FillStatus,
    MarketSimulator as MarketSimulatorProtocol
)

logger = logging.getLogger(__name__)


class SlippageModel(Protocol):
    """Enhanced slippage model interface."""
    
    def calculate_slippage(
        self,
        order: Order,
        market_price: Decimal,
        market_data: Dict[str, Any]
    ) -> Decimal:
        """Calculate slippage for order."""
        ...


class CommissionModel(Protocol):
    """Enhanced commission model interface."""
    
    def calculate_commission(
        self,
        order: Order,
        fill_price: Decimal,
        fill_quantity: Decimal
    ) -> Decimal:
        """Calculate commission for order."""
        ...


@dataclass
class MarketConditions:
    """Market conditions for execution simulation."""
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
    """Slippage model based on percentage of order size."""
    
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
            return market_price * self.base_slippage_pct
        
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
            return market_price * Decimal("0.001")  # Default 0.1%
        
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


class TieredCommissionModel:
    """Commission model with tiered rates based on volume."""
    
    def __init__(
        self,
        tiers: List[Tuple[Decimal, Decimal]],
        minimum_commission: Decimal = Decimal("1.0")
    ):
        """Initialize tiered commission model.
        
        Args:
            tiers: List of (volume_threshold, rate) tuples, sorted by threshold
            minimum_commission: Minimum commission per trade
        """
        self.tiers = sorted(tiers, key=lambda x: x[0])
        self.minimum_commission = minimum_commission
    
    def calculate_commission(
        self,
        order: Order,
        fill_price: Decimal,
        fill_quantity: Decimal
    ) -> Decimal:
        """Calculate commission based on volume tier."""
        trade_value = fill_price * fill_quantity
        
        # Find applicable tier
        commission_rate = self.tiers[0][1]  # Default to first tier
        for threshold, rate in self.tiers:
            if trade_value >= threshold:
                commission_rate = rate
            else:
                break
        
        commission = trade_value * commission_rate
        return max(commission, self.minimum_commission)


class PerShareCommissionModel:
    """Simple per-share commission model."""
    
    def __init__(
        self,
        rate_per_share: Decimal = Decimal("0.005"),
        minimum_commission: Decimal = Decimal("1.0"),
        maximum_commission: Decimal = Decimal("10.0")
    ):
        """Initialize per-share commission model.
        
        Args:
            rate_per_share: Commission per share
            minimum_commission: Minimum commission per trade
            maximum_commission: Maximum commission per trade
        """
        self.rate_per_share = rate_per_share
        self.minimum_commission = minimum_commission
        self.maximum_commission = maximum_commission
    
    def calculate_commission(
        self,
        order: Order,
        fill_price: Decimal,
        fill_quantity: Decimal
    ) -> Decimal:
        """Calculate per-share commission."""
        commission = fill_quantity * self.rate_per_share
        commission = max(commission, self.minimum_commission)
        commission = min(commission, self.maximum_commission)
        return commission


class ImprovedMarketSimulator(Component, Lifecycle, MarketSimulatorProtocol):
    """
    Improved market simulator with proper dependency injection.
    
    This simulator provides realistic order execution simulation with
    configurable slippage and commission models.
    """
    
    def __init__(
        self,
        component_id: str,
        slippage_model: Optional[SlippageModel] = None,
        commission_model: Optional[CommissionModel] = None,
        fill_probability: Decimal = Decimal("0.95"),
        partial_fill_enabled: bool = True,
        min_fill_ratio: Decimal = Decimal("0.1"),
        max_participation_rate: Decimal = Decimal("0.2")
    ):
        """Initialize market simulator with dependency injection.
        
        Args:
            component_id: Unique component identifier
            slippage_model: Slippage calculation model (injected)
            commission_model: Commission calculation model (injected)
            fill_probability: Probability of order being filled
            partial_fill_enabled: Whether partial fills are allowed
            min_fill_ratio: Minimum fill ratio for partial fills
            max_participation_rate: Maximum participation rate in market volume
        """
        self._component_id = component_id
        self._slippage_model = slippage_model or PercentageSlippageModel()
        self._commission_model = commission_model or PerShareCommissionModel()
        self._fill_probability = fill_probability
        self._partial_fill_enabled = partial_fill_enabled
        self._min_fill_ratio = min_fill_ratio
        self._max_participation_rate = max_participation_rate
        
        # Market data cache
        self._market_conditions: Dict[str, MarketConditions] = {}
        self._market_lock = asyncio.Lock()
        
        # Simulation statistics
        self._simulation_stats = {
            'total_simulations': 0,
            'successful_fills': 0,
            'partial_fills': 0,
            'rejected_fills': 0,
            'total_slippage': Decimal("0"),
            'total_commission': Decimal("0")
        }
        
        # Lifecycle state
        self._initialized = False
        self._running = False
        
        logger.info(f"ImprovedMarketSimulator created - ID: {component_id}")
    
    @property
    def component_id(self) -> str:
        """Component identifier."""
        return self._component_id
    
    def initialize(self, context: Dict[str, Any]) -> None:
        """Initialize the market simulator."""
        self._initialized = True
        logger.info(f"MarketSimulator initialized - ID: {self._component_id}")
    
    def start(self) -> None:
        """Start the market simulator."""
        if not self._initialized:
            raise RuntimeError("MarketSimulator not initialized")
        
        self._running = True
        logger.info(f"MarketSimulator started - ID: {self._component_id}")
    
    def stop(self) -> None:
        """Stop the market simulator."""
        self._running = False
        logger.info(f"MarketSimulator stopped - ID: {self._component_id}")
    
    def reset(self) -> None:
        """Reset simulator state."""
        # Clear market conditions
        self._market_conditions.clear()
        
        # Reset statistics
        self._simulation_stats = {
            'total_simulations': 0,
            'successful_fills': 0,
            'partial_fills': 0,
            'rejected_fills': 0,
            'total_slippage': Decimal("0"),
            'total_commission': Decimal("0")
        }
        
        logger.info(f"MarketSimulator reset - ID: {self._component_id}")
    
    def teardown(self) -> None:
        """Teardown the market simulator."""
        # Stop if running
        if self._running:
            self.stop()
        
        # Clear all state
        self.reset()
        
        logger.info(f"MarketSimulator torn down - ID: {self._component_id}")
    
    async def simulate_fill(
        self,
        order: Order,
        market_price: float,
        volume: float,
        spread: float = 0.01
    ) -> Optional[Fill]:
        """Simulate order fill with comprehensive market conditions."""
        if not self._running:
            logger.warning("MarketSimulator not running")
            return None
        
        self._simulation_stats['total_simulations'] += 1
        
        # Convert to Decimal for precise calculations
        market_price_decimal = Decimal(str(market_price))
        volume_decimal = Decimal(str(volume))
        spread_decimal = Decimal(str(spread))
        
        # Create market conditions
        conditions = await self._create_market_conditions(
            order.symbol, market_price_decimal, volume_decimal, spread_decimal
        )
        
        # Check fill probability
        if not self._should_fill_order(order, conditions):
            self._simulation_stats['rejected_fills'] += 1
            logger.debug(f"Order {order.order_id} not filled (probability/conditions)")
            return None
        
        # Determine fill price and quantity
        fill_result = await self._calculate_fill_execution(order, conditions)
        if not fill_result:
            self._simulation_stats['rejected_fills'] += 1
            return None
        
        fill_price, fill_quantity = fill_result
        
        # Calculate slippage and commission
        slippage = await self._calculate_slippage(order, market_price_decimal, conditions)
        commission = await self._calculate_commission(order, fill_price, fill_quantity)
        
        # Apply slippage to fill price
        final_fill_price = fill_price + slippage
        
        # Create fill - keep commission as Decimal to avoid conversion issues
        fill = Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=float(fill_quantity),
            price=float(final_fill_price),
            commission=commission,  # Keep as Decimal for consistent type handling
            slippage=float(slippage),
            fill_type=(
                FillType.FULL if fill_quantity >= Decimal(str(order.quantity))
                else FillType.PARTIAL
            ),
            status=FillStatus.FILLED,
            executed_at=datetime.now(),
            metadata={
                'market_price': float(market_price_decimal),
                'volume': float(volume_decimal),
                'spread': float(spread_decimal),
                'simulation_type': 'advanced',
                'conditions': {
                    'bid': float(conditions.bid),
                    'ask': float(conditions.ask),
                    'volatility': float(conditions.volatility),
                    'liquidity_factor': float(conditions.liquidity_factor)
                }
            }
        )
        
        # Update statistics
        self._update_simulation_stats(fill)
        
        logger.info(
            f"Simulated fill - Order: {order.order_id}, "
            f"Fill: {fill.quantity} @ {fill.price:.4f}, "
            f"Slippage: {fill.slippage:.4f}, Commission: {fill.commission:.2f}"
        )
        
        return fill
    
    def calculate_slippage(
        self,
        order: Order,
        market_price: float,
        volume: float
    ) -> float:
        """Calculate slippage for order (legacy interface)."""
        market_price_decimal = Decimal(str(market_price))
        volume_decimal = Decimal(str(volume))
        
        # Create basic market conditions
        conditions = MarketConditions(
            price=market_price_decimal,
            volume=volume_decimal,
            bid=market_price_decimal * Decimal("0.999"),
            ask=market_price_decimal * Decimal("1.001")
        )
        
        market_data = {'conditions': conditions}
        slippage = self._slippage_model.calculate_slippage(
            order, market_price_decimal, market_data
        )
        
        return float(slippage)
    
    def calculate_commission(self, order: Order, fill_price: float) -> float:
        """Calculate commission for order (legacy interface)."""
        fill_price_decimal = Decimal(str(fill_price))
        fill_quantity_decimal = Decimal(str(order.quantity))
        
        commission = self._commission_model.calculate_commission(
            order, fill_price_decimal, fill_quantity_decimal
        )
        
        return float(commission)
    
    async def update_market_data(self, symbol: str, market_data: Dict[str, Any]) -> None:
        """Update market conditions for a symbol."""
        async with self._market_lock:
            price = Decimal(str(market_data.get('price', 100)))
            volume = Decimal(str(market_data.get('volume', 10000)))
            bid = Decimal(str(market_data.get('bid', price * Decimal("0.999"))))
            ask = Decimal(str(market_data.get('ask', price * Decimal("1.001"))))
            volatility = Decimal(str(market_data.get('volatility', 0.02)))
            liquidity_factor = Decimal(str(market_data.get('liquidity_factor', 1.0)))
            
            self._market_conditions[symbol] = MarketConditions(
                price=price,
                volume=volume,
                bid=bid,
                ask=ask,
                volatility=volatility,
                liquidity_factor=liquidity_factor
            )
    
    def get_simulation_stats(self) -> Dict[str, Any]:
        """Get simulation statistics."""
        total_sims = self._simulation_stats['total_simulations']
        fill_rate = (
            self._simulation_stats['successful_fills'] / total_sims
            if total_sims > 0 else 0
        )
        
        return {
            "component_id": self._component_id,
            "running": self._running,
            "simulation_stats": {
                **self._simulation_stats,
                "fill_rate": float(fill_rate),
                "avg_slippage": float(
                    self._simulation_stats['total_slippage'] / 
                    max(self._simulation_stats['successful_fills'], 1)
                ),
                "avg_commission": float(
                    self._simulation_stats['total_commission'] / 
                    max(self._simulation_stats['successful_fills'], 1)
                )
            },
            "market_symbols": len(self._market_conditions)
        }
    
    # Private methods
    
    async def _create_market_conditions(
        self,
        symbol: str,
        market_price: Decimal,
        volume: Decimal,
        spread: Decimal
    ) -> MarketConditions:
        """Create market conditions for simulation."""
        async with self._market_lock:
            # Use cached conditions if available
            if symbol in self._market_conditions:
                cached = self._market_conditions[symbol]
                # Update price and volume but keep other characteristics
                return MarketConditions(
                    price=market_price,
                    volume=volume,
                    bid=market_price - spread / Decimal("2"),
                    ask=market_price + spread / Decimal("2"),
                    volatility=cached.volatility,
                    liquidity_factor=cached.liquidity_factor
                )
            
            # Create new conditions
            return MarketConditions(
                price=market_price,
                volume=volume,
                bid=market_price - spread / Decimal("2"),
                ask=market_price + spread / Decimal("2"),
                volatility=Decimal("0.02"),  # Default 2% volatility
                liquidity_factor=Decimal("1.0")  # Default liquidity
            )
    
    def _should_fill_order(self, order: Order, conditions: MarketConditions) -> bool:
        """Determine if order should be filled based on market conditions."""
        # Basic probability check
        if random.random() > float(self._fill_probability):
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
            
            # Check if stop triggered
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
    
    async def _calculate_fill_execution(
        self,
        order: Order,
        conditions: MarketConditions
    ) -> Optional[Tuple[Decimal, Decimal]]:
        """Calculate fill price and quantity."""
        # Determine base fill price
        fill_price = await self._get_fill_price(order, conditions)
        if fill_price is None:
            return None
        
        # Determine fill quantity
        fill_quantity = await self._get_fill_quantity(order, conditions)
        if fill_quantity <= 0:
            return None
        
        return fill_price, fill_quantity
    
    async def _get_fill_price(self, order: Order, conditions: MarketConditions) -> Optional[Decimal]:
        """Get fill price based on order type and market conditions."""
        if order.order_type == OrderType.MARKET:
            # Use appropriate side of the spread
            if order.side == OrderSide.BUY:
                return conditions.ask
            else:
                return conditions.bid
        
        elif order.order_type == OrderType.LIMIT:
            limit_price = Decimal(str(order.price))
            if order.side == OrderSide.BUY:
                return min(conditions.ask, limit_price)
            else:
                return max(conditions.bid, limit_price)
        
        elif order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            # For stop orders, use market price as base
            if order.order_type == OrderType.STOP_LIMIT:
                limit_price = Decimal(str(order.price))
                if order.side == OrderSide.BUY:
                    return min(conditions.ask, limit_price)
                else:
                    return max(conditions.bid, limit_price)
            else:
                # Regular stop order
                if order.side == OrderSide.BUY:
                    return conditions.ask
                else:
                    return conditions.bid
        
        return None
    
    async def _get_fill_quantity(self, order: Order, conditions: MarketConditions) -> Decimal:
        """Get fill quantity based on market liquidity and participation limits."""
        order_quantity = Decimal(str(order.quantity))
        
        # Check participation rate limit
        if conditions.volume > 0:
            max_quantity = conditions.volume * self._max_participation_rate
            order_quantity = min(order_quantity, max_quantity)
        
        # Handle partial fills
        if self._partial_fill_enabled and random.random() < 0.2:  # 20% chance
            min_quantity = order_quantity * self._min_fill_ratio
            fill_ratio = Decimal(str(random.uniform(float(self._min_fill_ratio), 1.0)))
            return order_quantity * fill_ratio
        
        return order_quantity
    
    async def _calculate_slippage(
        self,
        order: Order,
        market_price: Decimal,
        conditions: MarketConditions
    ) -> Decimal:
        """Calculate slippage using the configured model."""
        market_data = {'conditions': conditions}
        return self._slippage_model.calculate_slippage(order, market_price, market_data)
    
    async def _calculate_commission(
        self,
        order: Order,
        fill_price: Decimal,
        fill_quantity: Decimal
    ) -> Decimal:
        """Calculate commission using the configured model."""
        return self._commission_model.calculate_commission(order, fill_price, fill_quantity)
    
    def _update_simulation_stats(self, fill: Fill) -> None:
        """Update simulation statistics."""
        if fill.fill_type == FillType.FULL:
            self._simulation_stats['successful_fills'] += 1
        else:
            self._simulation_stats['partial_fills'] += 1
        
        self._simulation_stats['total_slippage'] += Decimal(str(fill.slippage))
        self._simulation_stats['total_commission'] += Decimal(str(fill.commission))


# Factory functions

def create_market_simulator(
    component_id: str,
    slippage_model: str = "percentage",
    commission_model: str = "per_share",
    **kwargs
) -> ImprovedMarketSimulator:
    """Factory function to create market simulator with specific models."""
    # Create slippage model
    if slippage_model == "percentage":
        slippage = PercentageSlippageModel(**kwargs.get('slippage_params', {}))
    elif slippage_model == "volume_impact":
        slippage = VolumeImpactSlippageModel(**kwargs.get('slippage_params', {}))
    else:
        raise ValueError(f"Unknown slippage model: {slippage_model}")
    
    # Create commission model
    if commission_model == "per_share":
        commission = PerShareCommissionModel(**kwargs.get('commission_params', {}))
    elif commission_model == "tiered":
        commission = TieredCommissionModel(**kwargs.get('commission_params', {}))
    else:
        raise ValueError(f"Unknown commission model: {commission_model}")
    
    return ImprovedMarketSimulator(
        component_id=component_id,
        slippage_model=slippage,
        commission_model=commission,
        **kwargs.get('simulator_params', {})
    )


def create_conservative_simulator(component_id: str) -> ImprovedMarketSimulator:
    """Create conservative market simulator for backtesting."""
    return create_market_simulator(
        component_id=component_id,
        slippage_model="percentage",
        commission_model="per_share",
        slippage_params={'base_slippage_pct': Decimal("0.002")},  # 0.2%
        commission_params={'rate_per_share': Decimal("0.01")},  # $0.01/share
        simulator_params={
            'fill_probability': Decimal("0.98"),
            'partial_fill_enabled': False
        }
    )


def create_realistic_simulator(component_id: str) -> ImprovedMarketSimulator:
    """Create realistic market simulator with advanced models."""
    return create_market_simulator(
        component_id=component_id,
        slippage_model="volume_impact",
        commission_model="tiered",
        slippage_params={
            'permanent_impact_factor': Decimal("0.0001"),
            'temporary_impact_factor': Decimal("0.0002")
        },
        commission_params={
            'tiers': [
                (Decimal("0"), Decimal("0.003")),      # $0-1k: 0.3%
                (Decimal("1000"), Decimal("0.002")),   # $1k-10k: 0.2%
                (Decimal("10000"), Decimal("0.001"))   # $10k+: 0.1%
            ]
        },
        simulator_params={
            'fill_probability': Decimal("0.95"),
            'partial_fill_enabled': True,
            'min_fill_ratio': Decimal("0.3")
        }
    )
