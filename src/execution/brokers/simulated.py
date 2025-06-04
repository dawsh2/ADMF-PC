"""
Canonical simulated broker implementation for ADMF-PC.

This unified simulated broker handles backtesting and paper trading,
following Protocol + Composition principles with clean separation of concerns.
"""

from typing import Dict, Optional, Any, List, Protocol
from datetime import datetime
from decimal import Decimal
import uuid
from dataclasses import dataclass, field
import logging

from ...core.components.protocols import Component, Lifecycle
from ..protocols import (
    Broker, Order, Fill, Position, OrderStatus,
    OrderSide, OrderType, FillType, FillStatus
)
from .slippage import SlippageModel, MarketConditions, PercentageSlippageModel
from .commission import CommissionModel, PercentageCommissionModel
from .liquidity import LiquidityModel, BasicLiquidityModel

logger = logging.getLogger(__name__)


@dataclass
class OrderTracker:
    """Tracks orders and fills without duplicating position state."""
    orders: Dict[str, Order] = field(default_factory=dict)
    order_status: Dict[str, OrderStatus] = field(default_factory=dict)
    fills: List[Fill] = field(default_factory=list)
    
    def get_pending_orders(self) -> List[Order]:
        """Get all pending orders."""
        return [
            order for order_id, order in self.orders.items()
            if self.order_status.get(order_id) in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]
        ]
    
    def get_fills_for_symbol(self, symbol: str) -> List[Fill]:
        """Get all fills for a specific symbol."""
        return [fill for fill in self.fills if fill.symbol == symbol]


class PortfolioStateProtocol(Protocol):
    """Protocol for portfolio state dependency injection (duck typing)."""
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for symbol."""
        ...
    
    def get_cash_balance(self) -> Decimal:
        """Get current cash balance."""
        ...
    
    def get_equity(self) -> Decimal:
        """Get total portfolio equity."""
        ...
    
    def update_position(self, symbol: str, quantity: Decimal, price: Decimal) -> None:
        """Update position from fill."""
        ...
    
    def update_cash(self, amount: Decimal) -> None:
        """Update cash balance."""
        ...


class MarketDataProtocol(Protocol):
    """Protocol for market data provider dependency injection."""
    
    def get_market_price(self, symbol: str) -> Decimal:
        """Get current market price for symbol."""
        ...
    
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market data for symbol."""
        ...


class SimulatedBroker(Component, Lifecycle):
    """
    Canonical simulated broker implementation using Protocol + Composition.
    
    This simulated broker:
    - Delegates position tracking to Risk module's PortfolioState
    - Uses dependency injection for clean separation of concerns
    - Supports backtesting and paper trading modes
    - Uses Decimal precision for financial calculations
    - Follows Protocol + Composition architecture
    """
    
    def __init__(
        self,
        component_id: str = None,
        portfolio_state: Optional[PortfolioStateProtocol] = None,
        slippage_model: Optional[SlippageModel] = None,
        commission_model: Optional[CommissionModel] = None,
        liquidity_model: Optional[LiquidityModel] = None,
        market_data_provider: Optional[MarketDataProtocol] = None,
        mode: str = "backtest"
    ):
        """Initialize broker with dependency injection.
        
        Args:
            component_id: Unique component identifier
            portfolio_state: Portfolio state from Risk module (injected)
            slippage_model: Slippage calculation model (injected)
            commission_model: Commission calculation model (injected)
            liquidity_model: Liquidity and fill probability model (injected)
            market_data_provider: Market data provider (injected)
            mode: Operating mode ('backtest' or 'live')
        """
        self._component_id = component_id or f"broker_{uuid.uuid4().hex[:8]}"
        self._portfolio_state = portfolio_state
        self._slippage_model = slippage_model or PercentageSlippageModel()
        self._commission_model = commission_model or PercentageCommissionModel()
        self._liquidity_model = liquidity_model or BasicLiquidityModel()
        self._market_data_provider = market_data_provider
        self._mode = mode
        
        # Order tracking (separate from position state)
        self._order_tracker = OrderTracker()
        
        # Configuration
        self._is_running = False
        
        logger.info(
            f"Broker initialized in {mode} mode",
            component_id=self._component_id,
            slippage_model=type(self._slippage_model).__name__,
            commission_model=type(self._commission_model).__name__,
            liquidity_model=type(self._liquidity_model).__name__
        )
    
    @property
    def component_id(self) -> str:
        """Get component ID."""
        return self._component_id
    
    # Lifecycle methods
    def initialize(self) -> None:
        """Initialize broker."""
        if self._portfolio_state is None:
            raise ValueError("PortfolioState must be injected for broker to function")
        
        if self._mode == "backtest" and self._market_data_provider is None:
            logger.warning("No market data provider for backtest mode")
        
        logger.debug(f"Broker {self._component_id} initialized")
    
    def start(self) -> None:
        """Start broker."""
        self._is_running = True
        logger.info(f"Broker {self._component_id} started")
    
    def stop(self) -> None:
        """Stop broker."""
        self._is_running = False
        logger.info(f"Broker {self._component_id} stopped")
    
    # Broker protocol implementation
    def submit_order(self, order: Order) -> str:
        """Submit order for execution."""
        if not self._is_running:
            raise RuntimeError("Broker is not running")
        
        # Validate order
        validation_result = self._validate_order(order)
        if not validation_result["valid"]:
            logger.warning(
                f"Order validation failed: {validation_result['reason']}",
                order_id=order.order_id,
                symbol=order.symbol
            )
            raise ValueError(f"Order validation failed: {validation_result['reason']}")
        
        # Track order
        self._order_tracker.orders[order.order_id] = order
        self._order_tracker.order_status[order.order_id] = OrderStatus.PENDING
        
        logger.info(
            f"Order submitted",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side.value,
            quantity=float(order.quantity),
            mode=self._mode
        )
        
        # In backtest mode, try immediate execution
        if self._mode == "backtest" and self._market_data_provider:
            self._try_execute_order(order)
        
        return order.order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        if order_id not in self._order_tracker.orders:
            return False
        
        current_status = self._order_tracker.order_status.get(order_id)
        if current_status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            return False
        
        self._order_tracker.order_status[order_id] = OrderStatus.CANCELLED
        
        logger.info(f"Order cancelled", order_id=order_id)
        return True
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get status of order."""
        return self._order_tracker.order_status.get(order_id)
    
    def get_positions(self) -> Dict[str, Position]:
        """Get all positions (delegated to portfolio state)."""
        if not self._portfolio_state:
            return {}
        
        # Delegate to portfolio state - it's the single source of truth
        positions = {}
        # Note: This would depend on the specific PortfolioState interface
        # Using duck typing to avoid circular imports
        if hasattr(self._portfolio_state, 'get_all_positions'):
            positions = self._portfolio_state.get_all_positions()
        
        return positions
    
    def get_account_value(self) -> Decimal:
        """Get total account value (delegated to portfolio state)."""
        if not self._portfolio_state:
            return Decimal(0)
        
        return self._portfolio_state.get_equity()
    
    def get_cash_balance(self) -> Decimal:
        """Get cash balance (delegated to portfolio state)."""
        if not self._portfolio_state:
            return Decimal(0)
        
        return self._portfolio_state.get_cash_balance()
    
    # Backtest-specific methods
    def process_market_data(self, market_data: Dict[str, Any]) -> List[Fill]:
        """Process market data and execute pending orders (backtest mode)."""
        if self._mode != "backtest":
            return []
        
        fills = []
        pending_orders = self._order_tracker.get_pending_orders()
        
        for order in pending_orders:
            fill = self._try_execute_order(order, market_data)
            if fill:
                fills.append(fill)
        
        return fills
    
    def get_fills(self) -> List[Fill]:
        """Get all fills."""
        return self._order_tracker.fills.copy()
    
    def get_fills_for_symbol(self, symbol: str) -> List[Fill]:
        """Get fills for specific symbol."""
        return self._order_tracker.get_fills_for_symbol(symbol)
    
    # Private methods
    def _validate_order(self, order: Order) -> Dict[str, Any]:
        """Validate order against portfolio state."""
        if not self._portfolio_state:
            return {"valid": False, "reason": "No portfolio state available"}
        
        # Basic validation
        if order.quantity <= 0:
            return {"valid": False, "reason": "Quantity must be positive"}
        
        # Check buying power for buy orders
        if order.side == OrderSide.BUY:
            # Calculate estimated commission using commission model
            estimated_commission = self._commission_model.calculate_commission(
                order, Decimal(str(order.price)), Decimal(str(order.quantity))
            )
            cash_needed = Decimal(str(order.quantity)) * Decimal(str(order.price)) + estimated_commission
            available_cash = self._portfolio_state.get_cash_balance()
            
            if cash_needed > available_cash:
                return {
                    "valid": False, 
                    "reason": f"Insufficient cash: need {cash_needed}, have {available_cash}"
                }
        
        # Check position for sell orders
        elif order.side == OrderSide.SELL:
            current_position = self._portfolio_state.get_position(order.symbol)
            available_quantity = current_position.quantity if current_position else Decimal(0)
            
            if order.quantity > available_quantity:
                return {
                    "valid": False,
                    "reason": f"Insufficient position: need {order.quantity}, have {available_quantity}"
                }
        
        return {"valid": True, "reason": ""}
    
    def _try_execute_order(self, order: Order, market_data: Dict[str, Any] = None) -> Optional[Fill]:
        """Try to execute order using composable models."""
        # Get market data from provider or use provided data
        if market_data is None and self._market_data_provider:
            market_data = self._market_data_provider.get_market_data(order.symbol)
            market_price = self._market_data_provider.get_market_price(order.symbol)
        elif market_data:
            market_price = Decimal(str(market_data.get('price', order.price)))
        else:
            # Fallback to order price
            market_price = Decimal(str(order.price))
            market_data = {
                'price': float(market_price),
                'volume': 10000,  # Default volume
                'volatility': 0.02
            }
        
        # Create market conditions
        conditions = MarketConditions(
            price=market_price,
            volume=Decimal(str(market_data.get('volume', 10000))),
            bid=market_price * Decimal("0.9995"),  # Simple spread
            ask=market_price * Decimal("1.0005"),
            volatility=Decimal(str(market_data.get('volatility', 0.02))),
            liquidity_factor=Decimal(str(market_data.get('liquidity_factor', 1.0)))
        )
        
        # Check if order should fill using liquidity model
        if not self._liquidity_model.should_fill_order(order, conditions):
            return None
        
        # Calculate fill quantity
        fill_quantity = self._liquidity_model.calculate_fill_quantity(order, conditions)
        if fill_quantity <= 0:
            return None
        
        # Calculate base fill price (before slippage)
        if order.order_type == OrderType.MARKET:
            base_price = conditions.ask if order.side == OrderSide.BUY else conditions.bid
        else:
            base_price = Decimal(str(order.price))
        
        # Calculate slippage
        slippage = self._slippage_model.calculate_slippage(
            order, base_price, {'conditions': conditions}
        )
        
        # Final fill price with slippage
        fill_price = base_price + slippage
        
        # Calculate commission
        commission = self._commission_model.calculate_commission(
            order, fill_price, fill_quantity
        )
        
        # Create fill
        fill = Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=float(fill_quantity),
            price=float(fill_price),
            commission=commission,
            slippage=float(slippage),
            fill_type=(
                FillType.FULL if fill_quantity >= Decimal(str(order.quantity))
                else FillType.PARTIAL
            ),
            status=FillStatus.FILLED,
            executed_at=datetime.now(),
            metadata={
                'market_price': float(market_price),
                'base_price': float(base_price),
                'conditions': {
                    'bid': float(conditions.bid),
                    'ask': float(conditions.ask),
                    'volume': float(conditions.volume),
                    'volatility': float(conditions.volatility)
                }
            }
        )
        
        # Process the fill
        self._process_fill(fill)
        return fill
    
    def _process_fill(self, fill: Fill) -> None:
        """Process a fill by updating portfolio state and tracking."""
        # Update portfolio state (single source of truth)
        if self._portfolio_state:
            # Calculate net quantity (positive for buy, negative for sell)
            net_quantity = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity
            
            # Update position
            self._portfolio_state.update_position(fill.symbol, net_quantity, fill.price)
            
            # Update cash (subtract for buy, add for sell, account for commission)
            # Commission is already calculated in the fill
            trade_value = Decimal(str(fill.quantity)) * Decimal(str(fill.price))
            commission = fill.commission if isinstance(fill.commission, Decimal) else Decimal(str(fill.commission))
            cash_impact = -(trade_value + commission) if fill.side == OrderSide.BUY else (trade_value - commission)
            self._portfolio_state.update_cash(cash_impact)
        
        # Track the fill
        self._order_tracker.fills.append(fill)
        
        # Update order status
        order_id = fill.order_id
        if order_id in self._order_tracker.orders:
            original_order = self._order_tracker.orders[order_id]
            if fill.quantity >= original_order.quantity:
                self._order_tracker.order_status[order_id] = OrderStatus.FILLED
            else:
                self._order_tracker.order_status[order_id] = OrderStatus.PARTIALLY_FILLED
        
        logger.info(
            f"Fill processed",
            order_id=fill.order_id,
            symbol=fill.symbol,
            side=fill.side.value,
            quantity=float(fill.quantity),
            price=float(fill.price),
            commission=float(fill.commission) if self._portfolio_state else 0
        )


# Factory functions for creating simulated brokers (follows Protocol + Composition)

def create_simulated_broker(
    mode: str = "backtest",
    portfolio_state: Optional[PortfolioStateProtocol] = None,
    slippage_model: Optional[SlippageModel] = None,
    commission_model: Optional[CommissionModel] = None,
    liquidity_model: Optional[LiquidityModel] = None,
    market_data_provider: Optional[MarketDataProtocol] = None,
    component_id: Optional[str] = None
) -> SimulatedBroker:
    """
    Factory function for creating simulated broker instances.
    
    Args:
        mode: Operating mode ('backtest' or 'paper')
        portfolio_state: Portfolio state dependency
        slippage_model: Slippage calculation model
        commission_model: Commission calculation model
        liquidity_model: Liquidity and fill probability model
        market_data_provider: Market data provider
        component_id: Optional component ID
        
    Returns:
        Configured SimulatedBroker instance
    """
    return SimulatedBroker(
        component_id=component_id,
        portfolio_state=portfolio_state,
        slippage_model=slippage_model,
        commission_model=commission_model,
        liquidity_model=liquidity_model,
        market_data_provider=market_data_provider,
        mode=mode
    )


def create_zero_commission_broker(
    mode: str = "backtest",
    portfolio_state: Optional[PortfolioStateProtocol] = None,
    market_data_provider: Optional[MarketDataProtocol] = None,
    component_id: Optional[str] = None
) -> SimulatedBroker:
    """Create broker with zero commission (like Alpaca)."""
    from .commission import ZeroCommissionModel
    from .liquidity import create_liquid_market_model
    
    return create_simulated_broker(
        mode=mode,
        portfolio_state=portfolio_state,
        slippage_model=PercentageSlippageModel(base_slippage_pct=Decimal("0.0005")),
        commission_model=ZeroCommissionModel(),
        liquidity_model=create_liquid_market_model(),
        market_data_provider=market_data_provider,
        component_id=component_id
    )


def create_traditional_broker(
    mode: str = "backtest",
    portfolio_state: Optional[PortfolioStateProtocol] = None,
    market_data_provider: Optional[MarketDataProtocol] = None,
    component_id: Optional[str] = None
) -> SimulatedBroker:
    """Create broker with traditional per-share commission."""
    from .commission import PerShareCommissionModel
    
    return create_simulated_broker(
        mode=mode,
        portfolio_state=portfolio_state,
        slippage_model=PercentageSlippageModel(base_slippage_pct=Decimal("0.001")),
        commission_model=PerShareCommissionModel(
            rate_per_share=Decimal("0.01"),
            minimum_commission=Decimal("4.95")
        ),
        liquidity_model=BasicLiquidityModel(),
        market_data_provider=market_data_provider,
        component_id=component_id
    )


def create_conservative_broker(
    mode: str = "backtest",
    portfolio_state: Optional[PortfolioStateProtocol] = None,
    market_data_provider: Optional[MarketDataProtocol] = None,
    component_id: Optional[str] = None
) -> SimulatedBroker:
    """Create conservative broker for realistic backtesting."""
    from .slippage import VolumeImpactSlippageModel
    from .commission import TieredCommissionModel
    
    return create_simulated_broker(
        mode=mode,
        portfolio_state=portfolio_state,
        slippage_model=VolumeImpactSlippageModel(
            permanent_impact_factor=Decimal("0.0002"),
            temporary_impact_factor=Decimal("0.0003")
        ),
        commission_model=TieredCommissionModel(
            tiers=[
                (Decimal("0"), Decimal("0.002")),      # 0.2% base
                (Decimal("10000"), Decimal("0.001"))   # 0.1% for large trades
            ]
        ),
        liquidity_model=BasicLiquidityModel(fill_probability=Decimal("0.90")),
        market_data_provider=market_data_provider,
        component_id=component_id
    )