"""Portfolio state tracking and management."""

from decimal import Decimal
from datetime import datetime
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass, field
import statistics
import logging

from .protocols import (
    PortfolioStateProtocol,
    PortfolioPosition as Position,
    RiskMetrics,
)
from ..core.events.types import Event
from ..execution.protocols import OrderProcessor

logger = logging.getLogger(__name__)


class PortfolioState(PortfolioStateProtocol):
    """Track and manage portfolio state."""
    
    def __init__(
        self,
        initial_capital: Decimal = Decimal("100000"),
        base_currency: str = "USD",
        order_manager: Optional[OrderProcessor] = None
    ):
        """Initialize portfolio state.
        
        Args:
            initial_capital: Starting capital
            base_currency: Base currency
            order_manager: Order manager for pending order queries (single source of truth)
        """
        self._initial_capital = initial_capital
        self._cash_balance = initial_capital
        self._base_currency = base_currency
        
        # Position tracking
        self._positions: Dict[str, Position] = {}
        
        # Order manager dependency injection - single source of truth for order state
        self._order_manager = order_manager
        
        # P&L tracking
        self._realized_pnl = Decimal(0)
        self._commission_paid = Decimal(0)
        
        # Order ID counter for uniqueness
        self._order_counter = 0
        
        # High water mark for drawdown
        self._high_water_mark = initial_capital
        self._max_drawdown = Decimal(0)
        
        # Historical values for risk calculations
        self._value_history: List[Decimal] = [initial_capital]
        self._returns_history: List[Decimal] = []
        
        # Last update timestamp
        self._last_update = datetime.now()
        
        # Track order fill commission
        self._last_commission = Decimal(0)
        
        # Container reference for event publishing
        self._container = None
        
        # Risk rules per strategy
        self._strategy_risk_rules: Dict[str, Dict[str, Any]] = {}
        
        # Exit memory: track signal state at time of risk exit
        # Key: (symbol, strategy_id), Value: signal value at exit
        self._exit_memory: Dict[Tuple[str, str], float] = {}
        
        # Exit memory configuration
        self._exit_memory_enabled = True  # Default enabled
        self._exit_memory_types = {'stop_loss', 'trailing_stop', 'take_profit'}  # Which exit types trigger memory
        
        # Track last signal values for each strategy
        self._last_signal_values: Dict[Tuple[str, str], float] = {}
        
        # Track last known prices for each symbol
        self._last_prices: Dict[str, Decimal] = {}
        
        # Risk manager reference (injected via set_risk_manager)
        self._risk_manager = None
        
        # Pending orders tracking (kept for backwards compatibility but delegated to order manager)
        self._pending_orders: Dict[str, Event] = {}
    
    def set_container(self, container: Any) -> None:
        """Set container reference for event publishing.
        
        Args:
            container: The container that owns this portfolio state
        """
        self._container = container
        bus_id = id(container.event_bus) if hasattr(container, 'event_bus') else 'no event bus'
        logger.info(f"Portfolio container reference set to: {container.name if hasattr(container, 'name') else container} with event bus {bus_id}")
    
    def set_risk_manager(self, risk_manager: Any) -> None:
        """Set risk manager reference for risk decisions.
        
        Args:
            risk_manager: The risk manager instance
        """
        self._risk_manager = risk_manager
        logger.info(f"Risk manager reference set: {type(risk_manager).__name__}")
    
    def set_strategy_risk_rules(self, strategy_id: str, risk_rules: Dict[str, Any]) -> None:
        """Set risk rules for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            risk_rules: Risk parameters (stop_loss, take_profit, trailing_stop, etc.)
        """
        self._strategy_risk_rules[strategy_id] = risk_rules
        logger.info(f"Set risk rules for {strategy_id}: {risk_rules}")
    
    def configure_exit_memory(self, enabled: bool = True, exit_types: Optional[set] = None) -> None:
        """Configure exit memory behavior.
        
        Args:
            enabled: Whether to enable exit memory (prevents re-entry until signal changes)
            exit_types: Set of exit types that trigger memory (default: {'stop_loss', 'trailing_stop', 'take_profit'})
        """
        self._exit_memory_enabled = enabled
        if exit_types is not None:
            self._exit_memory_types = exit_types
        logger.info(f"Exit memory configured: enabled={enabled}, types={self._exit_memory_types}")
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for symbol."""
        return self._positions.get(symbol)
    
    def can_create_order(self, symbol: str) -> bool:
        """Check if we should create a new order for this symbol.
        
        This prevents race conditions by ensuring we don't create multiple
        orders for the same symbol while one is pending.
        
        Args:
            symbol: The symbol to check
            
        Returns:
            True if no pending orders exist for this symbol
        """
        if not self._order_manager:
            # Fallback: if no order manager, allow order creation
            return True
            
        # Delegate to OrderManager - single source of truth
        pending_orders = self._order_manager.get_pending_orders()
        pending_for_symbol = [
            order_id for order_id in pending_orders
            if self._order_manager.get_order(order_id) and 
               self._order_manager.get_order(order_id).symbol == symbol
        ]
        return len(pending_for_symbol) == 0
    
    def add_pending_order(self, order_event: Event) -> None:
        """Add order to pending tracking - delegated to OrderManager.
        
        Args:
            order_event: Order event to track as pending
        """
        # Order tracking delegated to OrderManager - single source of truth
        # This method is kept for backward compatibility but delegates responsibility
        if self._order_manager:
            # OrderManager handles this internally when orders are submitted
            pass
    
    def remove_pending_order(self, order_id: str) -> Optional[Event]:
        """Remove order from pending tracking - delegated to OrderManager.
        
        Args:
            order_id: ID of order to remove
            
        Returns:
            None (OrderManager handles state internally)
        """
        # Order tracking delegated to OrderManager - single source of truth
        return None
    
    def get_pending_orders(self, symbol: Optional[str] = None) -> List[str]:
        """Get pending order IDs, optionally filtered by symbol.
        
        Args:
            symbol: Optional symbol to filter by
            
        Returns:
            List of pending order IDs
        """
        if not self._order_manager:
            return []
            
        # Delegate to OrderManager - single source of truth
        pending_orders = self._order_manager.get_pending_orders()
        
        if symbol is None:
            return list(pending_orders)
        else:
            # Filter by symbol
            return [
                order_id for order_id in pending_orders
                if self._order_manager.get_order(order_id) and 
                   self._order_manager.get_order(order_id).symbol == symbol
            ]
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all current positions."""
        return self._positions.copy()
    
    def get_cash_balance(self) -> Decimal:
        """Get current cash balance."""
        return self._cash_balance
    
    def get_total_value(self) -> Decimal:
        """Get total portfolio value."""
        positions_value = sum(
            pos.market_value for pos in self._positions.values()
        )
        return self._cash_balance + positions_value
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics."""
        # Calculate current values
        total_value = self.get_total_value()
        positions_value = sum(
            pos.market_value for pos in self._positions.values()
        )
        unrealized_pnl = sum(
            pos.unrealized_pnl for pos in self._positions.values()
        )
        
        # Update high water mark and drawdown
        if total_value > self._high_water_mark:
            self._high_water_mark = total_value
        
        current_drawdown = Decimal(0)
        if self._high_water_mark > 0:
            current_drawdown = (self._high_water_mark - total_value) / self._high_water_mark
        
        if current_drawdown > self._max_drawdown:
            self._max_drawdown = current_drawdown
        
        # Calculate Sharpe ratio if we have enough history
        sharpe_ratio = None
        if len(self._returns_history) >= 20:  # Need reasonable sample
            try:
                mean_return = statistics.mean(self._returns_history)
                std_return = statistics.stdev(self._returns_history)
                if std_return > 0:
                    # Annualized Sharpe (assuming daily returns)
                    sharpe_ratio = Decimal(str(mean_return / std_return * 252 ** 0.5))
            except:
                pass
        
        # Calculate simple VaR (would need more sophisticated calculation)
        var_95 = None
        if len(self._returns_history) >= 20:
            sorted_returns = sorted(self._returns_history)
            percentile_index = int(len(sorted_returns) * 0.05)
            var_95 = abs(sorted_returns[percentile_index]) * total_value
        
        # Calculate leverage
        leverage = positions_value / total_value if total_value > 0 else Decimal(0)
        
        # Calculate concentration
        concentration = {}
        if total_value > 0:
            for symbol, pos in self._positions.items():
                concentration[symbol] = pos.market_value / total_value
        
        return RiskMetrics(
            total_value=total_value,
            cash_balance=self._cash_balance,
            positions_value=positions_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=self._realized_pnl,
            max_drawdown=self._max_drawdown,
            current_drawdown=current_drawdown,
            sharpe_ratio=sharpe_ratio,
            var_95=var_95,
            leverage=leverage,
            concentration=concentration,
            timestamp=datetime.now()
        )
    
    def update_position(
        self,
        symbol: str,
        quantity_delta: Decimal,
        price: Decimal,
        timestamp: datetime,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Position:
        """Update position with a trade.
        
        Args:
            symbol: Symbol traded
            quantity_delta: Change in quantity (+ for buy, - for sell)
            price: Execution price
            timestamp: Trade timestamp
            metadata: Optional metadata to merge into position metadata
            
        Returns:
            Updated position
        """
        logger.info(f"[update_position] Starting for {symbol}: delta={quantity_delta}, price={price}, time={timestamp}")
        
        # Get or create position
        position = self._positions.get(symbol)
        
        if position is None:
            logger.info(f"  📍 Creating new position for {symbol}")
            # New position
            position = Position(
                symbol=symbol,
                quantity=quantity_delta,
                average_price=price,
                current_price=price,
                unrealized_pnl=Decimal(0),
                realized_pnl=Decimal(0),
                opened_at=timestamp,
                last_updated=timestamp,
                metadata={
                    'bars_held': 0,
                    'highest_price': str(price),  # For trailing stop (stored as string)
                }
            )
            
            # Merge any provided metadata
            if metadata:
                position.metadata.update(metadata)
                logger.info(f"  📝 Merged metadata: {metadata}")
            
            self._positions[symbol] = position
            logger.info(f"  📍 Position created and stored: {position}")
            
            # Emit POSITION_OPEN event
            if self._container:
                logger.info(f"  📍 Container reference exists: {self._container.name if hasattr(self._container, 'name') else self._container}")
                from ..core.events.types import Event, EventType
                position_event = Event(
                    event_type=EventType.POSITION_OPEN.value,
                    timestamp=timestamp,
                    payload={
                        'symbol': symbol,
                        'quantity': float(quantity_delta),
                        'entry_price': float(price),
                        'strategy_id': position.metadata.get('strategy_id', 'unknown'),
                        'metadata': position.metadata
                    },
                    source_id="portfolio",
                    container_id=self._container.container_id
                )
                logger.info(f"  📍 Publishing POSITION_OPEN event to bus {id(self._container.event_bus)}")
                self._container.publish_event(position_event, target_scope="parent")
                logger.info(f"✅ Published POSITION_OPEN event for {symbol} qty={quantity_delta} @ ${price}")
            else:
                logger.warning(f"❌ Cannot publish POSITION_OPEN event - no container reference set!")
        else:
            # Update existing position
            old_quantity = position.quantity
            new_quantity = old_quantity + quantity_delta
            
            logger.info(f"  📊 Position update: old_qty={old_quantity}, delta={quantity_delta}, new_qty={new_quantity}")
            logger.info(f"  📊 Types: old_qty={type(old_quantity)}, delta={type(quantity_delta)}, new_qty={type(new_quantity)}")
            logger.info(f"  📊 New quantity == 0? {new_quantity == 0}, abs(new_quantity) < 0.0001? {abs(float(new_quantity)) < 0.0001}")
            
            if new_quantity == 0:
                # Position closed
                # Get entry signal from metadata to handle LONG/SHORT correctly
                entry_signal = position.metadata.get('entry_signal', 1 if old_quantity > 0 else -1)
                # Use signal-based formula: signal * (exit - entry) * abs(quantity)
                # Convert all to Decimal to avoid type mismatch
                realized = Decimal(str(entry_signal)) * (price - position.average_price) * abs(old_quantity)
                self._realized_pnl += realized
                
                # Emit POSITION_CLOSE event before deleting
                if self._container:
                    from ..core.events.types import Event, EventType
                    close_event = Event(
                        event_type=EventType.POSITION_CLOSE.value,
                        timestamp=timestamp,
                        payload={
                            'symbol': symbol,
                            'quantity': float(old_quantity),
                            'entry_price': float(position.average_price),
                            'exit_price': float(price),
                            'realized_pnl': float(realized),
                            'strategy_id': position.metadata.get('strategy_id', 'unknown'),
                            'exit_type': position.metadata.get('exit_type'),
                            'exit_reason': position.metadata.get('exit_reason'),
                            'metadata': position.metadata
                        },
                        source_id="portfolio",
                        container_id=self._container.container_id
                    )
                    self._container.publish_event(close_event, target_scope="parent")
                    logger.info(f"✅ Published POSITION_CLOSE event for {symbol} pnl=${realized:.2f}")
                else:
                    logger.warning(f"❌ Cannot publish POSITION_CLOSE event - no container reference set!")
                
                del self._positions[symbol]
                
                # Update cash balance for the closing trade BEFORE returning
                cash_delta = -quantity_delta * price
                self._cash_balance += cash_delta
                
                # Update value history
                self._update_value_history()
                
                # Return closed position for reference
                position.quantity = Decimal(0)
                position.realized_pnl += realized
                position.unrealized_pnl = Decimal(0)
                position.last_updated = timestamp
                return position
            
            elif (old_quantity > 0 and quantity_delta > 0) or \
                 (old_quantity < 0 and quantity_delta < 0):
                # Adding to position
                total_cost = position.average_price * old_quantity + price * quantity_delta
                position.average_price = total_cost / new_quantity
                position.quantity = new_quantity
            
            else:
                # Partial close
                if abs(quantity_delta) > abs(old_quantity):
                    # Flipping position - need to emit CLOSE for old position and OPEN for new
                    # Get entry signal from metadata to handle LONG/SHORT correctly
                    entry_signal = position.metadata.get('entry_signal', 1 if old_quantity > 0 else -1)
                    # Use signal-based formula: signal * (exit - entry) * abs(quantity)
                    realized = Decimal(str(entry_signal)) * (price - position.average_price) * abs(old_quantity)
                    self._realized_pnl += realized
                    position.realized_pnl += realized
                    
                    # Emit POSITION_CLOSE event for the old position
                    if self._container:
                        from ..core.events.types import Event, EventType
                        close_event = Event(
                            event_type=EventType.POSITION_CLOSE.value,
                            timestamp=timestamp,
                            payload={
                                'symbol': symbol,
                                'quantity': float(old_quantity),
                                'entry_price': float(position.average_price),
                                'exit_price': float(price),
                                'realized_pnl': float(realized),
                                'strategy_id': position.metadata.get('strategy_id', 'unknown'),
                                'exit_type': position.metadata.get('exit_type'),
                                'exit_reason': position.metadata.get('exit_reason'),
                                'metadata': position.metadata
                            },
                            source_id="portfolio",
                            container_id=self._container.container_id
                        )
                        self._container.publish_event(close_event, target_scope="parent")
                        logger.info(f"✅ Published POSITION_CLOSE event for flipped position {symbol} old_qty={old_quantity} pnl=${realized:.2f}")
                    
                    # New position in opposite direction
                    position.quantity = new_quantity
                    position.average_price = price
                    position.opened_at = timestamp
                    position.metadata = {'bars_held': 0, 'highest_price': str(price)}  # Reset metadata for new position
                    
                    # Emit POSITION_OPEN event for the new position
                    if self._container:
                        from ..core.events.types import Event, EventType
                        open_event = Event(
                            event_type=EventType.POSITION_OPEN.value,
                            timestamp=timestamp,
                            payload={
                                'symbol': symbol,
                                'quantity': float(new_quantity),
                                'entry_price': float(price),
                                'strategy_id': position.metadata.get('strategy_id', 'unknown'),
                                'metadata': position.metadata
                            },
                            source_id="portfolio",
                            container_id=self._container.container_id
                        )
                        self._container.publish_event(open_event, target_scope="parent")
                        logger.info(f"✅ Published POSITION_OPEN event for flipped position {symbol} new_qty={new_quantity} @ ${price}")
                else:
                    # Reducing position
                    # Get entry signal from metadata to handle LONG/SHORT correctly
                    entry_signal = position.metadata.get('entry_signal', 1 if old_quantity > 0 else -1)
                    # Use signal-based formula: signal * (exit - entry) * abs(quantity)
                    realized = Decimal(str(entry_signal)) * (price - position.average_price) * abs(quantity_delta)
                    self._realized_pnl += realized
                    position.realized_pnl += realized
                    position.quantity = new_quantity
            
            # Update unrealized P&L
            position.unrealized_pnl = (position.current_price - position.average_price) * position.quantity
            position.last_updated = timestamp
        
        # Update cash balance - ensure all values are Decimal
        cash_delta = -quantity_delta * price  # Both should already be Decimal
        self._cash_balance += cash_delta
        
        # Apply commission if set (would be set by fill event handler)
        if self._last_commission > 0:
            self._cash_balance -= self._last_commission
            self._commission_paid += self._last_commission
            self._last_commission = Decimal(0)
        
        # Update value history
        self._update_value_history()
        
        return position
    
    def update_market_prices(self, prices: Dict[str, Decimal]) -> None:
        """Update market prices for positions.
        
        Args:
            prices: Dictionary of symbol -> current price
        """
        for symbol, price in prices.items():
            if symbol in self._positions:
                position = self._positions[symbol]
                position.current_price = price
                position.unrealized_pnl = (price - position.average_price) * position.quantity
                position.last_updated = datetime.now()
        
        # Update value history
        self._update_value_history()
    
    def _update_value_history(self) -> None:
        """Update value history for risk calculations."""
        current_value = self.get_total_value()
        self._value_history.append(current_value)
        
        # Calculate return if we have previous value
        if len(self._value_history) >= 2:
            prev_value = self._value_history[-2]
            if prev_value > 0:
                daily_return = (current_value - prev_value) / prev_value
                self._returns_history.append(daily_return)
        
        # Keep only recent history (e.g., 252 trading days)
        max_history = 252
        if len(self._value_history) > max_history:
            self._value_history = self._value_history[-max_history:]
        if len(self._returns_history) > max_history:
            self._returns_history = self._returns_history[-max_history:]
        
        self._last_update = datetime.now()
    
    def process_event(self, event: Event) -> None:
        """Process an event (fill, market data, etc)."""
        if event.event_type == "FILL":
            # Handle fill event
            payload = event.payload
            symbol = payload["symbol"]
            side = payload["side"]
            quantity = Decimal(str(payload["quantity"]))
            price = Decimal(str(payload["price"]))
            
            # Convert side to quantity delta
            if side.upper() == "BUY":
                quantity_delta = quantity
            else:
                quantity_delta = -quantity
            
            # Get strategy_id from payload metadata
            position_metadata = {}
            if 'metadata' in payload:
                strategy_id = payload['metadata'].get('strategy_id')
                if strategy_id:
                    position_metadata['strategy_id'] = strategy_id
            
            # Update position
            position = self.update_position(symbol, quantity_delta, price, event.timestamp, metadata=position_metadata)
            
            # Remove pending order if this was from an order
            order_id = payload.get("order_id")
            if order_id:
                self.remove_pending_order(order_id)
                
        elif event.event_type == "SIGNAL":
            # Handle signal event through risk manager
            payload = event.payload
            symbol = payload.get("symbol")
            direction = payload.get("direction")
            strength = payload.get("strength")
            strategy_id = payload.get("strategy_id")
            
            # Log signal reception for debugging
            logger.info(f"📨 Portfolio received SIGNAL: {symbol} {direction} strength={strength:.2f} from {strategy_id}")
            
            # Extract price from signal metadata
            price = None
            if 'price' in payload:
                price = Decimal(str(payload['price']))
            elif 'metadata' in payload and 'price' in payload['metadata']:
                price = Decimal(str(payload['metadata']['price']))
            
            if price:
                # Update last known price for this symbol
                self._last_prices[symbol] = price
                logger.debug(f"  💰 Updated last price for {symbol}: ${price}")
            else:
                logger.warning(f"  ⚠️ Signal missing price information: {payload}")
                return
            
            # Check if risk manager is available
            if not self._risk_manager:
                logger.warning("  ⚠️ No risk manager configured, cannot process signal")
                return
            
            # Store strategy risk rules if provided
            risk_rules = None
            if 'risk' in payload:
                risk_rules = payload['risk']
            elif 'metadata' in payload and 'parameters' in payload['metadata']:
                params = payload['metadata']['parameters']
                if '_risk' in params:
                    risk_rules = params['_risk']
            
            if risk_rules and strategy_id:
                self.set_strategy_risk_rules(strategy_id, risk_rules)
                logger.debug(f"  💰 Updated risk rules for {strategy_id}: {risk_rules}")
            
            # Track signal value for exit memory
            from ..strategy.types import SignalDirection
            
            # Map direction to numeric value for exit memory
            direction_value = 0.0
            if direction in [SignalDirection.LONG, "LONG"]:
                direction_value = 1.0
            elif direction in [SignalDirection.SHORT, "SHORT"]:
                direction_value = -1.0
            
            # Use strength if provided, otherwise use direction mapping
            if strength is not None:
                direction_value = float(strength)
            
            # Track this signal value
            base_strategy_id = strategy_id.replace("_exit", "") if strategy_id else strategy_id
            signal_key = (symbol, base_strategy_id)
            self._last_signal_values[signal_key] = direction_value
            
            # Prepare portfolio state for risk manager
            portfolio_state = {
                'positions': self._positions,
                'cash': self._cash_balance,
                'pending_orders': self._pending_orders,
                'last_prices': self._last_prices,
                'strategy_risk_rules': self._strategy_risk_rules,
                'initial_capital': self._initial_capital,
                'exit_memory': self._exit_memory,
                'exit_memory_enabled': self._exit_memory_enabled,
                'exit_memory_types': self._exit_memory_types,
                'last_signal_values': self._last_signal_values
            }
            
            # Call risk manager to evaluate signal and all positions
            logger.debug(f"  🎯 Calling risk manager to evaluate signal and check all positions")
            try:
                decisions = self._risk_manager.evaluate_signal(
                    signal=payload,
                    portfolio_state=portfolio_state,
                    timestamp=event.timestamp
                )
                
                logger.info(f"  📊 Risk manager returned {len(decisions)} decisions")
                
                # Execute each decision
                for i, decision in enumerate(decisions):
                    logger.debug(f"  Decision {i+1}: {decision.get('action')} for {decision.get('symbol')}")
                    
                    if decision['action'] == 'create_order':
                        self._create_order_from_decision(decision, timestamp=event.timestamp)
                    elif decision['action'] == 'update_metadata':
                        # Update position metadata (e.g., highest_price for trailing stops)
                        symbol = decision['symbol']
                        if symbol in self._positions:
                            self._positions[symbol].metadata.update(decision['updates'])
                            logger.debug(f"  📝 Updated metadata for {symbol}: {decision['updates']}")
                    elif decision['action'] == 'update_exit_memory':
                        # Store exit memory when risk exit occurs
                        memory_key = (decision['symbol'], decision['strategy_id'])
                        self._exit_memory[memory_key] = decision['signal_value']
                        logger.info(f"  💾 Stored exit memory: {memory_key} -> {decision['signal_value']}")
                    elif decision['action'] == 'clear_exit_memory':
                        # Clear exit memory when signal changes
                        memory_key = (decision['symbol'], decision['strategy_id'])
                        if memory_key in self._exit_memory:
                            del self._exit_memory[memory_key]
                            logger.info(f"  🗑️ Cleared exit memory for {memory_key}")
                            
            except Exception as e:
                logger.error(f"  ❌ Risk manager evaluation failed: {e}", exc_info=True)
            
    
    def on_fill(self, event: Any) -> None:
        """Handle FILL event from execution engine.
        
        Updates portfolio positions based on order fills.
        """
        logger.info(f"🎯 Portfolio on_fill called with event type: {type(event).__name__}")
        
        from ..core.events.types import Event
        
        if not isinstance(event, Event):
            logger.warning(f"⚠️ on_fill received non-Event object: {type(event)}")
            return
            
        if event.event_type != "FILL":
            logger.warning(f"⚠️ on_fill received wrong event type: {event.event_type}")
            return
        
        payload = event.payload
        logger.info(f"💰 Portfolio received FILL: {payload.get('side')} {payload.get('quantity')} {payload.get('symbol')} @ ${payload.get('price')}")
        
        # Convert to Fill object
        from ..execution.types import Fill, FillStatus, OrderSide
        
        # Handle side - could be lowercase or uppercase
        side_str = payload['side'].upper() if isinstance(payload['side'], str) else payload['side']
        
        # Handle status - could be lowercase or uppercase
        status_str = payload['status'].upper() if isinstance(payload['status'], str) else payload['status']
        
        fill = Fill(
            fill_id=payload['fill_id'],
            order_id=payload['order_id'],
            symbol=payload['symbol'],
            side=OrderSide[side_str],
            quantity=Decimal(str(payload['quantity'])),
            price=Decimal(str(payload['price'])),
            commission=Decimal(str(payload['commission'])),
            status=FillStatus[status_str],
            executed_at=datetime.fromisoformat(payload.get('filled_at', payload.get('executed_at')))
        )
        
        # Update position using the centralized method that emits events
        symbol = fill.symbol
        quantity = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity
        
        logger.info(f"  📊 Updating position for {symbol}: quantity_delta={quantity}")
        
        # Set commission for update_position to apply
        self._last_commission = fill.commission
        
        # Get metadata from fill if available
        metadata = payload.get('metadata', {})
        strategy_id = metadata.get('strategy_id')
        exit_type = metadata.get('exit_type')
        exit_reason = metadata.get('exit_reason')
        
        # If this is an exit fill, update position metadata BEFORE closing
        if exit_type or exit_reason:
            existing_position = self._positions.get(symbol)
            if existing_position:
                existing_position.metadata['exit_type'] = exit_type
                existing_position.metadata['exit_reason'] = exit_reason
                logger.info(f"  🏷️ Exit metadata: type={exit_type}, reason={exit_reason}")
        
        # Prepare metadata to pass to update_position
        position_metadata = {}
        if strategy_id:
            position_metadata['strategy_id'] = strategy_id
        
        # Pass through entry_signal if it exists
        if 'entry_signal' in metadata:
            position_metadata['entry_signal'] = metadata['entry_signal']
            logger.info(f"  📝 Propagating entry_signal: {metadata['entry_signal']}")
        
        # Update position - this will emit POSITION_OPEN/CLOSE events
        position = self.update_position(symbol, quantity, fill.price, fill.executed_at, metadata=position_metadata)
        logger.info(f"  📊 Position updated: {position}")
        
        # Note: Cash balance and commission already updated by update_position
        
        # Update high water mark
        total_value = self.get_total_value()
        if total_value > self._high_water_mark:
            self._high_water_mark = total_value
        
        logger.info(f"  💳 Cash balance: ${self._cash_balance:.2f}")
        logger.info(f"  📈 Total portfolio value: ${total_value:.2f}")
    
    def _create_order_from_decision(self, decision: Dict[str, Any], timestamp: datetime = None) -> None:
        """Create an order based on risk manager decision.
        
        Args:
            decision: Risk decision containing order details
            timestamp: Bar timestamp (if None, uses current time)
        """
        from ..execution.types import Order, OrderSide, OrderType
        
        # Extract order details
        symbol = decision['symbol']
        side = OrderSide[decision['side']]
        quantity = Decimal(str(decision['quantity']))
        order_type = OrderType[decision.get('order_type', 'MARKET')]
        price = Decimal(str(decision.get('price', 0)))
        
        # Create unique order ID
        self._order_counter += 1
        # Use bar timestamp if available, otherwise current time
        order_timestamp = timestamp if timestamp else datetime.now()
        order_id = f"ORD_{symbol}_{order_timestamp.strftime('%Y%m%d_%H%M%S')}_{self._order_counter:06d}"
        
        # Build metadata
        metadata = {'strategy_id': decision.get('strategy_id')}
        if 'exit_type' in decision:
            metadata['exit_type'] = decision['exit_type']
        if 'exit_reason' in decision:
            metadata['exit_reason'] = decision['exit_reason']
        # Add bar timestamp to metadata so execution can use it
        if timestamp:
            metadata['bar_timestamp'] = timestamp.isoformat()
        
        
        # DECISION_DEBUG - Temporary logging
        logger.info(f"[DECISION_DEBUG] Processing decision: type={decision.get('type')}, action={decision.get('action')}")
        logger.info(f"[DECISION_DEBUG] Full decision: {decision}")
        
        # For entry orders, store the current signal value
        # This will be used for exit memory to prevent re-entry on same signal
        if decision.get('type') == 'entry':
            # Derive signal value from order side
            # BUY = Long = 1.0, SELL = Short = -1.0
            side_str = decision.get('side', '').upper()
            if side_str == 'BUY':
                entry_signal = 1.0
            elif side_str == 'SELL':
                entry_signal = -1.0
            else:
                # Fallback to last known signal
                signal_key = (symbol, decision.get('strategy_id', 'unknown'))
                entry_signal = self._last_signal_values.get(signal_key, 0.0)
            
            metadata['entry_signal'] = entry_signal
            logger.info(f"  💾 Storing entry signal in order metadata: {entry_signal} (side: {side_str})")
        
        # Create order
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price,
            created_at=order_timestamp,
            metadata=metadata
        )
        
        logger.info(f"  📤 Creating {decision.get('type', '')} ORDER: {order.side} {order.quantity} {order.symbol} @ ${price} {order_type}")
        logger.info(f"  📝 Order metadata: {metadata}")
        
        # Publish ORDER event
        if self._container:
            from ..core.events.types import Event, EventType
            order_event = Event(
                event_type=EventType.ORDER.value,
                timestamp=order_timestamp,
                payload=order.to_dict(),
                source_id="portfolio",
                container_id=self._container.container_id
            )
            self._container.publish_event(order_event, target_scope="parent")
            logger.info(f"  ✅ Order published to execution engine")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        metrics = self.get_risk_metrics()
        initial = self._initial_capital
        current = metrics.total_value
        
        total_return = (current - initial) / initial if initial > 0 else Decimal(0)
        
        return {
            "initial_capital": str(initial),
            "current_value": str(current),
            "total_return": f"{total_return:.2%}",
            "realized_pnl": str(metrics.realized_pnl),
            "unrealized_pnl": str(metrics.unrealized_pnl),
            "commission_paid": str(self._commission_paid),
            "max_drawdown": f"{metrics.max_drawdown:.2%}",
            "current_drawdown": f"{metrics.current_drawdown:.2%}",
            "sharpe_ratio": str(metrics.sharpe_ratio) if metrics.sharpe_ratio else "N/A",
            "positions_count": len(self._positions),
            "pending_orders_count": len(self._pending_orders),
            "last_update": self._last_update.isoformat()
        }
