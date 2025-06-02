"""Signal processing pipeline for converting signals to orders."""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, Optional, List

import logging

from .protocols import (
    SignalProcessorProtocol,
    Signal,
    Order,
    OrderType,
    OrderSide,
    SignalType,
    PortfolioStateProtocol,
    PositionSizerProtocol,
    RiskLimitProtocol,
)


class SignalProcessor(SignalProcessorProtocol):
    """Process trading signals into executable orders."""
    
    def __init__(self):
        """Initialize signal processor."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._processed_signals = 0
        self._approved_orders = 0
        self._rejected_signals = 0
    
    def process_signal(
        self,
        signal: Signal,
        portfolio_state: PortfolioStateProtocol,
        position_sizer: PositionSizerProtocol,
        risk_limits: List[RiskLimitProtocol],
        market_data: Dict[str, Any]
    ) -> Optional[Order]:
        """Process signal into order.
        
        Pipeline:
        1. Validate signal
        2. Calculate position size
        3. Create order
        4. Check risk limits
        5. Return order or None
        
        Args:
            signal: Trading signal
            portfolio_state: Current portfolio state
            position_sizer: Position sizing strategy
            risk_limits: Risk limits to check
            market_data: Current market data
            
        Returns:
            Order if approved, None if vetoed
        """
        self._processed_signals += 1
        
        try:
            # 1. Validate signal
            if not self._validate_signal(signal, portfolio_state):
                self._rejected_signals += 1
                return None
            
            # 2. Calculate position size
            size = position_sizer.calculate_size(
                signal, portfolio_state, market_data
            )
            
            if size <= 0:
                self.logger.warning(
                    f"Signal rejected - Zero size - Signal: {signal}, Reason: Position size is zero or negative"
                )
                self._rejected_signals += 1
                return None
            
            # 3. Create order
            order = self._create_order(signal, size, market_data)
            
            # 4. Check risk limits
            risk_checks_passed = []
            for limit in risk_limits:
                passes, reason = limit.check_limit(order, portfolio_state, market_data)
                if not passes:
                    self.logger.warning(
                        f"Signal rejected - Risk limit - Signal: {signal}, Limit: {type(limit).__name__}, Reason: {reason}"
                    )
                    self._rejected_signals += 1
                    return None
                risk_checks_passed.append(type(limit).__name__)
            
            # Update order with passed checks
            order = Order(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                quantity=order.quantity,
                price=order.price,
                stop_price=order.stop_price,
                time_in_force=order.time_in_force,
                source_signal=order.source_signal,
                risk_checks_passed=risk_checks_passed,
                timestamp=order.timestamp,
                metadata=order.metadata
            )
            
            self._approved_orders += 1
            signal_type_str = signal.signal_type.value if hasattr(signal.signal_type, 'value') else str(signal.signal_type)
            self.logger.info(
                f"Signal processed - Type: {signal_type_str}, Symbol: {signal.symbol}, Order ID: {order.order_id}, Quantity: {order.quantity}, Risk checks: {len(risk_checks_passed)}"
            )
            
            return order
            
        except Exception as e:
            self.logger.error(
                f"Signal processing error - Signal: {signal}, Error: {str(e)}"
            )
            self._rejected_signals += 1
            return None
    
    def _validate_signal(
        self,
        signal: Signal,
        portfolio_state: PortfolioStateProtocol
    ) -> bool:
        """Validate signal before processing.
        
        Args:
            signal: Signal to validate
            portfolio_state: Current portfolio state
            
        Returns:
            True if valid, False otherwise
        """
        # Check signal strength
        if signal.strength == 0:
            self.logger.debug(f"Signal rejected - Zero strength - Signal: {signal}")
            return False
        
        # Check position logic
        position = portfolio_state.get_position(signal.symbol)
        
        # Exit signal but no position
        signal_type = signal.signal_type.value if hasattr(signal.signal_type, 'value') else signal.signal_type
        if signal_type in ["exit", "risk_exit"] and not position:
            self.logger.debug(
                f"Signal rejected - No position - Signal: {signal}, Reason: Exit signal but no position"
            )
            return False
        
        # Entry signal but already have position (unless rebalancing)
        if signal_type in ["entry", "entry_long", "entry_short"] and position and position.quantity != 0:
            # Allow if it's adding to position in same direction
            if (position.quantity > 0 and signal.side == OrderSide.BUY) or \
               (position.quantity < 0 and signal.side == OrderSide.SELL):
                return True
            else:
                self.logger.debug(
                    f"Signal rejected - Conflicting position - Signal: {signal}, Current position: {position.quantity}"
                )
                return False
        
        return True
    
    def _create_order(
        self,
        signal: Signal,
        size: Decimal,
        market_data: Dict[str, Any]
    ) -> Order:
        """Create order from signal and size.
        
        Args:
            signal: Source signal
            size: Position size
            market_data: Market data for pricing
            
        Returns:
            Created order
        """
        # Generate order ID
        order_id = f"ORD-{uuid.uuid4().hex[:8]}"
        
        # Determine order type and price
        order_type = OrderType.MARKET  # Default to market orders
        price = None
        stop_price = None
        
        # Get current market price
        current_price = market_data.get("prices", {}).get(signal.symbol)
        if current_price:
            current_price = Decimal(str(current_price))
            
            # Could implement limit order logic here based on signal metadata
            if signal.metadata.get("order_type") == "limit":
                order_type = OrderType.LIMIT
                # Place limit order slightly better than market
                if signal.side == OrderSide.BUY:
                    price = current_price * Decimal("0.999")  # 0.1% better
                else:
                    price = current_price * Decimal("1.001")
            
            # Stop loss orders for risk exits
            if signal.signal_type == SignalType.RISK_EXIT:
                order_type = OrderType.STOP
                stop_price = signal.metadata.get("stop_price")
                if stop_price:
                    stop_price = Decimal(str(stop_price))
        
        # Time in force
        time_in_force = signal.metadata.get("time_in_force", "GTC")
        
        # Create order
        return Order(
            order_id=order_id,
            symbol=signal.symbol,
            side=signal.side,
            order_type=order_type,
            quantity=size,
            price=price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            source_signal=signal,
            risk_checks_passed=[],  # Will be updated after risk checks
            timestamp=datetime.now(),
            metadata={
                "signal_strength": str(signal.strength),
                "signal_type": signal.signal_type.value if hasattr(signal.signal_type, 'value') else str(signal.signal_type),
                "strategy_id": signal.strategy_id,
                **signal.metadata
            }
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        approval_rate = 0
        if self._processed_signals > 0:
            approval_rate = self._approved_orders / self._processed_signals
        
        return {
            "processed_signals": self._processed_signals,
            "approved_orders": self._approved_orders,
            "rejected_signals": self._rejected_signals,
            "approval_rate": f"{approval_rate:.1%}"
        }


class SignalAggregator:
    """Aggregate signals from multiple strategies."""
    
    def __init__(self, aggregation_method: str = "weighted_average"):
        """Initialize signal aggregator.
        
        Args:
            aggregation_method: Method for combining signals
                - "weighted_average": Weight by signal strength
                - "majority_vote": Take majority direction
                - "unanimous": Require all signals to agree
                - "first": Take first signal only
        """
        self.aggregation_method = aggregation_method
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def aggregate_signals(
        self,
        signals: List[Signal],
        weights: Optional[Dict[str, Decimal]] = None
    ) -> List[Signal]:
        """Aggregate multiple signals for the same symbol.
        
        Args:
            signals: List of signals to aggregate
            weights: Optional strategy weights
            
        Returns:
            Aggregated signals
        """
        if not signals:
            return []
        
        # Group signals by symbol
        symbol_signals: Dict[str, List[Signal]] = {}
        for signal in signals:
            if signal.symbol not in symbol_signals:
                symbol_signals[signal.symbol] = []
            symbol_signals[signal.symbol].append(signal)
        
        # Aggregate each symbol
        aggregated = []
        for symbol, sig_list in symbol_signals.items():
            if len(sig_list) == 1:
                aggregated.append(sig_list[0])
            else:
                agg_signal = self._aggregate_symbol_signals(sig_list, weights)
                if agg_signal:
                    aggregated.append(agg_signal)
        
        return aggregated
    
    def _aggregate_symbol_signals(
        self,
        signals: List[Signal],
        weights: Optional[Dict[str, Decimal]] = None
    ) -> Optional[Signal]:
        """Aggregate signals for a single symbol.
        
        Args:
            signals: Signals for the same symbol
            weights: Optional strategy weights
            
        Returns:
            Aggregated signal or None
        """
        if self.aggregation_method == "first":
            return signals[0]
        
        elif self.aggregation_method == "weighted_average":
            # Calculate weighted average strength
            total_weight = Decimal(0)
            weighted_strength = Decimal(0)
            
            for signal in signals:
                weight = Decimal(1)
                if weights and signal.strategy_id in weights:
                    weight = weights[signal.strategy_id]
                
                total_weight += weight
                # Ensure both signal.strength and weight are Decimal for arithmetic
                signal_strength = Decimal(str(signal.strength)) if not isinstance(signal.strength, Decimal) else signal.strength
                weight_decimal = Decimal(str(weight)) if not isinstance(weight, Decimal) else weight
                weighted_strength += signal_strength * weight_decimal
            
            if total_weight == 0:
                return None
            
            avg_strength = weighted_strength / total_weight
            
            # Determine consensus side
            if avg_strength > 0:
                side = OrderSide.BUY
            elif avg_strength < 0:
                side = OrderSide.SELL
            else:
                return None  # No consensus
            
            # Use most recent signal as template
            template = max(signals, key=lambda s: s.timestamp)
            
            return Signal(
                signal_id=f"AGG-{uuid.uuid4().hex[:8]}",
                strategy_id="aggregated",
                symbol=template.symbol,
                signal_type=template.signal_type,
                side=side,
                strength=abs(avg_strength),
                timestamp=datetime.now(),
                metadata={
                    "aggregation_method": self.aggregation_method,
                    "source_strategies": [s.strategy_id for s in signals],
                    "original_strengths": {s.strategy_id: str(s.strength) for s in signals}
                }
            )
        
        elif self.aggregation_method == "majority_vote":
            # Count buy vs sell signals
            buy_count = sum(1 for s in signals if s.strength > 0)
            sell_count = sum(1 for s in signals if s.strength < 0)
            
            if buy_count > sell_count:
                side = OrderSide.BUY
                strength = Decimal(buy_count) / Decimal(len(signals))
            elif sell_count > buy_count:
                side = OrderSide.SELL
                strength = Decimal(sell_count) / Decimal(len(signals))
            else:
                return None  # Tie
            
            template = max(signals, key=lambda s: s.timestamp)
            
            return Signal(
                signal_id=f"AGG-{uuid.uuid4().hex[:8]}",
                strategy_id="aggregated",
                symbol=template.symbol,
                signal_type=template.signal_type,
                side=side,
                strength=strength,
                timestamp=datetime.now(),
                metadata={
                    "aggregation_method": self.aggregation_method,
                    "buy_votes": buy_count,
                    "sell_votes": sell_count,
                    "source_strategies": [s.strategy_id for s in signals]
                }
            )
        
        elif self.aggregation_method == "unanimous":
            # All signals must agree on direction
            first_side = OrderSide.BUY if signals[0].strength > 0 else OrderSide.SELL
            
            for signal in signals[1:]:
                signal_side = OrderSide.BUY if signal.strength > 0 else OrderSide.SELL
                if signal_side != first_side:
                    return None  # Not unanimous
            
            # Average strength of all signals - ensure all strengths are Decimal
            total_strength = sum(Decimal(str(abs(s.strength))) for s in signals)
            avg_strength = total_strength / Decimal(len(signals))
            
            template = max(signals, key=lambda s: s.timestamp)
            
            return Signal(
                signal_id=f"AGG-{uuid.uuid4().hex[:8]}",
                strategy_id="aggregated",
                symbol=template.symbol,
                signal_type=template.signal_type,
                side=first_side,
                strength=avg_strength,
                timestamp=datetime.now(),
                metadata={
                    "aggregation_method": self.aggregation_method,
                    "unanimous": True,
                    "source_strategies": [s.strategy_id for s in signals]
                }
            )
        
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")