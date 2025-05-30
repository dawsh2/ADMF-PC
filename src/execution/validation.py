"""
Enhanced validation infrastructure for the execution module.

This module provides validation classes and utilities that integrate
with the core infrastructure validation system.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
import logging

from ..core.infrastructure.validation import ValidationResult as CoreValidationResult
from .protocols import Order, Fill, OrderSide, OrderType, OrderStatus

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Enhanced validation result for execution module."""
    is_valid: bool
    reason: str
    details: Optional[Dict[str, Any]] = None
    severity: str = "error"  # error, warning, info
    
    def __post_init__(self):
        """Initialize details if not provided."""
        if self.details is None:
            self.details = {}
    
    @classmethod
    def success(cls, reason: str = "Valid") -> 'ValidationResult':
        """Create successful validation result."""
        return cls(True, reason, severity="info")
    
    @classmethod
    def failure(cls, reason: str, details: Optional[Dict[str, Any]] = None) -> 'ValidationResult':
        """Create failed validation result."""
        return cls(False, reason, details, severity="error")
    
    @classmethod
    def warning(cls, reason: str, details: Optional[Dict[str, Any]] = None) -> 'ValidationResult':
        """Create warning validation result."""
        return cls(True, reason, details, severity="warning")


class OrderValidator:
    """Comprehensive order validation."""
    
    def __init__(self):
        """Initialize order validator."""
        self.validation_rules = {
            'basic': self._validate_basic_order,
            'price': self._validate_price_fields,
            'quantity': self._validate_quantity,
            'symbol': self._validate_symbol,
            'order_type': self._validate_order_type_constraints,
            'timestamps': self._validate_timestamps
        }
    
    def validate_order(
        self,
        order: Order,
        rules: Optional[List[str]] = None,
        market_data: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Validate order with specified rules.
        
        Args:
            order: Order to validate
            rules: List of validation rules to apply (None for all)
            market_data: Market data for price validation
            
        Returns:
            ValidationResult with validation outcome
        """
        if rules is None:
            rules = list(self.validation_rules.keys())
        
        # Run all validation rules
        results = []
        for rule_name in rules:
            if rule_name in self.validation_rules:
                try:
                    result = self.validation_rules[rule_name](order, market_data)
                    results.append((rule_name, result))
                except Exception as e:
                    logger.error(f"Validation rule {rule_name} failed: {e}")
                    results.append((rule_name, ValidationResult.failure(
                        f"Validation rule error: {e}",
                        {"rule": rule_name, "exception": str(e)}
                    )))
        
        # Combine results
        return self._combine_validation_results(results)
    
    def _validate_basic_order(self, order: Order, market_data: Optional[Dict[str, Any]]) -> ValidationResult:
        """Basic order validation."""
        if not order.order_id:
            return ValidationResult.failure("Missing order ID")
        
        if not order.symbol:
            return ValidationResult.failure("Missing symbol")
        
        if not isinstance(order.side, OrderSide):
            return ValidationResult.failure("Invalid order side")
        
        if not isinstance(order.order_type, OrderType):
            return ValidationResult.failure("Invalid order type")
        
        return ValidationResult.success("Basic validation passed")
    
    def _validate_price_fields(self, order: Order, market_data: Optional[Dict[str, Any]]) -> ValidationResult:
        """Validate price-related fields."""
        # Limit orders require price
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if order.price is None:
                return ValidationResult.failure(
                    f"{order.order_type.name} order requires price",
                    {"order_type": order.order_type.name}
                )
            if order.price <= 0:
                return ValidationResult.failure(
                    f"Invalid price: {order.price}",
                    {"price": order.price}
                )
        
        # Stop orders require stop price
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if order.stop_price is None:
                return ValidationResult.failure(
                    f"{order.order_type.name} order requires stop price",
                    {"order_type": order.order_type.name}
                )
            if order.stop_price <= 0:
                return ValidationResult.failure(
                    f"Invalid stop price: {order.stop_price}",
                    {"stop_price": order.stop_price}
                )
        
        # Stop-limit price relationship validation
        if order.order_type == OrderType.STOP_LIMIT:
            if order.side == OrderSide.BUY:
                if order.stop_price > order.price:
                    return ValidationResult.failure(
                        "Buy stop-limit: stop price must be <= limit price",
                        {"stop_price": order.stop_price, "limit_price": order.price}
                    )
            else:  # SELL
                if order.stop_price < order.price:
                    return ValidationResult.failure(
                        "Sell stop-limit: stop price must be >= limit price",
                        {"stop_price": order.stop_price, "limit_price": order.price}
                    )
        
        # Market data price validation
        if market_data:
            market_price = market_data.get('price')
            if market_price and order.price:
                price_diff = abs(order.price - market_price) / market_price
                if price_diff > 0.5:  # 50% from market price
                    return ValidationResult.warning(
                        f"Order price significantly different from market price",
                        {
                            "order_price": order.price,
                            "market_price": market_price,
                            "difference_pct": price_diff * 100
                        }
                    )
        
        return ValidationResult.success("Price validation passed")
    
    def _validate_quantity(self, order: Order, market_data: Optional[Dict[str, Any]]) -> ValidationResult:
        """Validate order quantity."""
        if order.quantity <= 0:
            return ValidationResult.failure(
                f"Invalid quantity: {order.quantity}",
                {"quantity": order.quantity}
            )
        
        # Check for reasonable quantity bounds
        if order.quantity > 1000000:  # 1M shares
            return ValidationResult.warning(
                f"Very large quantity: {order.quantity}",
                {"quantity": order.quantity}
            )
        
        # Check fractional shares
        if order.quantity != int(order.quantity):
            # Some markets don't support fractional shares
            return ValidationResult.warning(
                f"Fractional quantity: {order.quantity}",
                {"quantity": order.quantity}
            )
        
        return ValidationResult.success("Quantity validation passed")
    
    def _validate_symbol(self, order: Order, market_data: Optional[Dict[str, Any]]) -> ValidationResult:
        """Validate trading symbol."""
        if not order.symbol.strip():
            return ValidationResult.failure("Empty symbol")
        
        # Basic symbol format validation
        if len(order.symbol) > 10:
            return ValidationResult.warning(
                f"Unusually long symbol: {order.symbol}",
                {"symbol": order.symbol, "length": len(order.symbol)}
            )
        
        # Check for invalid characters
        invalid_chars = set(order.symbol) - set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-')
        if invalid_chars:
            return ValidationResult.warning(
                f"Symbol contains unusual characters: {invalid_chars}",
                {"symbol": order.symbol, "invalid_chars": list(invalid_chars)}
            )
        
        return ValidationResult.success("Symbol validation passed")
    
    def _validate_order_type_constraints(self, order: Order, market_data: Optional[Dict[str, Any]]) -> ValidationResult:
        """Validate order type specific constraints."""
        # Market orders shouldn't have price
        if order.order_type == OrderType.MARKET:
            if order.price is not None:
                return ValidationResult.warning(
                    "Market order should not have price",
                    {"order_type": order.order_type.name, "price": order.price}
                )
            if order.stop_price is not None:
                return ValidationResult.warning(
                    "Market order should not have stop price",
                    {"order_type": order.order_type.name, "stop_price": order.stop_price}
                )
        
        return ValidationResult.success("Order type validation passed")
    
    def _validate_timestamps(self, order: Order, market_data: Optional[Dict[str, Any]]) -> ValidationResult:
        """Validate order timestamps."""
        if not order.created_at:
            return ValidationResult.failure("Missing creation timestamp")
        
        # Check if order is too old
        if order.created_at:
            age = datetime.now() - order.created_at
            if age.total_seconds() > 86400:  # 24 hours
                return ValidationResult.warning(
                    f"Order is very old: {age}",
                    {"created_at": order.created_at.isoformat(), "age_seconds": age.total_seconds()}
                )
        
        return ValidationResult.success("Timestamp validation passed")
    
    def _combine_validation_results(self, results: List[tuple]) -> ValidationResult:
        """Combine multiple validation results."""
        failures = [r for name, r in results if not r.is_valid]
        warnings = [r for name, r in results if r.is_valid and r.severity == "warning"]
        
        if failures:
            # Return first failure
            failure = failures[0]
            all_details = {}
            for name, result in results:
                all_details[name] = {
                    "valid": result.is_valid,
                    "reason": result.reason,
                    "details": result.details
                }
            
            return ValidationResult.failure(
                failure.reason,
                {"validation_results": all_details}
            )
        
        if warnings:
            # Return combined warnings
            warning_messages = [w.reason for w in warnings]
            all_details = {}
            for name, result in results:
                all_details[name] = {
                    "valid": result.is_valid,
                    "reason": result.reason,
                    "details": result.details
                }
            
            return ValidationResult.warning(
                f"Validation passed with warnings: {'; '.join(warning_messages)}",
                {"validation_results": all_details}
            )
        
        return ValidationResult.success("All validations passed")


class FillValidator:
    """Validation for fill/execution data."""
    
    def validate_fill(
        self,
        fill: Fill,
        order: Order,
        existing_fills: Optional[List[Fill]] = None
    ) -> ValidationResult:
        """Validate fill against order and existing fills.
        
        Args:
            fill: Fill to validate
            order: Original order
            existing_fills: Previous fills for this order
            
        Returns:
            ValidationResult with validation outcome
        """
        # Basic fill validation
        basic_result = self._validate_basic_fill(fill, order)
        if not basic_result.is_valid:
            return basic_result
        
        # Quantity validation
        quantity_result = self._validate_fill_quantity(fill, order, existing_fills)
        if not quantity_result.is_valid:
            return quantity_result
        
        # Price validation
        price_result = self._validate_fill_price(fill, order)
        if not price_result.is_valid:
            return price_result
        
        # Commission and slippage validation
        cost_result = self._validate_fill_costs(fill)
        if not cost_result.is_valid:
            return cost_result
        
        return ValidationResult.success("Fill validation passed")
    
    def _validate_basic_fill(self, fill: Fill, order: Order) -> ValidationResult:
        """Basic fill validation."""
        if fill.order_id != order.order_id:
            return ValidationResult.failure(
                "Fill order ID mismatch",
                {"fill_order_id": fill.order_id, "order_id": order.order_id}
            )
        
        if fill.symbol != order.symbol:
            return ValidationResult.failure(
                "Fill symbol mismatch",
                {"fill_symbol": fill.symbol, "order_symbol": order.symbol}
            )
        
        if fill.side != order.side:
            return ValidationResult.failure(
                "Fill side mismatch",
                {"fill_side": fill.side.name, "order_side": order.side.name}
            )
        
        return ValidationResult.success("Basic fill validation passed")
    
    def _validate_fill_quantity(
        self,
        fill: Fill,
        order: Order,
        existing_fills: Optional[List[Fill]]
    ) -> ValidationResult:
        """Validate fill quantity."""
        if fill.quantity <= 0:
            return ValidationResult.failure(
                f"Invalid fill quantity: {fill.quantity}",
                {"fill_quantity": fill.quantity}
            )
        
        # Check against existing fills
        if existing_fills:
            total_filled = sum(f.quantity for f in existing_fills) + fill.quantity
            if total_filled > order.quantity * 1.001:  # Allow small rounding
                return ValidationResult.failure(
                    f"Over-fill: {total_filled} > {order.quantity}",
                    {
                        "total_filled": total_filled,
                        "order_quantity": order.quantity,
                        "new_fill_quantity": fill.quantity
                    }
                )
        else:
            # First fill
            if fill.quantity > order.quantity * 1.001:  # Allow small rounding
                return ValidationResult.failure(
                    f"Fill quantity exceeds order: {fill.quantity} > {order.quantity}",
                    {"fill_quantity": fill.quantity, "order_quantity": order.quantity}
                )
        
        return ValidationResult.success("Fill quantity validation passed")
    
    def _validate_fill_price(self, fill: Fill, order: Order) -> ValidationResult:
        """Validate fill price."""
        if fill.price <= 0:
            return ValidationResult.failure(
                f"Invalid fill price: {fill.price}",
                {"fill_price": fill.price}
            )
        
        # Price improvement validation for limit orders
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and order.price:
            if order.side == OrderSide.BUY:
                if fill.price > order.price * 1.01:  # Allow 1% tolerance
                    return ValidationResult.failure(
                        f"Buy fill price above limit: {fill.price} > {order.price}",
                        {"fill_price": fill.price, "limit_price": order.price}
                    )
            else:  # SELL
                if fill.price < order.price * 0.99:  # Allow 1% tolerance
                    return ValidationResult.failure(
                        f"Sell fill price below limit: {fill.price} < {order.price}",
                        {"fill_price": fill.price, "limit_price": order.price}
                    )
        
        return ValidationResult.success("Fill price validation passed")
    
    def _validate_fill_costs(self, fill: Fill) -> ValidationResult:
        """Validate commission and slippage."""
        # Commission should be non-negative
        if fill.commission < 0:
            return ValidationResult.failure(
                f"Negative commission: {fill.commission}",
                {"commission": fill.commission}
            )
        
        # Commission reasonableness check
        trade_value = fill.quantity * fill.price
        commission_rate = fill.commission / trade_value if trade_value > 0 else 0
        if commission_rate > 0.1:  # 10% commission seems unreasonable
            return ValidationResult.warning(
                f"Very high commission rate: {commission_rate:.1%}",
                {"commission": fill.commission, "trade_value": trade_value, "rate": commission_rate}
            )
        
        # Slippage reasonableness check
        slippage_rate = abs(fill.slippage) / fill.price if fill.price > 0 else 0
        if slippage_rate > 0.05:  # 5% slippage seems high
            return ValidationResult.warning(
                f"High slippage: {slippage_rate:.1%}",
                {"slippage": fill.slippage, "price": fill.price, "rate": slippage_rate}
            )
        
        return ValidationResult.success("Fill costs validation passed")


class ExecutionValidator:
    """High-level execution validation coordinator."""
    
    def __init__(self):
        """Initialize execution validator."""
        self.order_validator = OrderValidator()
        self.fill_validator = FillValidator()
    
    def validate_order_execution_flow(
        self,
        order: Order,
        fills: List[Fill],
        market_data: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Validate complete order execution flow.
        
        Args:
            order: Original order
            fills: All fills for the order
            market_data: Market data at time of order
            
        Returns:
            ValidationResult for the complete flow
        """
        # Validate order
        order_result = self.order_validator.validate_order(order, market_data=market_data)
        if not order_result.is_valid:
            return ValidationResult.failure(
                f"Order validation failed: {order_result.reason}",
                {"order_validation": order_result.details}
            )
        
        # Validate each fill
        fill_results = []
        for i, fill in enumerate(fills):
            existing_fills = fills[:i]  # Previous fills
            fill_result = self.fill_validator.validate_fill(fill, order, existing_fills)
            fill_results.append((i, fill_result))
            
            if not fill_result.is_valid:
                return ValidationResult.failure(
                    f"Fill {i} validation failed: {fill_result.reason}",
                    {"fill_validation": fill_result.details, "fill_index": i}
                )
        
        # Validate execution completeness
        total_filled = sum(f.quantity for f in fills)
        if total_filled < order.quantity * 0.999:  # Allow small rounding
            return ValidationResult.warning(
                f"Order partially filled: {total_filled}/{order.quantity}",
                {"total_filled": total_filled, "order_quantity": order.quantity}
            )
        
        return ValidationResult.success("Execution flow validation passed")


# Factory functions

def create_order_validator() -> OrderValidator:
    """Create order validator."""
    return OrderValidator()


def create_fill_validator() -> FillValidator:
    """Create fill validator."""
    return FillValidator()


def create_execution_validator() -> ExecutionValidator:
    """Create execution validator."""
    return ExecutionValidator()


# Utility functions

def validate_order_quick(order: Order) -> bool:
    """Quick order validation returning boolean."""
    validator = OrderValidator()
    result = validator.validate_order(order, rules=['basic', 'quantity'])
    return result.is_valid


def validate_portfolio_integration(
    execution_module: Dict[str, Any],
    portfolio_state: Any
) -> ValidationResult:
    """Validate execution module integration with portfolio state."""
    try:
        broker = execution_module.get('broker')
        if not broker:
            return ValidationResult.failure("No broker in execution module")
        
        # Check portfolio state reference
        if not hasattr(broker, '_portfolio_state'):
            return ValidationResult.failure("Broker missing portfolio state reference")
        
        if broker._portfolio_state is not portfolio_state:
            return ValidationResult.failure("Broker portfolio state mismatch")
        
        return ValidationResult.success("Portfolio integration validation passed")
        
    except Exception as e:
        return ValidationResult.failure(
            f"Portfolio integration validation error: {e}",
            {"exception": str(e)}
        )
