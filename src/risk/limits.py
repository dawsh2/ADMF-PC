"""
Risk limit implementations for portfolio protection.

Pure functional implementations - no internal state, all state passed as parameters.
"""

from decimal import Decimal
from typing import Dict, Any, Optional, List, Set, Tuple
from datetime import datetime, timedelta


def check_max_position_limit(
    order: Dict[str, Any],
    portfolio_state: Dict[str, Any],
    risk_limits: Dict[str, Any],
    market_data: Dict[str, Any]
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Check if order violates position size limit.
    
    Pure function - no side effects or state mutations.
    
    Args:
        order: Proposed order with symbol, quantity, side, price
        portfolio_state: Current portfolio state with positions
        risk_limits: Risk parameters including max_position_value, max_position_percent
        market_data: Current market data with prices
        
    Returns:
        Tuple of (passes_check, reason_if_failed, metrics)
    """
    # Extract parameters
    max_position_value = risk_limits.get('max_position_value')
    max_position_percent = risk_limits.get('max_position_percent')
    
    if not (max_position_value or max_position_percent):
        return True, None, {}
    
    # Get current position
    positions = portfolio_state.get('positions', {})
    current_position = positions.get(order['symbol'], {})
    current_quantity = Decimal(str(current_position.get('quantity', 0)))
    
    # Calculate new position
    order_quantity = Decimal(str(order['quantity']))
    if order['side'] == 'buy':
        new_quantity = current_quantity + order_quantity
    else:
        new_quantity = current_quantity - order_quantity
    
    # Get price
    price = order.get('price')
    if not price:  # Market order
        price = market_data.get('prices', {}).get(order['symbol'])
        if not price:
            return False, "No price available for position limit check", {}
    
    price = Decimal(str(price))
    new_position_value = abs(new_quantity) * price
    
    metrics = {
        'current_quantity': float(current_quantity),
        'new_quantity': float(new_quantity),
        'position_value': float(new_position_value)
    }
    
    # Check value limit
    if max_position_value:
        max_value = Decimal(str(max_position_value))
        if new_position_value > max_value:
            reason = f"Position value {new_position_value} exceeds limit {max_value}"
            metrics['limit_exceeded'] = 'value'
            metrics['max_value'] = float(max_value)
            return False, reason, metrics
    
    # Check percentage limit
    if max_position_percent:
        portfolio_value = Decimal(str(portfolio_state.get('total_value', 0)))
        if portfolio_value == 0:
            return False, "Cannot calculate position percentage with zero portfolio value", metrics
        
        position_percent = new_position_value / portfolio_value
        max_percent = Decimal(str(max_position_percent))
        
        if position_percent > max_percent:
            reason = f"Position {position_percent:.1%} exceeds limit {max_percent:.1%}"
            metrics['limit_exceeded'] = 'percent'
            metrics['position_percent'] = float(position_percent)
            metrics['max_percent'] = float(max_percent)
            return False, reason, metrics
    
    return True, None, metrics


def check_max_drawdown_limit(
    order: Dict[str, Any],
    portfolio_state: Dict[str, Any],
    risk_limits: Dict[str, Any],
    market_data: Dict[str, Any]
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Check if current drawdown exceeds limit.
    
    Pure function - no side effects or state mutations.
    
    Args:
        order: Proposed order
        portfolio_state: Current portfolio state with metrics
        risk_limits: Risk parameters including max_drawdown
        market_data: Current market data
        
    Returns:
        Tuple of (passes_check, reason_if_failed, metrics)
    """
    max_drawdown = risk_limits.get('max_drawdown')
    if not max_drawdown:
        return True, None, {}
    
    # Get current drawdown from portfolio metrics
    metrics = portfolio_state.get('metrics', {})
    current_drawdown = Decimal(str(metrics.get('current_drawdown', 0)))
    max_dd = Decimal(str(max_drawdown))
    
    result_metrics = {
        'current_drawdown': float(current_drawdown),
        'max_drawdown': float(max_dd)
    }
    
    if current_drawdown > max_dd:
        reason = f"Current drawdown {current_drawdown:.1%} exceeds limit {max_dd:.1%}"
        result_metrics['limit_exceeded'] = True
        return False, reason, result_metrics
    
    return True, None, result_metrics


def check_var_limit(
    order: Dict[str, Any],
    portfolio_state: Dict[str, Any],
    risk_limits: Dict[str, Any],
    market_data: Dict[str, Any]
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Check if VaR exceeds limit.
    
    Pure function - no side effects or state mutations.
    
    Args:
        order: Proposed order
        portfolio_state: Current portfolio state
        risk_limits: Risk parameters including max_var, confidence_level
        market_data: Current market data
        
    Returns:
        Tuple of (passes_check, reason_if_failed, metrics)
    """
    max_var = risk_limits.get('max_var')
    if not max_var:
        return True, None, {}
    
    confidence_level = Decimal(str(risk_limits.get('confidence_level', 0.95)))
    metrics = portfolio_state.get('metrics', {})
    
    # Check if VaR is available
    var_95 = metrics.get('var_95')
    if var_95 is None:
        # Can't check, allow order
        return True, None, {'var_available': False}
    
    # Simple check: current VaR
    portfolio_value = Decimal(str(portfolio_state.get('total_value', 1)))
    var_fraction = Decimal(str(var_95)) / portfolio_value
    max_var_decimal = Decimal(str(max_var))
    
    result_metrics = {
        'var_fraction': float(var_fraction),
        'max_var': float(max_var_decimal),
        'confidence_level': float(confidence_level)
    }
    
    if var_fraction > max_var_decimal:
        reason = f"VaR {var_fraction:.1%} exceeds limit {max_var_decimal:.1%}"
        result_metrics['limit_exceeded'] = True
        return False, reason, result_metrics
    
    return True, None, result_metrics


def check_max_exposure_limit(
    order: Dict[str, Any],
    portfolio_state: Dict[str, Any],
    risk_limits: Dict[str, Any],
    market_data: Dict[str, Any]
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Check if order would exceed total exposure limit.
    
    Pure function - no side effects or state mutations.
    
    Args:
        order: Proposed order
        portfolio_state: Current portfolio state
        risk_limits: Risk parameters including max_exposure_pct
        market_data: Current market data
        
    Returns:
        Tuple of (passes_check, reason_if_failed, metrics)
    """
    max_exposure_pct = risk_limits.get('max_exposure_pct')
    if not max_exposure_pct:
        return True, None, {}
    
    # Get current metrics
    metrics = portfolio_state.get('metrics', {})
    portfolio_value = Decimal(str(portfolio_state.get('total_value', 0)))
    current_exposure = Decimal(str(metrics.get('positions_value', 0)))
    
    # Calculate order value
    price = order.get('price')
    if not price:
        price = market_data.get('prices', {}).get(order['symbol'])
        if not price:
            return False, "No price available for exposure check", {}
    
    price = Decimal(str(price))
    order_value = Decimal(str(order['quantity'])) * price
    
    # Calculate new exposure
    if order['side'] == 'buy':
        new_exposure = current_exposure + order_value
    else:
        # Selling reduces exposure
        new_exposure = current_exposure - order_value
    
    # Calculate exposure percentage
    exposure_pct = (new_exposure / portfolio_value) * 100 if portfolio_value > 0 else Decimal(0)
    max_exp = Decimal(str(max_exposure_pct))
    
    result_metrics = {
        'current_exposure': float(current_exposure),
        'new_exposure': float(new_exposure),
        'exposure_pct': float(exposure_pct),
        'max_exposure_pct': float(max_exp)
    }
    
    if exposure_pct > max_exp:
        reason = f"Total exposure {exposure_pct:.1f}% would exceed limit {max_exp:.1f}%"
        result_metrics['limit_exceeded'] = True
        return False, reason, result_metrics
    
    return True, None, result_metrics


def check_concentration_limit(
    order: Dict[str, Any],
    portfolio_state: Dict[str, Any],
    risk_limits: Dict[str, Any],
    market_data: Dict[str, Any]
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Check if order violates concentration limits.
    
    Pure function - no side effects or state mutations.
    
    Args:
        order: Proposed order
        portfolio_state: Current portfolio state
        risk_limits: Risk parameters including max_single_position
        market_data: Current market data
        
    Returns:
        Tuple of (passes_check, reason_if_failed, metrics)
    """
    max_single_position = risk_limits.get('max_single_position')
    if not max_single_position:
        return True, None, {}
    
    # Get portfolio metrics
    portfolio_value = Decimal(str(portfolio_state.get('total_value', 0)))
    if portfolio_value == 0:
        return False, "Cannot calculate concentration with zero portfolio value", {}
    
    # Calculate position after order
    positions = portfolio_state.get('positions', {})
    current_position = positions.get(order['symbol'], {})
    current_value = Decimal(str(current_position.get('market_value', 0)))
    
    # Get order value
    price = order.get('price')
    if not price:
        price = market_data.get('prices', {}).get(order['symbol'])
        if not price:
            return False, "No price available for concentration check", {}
    
    price = Decimal(str(price))
    order_value = Decimal(str(order['quantity'])) * price
    
    if order['side'] == 'buy':
        new_value = current_value + order_value
    else:
        new_value = current_value - order_value
    
    new_concentration = abs(new_value) / portfolio_value
    max_conc = Decimal(str(max_single_position))
    
    result_metrics = {
        'position_concentration': float(new_concentration),
        'max_concentration': float(max_conc)
    }
    
    if new_concentration > max_conc:
        reason = f"Position concentration {new_concentration:.1%} exceeds limit {max_conc:.1%}"
        result_metrics['limit_exceeded'] = True
        return False, reason, result_metrics
    
    return True, None, result_metrics


def check_leverage_limit(
    order: Dict[str, Any],
    portfolio_state: Dict[str, Any],
    risk_limits: Dict[str, Any],
    market_data: Dict[str, Any]
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Check if order would exceed leverage limit.
    
    Pure function - no side effects or state mutations.
    
    Args:
        order: Proposed order
        portfolio_state: Current portfolio state
        risk_limits: Risk parameters including max_leverage
        market_data: Current market data
        
    Returns:
        Tuple of (passes_check, reason_if_failed, metrics)
    """
    max_leverage = risk_limits.get('max_leverage')
    if not max_leverage:
        return True, None, {}
    
    # Get current leverage
    metrics = portfolio_state.get('metrics', {})
    current_leverage = Decimal(str(metrics.get('leverage', 1.0)))
    
    # For buy orders, check if we'd exceed leverage
    if order['side'] == 'buy':
        # Get order value
        price = order.get('price')
        if not price:
            price = market_data.get('prices', {}).get(order['symbol'])
            if not price:
                return False, "No price available for leverage check", {}
        
        price = Decimal(str(price))
        order_value = Decimal(str(order['quantity'])) * price
        
        # Calculate new positions value
        new_positions_value = Decimal(str(metrics.get('positions_value', 0))) + order_value
        
        # Calculate new leverage
        cash = Decimal(str(portfolio_state.get('cash', 0)))
        equity = Decimal(str(metrics.get('total_value', 1)))
        
        # Simple leverage calculation
        if cash < order_value:
            # Would need margin
            margin_needed = order_value - cash
            new_leverage = new_positions_value / equity
            
            max_lev = Decimal(str(max_leverage))
            
            result_metrics = {
                'current_leverage': float(current_leverage),
                'new_leverage': float(new_leverage),
                'max_leverage': float(max_lev),
                'margin_needed': float(margin_needed)
            }
            
            if new_leverage > max_lev:
                reason = f"Leverage {new_leverage:.2f}x exceeds limit {max_lev:.2f}x"
                result_metrics['limit_exceeded'] = True
                return False, reason, result_metrics
            
            return True, None, result_metrics
    
    return True, None, {'current_leverage': float(current_leverage)}


def check_daily_loss_limit(
    order: Dict[str, Any],
    portfolio_state: Dict[str, Any],
    risk_limits: Dict[str, Any],
    market_data: Dict[str, Any]
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Check if daily loss limit is exceeded.
    
    Pure function - no side effects or state mutations.
    
    Args:
        order: Proposed order
        portfolio_state: Current portfolio state
        risk_limits: Risk parameters including max_daily_loss
        market_data: Current market data
        
    Returns:
        Tuple of (passes_check, reason_if_failed, metrics)
    """
    max_daily_loss = risk_limits.get('max_daily_loss')
    if not max_daily_loss:
        return True, None, {}
    
    # Get current daily P&L from portfolio metrics
    metrics = portfolio_state.get('metrics', {})
    portfolio_value = Decimal(str(portfolio_state.get('total_value', 1)))
    
    # Get today's P&L (should be tracked in portfolio state)
    daily_pnl = Decimal(str(metrics.get('daily_pnl', 0)))
    
    # Simple check: if P&L is very negative
    daily_loss_fraction = -daily_pnl / portfolio_value if daily_pnl < 0 else Decimal(0)
    max_loss = Decimal(str(max_daily_loss))
    
    result_metrics = {
        'daily_pnl': float(daily_pnl),
        'daily_loss_fraction': float(daily_loss_fraction),
        'max_daily_loss': float(max_loss)
    }
    
    if daily_loss_fraction > max_loss:
        reason = f"Daily loss {daily_loss_fraction:.1%} exceeds limit {max_loss:.1%}"
        result_metrics['limit_exceeded'] = True
        return False, reason, result_metrics
    
    return True, None, result_metrics


def check_symbol_restrictions(
    order: Dict[str, Any],
    portfolio_state: Dict[str, Any],
    risk_limits: Dict[str, Any],
    market_data: Dict[str, Any]
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Check if symbol is allowed for trading.
    
    Pure function - no side effects or state mutations.
    
    Args:
        order: Proposed order
        portfolio_state: Current portfolio state
        risk_limits: Risk parameters including allowed_symbols, blocked_symbols
        market_data: Current market data
        
    Returns:
        Tuple of (passes_check, reason_if_failed, metrics)
    """
    allowed_symbols = risk_limits.get('allowed_symbols')
    blocked_symbols = risk_limits.get('blocked_symbols', set())
    
    symbol = order['symbol']
    
    # Check blocked symbols
    if symbol in blocked_symbols:
        return False, f"Symbol {symbol} is blocked for trading", {'blocked': True}
    
    # Check allowed symbols
    if allowed_symbols and symbol not in allowed_symbols:
        return False, f"Symbol {symbol} is not in allowed list", {'not_allowed': True}
    
    return True, None, {'symbol': symbol}


def check_all_limits(
    order: Dict[str, Any],
    portfolio_state: Dict[str, Any],
    risk_limits: Dict[str, Any],
    market_data: Dict[str, Any]
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Check all risk limits in order.
    
    Pure function that combines all limit checks.
    
    Args:
        order: Proposed order
        portfolio_state: Current portfolio state
        risk_limits: All risk parameters
        market_data: Current market data
        
    Returns:
        Tuple of (passes_all_checks, first_failure_reason, combined_metrics)
    """
    combined_metrics = {}
    
    # List of all limit check functions
    limit_checks = [
        check_symbol_restrictions,
        check_max_position_limit,
        check_max_drawdown_limit,
        check_var_limit,
        check_max_exposure_limit,
        check_concentration_limit,
        check_leverage_limit,
        check_daily_loss_limit,
    ]
    
    # Run each check
    for check_func in limit_checks:
        passed, reason, metrics = check_func(order, portfolio_state, risk_limits, market_data)
        
        # Add metrics to combined result
        check_name = check_func.__name__.replace('check_', '')
        combined_metrics[check_name] = metrics
        
        # Return on first failure
        if not passed:
            combined_metrics['failed_check'] = check_name
            return False, reason, combined_metrics
    
    # All checks passed
    return True, None, combined_metrics


# Backward compatibility wrapper for RiskLimits aggregator
class RiskLimits:
    """
    Stateless risk limits aggregator for backward compatibility.
    
    This is a thin wrapper that delegates to pure functions.
    """
    
    def __init__(self,
                 max_position_size: float = 10000,
                 max_portfolio_heat: float = 6.0,
                 max_correlation: float = 0.7,
                 max_sector_exposure: float = 0.4,
                 max_drawdown: float = 0.2,
                 max_var_95: float = 0.1):
        """
        Initialize risk limits configuration.
        
        Args:
            max_position_size: Maximum position size in dollars
            max_portfolio_heat: Maximum portfolio heat percentage
            max_correlation: Maximum correlation allowed
            max_sector_exposure: Maximum sector exposure
            max_drawdown: Maximum drawdown percentage
            max_var_95: Maximum 95% VaR
        """
        self.risk_limits = {
            'max_position_value': max_position_size if max_position_size > 0 else None,
            'max_exposure_pct': max_sector_exposure * 100 if max_sector_exposure > 0 else None,
            'max_drawdown': max_drawdown if max_drawdown > 0 else None,
            'max_var': max_var_95 if max_var_95 > 0 else None,
            'confidence_level': 0.95
        }
    
    def check_order(self, order: Dict[str, Any], portfolio_state: Dict[str, Any],
                   market_data: Dict[str, Any] = None) -> Tuple[bool, Optional[str]]:
        """
        Check if order passes all risk limits.
        
        Args:
            order: Order to check
            portfolio_state: Current portfolio state
            market_data: Current market data
            
        Returns:
            Tuple of (is_allowed, reason_if_rejected)
        """
        if market_data is None:
            market_data = {}
            
        passed, reason, _ = check_all_limits(
            order, portfolio_state, self.risk_limits, market_data
        )
        return passed, reason
    
    def get_limit_info(self) -> Dict[str, Any]:
        """Get information about configured limits."""
        return {
            "risk_limits": self.risk_limits,
            "checks_enabled": [
                name for name, value in self.risk_limits.items() 
                if value is not None
            ]
        }