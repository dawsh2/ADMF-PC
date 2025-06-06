"""
Risk validators for unified architecture.

These validators implement pure functions for use as lightweight services
in the event-driven architecture. All state is passed as parameters - 
no internal state is maintained.
"""

from typing import Dict, Any, Optional
from decimal import Decimal

from ..core.containers.discovery import risk_validator


@risk_validator(
    name='max_position_validator',
    validation_types=['position_size', 'position_value']
)
def validate_max_position(
    order: Dict[str, Any],
    portfolio_state: Dict[str, Any],
    risk_limits: Dict[str, Any],
    market_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate order against position size limits.
    
    Args:
        order: Order to validate
            - symbol: Asset symbol
            - quantity: Number of shares/contracts
            - side: 'buy' or 'sell'
            - price: Optional limit price
        portfolio_state: Current portfolio state
            - positions: Dict of symbol -> position info
            - cash: Available cash
            - total_value: Total portfolio value
        risk_limits: Risk parameters
            - max_position_value: Max position $ value
            - max_position_percent: Max position % of portfolio
        market_data: Current market data
            - close: Current price
            
    Returns:
        Validation result with approved flag and metrics
    """
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
    
    # Get price for position value calculation
    price = order.get('price')
    if not price:
        price = market_data.get('close', market_data.get('price'))
    
    if not price or price <= 0:
        return {
            'approved': False,
            'reason': 'No valid price available for position limit check',
            'risk_metrics': {}
        }
    
    price = Decimal(str(price))
    new_position_value = abs(new_quantity) * price
    
    # Check value limit
    max_value = risk_limits.get('max_position_value')
    if max_value and new_position_value > Decimal(str(max_value)):
        return {
            'approved': False,
            'reason': f'Position value ${new_position_value:.2f} exceeds limit ${max_value}',
            'adjusted_quantity': None,
            'risk_metrics': {
                'position_value': float(new_position_value),
                'limit_value': float(max_value)
            }
        }
    
    # Check percentage limit
    max_percent = risk_limits.get('max_position_percent')
    if max_percent:
        total_value = Decimal(str(portfolio_state.get('total_value', 0)))
        if total_value <= 0:
            return {
                'approved': False,
                'reason': 'Cannot calculate position percentage with zero portfolio value',
                'risk_metrics': {}
            }
        
        position_percent = new_position_value / total_value
        max_pct_decimal = Decimal(str(max_percent))
        
        if position_percent > max_pct_decimal:
            # Calculate adjusted quantity that would meet the limit
            allowed_value = total_value * max_pct_decimal
            adjusted_quantity = int(allowed_value / price)
            
            return {
                'approved': False,
                'reason': f'Position {position_percent:.1%} exceeds limit {max_percent:.1%}',
                'adjusted_quantity': adjusted_quantity if adjusted_quantity > 0 else None,
                'risk_metrics': {
                    'position_percent': float(position_percent),
                    'limit_percent': float(max_percent)
                }
            }
    
    # Order approved
    return {
        'approved': True,
        'adjusted_quantity': None,
        'reason': '',
        'risk_metrics': {
            'position_value': float(new_position_value),
            'position_percent': float(new_position_value / Decimal(str(portfolio_state.get('total_value', 1))))
        }
    }


@risk_validator(
    name='position_sizer',
    validation_types=['position_sizing']
)
def calculate_position_size(
    signal: Dict[str, Any],
    portfolio_state: Dict[str, Any],
    risk_params: Dict[str, Any],
    market_data: Dict[str, Any]
) -> float:
    """
    Calculate appropriate position size for a signal.
    
    Args:
        signal: Trading signal with direction and strength
        portfolio_state: Current portfolio state
        risk_params: Risk parameters for sizing
        market_data: Current market prices
        
    Returns:
        Position size (number of shares/contracts)
    """
    # Get portfolio value and price
    total_value = Decimal(str(portfolio_state.get('total_value', 0)))
    price = Decimal(str(market_data.get('close', market_data.get('price', 0))))
    
    if total_value <= 0 or price <= 0:
        return 0
    
    # Get position sizing parameters
    base_position_pct = Decimal(str(risk_params.get('base_position_percent', 0.02)))
    use_signal_strength = risk_params.get('use_signal_strength', True)
    
    # Calculate base position value
    position_value = total_value * base_position_pct
    
    # Adjust by signal strength if enabled
    if use_signal_strength and 'strength' in signal:
        strength = Decimal(str(signal['strength']))
        position_value *= strength
    
    # Convert to shares
    shares = int(position_value / price)
    
    return max(0, shares)


@risk_validator(
    name='drawdown_validator',
    validation_types=['drawdown', 'risk_limits']
)
def validate_drawdown(
    order: Dict[str, Any],
    portfolio_state: Dict[str, Any],
    risk_limits: Dict[str, Any],
    market_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate order against drawdown limits.
    
    Args:
        order: Order to validate
        portfolio_state: Current portfolio state with metrics
        risk_limits: Risk parameters
            - max_drawdown: Maximum allowed drawdown (0-1)
            - drawdown_halt_threshold: Stop trading threshold
        market_data: Current market data
        
    Returns:
        Validation result
    """
    # Get current drawdown from portfolio metrics
    metrics = portfolio_state.get('metrics', {})
    current_drawdown = Decimal(str(metrics.get('current_drawdown', 0)))
    
    # Get limits
    max_drawdown = Decimal(str(risk_limits.get('max_drawdown', 0.2)))
    halt_threshold = Decimal(str(risk_limits.get('drawdown_halt_threshold', max_drawdown)))
    
    # Check if we should halt all trading
    if current_drawdown >= halt_threshold:
        return {
            'approved': False,
            'reason': f'Drawdown {current_drawdown:.1%} exceeds halt threshold {halt_threshold:.1%}',
            'risk_metrics': {
                'current_drawdown': float(current_drawdown),
                'halt_threshold': float(halt_threshold)
            }
        }
    
    # Check if we should restrict new positions
    if current_drawdown >= max_drawdown:
        # Only allow closing positions
        positions = portfolio_state.get('positions', {})
        current_position = positions.get(order['symbol'], {})
        current_qty = current_position.get('quantity', 0)
        
        # Check if this order reduces position
        is_reducing = (
            (order['side'] == 'sell' and current_qty > 0) or
            (order['side'] == 'buy' and current_qty < 0)
        )
        
        if not is_reducing:
            return {
                'approved': False,
                'reason': f'Drawdown {current_drawdown:.1%} exceeds limit {max_drawdown:.1%} - only position reduction allowed',
                'risk_metrics': {
                    'current_drawdown': float(current_drawdown),
                    'max_drawdown': float(max_drawdown)
                }
            }
    
    # Order approved
    return {
        'approved': True,
        'reason': '',
        'risk_metrics': {
            'current_drawdown': float(current_drawdown),
            'max_drawdown': float(max_drawdown),
            'headroom': float(max_drawdown - current_drawdown)
        }
    }


@risk_validator(
    name='drawdown_position_sizer',
    validation_types=['position_sizing', 'drawdown_adjustment']
)
def calculate_drawdown_adjusted_size(
    signal: Dict[str, Any],
    portfolio_state: Dict[str, Any],
    risk_params: Dict[str, Any],
    market_data: Dict[str, Any]
) -> float:
    """
    Calculate position size considering drawdown.
    
    Reduces position size as drawdown increases.
    """
    # Get base size calculation
    base_size = calculate_position_size(
        signal, portfolio_state, risk_params, market_data
    )
    
    # Get drawdown metrics
    metrics = portfolio_state.get('metrics', {})
    current_drawdown = Decimal(str(metrics.get('current_drawdown', 0)))
    max_drawdown = Decimal(str(risk_params.get('max_drawdown', 0.2)))
    
    # Scale position size based on drawdown
    if current_drawdown >= max_drawdown:
        return 0  # No new positions
    
    # Linear scaling: full size at 0 drawdown, 0 size at max drawdown
    drawdown_factor = 1 - (current_drawdown / max_drawdown)
    adjusted_size = int(base_size * float(drawdown_factor))
    
    return max(0, adjusted_size)


@risk_validator(
    name='composite_validator',
    validation_types=['composite', 'all']
)
def validate_composite(
    order: Dict[str, Any],
    portfolio_state: Dict[str, Any],
    risk_limits: Dict[str, Any],
    market_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate order against all risk limits.
    
    Runs position and drawdown validators and returns the first failure
    or combined success result.
    """
    # Run position validator
    position_result = validate_max_position(
        order, portfolio_state, risk_limits, market_data
    )
    
    if not position_result['approved']:
        return position_result
    
    # Run drawdown validator
    drawdown_result = validate_drawdown(
        order, portfolio_state, risk_limits, market_data
    )
    
    if not drawdown_result['approved']:
        return drawdown_result
    
    # Combine risk metrics
    combined_metrics = {}
    combined_metrics.update(position_result.get('risk_metrics', {}))
    combined_metrics.update(drawdown_result.get('risk_metrics', {}))
    
    return {
        'approved': True,
        'adjusted_quantity': None,
        'reason': '',
        'risk_metrics': combined_metrics
    }


@risk_validator(
    name='composite_position_sizer',
    validation_types=['position_sizing', 'composite']
)
def calculate_composite_size(
    signal: Dict[str, Any],
    portfolio_state: Dict[str, Any],
    risk_params: Dict[str, Any],
    market_data: Dict[str, Any]
) -> float:
    """
    Calculate position size considering all risk factors.
    
    Uses the most conservative size from position and drawdown validators.
    """
    # Get sizes from each validator
    position_size = calculate_position_size(
        signal, portfolio_state, risk_params, market_data
    )
    
    drawdown_size = calculate_drawdown_adjusted_size(
        signal, portfolio_state, risk_params, market_data
    )
    
    # Return the most conservative
    return min(position_size, drawdown_size)


# Helper function to create validator objects for compatibility
def create_validator_wrapper(validator_func):
    """
    Create a wrapper object that provides the validate_order method interface.
    
    This allows pure validator functions to be used with components that expect
    objects with a validate_order method.
    """
    class ValidatorWrapper:
        def __init__(self):
            self.validator_func = validator_func
            
        def validate_order(self, order, portfolio_state, risk_limits, market_data):
            return self.validator_func(order, portfolio_state, risk_limits, market_data)
            
        def calculate_position_size(self, signal, portfolio_state, risk_params, market_data):
            # If the validator function is a position sizer, use it directly
            if 'position_size' in validator_func.__name__:
                return validator_func(signal, portfolio_state, risk_params, market_data)
            # Otherwise, use the default position sizer
            return calculate_position_size(signal, portfolio_state, risk_params, market_data)
    
    return ValidatorWrapper()