"""Position sizing strategies for risk management.

Pure functional implementations - no internal state, all state passed as parameters.
"""

from decimal import Decimal
from typing import Dict, Any, Optional, Tuple
import math


def apply_position_constraints(
    size: Decimal,
    min_size: Decimal = Decimal("0.01")
) -> Decimal:
    """Apply basic constraints to position size.
    
    Pure function - no side effects.
    
    Args:
        size: Raw position size
        min_size: Minimum position size
        
    Returns:
        Constrained position size
    """
    # Ensure minimum size
    if abs(size) < min_size:
        return Decimal(0)
    
    # Round to reasonable precision (e.g., shares)
    return size.quantize(Decimal("0.01"))


def calculate_fixed_position_size(
    signal: Dict[str, Any],
    portfolio_state: Dict[str, Any],
    sizing_params: Dict[str, Any],
    market_data: Dict[str, Any]
) -> Decimal:
    """Calculate fixed position size.
    
    Pure function - no side effects or state mutations.
    
    Args:
        signal: Trading signal with symbol, strength, direction
        portfolio_state: Current portfolio state
        sizing_params: Parameters including 'fixed_size', 'min_size'
        market_data: Current market data
        
    Returns:
        Fixed position size
    """
    # Get parameters
    fixed_size = Decimal(str(sizing_params.get('fixed_size', 100)))
    min_size = Decimal(str(sizing_params.get('min_size', 0.01)))
    
    # Adjust size based on signal strength
    signal_strength = Decimal(str(signal.get('strength', 1.0)))
    adjusted_size = fixed_size * abs(signal_strength)
    
    return apply_position_constraints(adjusted_size, min_size)


def calculate_percentage_position_size(
    signal: Dict[str, Any],
    portfolio_state: Dict[str, Any],
    sizing_params: Dict[str, Any],
    market_data: Dict[str, Any]
) -> Decimal:
    """Calculate position size as percentage of portfolio.
    
    Pure function - no side effects or state mutations.
    
    Args:
        signal: Trading signal with symbol, strength, direction
        portfolio_state: Current portfolio state with total_value, cash
        sizing_params: Parameters including 'percentage', 'use_leverage', 'min_size'
        market_data: Current market data with prices
        
    Returns:
        Position size based on portfolio percentage
    """
    # Get parameters
    percentage = Decimal(str(sizing_params.get('percentage', 0.02)))
    use_leverage = sizing_params.get('use_leverage', False)
    min_size = Decimal(str(sizing_params.get('min_size', 0.01)))
    
    # Get current portfolio value
    portfolio_value = Decimal(str(portfolio_state.get('total_value', 0)))
    if portfolio_value <= 0:
        return Decimal(0)
    
    # Get current price
    prices = market_data.get('prices', {})
    symbol = signal.get('symbol')
    price = prices.get(symbol) if symbol else None
    
    if not price or price <= 0:
        return Decimal(0)
    
    price = Decimal(str(price))
    
    # Calculate base position value
    position_value = portfolio_value * percentage
    
    # Adjust for signal strength
    signal_strength = Decimal(str(signal.get('strength', 1.0)))
    position_value *= abs(signal_strength)
    
    # Check leverage constraints
    if not use_leverage:
        cash_available = Decimal(str(portfolio_state.get('cash', 0)))
        position_value = min(position_value, cash_available)
    
    # Convert to shares
    shares = position_value / price
    
    return apply_position_constraints(shares, min_size)


def calculate_kelly_position_size(
    signal: Dict[str, Any],
    portfolio_state: Dict[str, Any],
    sizing_params: Dict[str, Any],
    market_data: Dict[str, Any]
) -> Decimal:
    """Calculate position size using Kelly Criterion.
    
    Pure function - no side effects or state mutations.
    
    Kelly formula: f = (p*b - q) / b
    where:
    - f = fraction of capital to bet
    - p = probability of winning
    - q = probability of losing (1-p)
    - b = ratio of win to loss
    
    Args:
        signal: Trading signal with symbol, strength, direction
        portfolio_state: Current portfolio state
        sizing_params: Parameters including 'win_rate', 'avg_win', 'avg_loss', 
                      'kelly_fraction', 'max_percentage', 'min_size'
        market_data: Current market data
        
    Returns:
        Kelly-optimized position size
    """
    # Get parameters
    win_rate = Decimal(str(sizing_params.get('win_rate', 0.5)))
    avg_win = Decimal(str(sizing_params.get('avg_win', 1.0)))
    avg_loss = Decimal(str(sizing_params.get('avg_loss', 1.0)))
    kelly_fraction = Decimal(str(sizing_params.get('kelly_fraction', 0.25)))
    max_percentage = Decimal(str(sizing_params.get('max_percentage', 0.25)))
    min_size = Decimal(str(sizing_params.get('min_size', 0.01)))
    
    # Validate parameters
    if avg_loss <= 0 or win_rate < 0 or win_rate > 1:
        return Decimal(0)
    
    # Calculate Kelly percentage
    b = avg_win / avg_loss
    q = Decimal(1) - win_rate
    
    kelly_percentage = (win_rate * b - q) / b
    
    # Apply Kelly fraction for safety
    kelly_percentage *= kelly_fraction
    
    # Apply maximum constraint
    kelly_percentage = min(kelly_percentage, max_percentage)
    
    # Ensure positive
    if kelly_percentage <= 0:
        return Decimal(0)
    
    # Adjust for signal strength
    signal_strength = Decimal(str(signal.get('strength', 1.0)))
    kelly_percentage *= abs(signal_strength)
    
    # Get portfolio value and price
    portfolio_value = Decimal(str(portfolio_state.get('total_value', 0)))
    if portfolio_value <= 0:
        return Decimal(0)
    
    prices = market_data.get('prices', {})
    symbol = signal.get('symbol')
    price = prices.get(symbol) if symbol else None
    
    if not price or price <= 0:
        return Decimal(0)
    
    price = Decimal(str(price))
    
    # Calculate position value and shares
    position_value = portfolio_value * kelly_percentage
    shares = position_value / price
    
    return apply_position_constraints(shares, min_size)


def calculate_volatility_based_position_size(
    signal: Dict[str, Any],
    portfolio_state: Dict[str, Any],
    sizing_params: Dict[str, Any],
    market_data: Dict[str, Any]
) -> Decimal:
    """Calculate position size based on volatility targeting.
    
    Pure function - no side effects or state mutations.
    
    Size inversely proportional to asset volatility to achieve
    equal risk contribution.
    
    Args:
        signal: Trading signal with symbol, strength, direction
        portfolio_state: Current portfolio state
        sizing_params: Parameters including 'target_volatility', 'lookback_days',
                      'max_percentage', 'min_size'
        market_data: Current market data with volatility
        
    Returns:
        Volatility-adjusted position size
    """
    # Get parameters
    target_volatility = Decimal(str(sizing_params.get('target_volatility', 0.15)))
    max_percentage = Decimal(str(sizing_params.get('max_percentage', 0.25)))
    min_size = Decimal(str(sizing_params.get('min_size', 0.01)))
    
    # Get symbol from signal
    symbol = signal.get('symbol')
    if not symbol:
        return Decimal(0)
    
    # Get asset volatility from market data
    volatility_data = market_data.get('volatility', {})
    asset_volatility = volatility_data.get(symbol)
    
    if not asset_volatility or asset_volatility <= 0:
        # Fallback to simple percentage sizing
        fallback_params = {
            'percentage': 0.02,
            'min_size': min_size
        }
        return calculate_percentage_position_size(
            signal, portfolio_state, fallback_params, market_data
        )
    
    asset_volatility = Decimal(str(asset_volatility))
    
    # Calculate position percentage based on volatility
    # Lower volatility = larger position
    position_percentage = target_volatility / asset_volatility
    
    # Apply maximum constraint
    position_percentage = min(position_percentage, max_percentage)
    
    # Adjust for signal strength
    signal_strength = Decimal(str(signal.get('strength', 1.0)))
    position_percentage *= abs(signal_strength)
    
    # Get portfolio value and price
    portfolio_value = Decimal(str(portfolio_state.get('total_value', 0)))
    if portfolio_value <= 0:
        return Decimal(0)
    
    prices = market_data.get('prices', {})
    price = prices.get(symbol)
    
    if not price or price <= 0:
        return Decimal(0)
    
    price = Decimal(str(price))
    
    # Calculate position value and shares
    position_value = portfolio_value * position_percentage
    shares = position_value / price
    
    return apply_position_constraints(shares, min_size)


def calculate_atr_based_position_size(
    signal: Dict[str, Any],
    portfolio_state: Dict[str, Any],
    sizing_params: Dict[str, Any],
    market_data: Dict[str, Any]
) -> Decimal:
    """Calculate position size based on ATR and fixed risk.
    
    Pure function - no side effects or state mutations.
    
    Position size = Risk Amount / (ATR * Multiplier)
    
    Args:
        signal: Trading signal with symbol, strength, direction
        portfolio_state: Current portfolio state
        sizing_params: Parameters including 'risk_per_trade', 'atr_multiplier',
                      'max_percentage', 'min_size'
        market_data: Current market data with ATR
        
    Returns:
        ATR-based position size
    """
    # Get parameters
    risk_per_trade = Decimal(str(sizing_params.get('risk_per_trade', 0.01)))
    atr_multiplier = Decimal(str(sizing_params.get('atr_multiplier', 2)))
    max_percentage = Decimal(str(sizing_params.get('max_percentage', 0.25)))
    min_size = Decimal(str(sizing_params.get('min_size', 0.01)))
    
    # Get symbol from signal
    symbol = signal.get('symbol')
    if not symbol:
        return Decimal(0)
    
    # Get ATR from market data
    atr_data = market_data.get('atr', {})
    atr = atr_data.get(symbol)
    
    if not atr or atr <= 0:
        # Fallback to percentage sizing
        fallback_params = {
            'percentage': float(max_percentage * Decimal('0.1')),
            'min_size': min_size
        }
        return calculate_percentage_position_size(
            signal, portfolio_state, fallback_params, market_data
        )
    
    atr = Decimal(str(atr))
    
    # Get portfolio value and price
    portfolio_value = Decimal(str(portfolio_state.get('total_value', 0)))
    if portfolio_value <= 0:
        return Decimal(0)
    
    prices = market_data.get('prices', {})
    price = prices.get(symbol)
    
    if not price or price <= 0:
        return Decimal(0)
    
    price = Decimal(str(price))
    
    # Calculate risk amount
    risk_amount = portfolio_value * risk_per_trade
    
    # Calculate stop distance
    stop_distance = atr * atr_multiplier
    
    if stop_distance <= 0:
        return Decimal(0)
    
    # Calculate shares based on risk
    shares = risk_amount / stop_distance
    
    # Check maximum position constraint
    position_value = shares * price
    max_value = portfolio_value * max_percentage
    
    if position_value > max_value:
        shares = max_value / price
    
    # Adjust for signal strength
    signal_strength = Decimal(str(signal.get('strength', 1.0)))
    shares *= abs(signal_strength)
    
    return apply_position_constraints(shares, min_size)


# Backward compatibility wrappers for OOP interface
class FixedPositionSizer:
    """Fixed position size strategy - backward compatibility wrapper."""
    
    def __init__(self, size: Decimal, min_size: Decimal = Decimal("0.01")):
        self.sizing_params = {
            'fixed_size': float(size),
            'min_size': float(min_size)
        }
    
    def calculate_size(
        self,
        signal: Any,
        portfolio_state: Any,
        market_data: Dict[str, Any]
    ) -> Decimal:
        # Convert signal to dict if needed
        signal_dict = {
            'symbol': getattr(signal, 'symbol', signal.get('symbol') if isinstance(signal, dict) else None),
            'strength': getattr(signal, 'strength', signal.get('strength', 1.0) if isinstance(signal, dict) else 1.0)
        }
        
        # Convert portfolio state to dict if needed
        if hasattr(portfolio_state, 'get_total_value'):
            portfolio_dict = {
                'total_value': float(portfolio_state.get_total_value()),
                'cash': float(getattr(portfolio_state, 'get_cash_balance', lambda: 0)())
            }
        else:
            portfolio_dict = portfolio_state
        
        return calculate_fixed_position_size(
            signal_dict, portfolio_dict, self.sizing_params, market_data
        )


class PercentagePositionSizer:
    """Percentage position sizing - backward compatibility wrapper."""
    
    def __init__(
        self,
        percentage: Decimal,
        use_leverage: bool = False,
        min_size: Decimal = Decimal("0.01")
    ):
        self.sizing_params = {
            'percentage': float(percentage),
            'use_leverage': use_leverage,
            'min_size': float(min_size)
        }
    
    def calculate_size(
        self,
        signal: Any,
        portfolio_state: Any,
        market_data: Dict[str, Any]
    ) -> Decimal:
        # Convert signal to dict if needed
        signal_dict = {
            'symbol': getattr(signal, 'symbol', signal.get('symbol') if isinstance(signal, dict) else None),
            'strength': getattr(signal, 'strength', signal.get('strength', 1.0) if isinstance(signal, dict) else 1.0)
        }
        
        # Convert portfolio state to dict if needed
        if hasattr(portfolio_state, 'get_total_value'):
            portfolio_dict = {
                'total_value': float(portfolio_state.get_total_value()),
                'cash': float(getattr(portfolio_state, 'get_cash_balance', lambda: 0)())
            }
        else:
            portfolio_dict = portfolio_state
        
        return calculate_percentage_position_size(
            signal_dict, portfolio_dict, self.sizing_params, market_data
        )


class KellyCriterionSizer:
    """Kelly Criterion sizing - backward compatibility wrapper."""
    
    def __init__(
        self,
        win_rate: Decimal,
        avg_win: Decimal,
        avg_loss: Decimal,
        kelly_fraction: Decimal = Decimal("0.25"),
        max_percentage: Decimal = Decimal("0.25"),
        min_size: Decimal = Decimal("0.01")
    ):
        self.sizing_params = {
            'win_rate': float(win_rate),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'kelly_fraction': float(kelly_fraction),
            'max_percentage': float(max_percentage),
            'min_size': float(min_size)
        }
    
    def calculate_size(
        self,
        signal: Any,
        portfolio_state: Any,
        market_data: Dict[str, Any]
    ) -> Decimal:
        # Convert signal to dict if needed
        signal_dict = {
            'symbol': getattr(signal, 'symbol', signal.get('symbol') if isinstance(signal, dict) else None),
            'strength': getattr(signal, 'strength', signal.get('strength', 1.0) if isinstance(signal, dict) else 1.0)
        }
        
        # Convert portfolio state to dict if needed
        if hasattr(portfolio_state, 'get_total_value'):
            portfolio_dict = {
                'total_value': float(portfolio_state.get_total_value()),
                'cash': float(getattr(portfolio_state, 'get_cash_balance', lambda: 0)())
            }
        else:
            portfolio_dict = portfolio_state
        
        return calculate_kelly_position_size(
            signal_dict, portfolio_dict, self.sizing_params, market_data
        )


class VolatilityBasedSizer:
    """Volatility-based sizing - backward compatibility wrapper."""
    
    def __init__(
        self,
        target_volatility: Decimal,
        lookback_days: int = 20,
        max_percentage: Decimal = Decimal("0.25"),
        min_size: Decimal = Decimal("0.01")
    ):
        self.sizing_params = {
            'target_volatility': float(target_volatility),
            'lookback_days': lookback_days,
            'max_percentage': float(max_percentage),
            'min_size': float(min_size)
        }
    
    def calculate_size(
        self,
        signal: Any,
        portfolio_state: Any,
        market_data: Dict[str, Any]
    ) -> Decimal:
        # Convert signal to dict if needed
        signal_dict = {
            'symbol': getattr(signal, 'symbol', signal.get('symbol') if isinstance(signal, dict) else None),
            'strength': getattr(signal, 'strength', signal.get('strength', 1.0) if isinstance(signal, dict) else 1.0)
        }
        
        # Convert portfolio state to dict if needed
        if hasattr(portfolio_state, 'get_total_value'):
            portfolio_dict = {
                'total_value': float(portfolio_state.get_total_value()),
                'cash': float(getattr(portfolio_state, 'get_cash_balance', lambda: 0)())
            }
        else:
            portfolio_dict = portfolio_state
        
        return calculate_volatility_based_position_size(
            signal_dict, portfolio_dict, self.sizing_params, market_data
        )