"""
Financial calculations for trading execution.

This module contains all decimal-based calculations specific to trading,
ensuring all financial calculations use Decimal throughout the system.
"""

from decimal import Decimal, ROUND_HALF_UP, getcontext
from typing import Union, Optional
import json


# Set global decimal context for financial calculations
getcontext().prec = 10  # 10 significant digits
getcontext().rounding = ROUND_HALF_UP


def ensure_decimal(value: Union[int, float, str, Decimal]) -> Decimal:
    """
    Ensure a value is converted to Decimal safely.
    
    Args:
        value: Value to convert (int, float, str, or Decimal)
        
    Returns:
        Decimal representation of the value
        
    Raises:
        ValueError: If value cannot be converted to Decimal
    """
    if isinstance(value, Decimal):
        return value
    elif isinstance(value, (int, float)):
        # Convert through string to avoid float precision issues
        return Decimal(str(value))
    elif isinstance(value, str):
        return Decimal(value)
    else:
        raise ValueError(f"Cannot convert {type(value)} to Decimal")


def round_price(price: Decimal, decimals: int = 2) -> Decimal:
    """
    Round a price to specified decimal places.
    
    Args:
        price: Price to round
        decimals: Number of decimal places (default 2)
        
    Returns:
        Rounded price
    """
    return price.quantize(Decimal(f'0.{"0" * decimals}'))


def round_quantity(quantity: Decimal, decimals: int = 0) -> Decimal:
    """
    Round a quantity to specified decimal places.
    
    Args:
        quantity: Quantity to round
        decimals: Number of decimal places (default 0 for whole shares)
        
    Returns:
        Rounded quantity
    """
    if decimals == 0:
        return quantity.quantize(Decimal('1'))
    else:
        return quantity.quantize(Decimal(f'0.{"0" * decimals}'))


def calculate_value(quantity: Decimal, price: Decimal) -> Decimal:
    """
    Calculate the value of a position.
    
    Args:
        quantity: Number of shares
        price: Price per share
        
    Returns:
        Total value (quantity * price)
    """
    return quantity * price


def calculate_commission(quantity: Decimal, price: Decimal, 
                        commission_rate: Decimal = Decimal('0.005')) -> Decimal:
    """
    Calculate commission for a trade.
    
    Args:
        quantity: Number of shares
        price: Price per share  
        commission_rate: Commission per share (default $0.005)
        
    Returns:
        Total commission
    """
    return abs(quantity) * commission_rate


def calculate_slippage(price: Decimal, slippage_bps: int = 5) -> Decimal:
    """
    Calculate slippage amount.
    
    Args:
        price: Base price
        slippage_bps: Slippage in basis points (default 5 bps)
        
    Returns:
        Slippage amount
    """
    slippage_pct = Decimal(str(slippage_bps)) / Decimal('10000')
    return price * slippage_pct


def safe_divide(numerator: Decimal, denominator: Decimal, 
                default: Optional[Decimal] = None) -> Optional[Decimal]:
    """
    Safely divide two decimals, handling zero denominator.
    
    Args:
        numerator: The numerator
        denominator: The denominator
        default: Default value if division by zero (default None)
        
    Returns:
        Result of division or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


def format_currency(value: Decimal, symbol: str = "$") -> str:
    """
    Format a decimal value as currency.
    
    Args:
        value: Value to format
        symbol: Currency symbol (default "$")
        
    Returns:
        Formatted currency string
    """
    rounded = round_price(value)
    return f"{symbol}{rounded:,.2f}"


def format_percentage(value: Decimal, decimals: int = 2) -> str:
    """
    Format a decimal value as percentage.
    
    Args:
        value: Value to format (0.15 = 15%)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    percentage = value * Decimal('100')
    return f"{percentage:.{decimals}f}%"


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types."""
    
    def default(self, obj):
        """Encode Decimal to string for JSON serialization."""
        if isinstance(obj, Decimal):
            return str(obj)
        return super().default(obj)


# Validation functions

def validate_price(price: Decimal) -> bool:
    """Validate that a price is positive."""
    return price > 0


def validate_quantity(quantity: Decimal) -> bool:
    """Validate that a quantity is non-negative."""
    return quantity >= 0


def validate_percentage(percentage: Decimal) -> bool:
    """Validate that a percentage is between 0 and 1."""
    return Decimal('0') <= percentage <= Decimal('1')


# Trading-specific calculations

def calculate_pnl(entry_price: Decimal, exit_price: Decimal, 
                  quantity: Decimal, side: str = 'long') -> Decimal:
    """
    Calculate profit/loss for a trade.
    
    Args:
        entry_price: Entry price
        exit_price: Exit price
        quantity: Number of shares
        side: 'long' or 'short'
        
    Returns:
        P&L amount
    """
    if side == 'long':
        return (exit_price - entry_price) * quantity
    else:  # short
        return (entry_price - exit_price) * quantity


def calculate_return_pct(entry_price: Decimal, exit_price: Decimal, 
                        side: str = 'long') -> Decimal:
    """
    Calculate return percentage for a trade.
    
    Args:
        entry_price: Entry price
        exit_price: Exit price
        side: 'long' or 'short'
        
    Returns:
        Return percentage (0.1 = 10%)
    """
    if entry_price == 0:
        return Decimal('0')
        
    if side == 'long':
        return (exit_price - entry_price) / entry_price
    else:  # short
        return (entry_price - exit_price) / entry_price