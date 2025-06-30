"""
Exit monitoring for stop loss, take profit, and trailing stop functionality.

This module provides pure functions for checking exit conditions on positions,
maintaining the stateless architecture while enabling risk management.
"""

from typing import Dict, Any, Optional, Tuple
from decimal import Decimal
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExitSignal:
    """Exit signal information."""
    should_exit: bool
    exit_type: Optional[str] = None  # 'stop_loss', 'take_profit', 'trailing_stop'
    reason: Optional[str] = None
    urgency: str = 'normal'  # 'immediate', 'high', 'normal'
    exit_price: Optional[Decimal] = None  # Exact price where exit should occur


def check_exit_conditions(
    position: Dict[str, Any],
    current_price: Decimal,
    risk_rules: Dict[str, Any]
) -> ExitSignal:
    """
    Check if position should be exited based on risk rules.
    
    Args:
        position: Position information including:
            - symbol: Trading symbol
            - quantity: Position size (positive for long, negative for short)
            - average_price: Entry price
            - metadata: Dict with highest_price, bars_held, etc.
        current_price: Current market price
        risk_rules: Risk parameters including:
            - stop_loss: Stop loss percentage (e.g., 0.001 for 0.1%)
            - take_profit: Take profit percentage (e.g., 0.0015 for 0.15%)
            - trailing_stop: Trailing stop percentage (e.g., 0.0005 for 0.05%)
            - max_holding_period: Maximum bars to hold position
            
    Returns:
        ExitSignal with exit decision and reason
    """
    if not position or position.get('quantity', 0) == 0:
        return ExitSignal(should_exit=False, reason="No position")
    
    quantity = Decimal(str(position['quantity']))
    entry_price = Decimal(str(position.get('average_price', 0)))
    
    if entry_price <= 0:
        logger.warning(f"Invalid entry price for {position.get('symbol')}: {entry_price}")
        return ExitSignal(should_exit=False, reason="Invalid entry price")
    
    # Calculate current P&L percentage
    if quantity > 0:  # Long position
        pnl_pct = (current_price - entry_price) / entry_price
    else:  # Short position
        pnl_pct = (entry_price - current_price) / entry_price
    
    # Check stop loss
    stop_loss = risk_rules.get('stop_loss')
    if stop_loss and pnl_pct <= -Decimal(str(stop_loss)):
        return ExitSignal(
            should_exit=True,
            exit_type='stop_loss',
            reason=f"Stop loss hit: {pnl_pct:.2%} <= -{stop_loss*100:.3f}%",
            urgency='immediate'
        )
    
    # Check take profit
    take_profit = risk_rules.get('take_profit')
    if take_profit and pnl_pct >= Decimal(str(take_profit)):
        return ExitSignal(
            should_exit=True,
            exit_type='take_profit',
            reason=f"Take profit hit: {pnl_pct:.2%} >= {take_profit*100:.3f}%",
            urgency='normal'
        )
    
    # Check trailing stop (only for positions in profit)
    trailing_stop = risk_rules.get('trailing_stop')
    if trailing_stop and quantity > 0:  # Currently only for long positions
        metadata = position.get('metadata', {})
        
        # Check if trailing stop price exists (set by risk manager)
        trailing_stop_price_str = metadata.get('trailing_stop_price')
        if trailing_stop_price_str:
            trailing_stop_price = Decimal(str(trailing_stop_price_str))
            if current_price <= trailing_stop_price:
                return ExitSignal(
                    should_exit=True,
                    exit_type='trailing_stop',
                    reason=f"Trailing stop hit: {current_price:.4f} <= {trailing_stop_price:.4f}",
                    urgency='high'
                )
        else:
            # If no trailing stop price yet, check if we should have one
            # This handles the initial case where position just went profitable
            highest_price = Decimal(str(metadata.get('highest_price', entry_price)))
            if highest_price > entry_price:
                # Position is in profit, calculate where trailing stop would be
                calc_trailing_stop = highest_price * (1 - Decimal(str(trailing_stop)))
                if current_price <= calc_trailing_stop:
                    return ExitSignal(
                        should_exit=True,
                        exit_type='trailing_stop',
                        reason=f"Trailing stop hit: {current_price:.4f} <= {calc_trailing_stop:.4f} (from high {highest_price:.4f})",
                        urgency='high'
                    )
    
    # Check max holding period
    max_holding_period = risk_rules.get('max_holding_period')
    if max_holding_period:
        bars_held = position.get('metadata', {}).get('bars_held', 0)
        if bars_held >= max_holding_period:
            return ExitSignal(
                should_exit=True,
                exit_type='time_based',
                reason=f"Max holding period reached: {bars_held} >= {max_holding_period} bars",
                urgency='normal'
            )
    
    # No exit conditions met
    return ExitSignal(
        should_exit=False,
        reason=f"Position OK (P&L: {pnl_pct:.2%})"
    )


def check_price_level_exits(
    position: Dict[str, Any],
    price_to_check: Decimal,
    risk_rules: Dict[str, Any],
    price_type: str = 'current'
) -> ExitSignal:
    """
    Check if position should be exited based on actual price levels.
    
    This function checks if the given price crosses stop loss or take profit
    price levels, similar to how the analysis notebook works.
    
    Args:
        position: Position information including entry price and quantity
        price_to_check: The price to check (could be high, low, or close)
        risk_rules: Risk parameters with stop_loss and take_profit percentages
        price_type: Type of price being checked ('high', 'low', or 'current')
            
    Returns:
        ExitSignal with exit decision based on price levels
    """
    if not position or position.get('quantity', 0) == 0:
        return ExitSignal(should_exit=False, reason="No position")
    
    quantity = Decimal(str(position['quantity']))
    entry_price = Decimal(str(position.get('average_price', 0)))
    
    if entry_price <= 0:
        return ExitSignal(should_exit=False, reason="Invalid entry price")
    
    # Calculate stop loss and take profit price levels
    stop_loss_pct = risk_rules.get('stop_loss', 0)
    take_profit_pct = risk_rules.get('take_profit', 0)
    
    if quantity > 0:  # Long position
        stop_price = entry_price * (1 - Decimal(str(stop_loss_pct))) if stop_loss_pct else Decimal('0')
        target_price = entry_price * (1 + Decimal(str(take_profit_pct))) if take_profit_pct else Decimal('999999')
        
        # For long positions:
        # - Stop loss is triggered when low <= stop_price
        # - Take profit is triggered when high >= target_price
        if price_type == 'low' and stop_price > 0 and price_to_check <= stop_price:
            pnl_pct = (stop_price - entry_price) / entry_price
            return ExitSignal(
                should_exit=True,
                exit_type='stop_loss',
                exit_price=stop_price,
                reason=f"Stop loss hit: price {price_to_check:.4f} <= stop {stop_price:.4f} ({pnl_pct:.2%})",
                urgency='immediate'
            )
        elif price_type == 'high' and target_price < 999999 and price_to_check >= target_price:
            pnl_pct = (target_price - entry_price) / entry_price
            return ExitSignal(
                should_exit=True,
                exit_type='take_profit',
                exit_price=target_price,
                reason=f"Take profit hit: price {price_to_check:.4f} >= target {target_price:.4f} ({pnl_pct:.2%})",
                urgency='normal'
            )
    else:  # Short position
        stop_price = entry_price * (1 + Decimal(str(stop_loss_pct))) if stop_loss_pct else Decimal('999999')
        target_price = entry_price * (1 - Decimal(str(take_profit_pct))) if take_profit_pct else Decimal('0')
        
        # For short positions:
        # - Stop loss is triggered when high >= stop_price
        # - Take profit is triggered when low <= target_price
        if price_type == 'high' and stop_price < 999999 and price_to_check >= stop_price:
            pnl_pct = (entry_price - stop_price) / entry_price
            return ExitSignal(
                should_exit=True,
                exit_type='stop_loss',
                exit_price=stop_price,
                reason=f"Stop loss hit: price {price_to_check:.4f} >= stop {stop_price:.4f} ({pnl_pct:.2%})",
                urgency='immediate'
            )
        elif price_type == 'low' and target_price > 0 and price_to_check <= target_price:
            pnl_pct = (entry_price - target_price) / entry_price
            return ExitSignal(
                should_exit=True,
                exit_type='take_profit',
                exit_price=target_price,
                reason=f"Take profit hit: price {price_to_check:.4f} <= target {target_price:.4f} ({pnl_pct:.2%})",
                urgency='normal'
            )
    
    # No exit conditions met at this price level
    return ExitSignal(
        should_exit=False,
        reason=f"Price {price_to_check:.4f} within position range"
    )


def calculate_exit_prices(
    entry_price: Decimal,
    risk_rules: Dict[str, Any],
    is_long: bool = True
) -> Dict[str, Optional[Decimal]]:
    """
    Calculate exit price levels based on risk rules.
    
    Args:
        entry_price: Position entry price
        risk_rules: Risk parameters
        is_long: True for long positions, False for short
        
    Returns:
        Dictionary with stop_loss_price and take_profit_price
    """
    exit_prices = {
        'stop_loss_price': None,
        'take_profit_price': None,
        'initial_trailing_stop_price': None
    }
    
    # Calculate stop loss price
    stop_loss = risk_rules.get('stop_loss')
    if stop_loss:
        if is_long:
            exit_prices['stop_loss_price'] = entry_price * (1 - Decimal(str(stop_loss)))
        else:
            exit_prices['stop_loss_price'] = entry_price * (1 + Decimal(str(stop_loss)))
    
    # Calculate take profit price
    take_profit = risk_rules.get('take_profit')
    if take_profit:
        if is_long:
            exit_prices['take_profit_price'] = entry_price * (1 + Decimal(str(take_profit)))
        else:
            exit_prices['take_profit_price'] = entry_price * (1 - Decimal(str(take_profit)))
    
    # Calculate initial trailing stop (same as stop loss initially)
    trailing_stop = risk_rules.get('trailing_stop')
    if trailing_stop and is_long:  # Currently only for long positions
        exit_prices['initial_trailing_stop_price'] = entry_price * (1 - Decimal(str(trailing_stop)))
    
    return exit_prices


def format_exit_levels(position: Dict[str, Any], risk_rules: Dict[str, Any]) -> str:
    """
    Format position exit levels for logging.
    
    Args:
        position: Current position
        risk_rules: Risk parameters
        
    Returns:
        Formatted string with exit levels
    """
    entry_price = Decimal(str(position.get('average_price', 0)))
    current_price = Decimal(str(position.get('current_price', entry_price)))
    quantity = Decimal(str(position.get('quantity', 0)))
    
    if entry_price <= 0 or quantity == 0:
        return "No valid position"
    
    is_long = quantity > 0
    exit_prices = calculate_exit_prices(entry_price, risk_rules, is_long)
    
    # Calculate current P&L
    if is_long:
        pnl_pct = (current_price - entry_price) / entry_price
    else:
        pnl_pct = (entry_price - current_price) / entry_price
    
    lines = [
        f"Position: {'LONG' if is_long else 'SHORT'} {abs(quantity)} @ {entry_price:.4f}",
        f"Current: {current_price:.4f} (P&L: {pnl_pct:.2%})"
    ]
    
    if exit_prices['stop_loss_price']:
        lines.append(f"Stop Loss: {exit_prices['stop_loss_price']:.4f}")
    
    if exit_prices['take_profit_price']:
        lines.append(f"Take Profit: {exit_prices['take_profit_price']:.4f}")
    
    # Check for active trailing stop
    metadata = position.get('metadata', {})
    if 'trailing_stop_price' in metadata:
        lines.append(f"Trailing Stop: {metadata['trailing_stop_price']:.4f} "
                    f"(High: {metadata.get('highest_price', entry_price):.4f})")
    
    return " | ".join(lines)