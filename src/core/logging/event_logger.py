"""
Event-specific logging utilities for ADMF-PC.

This module provides intelligent event logging that can be selectively
enabled for specific event types or the entire trade loop.
"""

import logging
from typing import Set, Optional, Any, Dict

# Global configuration
_enabled_event_types: Set[str] = set()
_trade_loop_enabled: bool = False

# Event type constants
class EventTypes:
    BAR = "BAR"
    INDICATOR = "INDICATOR" 
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"
    PORTFOLIO = "PORTFOLIO"
    TRADE_LOOP = "TRADE_LOOP"

# Trade loop includes all trading-related events
TRADE_LOOP_EVENTS = {
    EventTypes.BAR, 
    EventTypes.INDICATOR, 
    EventTypes.SIGNAL, 
    EventTypes.ORDER, 
    EventTypes.FILL, 
    EventTypes.PORTFOLIO
}

def configure_event_logging(enabled_events: list) -> None:
    """Configure which event types should be logged."""
    global _enabled_event_types, _trade_loop_enabled
    
    _enabled_event_types = set(enabled_events)
    _trade_loop_enabled = EventTypes.TRADE_LOOP in _enabled_event_types
    
    # If TRADE_LOOP is enabled, enable all trade-related events
    if _trade_loop_enabled:
        _enabled_event_types.update(TRADE_LOOP_EVENTS)

def should_log_event(event_type: str) -> bool:
    """Check if this event type should be logged."""
    # If no specific events configured, log everything (backward compatibility)
    if not _enabled_event_types:
        return True
    
    return event_type in _enabled_event_types

def log_event(
    logger: logging.Logger, 
    event_type: str, 
    message: str,
    **kwargs
) -> None:
    """Log an event if its type is enabled."""
    if should_log_event(event_type):
        extra_info = f" | {kwargs}" if kwargs else ""
        logger.info(f"[{event_type}] {message}{extra_info}")

def log_bar_event(logger: logging.Logger, symbol: str, timestamp: Any, price: float, bar_num: int = None) -> None:
    """Log a BAR event."""
    if should_log_event(EventTypes.BAR):
        bar_info = f" ({bar_num})" if bar_num else ""
        log_event(logger, EventTypes.BAR, f"{symbol} @ {timestamp}{bar_info} - Price: {price:.4f}")

def log_indicator_event(logger: logging.Logger, symbol: str, indicator: str, value: float) -> None:
    """Log an INDICATOR event.""" 
    if should_log_event(EventTypes.INDICATOR):
        log_event(logger, EventTypes.INDICATOR, f"{symbol} {indicator} = {value:.4f}")

def log_signal_event(logger: logging.Logger, signal) -> None:
    """Log a SIGNAL event."""
    if should_log_event(EventTypes.SIGNAL):
        # Handle both Signal objects and dictionaries for backward compatibility
        if hasattr(signal, 'symbol'):  # Signal dataclass object
            symbol = signal.symbol
            direction = signal.side  # Signal uses 'side' instead of 'direction'
            strength = float(signal.strength)
            reason = signal.metadata.get('reason', 'No reason')
        else:  # Dictionary (backward compatibility)
            symbol = signal.get('symbol', 'UNKNOWN')
            direction = signal.get('direction', 'UNKNOWN') 
            strength = signal.get('strength', 0)
            reason = signal.get('metadata', {}).get('reason', 'No reason')
        log_event(logger, EventTypes.SIGNAL, f"{symbol} {direction} (strength: {strength:.2f}) - {reason}")

def log_order_event(logger: logging.Logger, order) -> None:
    """Log an ORDER event."""
    if should_log_event(EventTypes.ORDER):
        # Handle both Order objects and dictionaries for backward compatibility
        if hasattr(order, 'symbol'):  # Order object
            symbol = order.symbol
            side = order.side.value if hasattr(order.side, 'value') else str(order.side)
            quantity = float(order.quantity)
            price = float(order.price) if order.price else 0.0
        else:  # Dictionary (backward compatibility)
            symbol = order.get('symbol', 'UNKNOWN')
            side = order.get('side', 'UNKNOWN')
            quantity = order.get('quantity', 0)
            price = order.get('price', 0)
        log_event(logger, EventTypes.ORDER, f"{symbol} {side} {quantity:.2f} @ {price:.4f}")

def log_fill_event(logger: logging.Logger, fill) -> None:
    """Log a FILL event."""
    if should_log_event(EventTypes.FILL):
        # Handle both Fill objects and dictionaries for backward compatibility
        if hasattr(fill, 'symbol'):  # Fill object
            symbol = fill.symbol
            side = fill.side.value if hasattr(fill.side, 'value') else str(fill.side)
            quantity = float(fill.quantity)
            price = float(fill.price)
        else:  # Dictionary (backward compatibility)
            symbol = fill.get('symbol', 'UNKNOWN')
            side = fill.get('side', 'UNKNOWN')
            quantity = fill.get('quantity', 0)
            price = fill.get('price', 0)
        log_event(logger, EventTypes.FILL, f"FILLED: {symbol} {side} {quantity:.2f} @ {price:.4f}")

def log_portfolio_event(logger: logging.Logger, cash: float, positions: Dict[str, Any]) -> None:
    """Log a PORTFOLIO event."""
    if should_log_event(EventTypes.PORTFOLIO):
        position_count = len(positions)
        total_value = cash + sum(pos.get('market_value', 0) for pos in positions.values())
        log_event(logger, EventTypes.PORTFOLIO, f"Cash: ${cash:.2f} | Positions: {position_count} | Total: ${total_value:.2f}")

def get_event_logger(name: str) -> logging.Logger:
    """Get a logger for event logging."""
    return logging.getLogger(name)

# Create module logger
logger = get_event_logger(__name__)