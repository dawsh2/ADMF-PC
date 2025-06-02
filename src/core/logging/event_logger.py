"""
File: src/core/logging/event_logger.py
Status: ACTIVE
Version: 1.0.0
Architecture Ref: COMPLEXITY_CHECKLIST.md#step0-documentation-infrastructure
Dependencies: logging, typing
Last Review: 2025-05-31
Next Review: 2025-08-31

Purpose: Provides intelligent event-specific logging utilities for ADMF-PC
trading pipeline as specified in COMPLEXITY_CHECKLIST.md Step 0. Enables
selective logging for specific event types (BAR, SIGNAL, ORDER, FILL, etc.)
and supports trade loop event grouping for system observability.

Key Concepts:
- Selective event type logging for performance optimization
- Trade loop event grouping for workflow debugging
- Backward compatibility for different object types
- Event flow tracking for BACKTEST_README.md#event-flow validation
- Performance-conscious logging with configurable enablement

Critical Dependencies:
- Supports event flow validation required for Step 0+ complexity steps
- Integrates with structured logging infrastructure for system observability
- Enables debugging of trading pipeline event sequences
"""

import logging
from typing import Set, Optional, Any, Dict

# Global configuration
_enabled_event_types: Set[str] = set()
_trade_loop_enabled: bool = False

# Event type constants
class EventTypes:
    """
    Event type constants for selective logging configuration.
    
    This class defines all supported event types in the ADMF-PC trading
    pipeline, enabling selective logging and performance optimization.
    Essential for event flow validation and debugging.
    
    Architecture Context:
        - Part of: Event-Specific Logging (COMPLEXITY_CHECKLIST.md#logging)
        - Supports: Trading pipeline event flow (BACKTEST_README.md#event-flow)
        - Enables: Selective performance-conscious logging
        - Used by: Event logging functions and configuration
    
    Example:
        configure_event_logging([EventTypes.SIGNAL, EventTypes.ORDER])
        log_signal_event(logger, signal)  # Will be logged
        log_bar_event(logger, bar)       # Will be ignored
    """
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
    """
    Configure which event types should be logged.
    
    Args:
        enabled_events: List of event types to enable for logging
        
    Returns:
        None
        
    Example:
        configure_event_logging([EventTypes.SIGNAL, EventTypes.ORDER])
        configure_event_logging([EventTypes.TRADE_LOOP])  # Enables all trading events
    """
    global _enabled_event_types, _trade_loop_enabled
    
    _enabled_event_types = set(enabled_events)
    _trade_loop_enabled = EventTypes.TRADE_LOOP in _enabled_event_types
    
    # If TRADE_LOOP is enabled, enable all trade-related events
    if _trade_loop_enabled:
        _enabled_event_types.update(TRADE_LOOP_EVENTS)

def should_log_event(event_type: str) -> bool:
    """
    Check if this event type should be logged.
    
    Args:
        event_type: Event type to check (e.g., EventTypes.SIGNAL)
        
    Returns:
        True if event should be logged, False otherwise
        
    Example:
        if should_log_event(EventTypes.SIGNAL):
            log_signal_event(logger, signal)
    """
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
    """
    Log an event if its type is enabled.
    
    Args:
        logger: Logger instance to use
        event_type: Type of event being logged
        message: Event message
        **kwargs: Additional context for the event
        
    Returns:
        None
        
    Example:
        log_event(logger, EventTypes.SIGNAL, "BUY signal generated", symbol="SPY")
    """
    if should_log_event(event_type):
        extra_info = f" | {kwargs}" if kwargs else ""
        logger.info(f"[{event_type}] {message}{extra_info}")

def log_bar_event(logger: logging.Logger, symbol: str, timestamp: Any, price: float, bar_num: int = None) -> None:
    """
    Log a BAR event for market data processing.
    
    Args:
        logger: Logger instance to use
        symbol: Trading symbol (e.g., "SPY")
        timestamp: Bar timestamp
        price: Bar price
        bar_num: Optional bar number for sequence tracking
        
    Returns:
        None
        
    Example:
        log_bar_event(logger, "SPY", datetime.now(), 100.50, 42)
    """
    if should_log_event(EventTypes.BAR):
        bar_info = f" ({bar_num})" if bar_num else ""
        log_event(logger, EventTypes.BAR, f"{symbol} @ {timestamp}{bar_info} - Price: {price:.4f}")

def log_indicator_event(logger: logging.Logger, symbol: str, indicator: str, value) -> None:
    """
    Log an INDICATOR event for technical analysis.
    
    Args:
        logger: Logger instance to use
        symbol: Trading symbol (e.g., "SPY")
        indicator: Indicator name (e.g., "SMA_20", "RSI")
        value: Calculated indicator value (float or dict)
        
    Returns:
        None
        
    Example:
        log_indicator_event(logger, "SPY", "RSI_14", 65.4)
        log_indicator_event(logger, "SPY", "BB_20", {"upper": 520, "middle": 510, "lower": 500})
    """ 
    if should_log_event(EventTypes.INDICATOR):
        if isinstance(value, dict):
            # Format dict values for Bollinger Bands etc
            value_str = ", ".join(f"{k}={v:.2f}" if isinstance(v, (int, float)) else f"{k}={v}" 
                                for k, v in value.items())
            log_event(logger, EventTypes.INDICATOR, f"{symbol} {indicator} = {{{value_str}}}")
        elif isinstance(value, (int, float)):
            log_event(logger, EventTypes.INDICATOR, f"{symbol} {indicator} = {value:.4f}")
        else:
            log_event(logger, EventTypes.INDICATOR, f"{symbol} {indicator} = {value}")

def log_signal_event(logger: logging.Logger, signal) -> None:
    """
    Log a SIGNAL event for trading strategy output.
    
    Args:
        logger: Logger instance to use
        signal: Signal object or dictionary with signal information
        
    Returns:
        None
        
    Example:
        signal = Signal(symbol="SPY", side=OrderSide.BUY, strength=0.8)
        log_signal_event(logger, signal)
    """
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
    """
    Log an ORDER event for trade execution.
    
    Args:
        logger: Logger instance to use
        order: Order object or dictionary with order information
        
    Returns:
        None
        
    Example:
        order = Order(symbol="SPY", side=OrderSide.BUY, quantity=100)
        log_order_event(logger, order)
    """
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
    """
    Log a FILL event for trade execution completion.
    
    Args:
        logger: Logger instance to use
        fill: Fill object or dictionary with execution information
        
    Returns:
        None
        
    Example:
        fill = Fill(symbol="SPY", side=OrderSide.BUY, quantity=100, price=100.50)
        log_fill_event(logger, fill)
    """
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
    """
    Log a PORTFOLIO event for portfolio state tracking.
    
    Args:
        logger: Logger instance to use
        cash: Current cash balance
        positions: Dictionary of current positions
        
    Returns:
        None
        
    Example:
        positions = {"SPY": {"quantity": 100, "market_value": 10050}}
        log_portfolio_event(logger, 50000.0, positions)
    """
    if should_log_event(EventTypes.PORTFOLIO):
        position_count = len(positions)
        total_value = cash + sum(pos.get('market_value', 0) for pos in positions.values())
        log_event(logger, EventTypes.PORTFOLIO, f"Cash: ${cash:.2f} | Positions: {position_count} | Total: ${total_value:.2f}")

def get_event_logger(name: str) -> logging.Logger:
    """
    Get a logger for event logging.
    
    Args:
        name: Logger name (typically module or component name)
        
    Returns:
        Configured logger instance for event logging
        
    Example:
        logger = get_event_logger("strategy.momentum")
        log_signal_event(logger, signal)
    """
    return logging.getLogger(name)

# Create module logger
logger = get_event_logger(__name__)