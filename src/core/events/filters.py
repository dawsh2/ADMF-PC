"""
Filter functions for event subscriptions.

This module provides filter functions for mandatory event filtering,
particularly for SIGNAL and FILL events to prevent cross-contamination
in multi-container systems.
"""

from typing import Callable, List, Set, Optional, Any, TYPE_CHECKING
from .types import Event

if TYPE_CHECKING:
    from ..containers.protocols import OrderTrackingProtocol


# Strategy filtering (existing functionality)

def strategy_filter(strategy_ids: List[str]) -> Callable[[Event], bool]:
    """
    Create a filter for specific strategy IDs.
    
    Used for SIGNAL event filtering to ensure containers only receive
    signals from their assigned strategies.
    
    Args:
        strategy_ids: List of strategy IDs to accept
        
    Returns:
        Filter function
        
    Example:
        filter_func = strategy_filter(['momentum_1', 'pairs_1'])
    """
    strategy_set = set(strategy_ids)
    
    def filter_func(event: Event) -> bool:
        return event.payload.get('strategy_id') in strategy_set
    
    return filter_func


# Container-based filtering for FILL events

def container_filter(container_id: str) -> Callable[[Event], bool]:
    """
    Create filter for FILL events belonging to specific container.
    
    This is the primary filter for FILL events to prevent cross-contamination
    between portfolio containers.
    
    Args:
        container_id: Container ID that should receive these fills
        
    Returns:
        Filter function for container-specific fills
        
    Example:
        # Portfolio only receives its own fills
        bus.subscribe(
            EventType.FILL.value,
            portfolio.handle_fill,
            filter_func=container_filter('portfolio_1')
        )
    """
    def filter_func(event: Event) -> bool:
        # Check multiple possible sources of container identification
        event_container = (
            event.payload.get('container_id') or          # Explicit in payload
            event.payload.get('portfolio_id') or          # Alternative naming
            event.container_id or                         # Event metadata
            event.payload.get('order', {}).get('container_id')  # Nested in order
        )
        return event_container == container_id
    
    return filter_func


def order_filter(order_ids: List[str]) -> Callable[[Event], bool]:
    """
    Create filter for FILL events from specific orders.
    
    Alternative to container filtering when you know specific order IDs.
    
    Args:
        order_ids: List of order IDs to accept fills for
        
    Returns:
        Filter function for order-specific fills
        
    Example:
        # Only receive fills for our pending orders
        bus.subscribe(
            EventType.FILL.value,
            portfolio.handle_fill,
            filter_func=order_filter(portfolio.pending_order_ids)
        )
    """
    order_set = set(order_ids)
    
    def filter_func(event: Event) -> bool:
        order_id = (
            event.payload.get('order_id') or
            event.payload.get('order', {}).get('id') or
            event.correlation_id  # Sometimes order_id is correlation_id
        )
        return order_id in order_set
    
    return filter_func


def order_ownership_filter(container: 'OrderTrackingProtocol') -> Callable[[Event], bool]:
    """
    Create filter for FILL events based on order ownership.
    
    This filter uses the OrderTrackingProtocol to check if a container
    owns the order associated with the FILL event.
    
    Args:
        container: Container implementing OrderTrackingProtocol
        
    Returns:
        Filter function checking order ownership
        
    Example:
        # Filter based on order tracking
        bus.subscribe(
            EventType.FILL.value,
            portfolio.handle_fill,
            filter_func=order_ownership_filter(portfolio)
        )
    """
    def filter_func(event: Event) -> bool:
        order_id = event.payload.get('order_id')
        symbol = event.payload.get('symbol')
        timeframe = event.payload.get('timeframe', '1m')  # Default timeframe
        
        if not order_id or not symbol:
            return False
            
        # Get container_id from the container
        container_id = getattr(container, 'container_id', None)
        if not container_id:
            return False
            
        return container.has_pending_orders(container_id, symbol, timeframe)
    
    return filter_func


# Composite filters

def portfolio_symbol_filter(container_id: str, symbols: List[str]) -> Callable[[Event], bool]:
    """
    Combined filter for container + symbol filtering.
    
    More restrictive - only fills for this container AND these symbols.
    
    Args:
        container_id: Container that should receive fills
        symbols: Symbols this container trades
        
    Returns:
        Combined filter function
        
    Example:
        # Portfolio only gets fills for its container and symbols
        bus.subscribe(
            EventType.FILL.value,
            portfolio.handle_fill,
            filter_func=portfolio_symbol_filter('portfolio_1', ['AAPL', 'MSFT'])
        )
    """
    container_check = container_filter(container_id)
    symbol_set = set(symbols)
    
    def filter_func(event: Event) -> bool:
        # Must pass container filter
        if not container_check(event):
            return False
        
        # Must be for one of our symbols
        symbol = event.payload.get('symbol')
        return symbol in symbol_set
    
    return filter_func


def combine_filters(*filters: Callable[[Event], bool]) -> Callable[[Event], bool]:
    """
    Combine multiple filters with AND logic.
    
    All filters must pass for the event to be delivered.
    
    Args:
        *filters: Variable number of filter functions
        
    Returns:
        Combined filter function
        
    Example:
        # Only receive momentum signals for tech stocks
        filter_func = combine_filters(
            strategy_filter(['momentum_1']),
            symbol_filter(['AAPL', 'MSFT', 'GOOGL'])
        )
    """
    def combined(event: Event) -> bool:
        return all(f(event) for f in filters)
    return combined


def any_of_filters(*filters: Callable[[Event], bool]) -> Callable[[Event], bool]:
    """
    Combine multiple filters with OR logic.
    
    Any filter passing allows the event to be delivered.
    
    Args:
        *filters: Variable number of filter functions
        
    Returns:
        Combined filter function
        
    Example:
        # Receive signals from either momentum OR mean reversion strategies
        filter_func = any_of_filters(
            strategy_filter(['momentum_1']),
            strategy_filter(['mean_reversion_1'])
        )
    """
    def any_of(event: Event) -> bool:
        return any(f(event) for f in filters)
    return any_of


# Additional utility filters

def symbol_filter(symbols: List[str]) -> Callable[[Event], bool]:
    """
    Create a filter for specific symbols.
    
    Args:
        symbols: List of symbols to accept
        
    Returns:
        Filter function
        
    Example:
        filter_func = symbol_filter(['AAPL', 'MSFT', 'GOOGL'])
    """
    symbol_set = set(symbols)
    
    def filter_func(event: Event) -> bool:
        return event.payload.get('symbol') in symbol_set
    
    return filter_func


def metadata_filter(key: str, value: Any) -> Callable[[Event], bool]:
    """
    Create a filter for specific metadata values.
    
    Args:
        key: Metadata key to check
        value: Required value
        
    Returns:
        Filter function
        
    Example:
        # Only events from backtest containers
        filter_func = metadata_filter('container_type', 'backtest')
    """
    def filter_func(event: Event) -> bool:
        return event.metadata.get(key) == value
    
    return filter_func