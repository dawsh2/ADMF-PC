"""
Filter helper functions for event subscriptions.

These helpers make it easy to create and compose filters for
common subscription patterns.
"""

from typing import Callable, List, Optional, Set, Any
from ..types import Event


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


def strategy_filter(strategy_ids: List[str]) -> Callable[[Event], bool]:
    """
    Create a filter for specific strategy IDs.
    
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


def classification_filter(classifications: List[str]) -> Callable[[Event], bool]:
    """
    Create a filter for specific market classifications.
    
    Args:
        classifications: List of classifications to accept
        
    Returns:
        Filter function
        
    Example:
        # Only trade in trending markets
        filter_func = classification_filter(['strong_uptrend', 'strong_downtrend'])
    """
    classification_set = set(classifications)
    
    def filter_func(event: Event) -> bool:
        return event.payload.get('classification') in classification_set
    
    return filter_func


def strength_filter(min_strength: float, max_strength: float = 1.0) -> Callable[[Event], bool]:
    """
    Create a filter for signal strength.
    
    Args:
        min_strength: Minimum signal strength
        max_strength: Maximum signal strength
        
    Returns:
        Filter function
        
    Example:
        # Only high conviction signals
        filter_func = strength_filter(0.8)
    """
    def filter_func(event: Event) -> bool:
        strength = event.payload.get('strength', 0.0)
        return min_strength <= strength <= max_strength
    
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


def payload_filter(key: str, value: Any) -> Callable[[Event], bool]:
    """
    Create a filter for specific payload values.
    
    Args:
        key: Payload key to check
        value: Required value
        
    Returns:
        Filter function
        
    Example:
        # Only BUY signals
        filter_func = payload_filter('direction', 'BUY')
    """
    def filter_func(event: Event) -> bool:
        return event.payload.get(key) == value
    
    return filter_func


def custom_filter(predicate: Callable[[Event], bool]) -> Callable[[Event], bool]:
    """
    Create a filter with custom logic.
    
    This is just a wrapper for clarity when building filter compositions.
    
    Args:
        predicate: Custom predicate function
        
    Returns:
        Filter function
        
    Example:
        # Complex custom logic
        filter_func = custom_filter(
            lambda e: e.payload.get('volume', 0) > 1000000 and 
                     e.payload.get('price', 0) > 50
        )
    """
    return predicate


# ============================================
# Example Usage Patterns
# ============================================

def create_portfolio_filter(
    strategy_ids: List[str],
    symbols: Optional[List[str]] = None,
    min_strength: float = 0.0,
    classifications: Optional[List[str]] = None
) -> Callable[[Event], bool]:
    """
    Create a comprehensive portfolio filter.
    
    This is an example of how to compose filters for a portfolio container.
    
    Args:
        strategy_ids: Required list of strategy IDs
        symbols: Optional symbol whitelist
        min_strength: Minimum signal strength
        classifications: Optional classification whitelist
        
    Returns:
        Combined filter function
        
    Example:
        filter_func = create_portfolio_filter(
            strategy_ids=['momentum_1', 'pairs_1'],
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            min_strength=0.7,
            classifications=['trending', 'breakout']
        )
    """
    filters = [strategy_filter(strategy_ids)]
    
    if symbols:
        filters.append(symbol_filter(symbols))
    
    if min_strength > 0:
        filters.append(strength_filter(min_strength))
    
    if classifications:
        filters.append(classification_filter(classifications))
    
    return combine_filters(*filters)