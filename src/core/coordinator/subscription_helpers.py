"""
Helper functions for setting up structured event subscriptions.

This module provides utilities for creating subscription filters
that work with both legacy string-based and new structured events.
"""

from typing import Dict, Any, List, Optional, Callable
from ..events import Event, EventType
from ..events.structured_events import SubscriptionDescriptor, StructuredEventFilter
import logging

logger = logging.getLogger(__name__)


def create_portfolio_subscription_filter(
    managed_strategies: List[Dict[str, Any]],
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None
) -> Callable[[Event], bool]:
    """
    Create a filter function for portfolio signal subscriptions.
    
    Handles both:
    - New structured events with embedded parameters
    - Legacy events with string-based strategy_id
    
    Args:
        managed_strategies: List of strategy configurations the portfolio manages
        symbol: Optional symbol filter
        timeframe: Optional timeframe filter
        
    Returns:
        Filter function for event bus subscription
    
    Example:
        managed_strategies = [
            {'type': 'sma_crossover', 'params': {'fast_period': 5, 'slow_period': 20}},
            {'type': 'rsi_threshold', 'params': {'period': 14, 'threshold': 30}}
        ]
        filter_func = create_portfolio_subscription_filter(managed_strategies, 'SPY', '1m')
    """
    # Create descriptors for each managed strategy
    descriptors = []
    
    for strategy_config in managed_strategies:
        criteria = {}
        
        # Add symbol/timeframe filters if specified
        if symbol:
            criteria['symbol'] = symbol
        if timeframe:
            criteria['timeframe'] = timeframe
        
        # Handle different strategy config formats
        if isinstance(strategy_config, dict):
            if 'type' in strategy_config:
                criteria['strategy_type'] = strategy_config['type']
            elif 'name' in strategy_config:
                criteria['strategy_type'] = strategy_config['name']
            
            # Add parameter matching if specified
            if 'params' in strategy_config and strategy_config['params']:
                criteria['parameters'] = strategy_config['params']
        
        elif isinstance(strategy_config, str):
            # Simple string format - just match strategy type
            criteria['strategy_type'] = strategy_config
        
        if criteria:
            descriptors.append(SubscriptionDescriptor(criteria))
    
    def filter_func(event: Event) -> bool:
        """Filter function that handles both event formats."""
        if event.event_type != EventType.SIGNAL.value:
            return False
        
        # Check if this is a structured event (v2.0)
        if event.metadata.get('version') == '2.0':
            # Use structured matching
            return any(desc.matches(event) for desc in descriptors)
        
        else:
            # Legacy matching based on strategy_id string
            strategy_id = event.payload.get('strategy_id', '')
            
            # First check if strategy_type is directly in payload
            payload_strategy_type = event.payload.get('strategy_type', '')
            
            # Debug logging for parameter matching
            signal_params = event.payload.get('parameters', {})
            if managed_strategies:
                logger.debug(f"Filter check - Signal {strategy_id}: strategy_type={payload_strategy_type}, params={signal_params}, managed={managed_strategies}")
            
            # Extract strategy type from legacy ID
            # Format: "SYMBOL_TIMEFRAME_STRATEGY_TYPE_PARAMS..."
            parts = strategy_id.split('_')
            
            for strategy_config in managed_strategies:
                if isinstance(strategy_config, dict):
                    strategy_type = strategy_config.get('type') or strategy_config.get('name')
                else:
                    strategy_type = strategy_config
                
                # Check if strategy type matches directly from payload
                if payload_strategy_type and strategy_type == payload_strategy_type:
                    return True
                
                # Check if strategy type appears in the ID
                if strategy_type in parts or strategy_id.endswith(f'_{strategy_type}'):
                    return True
                
                # Check if the managed strategy is a parameterized version (e.g., bollinger_bands_10_10)
                # and the payload strategy type is the base type (e.g., bollinger_bands)
                if payload_strategy_type and str(strategy_type).startswith(payload_strategy_type):
                    logger.debug(f"Match by prefix: {strategy_type} starts with {payload_strategy_type}")
                    # Don't return True here - continue to parameter checking below
                    # return True
                
                # Check if managed strategy name encodes parameters (e.g., bollinger_bands_13_10)
                # and the signal contains matching parameters
                if '_' in str(strategy_type) and payload_strategy_type:
                    # Try to extract parameters from the strategy name
                    parts = str(strategy_type).split('_')
                    
                    # Handle different naming conventions
                    # Format: strategy_type_param1_param2 or strategy_type_param1_param2_f0
                    # Remove filter suffix if present
                    if parts and parts[-1].startswith('f') and parts[-1][1:].isdigit():
                        parts = parts[:-1]  # Remove filter suffix
                    
                    if len(parts) >= 3:  # At least strategy_type + 2 params
                        # Extract base type (handle multi-word strategy types)
                        if payload_strategy_type == 'bollinger_bands' and len(parts) >= 4:
                            base_type = '_'.join(parts[0:2])  # bollinger_bands
                            param_parts = parts[2:]  # [13, 10]
                        else:
                            # For other strategies, assume single-word type
                            base_type = parts[0]
                            param_parts = parts[1:]
                        
                        # Check if base type matches
                        if base_type == payload_strategy_type:
                            # Match parameters based on strategy type
                            signal_params = event.payload.get('parameters', {})
                            
                            if payload_strategy_type == 'bollinger_bands' and len(param_parts) >= 2:
                                # Bollinger bands: period_stddev
                                expected_period = param_parts[0]
                                expected_std = param_parts[1]
                                
                                actual_period = str(signal_params.get('period', ''))
                                actual_std = str(signal_params.get('std_dev', '')).replace('.', '')
                                
                                logger.debug(f"Comparing bollinger_bands params: expected period={expected_period} vs actual={actual_period}, expected std={expected_std} vs actual={actual_std}")
                                if expected_period == actual_period and expected_std == actual_std:
                                    logger.debug(f"Matched {strategy_type} - period: {expected_period}, std_dev: {expected_std}")
                                    return True
                            
                            elif payload_strategy_type == 'keltner_channel' and len(param_parts) >= 2:
                                # Keltner channel: period_multiplier
                                expected_period = param_parts[0]
                                expected_mult = param_parts[1]
                                
                                actual_period = str(signal_params.get('period', ''))
                                actual_mult = str(signal_params.get('multiplier', '')).replace('.', '')
                                
                                if expected_period == actual_period and expected_mult == actual_mult:
                                    logger.debug(f"Matched {strategy_type} - period: {expected_period}, multiplier: {expected_mult}")
                                    return True
                            
                            # Add more strategy-specific parameter matching as needed
            
            # Default 'default' strategy matches all
            if 'default' in [str(s) for s in managed_strategies]:
                return True
        
        return False
    
    return filter_func


def create_regime_filter(
    regimes: List[str],
    classifier_type: Optional[str] = None,
    min_confidence: float = 0.0
) -> Callable[[Event], bool]:
    """
    Create filter for classification events.
    
    Args:
        regimes: List of regime names to match
        classifier_type: Optional classifier type filter
        min_confidence: Minimum confidence threshold
        
    Returns:
        Filter function for classification events
    """
    criteria = {
        'regime': regimes,  # List membership check
    }
    
    if classifier_type:
        criteria['classifier_type'] = classifier_type
    
    descriptor = SubscriptionDescriptor(criteria)
    
    def filter_func(event: Event) -> bool:
        if event.event_type != EventType.CLASSIFICATION.value:
            return False
        
        # Check descriptor match
        if not descriptor.matches(event):
            return False
        
        # Additional confidence check
        confidence = event.payload.get('confidence', 0.0)
        return confidence >= min_confidence
    
    return filter_func


def create_composite_filter(
    signal_criteria: Optional[Dict[str, Any]] = None,
    classification_criteria: Optional[Dict[str, Any]] = None
) -> Callable[[Event], bool]:
    """
    Create a filter that matches multiple event types.
    
    Useful for components that process both signals and classifications.
    """
    filters = []
    
    if signal_criteria:
        signal_desc = SubscriptionDescriptor(signal_criteria)
        filters.append(('signal', signal_desc))
    
    if classification_criteria:
        class_desc = SubscriptionDescriptor(classification_criteria)
        filters.append(('classification', class_desc))
    
    def filter_func(event: Event) -> bool:
        for event_type, descriptor in filters:
            if event_type == 'signal' and event.event_type == EventType.SIGNAL.value:
                return descriptor.matches(event)
            elif event_type == 'classification' and event.event_type == EventType.CLASSIFICATION.value:
                return descriptor.matches(event)
        
        return False
    
    return filter_func


def migrate_legacy_subscription(legacy_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert legacy subscription config to structured format.
    
    Helps with migration from old to new system.
    """
    structured_config = {}
    
    # Extract strategy type from various legacy formats
    if 'strategy_id' in legacy_config:
        # Parse strategy_id to extract components
        strategy_id = legacy_config['strategy_id']
        parts = strategy_id.split('_')
        
        if len(parts) >= 3:
            structured_config['symbol'] = parts[0]
            structured_config['timeframe'] = parts[1]
            structured_config['strategy_type'] = parts[2]
    
    elif 'managed_strategies' in legacy_config:
        # Convert managed_strategies list
        strategies = []
        for s in legacy_config['managed_strategies']:
            if isinstance(s, str):
                strategies.append({'type': s})
            else:
                strategies.append(s)
        structured_config['managed_strategies'] = strategies
    
    return structured_config