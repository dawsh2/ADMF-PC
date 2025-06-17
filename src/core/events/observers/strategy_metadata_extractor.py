"""
Strategy Metadata Extractor

Recursively extracts metadata from strategies, including nested ensemble strategies.
This ensures that all sub-strategies are properly documented in metadata.json.
"""

from typing import Dict, Any, List, Optional, Set
import logging
from ....core.components.discovery import get_component_registry

logger = logging.getLogger(__name__)


def extract_recursive_strategy_metadata(
    strategy_config: Dict[str, Any], 
    seen_strategies: Optional[Set[str]] = None,
    depth: int = 0,
    max_depth: int = 10
) -> Dict[str, Any]:
    """
    Recursively extract metadata from a strategy configuration.
    
    This handles ensemble strategies that contain sub-strategies,
    which themselves may be ensembles containing more sub-strategies.
    
    Args:
        strategy_config: Strategy configuration dict with 'type', 'name', 'params'
        seen_strategies: Set of already processed strategy names to avoid cycles
        depth: Current recursion depth
        max_depth: Maximum recursion depth to prevent infinite loops
        
    Returns:
        Complete metadata including all nested strategies
    """
    if seen_strategies is None:
        seen_strategies = set()
    
    if depth >= max_depth:
        logger.warning(f"Max recursion depth {max_depth} reached while extracting strategy metadata")
        return strategy_config
    
    strategy_type = strategy_config.get('type', '')
    strategy_name = strategy_config.get('name', '')
    params = strategy_config.get('params', {})
    
    # If type is missing, infer it from the name
    if not strategy_type and strategy_name:
        # Common pattern: strategy name IS the type for sub-strategies
        strategy_type = strategy_name
    
    # Create base metadata
    metadata = {
        'type': strategy_type,
        'name': strategy_name,
        'params': params.copy(),
        'depth': depth
    }
    
    # Check if this is an ensemble strategy
    is_ensemble = 'ensemble' in strategy_type.lower() or 'ensemble' in strategy_name.lower()
    
    if is_ensemble:
        # Extract sub-strategies from ensemble
        sub_strategies = []
        
        # Handle DEFAULT_REGIME_STRATEGIES case
        if strategy_type == 'duckdb_ensemble' and 'regime_strategies' not in params:
            # This ensemble uses DEFAULT_REGIME_STRATEGIES from the code
            try:
                # Try to import the DEFAULT_REGIME_STRATEGIES
                from ....strategy.strategies.ensemble.duckdb_ensemble import DEFAULT_REGIME_STRATEGIES
                metadata['uses_default_strategies'] = True
                # Note: default strategies are processed into regime_strategies below
                
                # Process the default strategies as if they were in the config
                metadata['regime_strategies'] = {}
                for regime, strategies in DEFAULT_REGIME_STRATEGIES.items():
                    metadata['regime_strategies'][regime] = []
                    
                    for sub_strategy in strategies:
                        # Avoid infinite recursion
                        sub_key = f"{sub_strategy.get('name', '')}_{sub_strategy.get('params', {})}"
                        if sub_key not in seen_strategies:
                            seen_strategies.add(sub_key)
                            
                            # Recursively extract sub-strategy metadata
                            sub_metadata = extract_recursive_strategy_metadata(
                                sub_strategy,
                                seen_strategies,
                                depth + 1,
                                max_depth
                            )
                            metadata['regime_strategies'][regime].append(sub_metadata)
                            sub_strategies.append(sub_metadata)
                
                logger.debug(f"Loaded {len(sub_strategies)} default strategies for duckdb_ensemble")
                
            except ImportError as e:
                logger.warning(f"Could not import DEFAULT_REGIME_STRATEGIES: {e}")
                metadata['uses_default_strategies'] = True
                metadata['note'] = 'Uses DEFAULT_REGIME_STRATEGIES from duckdb_ensemble.py (could not import)'
        
        # Common patterns for ensemble strategy parameters
        if 'regime_strategies' in params:
            # DuckDB ensemble pattern: regime_strategies is a dict of regime -> strategy list
            regime_strategies = params.get('regime_strategies', {})
            metadata['regime_strategies'] = {}
            
            for regime, strategies in regime_strategies.items():
                metadata['regime_strategies'][regime] = []
                
                for sub_strategy in strategies:
                    # Avoid infinite recursion
                    sub_key = f"{sub_strategy.get('name', '')}_{sub_strategy.get('params', {})}"
                    if sub_key not in seen_strategies:
                        seen_strategies.add(sub_key)
                        
                        # Recursively extract sub-strategy metadata
                        sub_metadata = extract_recursive_strategy_metadata(
                            sub_strategy,
                            seen_strategies,
                            depth + 1,
                            max_depth
                        )
                        metadata['regime_strategies'][regime].append(sub_metadata)
                        sub_strategies.append(sub_metadata)
        
        elif 'strategies' in params:
            # Generic ensemble pattern: strategies is a list
            strategies = params.get('strategies', [])
            metadata['strategies'] = []
            
            for sub_strategy in strategies:
                # Avoid infinite recursion
                sub_key = f"{sub_strategy.get('name', '')}_{sub_strategy.get('params', {})}"
                if sub_key not in seen_strategies:
                    seen_strategies.add(sub_key)
                    
                    # Recursively extract sub-strategy metadata
                    sub_metadata = extract_recursive_strategy_metadata(
                        sub_strategy,
                        seen_strategies,
                        depth + 1,
                        max_depth
                    )
                    metadata['strategies'].append(sub_metadata)
                    sub_strategies.append(sub_metadata)
        
        elif 'sub_strategies' in params:
            # Alternative ensemble pattern: sub_strategies
            sub_strategies_config = params.get('sub_strategies', [])
            metadata['sub_strategies'] = []
            
            for sub_strategy in sub_strategies_config:
                # Avoid infinite recursion
                sub_key = f"{sub_strategy.get('name', '')}_{sub_strategy.get('params', {})}"
                if sub_key not in seen_strategies:
                    seen_strategies.add(sub_key)
                    
                    # Recursively extract sub-strategy metadata
                    sub_metadata = extract_recursive_strategy_metadata(
                        sub_strategy,
                        seen_strategies,
                        depth + 1,
                        max_depth
                    )
                    metadata['sub_strategies'].append(sub_metadata)
                    sub_strategies.append(sub_metadata)
        
        # Add summary statistics
        metadata['is_ensemble'] = True
        metadata['total_sub_strategies'] = len(sub_strategies)
        metadata['unique_strategy_types'] = len(set(s.get('type', '') for s in sub_strategies))
    else:
        metadata['is_ensemble'] = False
    
    return metadata


def extract_all_strategies_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract metadata for all strategies in a configuration.
    
    Args:
        config: Complete configuration dict containing 'strategies' list
        
    Returns:
        Dict with metadata for each strategy including all nested sub-strategies
    """
    strategies = config.get('strategies', [])
    classifiers = config.get('classifiers', [])
    
    metadata = {
        'strategies': {},
        'classifiers': {},
        'summary': {
            'total_strategies': 0,
            'total_ensemble_strategies': 0,
            'total_leaf_strategies': 0,
            'max_nesting_depth': 0
        }
    }
    
    # Process strategies
    for strategy_config in strategies:
        strategy_name = strategy_config.get('name', 'unnamed')
        strategy_metadata = extract_recursive_strategy_metadata(strategy_config)
        metadata['strategies'][strategy_name] = strategy_metadata
        
        # Update summary
        metadata['summary']['total_strategies'] += 1
        if strategy_metadata.get('is_ensemble', False):
            metadata['summary']['total_ensemble_strategies'] += 1
        else:
            metadata['summary']['total_leaf_strategies'] += 1
        
        # Calculate max depth
        max_depth = calculate_max_depth(strategy_metadata)
        if max_depth > metadata['summary']['max_nesting_depth']:
            metadata['summary']['max_nesting_depth'] = max_depth
    
    # Process classifiers
    for classifier_config in classifiers:
        classifier_name = classifier_config.get('name', 'unnamed')
        metadata['classifiers'][classifier_name] = {
            'type': classifier_config.get('type', ''),
            'name': classifier_name,
            'params': classifier_config.get('params', {})
        }
    
    # Component inventory can be generated on-demand to save space
    # metadata['component_inventory'] = generate_component_inventory(metadata)
    
    return metadata


def calculate_max_depth(strategy_metadata: Dict[str, Any], current_depth: int = 0) -> int:
    """Calculate the maximum nesting depth of a strategy."""
    if not strategy_metadata.get('is_ensemble', False):
        return current_depth
    
    max_sub_depth = current_depth
    
    # Check different ensemble patterns
    for key in ['regime_strategies', 'strategies', 'sub_strategies']:
        if key in strategy_metadata:
            sub_strategies_dict = strategy_metadata[key]
            
            if isinstance(sub_strategies_dict, dict):
                # regime_strategies pattern
                for regime, strategies in sub_strategies_dict.items():
                    for sub_strategy in strategies:
                        sub_depth = calculate_max_depth(sub_strategy, current_depth + 1)
                        max_sub_depth = max(max_sub_depth, sub_depth)
            elif isinstance(sub_strategies_dict, list):
                # strategies or sub_strategies pattern
                for sub_strategy in sub_strategies_dict:
                    sub_depth = calculate_max_depth(sub_strategy, current_depth + 1)
                    max_sub_depth = max(max_sub_depth, sub_depth)
    
    return max_sub_depth


def generate_component_inventory(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate an inventory of all unique components used across all strategies.
    """
    inventory = {
        'unique_strategy_types': set(),
        'unique_strategy_names': set(),
        'strategy_usage_count': {},
        'parameter_variations': {}
    }
    
    def process_strategy(strategy_meta: Dict[str, Any]):
        """Process a single strategy and its sub-strategies."""
        strategy_type = strategy_meta.get('type', '')
        strategy_name = strategy_meta.get('name', '')
        
        inventory['unique_strategy_types'].add(strategy_type)
        inventory['unique_strategy_names'].add(strategy_name)
        
        # Count usage
        if strategy_type not in inventory['strategy_usage_count']:
            inventory['strategy_usage_count'][strategy_type] = 0
        inventory['strategy_usage_count'][strategy_type] += 1
        
        # Track parameter variations
        if strategy_type not in inventory['parameter_variations']:
            inventory['parameter_variations'][strategy_type] = []
        
        # Store parameters without the sub-strategy references
        params_copy = strategy_meta.get('params', {}).copy()
        for key in ['regime_strategies', 'strategies', 'sub_strategies']:
            params_copy.pop(key, None)
        
        if params_copy not in inventory['parameter_variations'][strategy_type]:
            inventory['parameter_variations'][strategy_type].append(params_copy)
        
        # Process sub-strategies
        if strategy_meta.get('is_ensemble', False):
            for key in ['regime_strategies', 'strategies', 'sub_strategies']:
                if key in strategy_meta:
                    sub_strategies_dict = strategy_meta[key]
                    
                    if isinstance(sub_strategies_dict, dict):
                        for regime, strategies in sub_strategies_dict.items():
                            for sub_strategy in strategies:
                                process_strategy(sub_strategy)
                    elif isinstance(sub_strategies_dict, list):
                        for sub_strategy in sub_strategies_dict:
                            process_strategy(sub_strategy)
    
    # Process all strategies
    for strategy_name, strategy_meta in metadata.get('strategies', {}).items():
        process_strategy(strategy_meta)
    
    # Convert sets to lists for JSON serialization
    inventory['unique_strategy_types'] = sorted(list(inventory['unique_strategy_types']))
    inventory['unique_strategy_names'] = sorted(list(inventory['unique_strategy_names']))
    
    # Add summary
    inventory['summary'] = {
        'total_unique_types': len(inventory['unique_strategy_types']),
        'total_unique_names': len(inventory['unique_strategy_names']),
        'most_used_strategy': max(inventory['strategy_usage_count'].items(), 
                                  key=lambda x: x[1])[0] if inventory['strategy_usage_count'] else None
    }
    
    return inventory


def update_metadata_with_recursive_strategies(
    existing_metadata: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update existing metadata with recursive strategy information.
    
    This can be used to enhance metadata that was already created by the tracer.
    """
    # Extract recursive strategy metadata
    strategy_metadata = extract_all_strategies_metadata(config)
    
    # Merge with existing metadata
    existing_metadata['strategy_metadata'] = strategy_metadata
    
    # Add data source config if available
    if 'data_source_config' in existing_metadata:
        existing_metadata['data_source'] = existing_metadata.pop('data_source_config')
    
    return existing_metadata