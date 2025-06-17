"""
Metadata Utilities

Utilities for computing metadata on-demand to reduce storage size.
"""

from typing import Dict, Any
from .strategy_metadata_extractor import generate_component_inventory


def compute_component_inventory(strategy_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute component inventory on-demand from strategy metadata.
    
    This generates the same information that was previously stored in metadata.json
    but computes it dynamically to save storage space.
    """
    return generate_component_inventory(strategy_metadata)


def get_strategy_summary(strategy_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a summary of strategy metadata including computed statistics.
    
    This combines the stored metadata with computed inventory information.
    """
    summary = strategy_metadata.get('summary', {})
    
    # Add computed component inventory
    component_inventory = compute_component_inventory(strategy_metadata)
    
    # Enhance summary with key inventory stats
    summary.update({
        'unique_strategy_types': component_inventory['summary']['total_unique_types'],
        'unique_strategy_names': component_inventory['summary']['total_unique_names'],
        'most_used_strategy': component_inventory['summary']['most_used_strategy']
    })
    
    return {
        'summary': summary,
        'component_inventory': component_inventory
    }


def analyze_metadata_size(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze metadata file size and provide optimization suggestions.
    """
    import json
    
    # Calculate size of different sections
    total_size = len(json.dumps(metadata, indent=2))
    
    size_breakdown = {}
    
    # Analyze major sections
    for key, value in metadata.items():
        section_size = len(json.dumps({key: value}, indent=2))
        size_breakdown[key] = {
            'bytes': section_size,
            'percentage': (section_size / total_size) * 100
        }
    
    # Calculate savings from optimizations
    optimization_savings = {}
    
    if 'strategy_metadata' in metadata:
        strategy_meta = metadata['strategy_metadata']
        
        # Check for default_regime_strategies duplication
        strategies = strategy_meta.get('strategies', {})
        for strategy_name, strategy_data in strategies.items():
            if 'default_regime_strategies' in strategy_data:
                default_size = len(json.dumps(strategy_data['default_regime_strategies']))
                optimization_savings['remove_default_regime_strategies'] = {
                    'bytes_saved': default_size,
                    'percentage_saved': (default_size / total_size) * 100
                }
                break
        
        # Check for component_inventory storage
        if 'component_inventory' in strategy_meta:
            inventory_size = len(json.dumps(strategy_meta['component_inventory']))
            optimization_savings['compute_inventory_on_demand'] = {
                'bytes_saved': inventory_size,
                'percentage_saved': (inventory_size / total_size) * 100
            }
    
    return {
        'total_size_bytes': total_size,
        'size_breakdown': size_breakdown,
        'optimization_savings': optimization_savings,
        'optimized_size_estimate': total_size - sum(
            saving['bytes_saved'] for saving in optimization_savings.values()
        )
    }