"""
Filter Metadata Enhancer

Enhances strategy metadata with filter information from the configuration.
This allows us to track which filters are applied to each compiled strategy.
"""

import logging
from typing import Dict, Any, Optional, List
import yaml
import json

logger = logging.getLogger(__name__)


class FilterMetadataEnhancer:
    """Enhances metadata with filter information from strategy configurations."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.strategy_filter_map = {}
        self.filter_descriptions = {}
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """Load and parse the configuration to extract filter information."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract filter information from parameter_space
            if 'parameter_space' in config and 'strategies' in config['parameter_space']:
                self._parse_strategies(config['parameter_space']['strategies'])
            
            # New clean syntax
            elif 'strategy' in config:
                self._parse_clean_syntax_strategies(config['strategy'])
                
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
    
    def _parse_clean_syntax_strategies(self, strategies: List[Dict[str, Any]]) -> None:
        """Parse strategies in the new clean YAML syntax."""
        strategy_index = 0
        
        for strategy_group in strategies:
            for strategy_type, strategy_config in strategy_group.items():
                if 'filter' not in strategy_config:
                    # No filters - baseline
                    self.strategy_filter_map[f"compiled_strategy_{strategy_index}"] = {
                        'filter_type': 'baseline',
                        'description': 'No filters applied'
                    }
                    strategy_index += 1
                    continue
                
                filters = strategy_config.get('filter', [])
                if not isinstance(filters, list):
                    filters = [filters]
                
                # Generate combinations for each filter
                for filter_spec in filters:
                    filter_info = self._parse_filter_spec(filter_spec)
                    
                    # Calculate number of variations this filter generates
                    variations = self._count_filter_variations(filter_spec, strategy_config)
                    
                    for _ in range(variations):
                        self.strategy_filter_map[f"compiled_strategy_{strategy_index}"] = filter_info
                        strategy_index += 1
    
    def _parse_filter_spec(self, filter_spec) -> Dict[str, Any]:
        """Parse a filter specification and return description."""
        if filter_spec is None:
            return {'filter_type': 'baseline', 'description': 'No filters applied'}
        
        # Single filter dict
        if isinstance(filter_spec, dict):
            if 'rsi_below' in filter_spec:
                return {'filter_type': 'rsi', 'description': 'RSI below threshold filter'}
            elif 'volume_above' in filter_spec:
                return {'filter_type': 'volume', 'description': 'Volume above average filter'}
            elif 'volatility_above' in filter_spec:
                return {'filter_type': 'volatility', 'description': 'Volatility regime filter'}
            elif 'time_exclude' in filter_spec:
                return {'filter_type': 'time', 'description': 'Time of day exclusion filter'}
            elif 'long' in filter_spec or 'short' in filter_spec:
                return {'filter_type': 'directional', 'description': 'Directional filter (long/short specific)'}
            elif 'price_below_vwap' in filter_spec or 'price_above_vwap' in filter_spec:
                return {'filter_type': 'vwap', 'description': 'VWAP positioning filter'}
        
        # Combined filters (list)
        elif isinstance(filter_spec, list):
            filter_types = []
            for f in filter_spec:
                if isinstance(f, dict):
                    for key in f:
                        if 'rsi' in key:
                            filter_types.append('rsi')
                        elif 'volume' in key:
                            filter_types.append('volume')
                        elif 'volatility' in key or 'atr' in key:
                            filter_types.append('volatility')
                        elif 'vwap' in key:
                            filter_types.append('vwap')
                        elif 'time' in key:
                            filter_types.append('time')
            
            return {
                'filter_type': 'combined',
                'description': f'Combined filters: {", ".join(set(filter_types))}'
            }
        
        return {'filter_type': 'unknown', 'description': str(filter_spec)[:100]}
    
    def _count_filter_variations(self, filter_spec, strategy_config: Dict[str, Any]) -> int:
        """Count how many strategy variations a filter generates."""
        # This is simplified - in reality would need to parse parameter ranges
        # For now, return 1 as placeholder
        return 1
    
    def _parse_strategies(self, strategies: List[Dict[str, Any]]) -> None:
        """Parse strategies in the old format."""
        strategy_index = 0
        
        for strategy in strategies:
            if 'filter' not in strategy:
                # No filter - baseline
                param_overrides = strategy.get('param_overrides', {})
                variations = 1
                for param, values in param_overrides.items():
                    if isinstance(values, list):
                        variations *= len(values)
                
                for _ in range(variations):
                    self.strategy_filter_map[f"compiled_strategy_{strategy_index}"] = {
                        'filter_type': 'baseline',
                        'description': 'No filters applied',
                        'parameters': param_overrides
                    }
                    strategy_index += 1
            else:
                # Has filter
                filter_str = strategy.get('filter', '')
                filter_params = strategy.get('filter_params', {})
                
                # Determine filter type
                filter_info = self._determine_filter_type(filter_str, filter_params)
                
                # Calculate variations
                param_overrides = strategy.get('param_overrides', {})
                variations = 1
                for param, values in param_overrides.items():
                    if isinstance(values, list):
                        variations *= len(values)
                
                # Filter param variations
                for param, values in filter_params.items():
                    if isinstance(values, list):
                        variations *= len(values)
                
                for _ in range(variations):
                    self.strategy_filter_map[f"compiled_strategy_{strategy_index}"] = filter_info
                    strategy_index += 1
    
    def _determine_filter_type(self, filter_str: str, filter_params: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the type of filter from the filter string."""
        filter_str_lower = filter_str.lower()
        
        # Check for specific patterns
        if 'atr' in filter_str_lower and 'atr_sma' in filter_str_lower:
            return {
                'filter_type': 'volatility_regime',
                'description': 'Volatility regime filter (ATR-based)',
                'filter_params': filter_params
            }
        elif 'vwap' in filter_str_lower:
            if 'signal > 0' in filter_str and 'signal < 0' in filter_str:
                return {
                    'filter_type': 'vwap_directional',
                    'description': 'VWAP directional positioning (long below, short above)',
                    'filter_params': filter_params
                }
            else:
                return {
                    'filter_type': 'vwap',
                    'description': 'VWAP-based filter',
                    'filter_params': filter_params
                }
        elif 'bar_of_day' in filter_str_lower:
            return {
                'filter_type': 'time_of_day',
                'description': 'Time of day filter (avoid midday)',
                'filter_params': filter_params
            }
        elif 'rsi' in filter_str_lower:
            if 'signal > 0' in filter_str and 'signal < 0' in filter_str:
                return {
                    'filter_type': 'rsi_directional',
                    'description': 'Directional RSI filter',
                    'filter_params': filter_params
                }
            else:
                return {
                    'filter_type': 'rsi',
                    'description': 'RSI threshold filter',
                    'filter_params': filter_params
                }
        elif 'volume' in filter_str_lower:
            return {
                'filter_type': 'volume',
                'description': 'Volume filter',
                'filter_params': filter_params
            }
        elif 'signal <= 0' in filter_str:
            return {
                'filter_type': 'long_only',
                'description': 'Long-only filter variant',
                'filter_params': filter_params
            }
        else:
            # Try to extract key indicators
            indicators = []
            if 'atr' in filter_str_lower:
                indicators.append('volatility')
            if 'sma' in filter_str_lower:
                indicators.append('trend')
            if 'close' in filter_str_lower:
                indicators.append('price')
            
            if len(indicators) > 1:
                return {
                    'filter_type': 'combined',
                    'description': f'Combined filter: {", ".join(indicators)}',
                    'filter_params': filter_params
                }
            else:
                return {
                    'filter_type': 'custom',
                    'description': filter_str[:100],
                    'filter_params': filter_params
                }
    
    def enhance_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance existing metadata with filter information."""
        if 'components' not in metadata:
            return metadata
        
        enhanced_metadata = metadata.copy()
        
        for component_name, component_data in enhanced_metadata['components'].items():
            if component_name.startswith('SPY_5m_compiled_strategy_'):
                # Extract strategy number
                strategy_key = component_name.replace('SPY_5m_', '')
                
                if strategy_key in self.strategy_filter_map:
                    filter_info = self.strategy_filter_map[strategy_key]
                    
                    # Add filter information to component metadata
                    component_data['filter_type'] = filter_info['filter_type']
                    component_data['filter_description'] = filter_info['description']
                    
                    if 'filter_params' in filter_info:
                        component_data['filter_params'] = filter_info['filter_params']
        
        # Add summary of filter types
        filter_summary = {}
        for component_name, component_data in enhanced_metadata['components'].items():
            if 'filter_type' in component_data:
                filter_type = component_data['filter_type']
                if filter_type not in filter_summary:
                    filter_summary[filter_type] = 0
                filter_summary[filter_type] += 1
        
        enhanced_metadata['filter_summary'] = filter_summary
        
        return enhanced_metadata


def enhance_metadata_with_filters(metadata_path: str, config_path: str, output_path: Optional[str] = None) -> None:
    """
    Enhance an existing metadata.json file with filter information.
    
    Args:
        metadata_path: Path to existing metadata.json
        config_path: Path to the configuration file that generated the strategies
        output_path: Optional output path (defaults to overwriting metadata_path)
    """
    # Load existing metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Create enhancer and enhance metadata
    enhancer = FilterMetadataEnhancer(config_path)
    enhanced_metadata = enhancer.enhance_metadata(metadata)
    
    # Write enhanced metadata
    output_path = output_path or metadata_path
    with open(output_path, 'w') as f:
        json.dump(enhanced_metadata, f, indent=2)
    
    logger.info(f"Enhanced metadata written to {output_path}")


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) < 3:
        print("Usage: python filter_metadata_enhancer.py <metadata.json> <config.yaml> [output.json]")
        sys.exit(1)
    
    metadata_path = sys.argv[1]
    config_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    enhance_metadata_with_filters(metadata_path, config_path, output_path)