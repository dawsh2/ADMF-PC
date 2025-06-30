"""
Composite Feature Support for the Feature System

This module provides support for features that depend on other features,
such as SMA of ATR (atr_sma_50) or EMA of RSI (rsi_ema_20).
"""

from typing import Dict, Any, List, Optional, Set, Tuple
import re
import logging
from .feature_spec import FeatureSpec

logger = logging.getLogger(__name__)


class CompositeFeatureParser:
    """Parser for composite feature patterns in filter expressions."""
    
    # Patterns for composite features
    COMPOSITE_PATTERNS = [
        # {indicator}_{ma_type}_{period} -> ma_type of indicator
        (r'(\w+)_(sma|ema|wma)_(\d+)', '{1}', '{0}', '{2}'),
        # {indicator}_stddev_{period} -> stddev of indicator  
        (r'(\w+)_stddev_(\d+)', 'stddev', '{0}', '{1}'),
    ]
    
    # Known base indicators that can be sources
    KNOWN_INDICATORS = {
        'atr', 'rsi', 'obv', 'adx', 'cci', 'mfi', 'roc',
        'volume', 'macd', 'macd_signal', 'macd_histogram'
    }
    
    @classmethod
    def parse_composite_feature(cls, feature_name: str) -> Optional[Tuple[str, FeatureSpec, List[FeatureSpec]]]:
        """
        Parse a composite feature name and return operation type, feature spec, and dependencies.
        
        Args:
            feature_name: Feature name like "atr_sma_50"
            
        Returns:
            Tuple of (operation_type, feature_spec, dependencies) or None if not composite
        """
        for pattern, op_template, source_template, param_template in cls.COMPOSITE_PATTERNS:
            match = re.match(pattern, feature_name)
            if match:
                groups = match.groups()
                
                # Extract components
                if len(groups) == 3:  # e.g., atr_sma_50
                    source = groups[0]
                    operation = groups[1]
                    period = int(groups[2])
                    
                    # Only recognize known indicators
                    if source not in cls.KNOWN_INDICATORS:
                        continue
                        
                    # Build dependency
                    dependencies = []
                    if source in ['atr', 'rsi', 'cci', 'mfi']:
                        # These need a period parameter
                        default_period = {'atr': 14, 'rsi': 14, 'cci': 20, 'mfi': 14}.get(source, 14)
                        dependencies.append(
                            FeatureSpec(feature_type=source, params={'period': default_period})
                        )
                    else:
                        # These don't need parameters or use defaults
                        dependencies.append(
                            FeatureSpec(feature_type=source, params={})
                        )
                    
                    # For now, return a pseudo-feature that will be handled specially
                    # In the future, this would create a proper composite feature type
                    logger.info(f"Parsed composite feature {feature_name}: {operation} of {source} with period {period}")
                    
                    # Return None for now since we don't have composite feature implementation
                    # But log the dependencies so they can be manually configured
                    logger.warning(
                        f"Composite feature {feature_name} detected but not yet supported. "
                        f"Please manually configure: {source} and then {operation} with period {period}"
                    )
                    return None
                    
        return None
    
    @classmethod
    def extract_required_features(cls, filter_expr: str) -> Set[str]:
        """
        Extract all features referenced in a filter expression, including composites.
        
        Args:
            filter_expr: Filter expression like "atr_sma_50 > 0.5 and rsi < 30"
            
        Returns:
            Set of feature names that need to be configured
        """
        # Find all potential feature references
        # Pattern matches word_word_number or word_number
        pattern = r'\b([a-z_]+(?:_\d+)?)\b'
        matches = re.findall(pattern, filter_expr, re.IGNORECASE)
        
        # Filter out operators and keywords
        keywords = {
            'signal', 'or', 'and', 'not', 'if', 'else', 'true', 'false',
            'abs', 'min', 'max', 'log', 'exp', 'sqrt', 'price', 'volume',
            'open', 'high', 'low', 'close', 'bar_of_day'
        }
        
        required_features = set()
        for match in matches:
            if match not in keywords and not match.isdigit():
                # Check if it's a composite feature
                composite_result = cls.parse_composite_feature(match)
                if composite_result:
                    # Add both the composite and its dependencies
                    # For now, just log since we can't handle composites yet
                    logger.info(f"Found composite feature reference: {match}")
                else:
                    # Regular feature
                    required_features.add(match)
        
        return required_features


def suggest_composite_feature_config(feature_name: str) -> Optional[Dict[str, Any]]:
    """
    Suggest configuration for a composite feature.
    
    Args:
        feature_name: Feature name like "atr_sma_50"
        
    Returns:
        Suggested configuration dict or None
    """
    # Parse the composite feature
    parts = feature_name.split('_')
    
    if len(parts) == 3 and parts[2].isdigit():
        source = parts[0]
        operation = parts[1]
        period = int(parts[2])
        
        if operation in ['sma', 'ema', 'wma'] and source in CompositeFeatureParser.KNOWN_INDICATORS:
            # Suggest a configuration
            config = {
                'base_feature': {
                    'name': f'{source}_base',
                    'type': source,
                    'params': {}
                },
                'composite_feature': {
                    'name': feature_name,
                    'type': operation,
                    'params': {'period': period},
                    'note': f'Manually compute {operation.upper()} of {source} with period {period}'
                }
            }
            
            # Add default parameters for known indicators
            if source == 'atr':
                config['base_feature']['params']['period'] = 14
            elif source == 'rsi':
                config['base_feature']['params']['period'] = 14
            elif source == 'cci':
                config['base_feature']['params']['period'] = 20
                
            return config
    
    return None


# Temporary workaround functions for filter context
def create_composite_feature_functions(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create functions that compute composite features on the fly.
    
    This is a temporary workaround until proper composite feature support is implemented.
    """
    functions = {}
    
    # Generic SMA of any feature
    def sma_of(source: str, period: int = 20) -> float:
        """Compute SMA of another feature (if available in history)."""
        # For now, just return the current value or 0
        # In the future, this would access historical values
        return features.get(source, 0)
    
    # Generic EMA of any feature  
    def ema_of(source: str, period: int = 20) -> float:
        """Compute EMA of another feature (if available in history)."""
        return features.get(source, 0)
    
    functions['sma_of'] = sma_of
    functions['ema_of'] = ema_of
    
    return functions