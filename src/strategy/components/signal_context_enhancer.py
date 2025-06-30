"""
Signal Context Enhancer

Automatically adds feature/classifier context to signal metadata.
This allows post-hoc analysis of signals with their market context.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class SignalContextEnhancer:
    """
    Enhances signal metadata with specified features and classifier outputs.
    
    Can be configured to automatically include:
    - Key feature values (SMA 200, VWAP, ATR, etc.)
    - Classifier outputs (trend regime, volatility state, etc.)
    - Relative positions (price vs VWAP, price vs SMA, etc.)
    """
    
    def __init__(self, context_config: Optional[Dict[str, Any]] = None):
        """
        Initialize with context configuration.
        
        Args:
            context_config: Dict specifying what context to capture
                {
                    'features': ['sma_200', 'vwap', 'atr_14'],
                    'classifiers': ['trend_regime', 'volatility_state'],
                    'relative': [
                        {'price_vs': 'sma_200'},
                        {'price_vs': 'vwap'}
                    ]
                }
        """
        self.config = context_config or self._default_config()
        
    def _default_config(self) -> Dict[str, Any]:
        """Default context to capture with every signal."""
        return {
            'features': [
                'sma_200',      # Long-term trend
                'vwap',         # Intraday reference
                'atr_14',       # Volatility
                'volume'        # Liquidity
            ],
            'classifiers': [],  # Will be populated if available
            'relative': [
                {'price_vs': 'sma_200', 'name': 'price_vs_sma200'},
                {'price_vs': 'vwap', 'name': 'price_vs_vwap'}
            ]
        }
    
    def enhance_signal(self, 
                      signal_dict: Dict[str, Any], 
                      features: Dict[str, Any],
                      bar: Dict[str, Any],
                      classifiers: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhance signal with context metadata.
        
        Args:
            signal_dict: The signal dictionary from strategy
            features: Available features
            bar: Current bar data
            classifiers: Optional classifier outputs
            
        Returns:
            Enhanced signal dictionary with context in metadata
        """
        if 'metadata' not in signal_dict:
            signal_dict['metadata'] = {}
        
        # Add context section
        context = {}
        
        # 1. Add specified features
        for feature_name in self.config.get('features', []):
            if feature_name in features:
                context[f'feature_{feature_name}'] = features[feature_name]
        
        # 2. Add classifier outputs
        if classifiers:
            for classifier_name in self.config.get('classifiers', []):
                if classifier_name in classifiers:
                    context[f'classifier_{classifier_name}'] = classifiers[classifier_name]
        
        # 3. Add relative positions
        price = bar.get('close', 0)
        for rel_config in self.config.get('relative', []):
            ref_feature = rel_config.get('price_vs')
            name = rel_config.get('name', f'price_vs_{ref_feature}')
            
            if ref_feature in features and features[ref_feature] is not None:
                ref_value = features[ref_feature]
                if ref_value > 0:
                    # Calculate percentage above/below
                    relative_pos = ((price - ref_value) / ref_value) * 100
                    context[name] = round(relative_pos, 4)
        
        # 4. Add market microstructure
        context['spread_pct'] = ((bar.get('high', price) - bar.get('low', price)) / price * 100) if price > 0 else 0
        context['volume'] = bar.get('volume', 0)
        
        # Add to signal metadata
        signal_dict['metadata']['context'] = context
        
        return signal_dict


def create_context_enhancer_from_config(config: Dict[str, Any]) -> Optional[SignalContextEnhancer]:
    """
    Create context enhancer from strategy config.
    
    Config example:
        signal_context:
            features: ['sma_200', 'vwap', 'atr_14']
            classifiers: ['market_regime']
            relative: [{'price_vs': 'vwap'}]
    """
    if 'signal_context' not in config:
        return None
        
    return SignalContextEnhancer(config['signal_context'])