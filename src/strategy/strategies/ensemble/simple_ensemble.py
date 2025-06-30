"""
Simple Ensemble Strategy

Combines multiple strategies with weighted voting and threshold logic.
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from ...types import Signal
from ...protocols import FeatureProvider, StrategyProtocol
from ....core.components.discovery import strategy

logger = logging.getLogger(__name__)


@strategy('ensemble')
class SimpleEnsemble(StrategyProtocol):
    """
    Simple ensemble that combines multiple strategies with weighted voting.
    
    Parameters:
        strategies: List of strategy configurations with types, params, and weights
        combination_method: How to combine signals ('weighted_vote', 'majority', 'unanimous')
        threshold: Minimum weighted signal value to generate a trade (can be expression)
    """
    
    def __init__(self, params: Dict[str, Any]):
        self.strategies = params.get('strategies', [])
        self.combination_method = params.get('combination_method', 'weighted_vote')
        self.threshold = params.get('threshold', 0.0)
        
        # Compile sub-strategies
        self._compiled_strategies = []
        self._weights = []
        
        total_weight = 0.0
        for strategy_config in self.strategies:
            strategy_type = strategy_config['type']
            strategy_params = strategy_config.get('param_overrides', {})
            weight = float(strategy_config.get('weight', 1.0))
            
            # Get strategy from component registry
            from ...components.discovery import get_component_registry
            registry = get_component_registry()
            
            component_info = registry.get_component(strategy_type)
            if not component_info or component_info.component_type != 'strategy':
                logger.warning(f"Strategy '{strategy_type}' not found in registry")
                continue
                
            # Get strategy factory function
            strategy_func = component_info.factory
            if not strategy_func:
                logger.warning(f"Strategy '{strategy_type}' has no factory function")
                continue
                
            # Store the strategy function with its parameters bound
            self._compiled_strategies.append((strategy_func, strategy_params))
            self._weights.append(weight)
            total_weight += weight
            
        # Normalize weights
        if total_weight > 0:
            self._weights = [w / total_weight for w in self._weights]
        else:
            self._weights = [1.0 / len(self._weights)] * len(self._weights) if self._weights else []
            
        logger.info(f"Ensemble initialized with {len(self._compiled_strategies)} strategies")
        
    def generate_signal(self, 
                       features: FeatureProvider,
                       data: pd.Series,
                       **kwargs) -> Optional[Signal]:
        """
        Generate ensemble signal by combining sub-strategy signals.
        """
        if not self._compiled_strategies:
            return None
            
        # Collect signals from all strategies
        signals = []
        weights = []
        
        for i, ((strategy_func, params), weight) in enumerate(zip(self._compiled_strategies, self._weights)):
            try:
                # Call strategy function with features, data, and params
                signal = strategy_func(features, data, params)
                if signal is not None:
                    signals.append(float(signal))
                    weights.append(weight)
            except Exception as e:
                logger.error(f"Strategy {i} failed: {e}")
                continue
                
        if not signals:
            return None
            
        # Combine signals based on method
        if self.combination_method == 'weighted_vote':
            # Weighted average of signals
            combined_signal = np.average(signals, weights=weights)
        elif self.combination_method == 'majority':
            # Majority vote (> 50% must agree on direction)
            positive = sum(1 for s in signals if s > 0)
            negative = sum(1 for s in signals if s < 0)
            total = len(signals)
            
            if positive > total / 2:
                combined_signal = 1.0
            elif negative > total / 2:
                combined_signal = -1.0
            else:
                combined_signal = 0.0
        elif self.combination_method == 'unanimous':
            # All must agree
            if all(s > 0 for s in signals):
                combined_signal = 1.0
            elif all(s < 0 for s in signals):
                combined_signal = -1.0
            else:
                combined_signal = 0.0
        else:
            # Default to weighted vote
            combined_signal = np.average(signals, weights=weights)
            
        # Apply threshold logic
        if self._should_trade(combined_signal, features, data):
            # Normalize to -1, 0, 1
            if combined_signal > 0:
                return Signal.LONG
            elif combined_signal < 0:
                return Signal.SHORT
            else:
                return Signal.NEUTRAL
        else:
            return Signal.NEUTRAL
            
    def _should_trade(self, signal_value: float, features: FeatureProvider, data: pd.Series) -> bool:
        """
        Evaluate threshold condition to determine if trade should be taken.
        """
        if isinstance(self.threshold, str):
            # Parse threshold expression
            # Support for expressions like "0.5 AND volume > sma(volume, 20) * 1.2"
            return self._evaluate_threshold_expression(signal_value, self.threshold, features, data)
        else:
            # Simple numeric threshold
            return abs(signal_value) >= float(self.threshold)
            
    def _evaluate_threshold_expression(self, signal_value: float, expression: str, 
                                     features: FeatureProvider, data: pd.Series) -> bool:
        """
        Evaluate complex threshold expression.
        """
        try:
            # Replace signal value placeholder
            expr = expression.replace('signal', str(signal_value))
            
            # Create evaluation context
            context = {
                'signal': signal_value,
                'volume': data.get('volume', 0),
                'close': data.get('close', 0),
                'open': data.get('open', 0),
                'high': data.get('high', 0),
                'low': data.get('low', 0),
                # Add common features
                'sma': lambda period: features.get(f'sma_{period}', 0),
                'ema': lambda period: features.get(f'ema_{period}', 0),
                'rsi': lambda period: features.get(f'rsi_{period}', 50),
                'atr': lambda period: features.get(f'atr_{period}', 0),
                'vwap': lambda: features.get('vwap', data.get('close', 0)),
                # Math functions
                'abs': abs,
                'min': min,
                'max': max,
            }
            
            # Evaluate expression
            result = eval(expr, {"__builtins__": {}}, context)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to evaluate threshold expression '{expression}': {e}")
            # Fall back to simple comparison
            return abs(signal_value) >= 0.5
            
    def get_required_features(self) -> List[str]:
        """Get all features required by sub-strategies."""
        features = []
        # For now, we'll need to implement feature discovery differently
        # since we're using factory functions rather than instances
        return features
        
    def get_parameters(self) -> Dict[str, Any]:
        """Get ensemble parameters."""
        return {
            'strategies': self.strategies,
            'combination_method': self.combination_method,
            'threshold': self.threshold
        }