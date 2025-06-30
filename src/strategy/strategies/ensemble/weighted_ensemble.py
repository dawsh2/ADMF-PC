"""
Weighted Ensemble Strategy with Signal Attribution

Tracks which sub-strategies contribute to each signal.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from ...classification_types import Signal
from ...protocols import FeatureProvider, StrategyProtocol
from ...validation import strategy
from ....core.events.bus import get_event_bus
from ....core.events.types import Event

logger = logging.getLogger(__name__)


@strategy('weighted_ensemble', 
    description="Ensemble with signal attribution tracking",
    parameters={
        'strategies': 'List of strategy configurations with types, params, and weights',
        'combination_method': 'How to combine signals (weighted_vote, majority, unanimous)',
        'threshold': 'Minimum signal value or expression to generate trade',
        'track_attribution': 'Whether to emit attribution metadata (default: True)'
    })
def weighted_ensemble(features: Dict[str, float], bar: pd.Series, params: Dict[str, Any]) -> Optional[Signal]:
    """
    Ensemble strategy that tracks sub-strategy contributions.
    
    Emits additional metadata showing which strategies contributed to each signal.
    """
    strategies = params.get('strategies', [])
    combination_method = params.get('combination_method', 'weighted_vote')
    threshold = params.get('threshold', 0.0)
    track_attribution = params.get('track_attribution', True)
    
    if not strategies:
        return None
    
    # Collect signals from all sub-strategies
    sub_signals = []
    weights = []
    attributions = {}
    
    # Get component registry
    from ....core.components.discovery import get_component_registry
    registry = get_component_registry()
    
    for i, strategy_config in enumerate(strategies):
        strategy_type = strategy_config.get('type')
        strategy_params = strategy_config.get('param_overrides', {})
        weight = float(strategy_config.get('weight', 1.0))
        
        # Get strategy function
        component_info = registry.get_component(strategy_type)
        if not component_info or component_info.component_type != 'strategy':
            logger.warning(f"Strategy '{strategy_type}' not found")
            continue
            
        strategy_func = component_info.factory
        if not strategy_func:
            continue
        
        try:
            # Call strategy
            signal = strategy_func(features, bar, strategy_params)
            if signal is not None:
                signal_value = float(signal)
                sub_signals.append(signal_value)
                weights.append(weight)
                
                # Track attribution
                if signal_value != 0:
                    attributions[strategy_type] = {
                        'signal': signal_value,
                        'weight': weight,
                        'params': strategy_params
                    }
                    
        except Exception as e:
            logger.error(f"Strategy {strategy_type} failed: {e}")
            continue
    
    if not sub_signals:
        return None
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight > 0:
        normalized_weights = [w / total_weight for w in weights]
    else:
        normalized_weights = [1.0 / len(weights)] * len(weights)
    
    # Combine signals
    if combination_method == 'weighted_vote':
        combined_signal = np.average(sub_signals, weights=normalized_weights)
    elif combination_method == 'majority':
        positive = sum(1 for s in sub_signals if s > 0)
        negative = sum(1 for s in sub_signals if s < 0)
        if positive > len(sub_signals) / 2:
            combined_signal = 1.0
        elif negative > len(sub_signals) / 2:
            combined_signal = -1.0
        else:
            combined_signal = 0.0
    else:  # unanimous
        if all(s > 0 for s in sub_signals):
            combined_signal = 1.0
        elif all(s < 0 for s in sub_signals):
            combined_signal = -1.0
        else:
            combined_signal = 0.0
    
    # Apply threshold
    if isinstance(threshold, str):
        # Evaluate threshold expression
        try:
            context = {
                'signal': combined_signal,
                'volume': bar.get('volume', 0),
                'close': bar.get('close', 0),
                **{k: v for k, v in features.items()}
            }
            should_trade = eval(threshold, {"__builtins__": {}}, context)
        except:
            should_trade = abs(combined_signal) >= 0.5
    else:
        should_trade = abs(combined_signal) >= float(threshold)
    
    # Emit attribution event if tracking is enabled
    if track_attribution and should_trade and attributions:
        try:
            event_bus = get_event_bus()
            if event_bus:
                attribution_event = Event(
                    type='ENSEMBLE_ATTRIBUTION',
                    source='weighted_ensemble',
                    data={
                        'timestamp': bar.get('timestamp'),
                        'symbol': bar.get('symbol'),
                        'combined_signal': combined_signal,
                        'attributions': attributions,
                        'weights': dict(zip([s['type'] for s in strategies], normalized_weights))
                    }
                )
                event_bus.publish(attribution_event)
        except Exception as e:
            logger.debug(f"Could not emit attribution event: {e}")
    
    # Return final signal
    if should_trade:
        if combined_signal > 0:
            return Signal.LONG
        elif combined_signal < 0:
            return Signal.SHORT
    
    return Signal.NEUTRAL