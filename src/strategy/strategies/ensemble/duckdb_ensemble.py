"""
DuckDB Ensemble Strategy

An adaptive ensemble strategy that dynamically switches between different
strategies based on detected market regimes. Uses equal weighting (1/n)
for all active strategies in the current regime.
"""

from typing import Dict, Any, Optional, List
import logging
from ....core.components.discovery import strategy, get_component_registry

logger = logging.getLogger(__name__)


# Default strategy configurations per regime based on our analysis
DEFAULT_REGIME_STRATEGIES = {
    'low_vol_bullish': [
        {'name': 'dema_crossover', 'params': {'fast_dema_period': 19, 'slow_dema_period': 35}},
        {'name': 'macd_crossover', 'params': {'fast_ema': 12, 'slow_ema': 35, 'signal_ema': 9}},
        {'name': 'dema_crossover', 'params': {'fast_dema_period': 7, 'slow_dema_period': 35}},
        {'name': 'macd_crossover', 'params': {'fast_ema': 15, 'slow_ema': 35, 'signal_ema': 7}},
        {'name': 'cci_threshold', 'params': {'cci_period': 11, 'threshold': -40}},
        # Adding top performing pivot channel bounces
        {'name': 'pivot_channel_bounces', 'params': {'sr_period': 20, 'min_touches': 3, 'bounce_threshold': 0.003}}
    ],
    'low_vol_bearish': [
        {'name': 'stochastic_crossover', 'params': {'k_period': 27, 'd_period': 5}},
        {'name': 'cci_threshold', 'params': {'cci_period': 11, 'threshold': -20}},
        {'name': 'ema_sma_crossover', 'params': {'ema_period': 11, 'sma_period': 15}},
        {'name': 'keltner_breakout', 'params': {'period': 11, 'multiplier': 1.5}},
        {'name': 'rsi_bands', 'params': {'rsi_period': 7, 'oversold': 25, 'overbought': 70}},
        # Adding pivot channel bounces - moderate performer in bearish regime
        {'name': 'pivot_channel_bounces', 'params': {'sr_period': 20, 'min_touches': 2, 'bounce_threshold': 0.001}}
    ],
    'neutral': [
        # Mix of trend-following and mean-reversion strategies for diversity
        {'name': 'stochastic_rsi', 'params': {'rsi_period': 21, 'stoch_period': 21, 'oversold': 15, 'overbought': 80}},
        {'name': 'stochastic_rsi', 'params': {'rsi_period': 14, 'stoch_period': 14, 'oversold': 20, 'overbought': 80}},
        {'name': 'dema_crossover', 'params': {'fast_dema_period': 19, 'slow_dema_period': 35}},
        {'name': 'sma_crossover', 'params': {'fast_period': 10, 'slow_period': 30}},
        {'name': 'rsi_bands', 'params': {'rsi_period': 14, 'oversold': 30, 'overbought': 70}},
        {'name': 'rsi_bands', 'params': {'rsi_period': 21, 'oversold': 25, 'overbought': 75}},
        {'name': 'bollinger_mean_reversion', 'params': {'period': 20, 'std_dev': 2.0}},
        {'name': 'bollinger_breakout', 'params': {'period': 15, 'std_dev': 1.5}},
        {'name': 'cci_threshold', 'params': {'cci_period': 14, 'threshold': 100}},
        {'name': 'cci_threshold', 'params': {'cci_period': 14, 'threshold': -100}},
        {'name': 'williams_r', 'params': {'williams_period': 14, 'oversold': -80, 'overbought': -20}},
        {'name': 'ultimate_oscillator', 'params': {'period1': 7, 'period2': 14, 'period3': 28, 'oversold': 30, 'overbought': 70}},
    ],
    # Handle other possible regimes with neutral strategies
    'high_vol_bullish': [
        # Breakout strategies for volatile uptrends
        {'name': 'keltner_breakout', 'params': {'period': 19, 'multiplier': 2.5}},
        {'name': 'bollinger_breakout', 'params': {'period': 20, 'std_dev': 2.0}},
        {'name': 'bollinger_breakout', 'params': {'period': 15, 'std_dev': 1.5}},
        {'name': 'atr_channel_breakout', 'params': {'atr_period': 14, 'channel_period': 20, 'atr_multiplier': 2.0}},
        # Momentum strategies for volatile conditions
        {'name': 'dema_crossover', 'params': {'fast_dema_period': 7, 'slow_dema_period': 21}},
        {'name': 'macd_crossover', 'params': {'fast_ema': 12, 'slow_ema': 26, 'signal_ema': 9}}
    ],
    'high_vol_bearish': [
        # Breakout and momentum strategies for volatile downtrends
        {'name': 'keltner_breakout', 'params': {'period': 19, 'multiplier': 2.5}},
        {'name': 'atr_channel_breakout', 'params': {'atr_period': 14, 'channel_period': 20, 'atr_multiplier': 2.0}},
        {'name': 'bollinger_breakout', 'params': {'period': 20, 'std_dev': 2.0}},
        {'name': 'bollinger_breakout', 'params': {'period': 15, 'std_dev': 1.5}},
        # Mean reversion can work in volatile bear markets
        {'name': 'rsi_bands', 'params': {'rsi_period': 14, 'oversold': 20, 'overbought': 80}},
        {'name': 'stochastic_rsi', 'params': {'rsi_period': 14, 'stoch_period': 14, 'oversold': 10, 'overbought': 90}}
    ]
}


@strategy(
    name='duckdb_ensemble',
    feature_config=[
        # Features will be inferred recursively by topology builder
        # This list will be combined with recursive inference results
    ]
)
def duckdb_ensemble(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Adaptive ensemble strategy that switches strategies based on market regime.
    
    Parameters:
        regime_strategies: Dict mapping regime names to strategy configurations
        classifier_name: Name of the classifier to use for regime detection
        aggregation_method: How to combine signals ('equal_weight', 'confidence_weight')
        min_agreement: Minimum fraction of strategies that must agree (0-1)
    """
    # Skip execution if we're just checking features (bar will be None or empty)
    if not bar:
        logger.debug("🎯 ENSEMBLE called with no bar data - skipping (feature check mode)")
        return None
    
    symbol = bar.get('symbol', 'UNKNOWN')
    timestamp = bar.get('timestamp')
    
    # Get configuration
    regime_strategies = params.get('regime_strategies', DEFAULT_REGIME_STRATEGIES)
    classifier_name = params.get('classifier_name', 'volatility_momentum_classifier')
    aggregation_method = params.get('aggregation_method', 'equal_weight')
    min_agreement = params.get('min_agreement', 0.3)  # At least 30% of strategies must agree
    signal_persistence = params.get('signal_persistence', 3)  # Hold signal for N bars minimum
    
    # Get classifier from registry and call it to get current regime
    registry = get_component_registry()
    
    # Get all classifiers
    all_classifiers = registry.get_components_by_type('classifier')
    classifier_names = [c.name for c in all_classifiers]
    logger.debug(f"Available classifiers in registry: {classifier_names}")
    
    # Find the specific classifier
    classifier_info = None
    for clf in all_classifiers:
        if clf.name == classifier_name:
            classifier_info = clf
            break
    
    if not classifier_info:
        logger.error(f"Classifier '{classifier_name}' not found. Available: {classifier_names}")
        # Fallback: use 'neutral' regime when no classifier is available
        current_regime = 'neutral'
        logger.debug(f"No classifier found, using fallback regime: {current_regime}")
    else:
        classifier_func = classifier_info.factory
        
        # Get classifier parameters from the config
        # Look for classifier in the parent context's classifier list
        classifier_params = {}
        
        # Check if we have classifier parameters in the config
        classifiers_config = params.get('_classifiers_config', [])
        for clf_config in classifiers_config:
            if clf_config.get('name') == classifier_name or clf_config.get('type') == classifier_name:
                classifier_params = clf_config.get('params', {})
                logger.debug(f"Found classifier params from config: {classifier_params}")
                break
        
        # Fallback to defaults if not found
        if not classifier_params:
            # Use the actual period values from our minimal test config
            classifier_params = {
                'vol_threshold': params.get('vol_threshold', 0.8),  # Match minimal test config
                'rsi_overbought': params.get('rsi_overbought', 60), # Match minimal test config
                'rsi_oversold': params.get('rsi_oversold', 40),
                'atr_period': params.get('atr_period', 14),  # Match minimal test config
                'rsi_period': params.get('rsi_period', 14),  # Match minimal test config
                'sma_period': params.get('sma_period', 20)   # Match minimal test config
            }
            logger.debug(f"Using default classifier params: {classifier_params}")
        
        try:
            classifier_output = classifier_func(features, classifier_params)
            current_regime = classifier_output.get('regime')
            logger.debug(f"Classifier {classifier_name} returned regime: {current_regime}")
        except Exception as e:
            logger.error(f"Error calling classifier {classifier_name}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    if current_regime is None:
        logger.debug(f"No regime returned by classifier: {classifier_name}")
        return None
    
    # Get strategies for current regime
    active_strategies = regime_strategies.get(current_regime, [])
    
    if not active_strategies:
        logger.info(f"No strategies configured for regime: {current_regime}")
        logger.info(f"Available regimes in config: {list(regime_strategies.keys())}")
        return None
    
    logger.debug(f"🔄 REGIME: {current_regime} → {len(active_strategies)} strategies")
    strategy_names = [s['name'] for s in active_strategies]
    logger.debug(f"🔄 STRATEGIES: {strategy_names}")
    
    # Get all strategies from registry
    all_strategies = registry.get_components_by_type('strategy')
    strategy_names = [s.name for s in all_strategies]
    logger.debug(f"Available strategies in registry: {len(strategy_names)} total")
    
    # Collect signals from active strategies
    signals = []
    strategy_metadata = []
    
    for strategy_config in active_strategies:
        strategy_name = strategy_config['name']
        strategy_params = strategy_config['params']
        
        # Find strategy in registry
        strategy_info = None
        for strat in all_strategies:
            if strat.name == strategy_name:
                strategy_info = strat
                break
        
        if not strategy_info:
            logger.error(f"Strategy '{strategy_name}' not found in registry. Available: {strategy_names[:10]}...")
            continue
        
        strategy_func = strategy_info.factory
        
        try:
            # Call strategy with its specific parameters
            signal = strategy_func(features, bar, strategy_params)
            
            if signal:
                signal_value = signal.get('signal_value', 0)
                logger.debug(f"Strategy {strategy_name} returned signal: {signal_value}")
                if signal_value != 0:
                    signals.append(signal_value)
                    strategy_metadata.append({
                        'strategy': strategy_name,
                        'signal': signal_value,
                        'params': strategy_params
                    })
            else:
                logger.debug(f"Strategy {strategy_name} returned None")
        except Exception as e:
            logger.error(f"Error executing strategy {strategy_name}: {str(e)}")
            continue
    
    # Check if we have enough signals
    if not signals:
        logger.info(f"No signals generated for regime: {current_regime} from {len(active_strategies)} strategies")
        return None
    
    # Aggregate signals using equal weighting
    if aggregation_method == 'equal_weight':
        # Count bullish (+1) and bearish (-1) signals
        bullish_count = sum(1 for s in signals if s > 0)
        bearish_count = sum(1 for s in signals if s < 0)
        total_count = len(signals)
        
        # Calculate net signal
        if bullish_count > bearish_count:
            agreement_ratio = bullish_count / total_count
            if agreement_ratio >= min_agreement:
                ensemble_signal = 1
            else:
                ensemble_signal = 0
        elif bearish_count > bullish_count:
            agreement_ratio = bearish_count / total_count
            if agreement_ratio >= min_agreement:
                ensemble_signal = -1
            else:
                ensemble_signal = 0
        else:
            ensemble_signal = 0
            agreement_ratio = 0
    else:
        # Future: Add other aggregation methods like confidence weighting
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    # Return ensemble signal
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '1m')
    
    return {
        'signal_value': ensemble_signal,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'duckdb_ensemble',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'regime': current_regime,
            'active_strategies': len(active_strategies),
            'signals_generated': len(signals),
            'bullish_signals': bullish_count if aggregation_method == 'equal_weight' else 0,
            'bearish_signals': bearish_count if aggregation_method == 'equal_weight' else 0,
            'agreement_ratio': agreement_ratio if aggregation_method == 'equal_weight' else 0,
            'strategy_details': strategy_metadata,
            'price': bar.get('close', 0)
        }
    }


def create_custom_ensemble(regime_strategy_map: Dict[str, List[Dict[str, Any]]], 
                          classifier_name: str = 'volatility_momentum_classifier',
                          min_agreement: float = 0.3) -> Dict[str, Any]:
    """
    Helper function to create custom ensemble configurations.
    
    Args:
        regime_strategy_map: Dict mapping regime names to lists of strategy configs
        classifier_name: Name of the classifier to use
        min_agreement: Minimum agreement threshold
        
    Returns:
        Configuration dict for duckdb_ensemble strategy
    """
    return {
        'type': 'duckdb_ensemble',
        'name': 'adaptive_ensemble',
        'params': {
            'regime_strategies': regime_strategy_map,
            'classifier_name': classifier_name,
            'aggregation_method': 'equal_weight',
            'min_agreement': min_agreement
        }
    }


# Pre-configured ensemble variations
CONSERVATIVE_ENSEMBLE = create_custom_ensemble(
    {
        'low_vol_bullish': [
            {'name': 'dema_crossover', 'params': {'fast_dema_period': 19, 'slow_dema_period': 35}},
            {'name': 'macd_crossover', 'params': {'fast_ema': 12, 'slow_ema': 35, 'signal_ema': 9}}
        ],
        'low_vol_bearish': [
            {'name': 'stochastic_crossover', 'params': {'k_period': 27, 'd_period': 5}},
            {'name': 'cci_threshold', 'params': {'cci_period': 11, 'threshold': -20}}
        ],
        'neutral': [
            {'name': 'stochastic_rsi', 'params': {'rsi_period': 21, 'stoch_period': 21, 'oversold': 15, 'overbought': 80}}
        ]
    },
    min_agreement=0.5  # Higher agreement required
)

AGGRESSIVE_ENSEMBLE = create_custom_ensemble(
    DEFAULT_REGIME_STRATEGIES,
    min_agreement=0.2  # Lower agreement threshold
)