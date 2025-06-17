"""
Two-Layer Ensemble Strategy

A two-layer adaptive ensemble strategy with:
1. Baseline Layer (60%): Always-active cross-regime performers
2. Regime Booster Layer (40%): Regime-specific strategies that activate conditionally

This architecture provides stability through the baseline layer while optimizing
performance through regime-specific boosters.
"""

from typing import Dict, Any, Optional, List
import logging
from ....core.components.discovery import strategy, get_component_registry

logger = logging.getLogger(__name__)

# Baseline strategies - always active regardless of regime (cross-regime performers)
DEFAULT_BASELINE_STRATEGIES = [
    {'name': 'dema_crossover', 'params': {'fast_dema_period': 19, 'slow_dema_period': 15}},
    {'name': 'elder_ray', 'params': {'ema_period': 13, 'bull_threshold': 0, 'bear_threshold': -0.001}},
    {'name': 'sma_crossover', 'params': {'fast_period': 19, 'slow_period': 15}},
    {'name': 'stochastic_crossover', 'params': {'k_period': 5, 'd_period': 7}},
    {'name': 'pivot_channel_bounces', 'params': {'sr_period': 15, 'min_touches': 2, 'bounce_threshold': 0.001}}
]

# Regime-specific booster strategies - activate only when specific regime detected
DEFAULT_REGIME_BOOSTERS = {
    'bull_ranging': [
        {'name': 'roc_threshold', 'params': {'period': 5, 'threshold': 0.05}},
        {'name': 'rsi_threshold', 'params': {'period': 27, 'threshold': 50}},
        {'name': 'atr_channel_breakout', 'params': {'atr_period': 7, 'channel_period': 30, 'atr_multiplier': 1.5}},
        {'name': 'mfi_bands', 'params': {'period': 7, 'oversold': 25, 'overbought': 85}}
    ],
    'bear_ranging': [
        {'name': 'trendline_bounces', 'params': {'lookback': 5, 'min_touches': 2, 'threshold': 0.0005, 'strength': 0.1}},
        {'name': 'tema_sma_crossover', 'params': {'tema_period': 5, 'sma_period': 23}},
        {'name': 'rsi_threshold', 'params': {'period': 27, 'threshold': 50}},
        {'name': 'macd_crossover', 'params': {'fast_ema': 5, 'slow_ema': 35, 'signal_ema': 9}}
    ],
    'neutral': [
        {'name': 'ichimoku', 'params': {'conversion_period': 9, 'base_period': 35}},
        {'name': 'williams_r', 'params': {'williams_period': 21, 'oversold': -80, 'overbought': -20}},
        {'name': 'ema_sma_crossover', 'params': {'ema_period': 5, 'sma_period': 50}},
        {'name': 'aroon_crossover', 'params': {'period': 14}}
    ],
    # Fallback regimes
    'high_vol_bullish': [
        {'name': 'roc_threshold', 'params': {'period': 5, 'threshold': 0.05}},
        {'name': 'atr_channel_breakout', 'params': {'atr_period': 7, 'channel_period': 30, 'atr_multiplier': 1.5}}
    ],
    'high_vol_bearish': [
        {'name': 'tema_sma_crossover', 'params': {'tema_period': 5, 'sma_period': 23}},
        {'name': 'macd_crossover', 'params': {'fast_ema': 5, 'slow_ema': 35, 'signal_ema': 9}}
    ],
    'low_vol_bullish': [
        {'name': 'roc_threshold', 'params': {'period': 5, 'threshold': 0.05}},
        {'name': 'rsi_threshold', 'params': {'period': 27, 'threshold': 50}}
    ],
    'low_vol_bearish': [
        {'name': 'trendline_bounces', 'params': {'lookback': 5, 'min_touches': 2, 'threshold': 0.0005, 'strength': 0.1}},
        {'name': 'tema_sma_crossover', 'params': {'tema_period': 5, 'sma_period': 23}}
    ]
}


@strategy(
    name='two_layer_ensemble',
    feature_config=[
        # Features will be inferred recursively by topology builder
    ]
)
def two_layer_ensemble(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Two-layer ensemble strategy with baseline + regime boosters.
    
    Architecture:
    - Baseline Layer (default 60%): Always-active cross-regime performers
    - Regime Booster Layer (default 40%): Regime-specific strategies
    
    Parameters:
        baseline_strategies: List of always-active strategy configurations
        regime_boosters: Dict mapping regime names to booster strategy configurations
        classifier_name: Name of the classifier to use for regime detection
        baseline_allocation: Fraction allocated to baseline layer (0.0-1.0)
        baseline_aggregation: How to combine baseline signals ('equal_weight', 'consensus')
        booster_aggregation: How to combine booster signals ('equal_weight', 'consensus')
        min_baseline_agreement: Minimum fraction of baseline strategies that must agree
        min_booster_agreement: Minimum fraction of booster strategies that must agree
    """
    # Skip execution if we're just checking features (bar will be None or empty)
    if not bar:
        logger.debug("🎯 TWO_LAYER_ENSEMBLE called with no bar data - skipping (feature check mode)")
        return None
    
    symbol = bar.get('symbol', 'UNKNOWN')
    timestamp = bar.get('timestamp')
    
    # Get configuration
    baseline_strategies = params.get('baseline_strategies', DEFAULT_BASELINE_STRATEGIES)
    regime_boosters = params.get('regime_boosters', DEFAULT_REGIME_BOOSTERS)
    classifier_name = params.get('classifier_name', 'market_regime_detector')
    baseline_allocation = params.get('baseline_allocation', 0.6)  # 60% to baseline
    booster_allocation = 1.0 - baseline_allocation  # 40% to boosters
    
    baseline_aggregation = params.get('baseline_aggregation', 'equal_weight')
    booster_aggregation = params.get('booster_aggregation', 'equal_weight')
    min_baseline_agreement = params.get('min_baseline_agreement', 0.3)
    min_booster_agreement = params.get('min_booster_agreement', 0.3)
    
    # Get component registry
    registry = get_component_registry()
    
    # Get current regime from classifier
    current_regime = None
    all_classifiers = registry.get_components_by_type('classifier')
    classifier_info = None
    
    for clf in all_classifiers:
        if clf.name == classifier_name:
            classifier_info = clf
            break
    
    if not classifier_info:
        logger.error(f"Classifier '{classifier_name}' not found. Available: {[c.name for c in all_classifiers]}")
        current_regime = 'neutral'  # Fallback
    else:
        try:
            # Get classifier parameters from config
            classifier_params = {}
            classifiers_config = params.get('_classifiers_config', [])
            for clf_config in classifiers_config:
                if clf_config.get('name') == classifier_name or clf_config.get('type') == classifier_name:
                    classifier_params = clf_config.get('params', {})
                    break
            
            # Use defaults if not found
            if not classifier_params:
                classifier_params = {
                    'trend_sma_period': params.get('trend_sma_period', 50),
                    'trend_threshold': params.get('trend_threshold', 0.006),
                    'bull_bear_sma_period': params.get('bull_bear_sma_period', 12),
                    'volatility_period': params.get('volatility_period', 20),
                    'volatility_threshold': params.get('volatility_threshold', 1.5)
                }
            
            classifier_func = classifier_info.factory
            classifier_output = classifier_func(features, classifier_params)
            current_regime = classifier_output.get('regime')
            logger.debug(f"Classifier {classifier_name} returned regime: {current_regime}")
            
        except Exception as e:
            logger.error(f"Error calling classifier {classifier_name}: {str(e)}")
            current_regime = 'neutral'  # Fallback
    
    if current_regime is None:
        current_regime = 'neutral'  # Fallback
    
    # LAYER 1: Execute baseline strategies (always active)
    baseline_signals = []
    baseline_metadata = []
    
    all_strategies = registry.get_components_by_type('strategy')
    
    # Handle case where baseline_strategies might be a single dict instead of a list
    if isinstance(baseline_strategies, dict):
        baseline_strategies = [baseline_strategies]
    
    logger.debug(f"🌐 BASELINE LAYER: Executing {len(baseline_strategies)} always-active strategies")
    
    for strategy_config in baseline_strategies:
        strategy_name = strategy_config['name']
        strategy_params = strategy_config['params']
        
        # Find strategy in registry
        strategy_info = None
        for strat in all_strategies:
            if strat.name == strategy_name:
                strategy_info = strat
                break
        
        if not strategy_info:
            logger.error(f"Baseline strategy '{strategy_name}' not found in registry")
            continue
        
        try:
            strategy_func = strategy_info.factory
            signal = strategy_func(features, bar, strategy_params)
            
            if signal:
                signal_value = signal.get('signal_value', 0)
                if signal_value != 0:
                    baseline_signals.append(signal_value)
                    baseline_metadata.append({
                        'layer': 'baseline',
                        'strategy': strategy_name,
                        'signal': signal_value,
                        'params': strategy_params
                    })
                    logger.debug(f"  Baseline {strategy_name}: {signal_value}")
        except Exception as e:
            logger.error(f"Error executing baseline strategy {strategy_name}: {str(e)}")
            continue
    
    # LAYER 2: Execute regime booster strategies (conditional)
    booster_signals = []
    booster_metadata = []
    
    active_boosters = regime_boosters.get(current_regime, [])
    logger.debug(f"🎯 REGIME BOOSTER LAYER: Regime '{current_regime}' → {len(active_boosters)} boosters")
    
    for strategy_config in active_boosters:
        strategy_name = strategy_config['name']
        strategy_params = strategy_config['params']
        
        # Find strategy in registry
        strategy_info = None
        for strat in all_strategies:
            if strat.name == strategy_name:
                strategy_info = strat
                break
        
        if not strategy_info:
            logger.error(f"Booster strategy '{strategy_name}' not found in registry")
            continue
        
        try:
            strategy_func = strategy_info.factory
            signal = strategy_func(features, bar, strategy_params)
            
            if signal:
                signal_value = signal.get('signal_value', 0)
                if signal_value != 0:
                    booster_signals.append(signal_value)
                    booster_metadata.append({
                        'layer': 'booster',
                        'regime': current_regime,
                        'strategy': strategy_name,
                        'signal': signal_value,
                        'params': strategy_params
                    })
                    logger.debug(f"  Booster {strategy_name}: {signal_value}")
        except Exception as e:
            logger.error(f"Error executing booster strategy {strategy_name}: {str(e)}")
            continue
    
    # Aggregate baseline signals
    baseline_ensemble_signal = 0
    baseline_agreement = 0
    
    if baseline_signals:
        if baseline_aggregation == 'equal_weight':
            bullish_count = sum(1 for s in baseline_signals if s > 0)
            bearish_count = sum(1 for s in baseline_signals if s < 0)
            total_count = len(baseline_signals)
            
            if bullish_count > bearish_count:
                baseline_agreement = bullish_count / total_count
                baseline_ensemble_signal = 1 if baseline_agreement >= min_baseline_agreement else 0
            elif bearish_count > bullish_count:
                baseline_agreement = bearish_count / total_count
                baseline_ensemble_signal = -1 if baseline_agreement >= min_baseline_agreement else 0
            else:
                baseline_ensemble_signal = 0
                baseline_agreement = 0
    
    # Aggregate booster signals
    booster_ensemble_signal = 0
    booster_agreement = 0
    
    if booster_signals:
        if booster_aggregation == 'equal_weight':
            bullish_count = sum(1 for s in booster_signals if s > 0)
            bearish_count = sum(1 for s in booster_signals if s < 0)
            total_count = len(booster_signals)
            
            if bullish_count > bearish_count:
                booster_agreement = bullish_count / total_count
                booster_ensemble_signal = 1 if booster_agreement >= min_booster_agreement else 0
            elif bearish_count > bullish_count:
                booster_agreement = bearish_count / total_count
                booster_ensemble_signal = -1 if booster_agreement >= min_booster_agreement else 0
            else:
                booster_ensemble_signal = 0
                booster_agreement = 0
    
    # Combine layers with allocation weighting
    if baseline_ensemble_signal == 0 and booster_ensemble_signal == 0:
        final_signal = 0
    elif baseline_ensemble_signal == 0:
        final_signal = booster_ensemble_signal  # Only booster signal
    elif booster_ensemble_signal == 0:
        final_signal = baseline_ensemble_signal  # Only baseline signal
    else:
        # Both layers have signals - use allocation weighting for agreement
        if baseline_ensemble_signal == booster_ensemble_signal:
            final_signal = baseline_ensemble_signal  # Same direction - reinforce
        else:
            # Conflicting signals - weight by allocation and agreement
            baseline_strength = baseline_allocation * baseline_agreement
            booster_strength = booster_allocation * booster_agreement
            
            if baseline_strength > booster_strength:
                final_signal = baseline_ensemble_signal
            elif booster_strength > baseline_strength:
                final_signal = booster_ensemble_signal
            else:
                final_signal = 0  # Tie - no signal
    
    # Log layer results
    logger.debug(f"🌐 Baseline Layer: {len(baseline_signals)} signals → {baseline_ensemble_signal} (agreement: {baseline_agreement:.1%})")
    logger.debug(f"🎯 Booster Layer: {len(booster_signals)} signals → {booster_ensemble_signal} (agreement: {booster_agreement:.1%})")
    logger.debug(f"🔄 Final Signal: {final_signal}")
    
    # Return ensemble signal
    timeframe = bar.get('timeframe', '1m')
    
    return {
        'signal_value': final_signal,
        'timestamp': timestamp,
        'strategy_id': 'two_layer_ensemble',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'regime': current_regime,
            'baseline_layer': {
                'strategies_executed': len(baseline_strategies),
                'signals_generated': len(baseline_signals),
                'ensemble_signal': baseline_ensemble_signal,
                'agreement_ratio': baseline_agreement,
                'allocation': baseline_allocation
            },
            'booster_layer': {
                'strategies_executed': len(active_boosters),
                'signals_generated': len(booster_signals),
                'ensemble_signal': booster_ensemble_signal,
                'agreement_ratio': booster_agreement,
                'allocation': booster_allocation
            },
            'signal_combination': {
                'baseline_signal': baseline_ensemble_signal,
                'booster_signal': booster_ensemble_signal,
                'final_signal': final_signal
            },
            'strategy_details': baseline_metadata + booster_metadata,
            'price': bar.get('close', 0)
        }
    }


def create_two_layer_config(
    baseline_strategies: List[Dict[str, Any]] = None,
    regime_boosters: Dict[str, List[Dict[str, Any]]] = None,
    classifier_name: str = 'market_regime_detector',
    baseline_allocation: float = 0.6
) -> Dict[str, Any]:
    """
    Helper function to create two-layer ensemble configurations.
    
    Args:
        baseline_strategies: List of always-active strategy configs
        regime_boosters: Dict mapping regime names to booster strategy configs
        classifier_name: Name of the classifier to use
        baseline_allocation: Fraction allocated to baseline layer (0.0-1.0)
        
    Returns:
        Configuration dict for two_layer_ensemble strategy
    """
    return {
        'type': 'two_layer_ensemble',
        'name': 'baseline_plus_regime_boosters',
        'params': {
            'baseline_strategies': baseline_strategies or DEFAULT_BASELINE_STRATEGIES,
            'regime_boosters': regime_boosters or DEFAULT_REGIME_BOOSTERS,
            'classifier_name': classifier_name,
            'baseline_allocation': baseline_allocation,
            'baseline_aggregation': 'equal_weight',
            'booster_aggregation': 'equal_weight',
            'min_baseline_agreement': 0.3,
            'min_booster_agreement': 0.3
        }
    }


# Pre-configured ensemble variations
CONSERVATIVE_TWO_LAYER = create_two_layer_config(
    baseline_allocation=0.8,  # 80% baseline for stability
    classifier_name='market_regime_detector'
)

AGGRESSIVE_TWO_LAYER = create_two_layer_config(
    baseline_allocation=0.4,  # 40% baseline, 60% boosters for optimization
    classifier_name='market_regime_detector'
)

BALANCED_TWO_LAYER = create_two_layer_config(
    baseline_allocation=0.6,  # 60% baseline, 40% boosters (default)
    classifier_name='market_regime_detector'
)