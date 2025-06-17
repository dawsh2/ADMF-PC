"""
Two-Layer Ensemble Strategy with Debug Output

Debug version that logs regime changes and active strategies.
"""

from typing import Dict, Any, Optional, List
import logging
from ....core.components.discovery import strategy, get_component_registry

logger = logging.getLogger(__name__)

# Import configurations from main implementation
from .two_layer_ensemble import (
    DEFAULT_BASELINE_STRATEGIES,
    DEFAULT_REGIME_BOOSTERS,
    create_two_layer_config,
    CONSERVATIVE_TWO_LAYER,
    AGGRESSIVE_TWO_LAYER,
    BALANCED_TWO_LAYER
)

# Global state for tracking regime changes (reset between runs)
_debug_state = {
    'previous_regime': None,
    'regime_change_count': 0,
    'bar_count': 0,
    'previous_signal': 0,
    'regime_history': []
}


@strategy(
    name='two_layer_ensemble_debug',
    feature_config=[
        # Features will be inferred recursively by topology builder
    ]
)
def two_layer_ensemble_debug(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Debug version of two-layer ensemble with regime change logging.
    """
    global _debug_state
    
    # Skip execution if we're just checking features
    if not bar:
        return None
    
    symbol = bar.get('symbol', 'UNKNOWN')
    timestamp = bar.get('timestamp')
    
    # Get actual bar index from features (provided by ComponentState)
    # This will be the actual number of bars streamed, not relative to when strategy became ready
    bar_idx = features.get('actual_bar_count', features.get('original_bar_index', bar.get('bar_idx', _debug_state['bar_count'])))
    _debug_state['bar_count'] += 1
    
    # Get configuration
    baseline_strategies = params.get('baseline_strategies', DEFAULT_BASELINE_STRATEGIES)
    regime_boosters = params.get('regime_boosters', DEFAULT_REGIME_BOOSTERS)
    classifier_name = params.get('classifier_name', 'market_regime_classifier')
    baseline_allocation = params.get('baseline_allocation', 0.25)  # 25% to baseline
    booster_allocation = 1.0 - baseline_allocation  # 75% to boosters
    
    baseline_aggregation = params.get('baseline_aggregation', 'equal_weight')
    booster_aggregation = params.get('booster_aggregation', 'equal_weight')
    min_baseline_agreement = params.get('min_baseline_agreement', 0.3)
    min_booster_agreement = params.get('min_booster_agreement', 0.3)
    
    # Get component registry
    registry = get_component_registry()
    
    # Get current regime by calling classifier directly (like regular two_layer_ensemble)
    # This is the architectural issue - strategies shouldn't call classifiers inline
    # but that's how two_layer_ensemble is implemented
    all_classifiers = registry.get_components_by_type('classifier')
    classifier_info = None
    
    for clf in all_classifiers:
        if clf.name == classifier_name:
            classifier_info = clf
            break
    
    if not classifier_info:
        logger.error(f"Classifier '{classifier_name}' not found")
        current_regime = 'neutral'
    else:
        try:
            classifier_params = {
                'trend_threshold': params.get('trend_threshold', 0.006),
                'vol_threshold': params.get('vol_threshold', 0.8),
                'sma_short': params.get('sma_short', 12),
                'sma_long': params.get('sma_long', 50),
                'atr_period': params.get('atr_period', 20),
                'rsi_period': params.get('rsi_period', 14)
            }
            
            classifier_func = classifier_info.factory
            classifier_output = classifier_func(features, classifier_params)
            current_regime = classifier_output.get('regime', 'neutral')
            
        except Exception as e:
            logger.error(f"Error calling classifier {classifier_name}: {str(e)}")
            current_regime = 'neutral'
    
    # DEBUG: Log regime changes with crown emoji
    if current_regime != _debug_state['previous_regime']:
        _debug_state['regime_change_count'] += 1
        print(f"\n👑 REGIME CHANGE DETECTED! (Change #{_debug_state['regime_change_count']})")
        print(f"   📍 Bar: {bar_idx}")
        print(f"   📈 Price: ${bar.get('close', 0):.2f}")
        print(f"   🔄 Transition: {_debug_state['previous_regime']} → {current_regime}")
        print(f"   ⏱️  Timestamp: {timestamp}")
        
        # Show what strategies will be active
        active_boosters = regime_boosters.get(current_regime, [])
        print(f"\n   🎯 REGIME-SPECIFIC BOOSTERS ({len(active_boosters)} strategies, {booster_allocation*100:.0f}% weight):")
        for i, strategy_config in enumerate(active_boosters, 1):
            print(f"      {i}. {strategy_config['name']} - params: {strategy_config['params']}")
        
        print(f"\n   🌐 BASELINE STRATEGIES ({len(baseline_strategies)} strategies, {baseline_allocation*100:.0f}% weight):")
        for i, strategy_config in enumerate(baseline_strategies, 1):
            if isinstance(baseline_strategies, dict):
                # Single strategy case
                print(f"      1. {baseline_strategies['name']} - params: {baseline_strategies['params']}")
                break
            else:
                print(f"      {i}. {strategy_config['name']} - params: {strategy_config['params']}")
        
        print(f"   {'='*70}\n")
        
        _debug_state['previous_regime'] = current_regime
    
    # LAYER 1: Execute baseline strategies (always active)
    baseline_signals = []
    baseline_metadata = []
    
    all_strategies = registry.get_components_by_type('strategy')
    
    # Handle case where baseline_strategies might be a single dict
    if isinstance(baseline_strategies, dict):
        baseline_strategies = [baseline_strategies]
    
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
        except Exception as e:
            logger.error(f"Error executing baseline strategy {strategy_name}: {str(e)}")
            continue
    
    # LAYER 2: Execute regime booster strategies (conditional)
    booster_signals = []
    booster_metadata = []
    
    active_boosters = regime_boosters.get(current_regime, [])
    
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
    
    # DEBUG: Log signal generation on signal changes or periodic intervals
    if bar_idx % 1000 == 0 or (final_signal != _debug_state.get('previous_signal', 0)):
        print(f"\n📊 SIGNAL DEBUG at bar {bar_idx} (regime: {current_regime}):")
        print(f"   🌐 Baseline: {len(baseline_signals)} signals → {baseline_ensemble_signal} (agreement: {baseline_agreement:.1%})")
        if baseline_signals:
            signal_summary = ', '.join([f"{m['strategy']}:{m['signal']}" for m in baseline_metadata])
            print(f"      Details: {signal_summary}")
        print(f"   🎯 Boosters: {len(booster_signals)} signals → {booster_ensemble_signal} (agreement: {booster_agreement:.1%})")
        if booster_signals:
            signal_summary = ', '.join([f"{m['strategy']}:{m['signal']}" for m in booster_metadata])
            print(f"      Details: {signal_summary}")
        print(f"   🔄 Final Signal: {final_signal}")
        if baseline_ensemble_signal != 0 and booster_ensemble_signal != 0 and baseline_ensemble_signal != booster_ensemble_signal:
            print(f"   ⚔️  CONFLICT RESOLVED: Baseline({baseline_ensemble_signal}) vs Booster({booster_ensemble_signal}) → {final_signal}")
        
        # Update previous signal for change detection
        _debug_state['previous_signal'] = final_signal
    
    # Return ensemble signal
    timeframe = bar.get('timeframe', '1m')
    
    return {
        'signal_value': final_signal,
        'timestamp': timestamp,
        'strategy_id': 'two_layer_ensemble_debug',
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


def reset_debug_state():
    """Reset debug state between runs."""
    global _debug_state
    _debug_state = {
        'previous_regime': None,
        'regime_change_count': 0,
        'bar_count': 0,
        'previous_signal': 0,
        'regime_history': []
    }


# Export the reset function
__all__ = ['two_layer_ensemble_debug', 'reset_debug_state']