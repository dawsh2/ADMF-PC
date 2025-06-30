"""
Two-Layer Ensemble Strategy with Enhanced Debug Logging

Shows detailed information about:
- Baseline strategy calls and signals
- Regime-specific strategy calls and signals  
- Dynamic switching based on classifier state
- Signal aggregation logic
- Layer conflict resolution
"""

from typing import Dict, Any, Optional, List
from src.core.features.feature_spec import FeatureSpec
import logging
from ....core.components.discovery import strategy, get_component_registry

logger = logging.getLogger(__name__)

# Import configurations from main implementation
from .two_layer_ensemble import (
    DEFAULT_BASELINE_STRATEGIES,
    DEFAULT_REGIME_BOOSTERS,
    create_two_layer_config
)

# Global state for tracking detailed debug info
_debug_state = {
    'bar_count': 0,
    'regime_history': [],
    'signal_history': [],
    'baseline_calls': 0,
    'booster_calls': 0,
    'regime_changes': 0,
    'signal_changes': 0,
    'conflicts': 0,
    'last_regime': None,
    'last_signal': None
}


@strategy(
    name='two_layer_ensemble_enhanced_debug',
    feature_discovery=lambda params: [FeatureSpec('sma', {'period': params.get('sma_period', 20)}), FeatureSpec('ema', {'period': params.get('ema_period', 20)}), FeatureSpec('macd', {'fast_period': params.get('fast_period', 12), 'slow_period': params.get('slow_period', 26), 'signal_period': params.get('signal_period', 9)}), FeatureSpec('stochastic', {}), FeatureSpec('dema', {}), FeatureSpec('vortex', {}), FeatureSpec('tema', {}), FeatureSpec('ichimoku', {}), FeatureSpec('rsi', {'period': params.get('rsi_period', 14)}), FeatureSpec('atr', {})],
    param_feature_mapping={
        # Baseline strategies
        'fast_period': 'sma_{fast_period}',
        'slow_period': 'sma_{slow_period}', 
        'fast_ema_period': 'ema_{fast_ema_period}',
        'slow_ema_period': 'ema_{slow_ema_period}',
        'fast_ema': 'macd_{fast_ema}_{slow_ema}_{signal_ema}',
        'slow_ema': 'macd_{fast_ema}_{slow_ema}_{signal_ema}',
        'signal_ema': 'macd_{fast_ema}_{slow_ema}_{signal_ema}',
        
        # Booster strategies
        'stochastic_k_period': 'stochastic_{stochastic_k_period}_{stochastic_d_period}',
        'stochastic_d_period': 'stochastic_{stochastic_k_period}_{stochastic_d_period}',
        'fast_dema_period': 'dema_{fast_dema_period}',
        'slow_dema_period': 'dema_{slow_dema_period}',
        'vortex_period': 'vortex_{vortex_period}',
        'tema_period': 'tema_{tema_period}',
        'dema_period': 'dema_{dema_period}',
        'ema_period': 'ema_{ema_period}',
        'sma_period': 'sma_{sma_period}',
        'conversion_period': 'ichimoku_{conversion_period}_{base_period}',
        'base_period': 'ichimoku_{conversion_period}_{base_period}',
        
        # Classifier
        'sma_short': 'sma_{sma_short}',
        'sma_long': 'sma_{sma_long}',
        'atr_period': 'atr_{atr_period}',
        'rsi_period': 'rsi_{rsi_period}'
    }
)
def two_layer_ensemble_enhanced_debug(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Enhanced debug version with detailed logging of all strategy interactions.
    """
    global _debug_state
    
    # Skip if just checking features
    if not bar:
        return None
    
    symbol = bar.get('symbol', 'UNKNOWN')
    timestamp = bar.get('timestamp')
    
    # Get actual bar index from features (provided by ComponentState)
    # This will be the actual number of bars streamed, not relative to when strategy became ready
    bar_idx = features.get('actual_bar_count', features.get('original_bar_index', bar.get('bar_idx', _debug_state['bar_count'])))
    _debug_state['bar_count'] += 1
    
    # Configuration
    baseline_strategies = params.get('baseline_strategies', DEFAULT_BASELINE_STRATEGIES)
    regime_boosters = params.get('regime_boosters', DEFAULT_REGIME_BOOSTERS)
    classifier_name = params.get('classifier_name', 'market_regime_classifier')
    baseline_allocation = params.get('baseline_allocation', 0.25)
    booster_allocation = 1.0 - baseline_allocation
    
    baseline_aggregation = params.get('baseline_aggregation', 'equal_weight')
    booster_aggregation = params.get('booster_aggregation', 'equal_weight')
    min_baseline_agreement = params.get('min_baseline_agreement', 0.3)
    min_booster_agreement = params.get('min_booster_agreement', 0.3)
    
    # Get component registry
    registry = get_component_registry()
    
    # Get current regime by calling classifier (like regular two_layer_ensemble)
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
    
    # Log regime changes
    if current_regime != _debug_state['last_regime']:
        _debug_state['regime_changes'] += 1
        print(f"\n{'='*80}")
        print(f"🎯 REGIME CHANGE #{_debug_state['regime_changes']} at bar {bar_idx}")
        print(f"   Previous: {_debug_state['last_regime']} → Current: {current_regime}")
        print(f"   Price: ${bar.get('close', 0):.2f}")
        print(f"{'='*80}")
        _debug_state['last_regime'] = current_regime
    
    # Get all strategies
    all_strategies = registry.get_components_by_type('strategy')
    
    # Handle baseline strategies
    if isinstance(baseline_strategies, dict):
        baseline_strategies = [baseline_strategies]
    
    # LAYER 1: Execute baseline strategies
    print(f"\n📊 Bar {bar_idx} - BASELINE LAYER (25% weight):")
    baseline_signals = []
    baseline_metadata = []
    
    for i, strategy_config in enumerate(baseline_strategies):
        strategy_name = strategy_config['name']
        strategy_params = strategy_config['params']
        
        # Find strategy
        strategy_info = None
        for strat in all_strategies:
            if strat.name == strategy_name:
                strategy_info = strat
                break
        
        if not strategy_info:
            print(f"   ❌ Strategy '{strategy_name}' not found")
            continue
        
        try:
            strategy_func = strategy_info.factory
            signal = strategy_func(features, bar, strategy_params)
            _debug_state['baseline_calls'] += 1
            
            if signal:
                signal_value = signal.get('signal_value', 0)
                print(f"   {i+1}. {strategy_name}: signal={signal_value}")
                
                if signal_value != 0:
                    baseline_signals.append(signal_value)
                    baseline_metadata.append({
                        'layer': 'baseline',
                        'strategy': strategy_name,
                        'signal': signal_value,
                        'params': strategy_params
                    })
            else:
                print(f"   {i+1}. {strategy_name}: no signal")
                
        except Exception as e:
            print(f"   ❌ Error in {strategy_name}: {str(e)}")
    
    # LAYER 2: Execute regime booster strategies
    active_boosters = regime_boosters.get(current_regime, [])
    print(f"\n🚀 Bar {bar_idx} - REGIME BOOSTERS for '{current_regime}' (75% weight):")
    booster_signals = []
    booster_metadata = []
    
    for i, strategy_config in enumerate(active_boosters):
        strategy_name = strategy_config['name']
        strategy_params = strategy_config['params']
        
        # Find strategy
        strategy_info = None
        for strat in all_strategies:
            if strat.name == strategy_name:
                strategy_info = strat
                break
        
        if not strategy_info:
            print(f"   ❌ Strategy '{strategy_name}' not found")
            continue
        
        try:
            strategy_func = strategy_info.factory
            signal = strategy_func(features, bar, strategy_params)
            _debug_state['booster_calls'] += 1
            
            if signal:
                signal_value = signal.get('signal_value', 0)
                print(f"   {i+1}. {strategy_name}: signal={signal_value}")
                
                if signal_value != 0:
                    booster_signals.append(signal_value)
                    booster_metadata.append({
                        'layer': 'booster',
                        'regime': current_regime,
                        'strategy': strategy_name,
                        'signal': signal_value,
                        'params': strategy_params
                    })
            else:
                print(f"   {i+1}. {strategy_name}: no signal")
                
        except Exception as e:
            print(f"   ❌ Error in {strategy_name}: {str(e)}")
    
    # Aggregate baseline signals
    baseline_ensemble_signal = 0
    baseline_agreement = 0
    
    if baseline_signals:
        if baseline_aggregation == 'equal_weight':
            bullish_count = sum(1 for s in baseline_signals if s > 0)
            bearish_count = sum(1 for s in baseline_signals if s < 0)
            total_count = len(baseline_signals)
            
            print(f"\n📈 Baseline aggregation: {bullish_count} bullish, {bearish_count} bearish of {total_count}")
            
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
            
            print(f"📈 Booster aggregation: {bullish_count} bullish, {bearish_count} bearish of {total_count}")
            
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
    print(f"\n🔄 SIGNAL COMBINATION:")
    print(f"   Baseline signal: {baseline_ensemble_signal} (agreement: {baseline_agreement:.1%})")
    print(f"   Booster signal: {booster_ensemble_signal} (agreement: {booster_agreement:.1%})")
    
    if baseline_ensemble_signal == 0 and booster_ensemble_signal == 0:
        final_signal = 0
        print(f"   → Final: 0 (both layers neutral)")
    elif baseline_ensemble_signal == 0:
        final_signal = booster_ensemble_signal
        print(f"   → Final: {final_signal} (only booster signal)")
    elif booster_ensemble_signal == 0:
        final_signal = baseline_ensemble_signal
        print(f"   → Final: {final_signal} (only baseline signal)")
    else:
        # Both layers have signals
        if baseline_ensemble_signal == booster_ensemble_signal:
            final_signal = baseline_ensemble_signal
            print(f"   → Final: {final_signal} (both layers agree)")
        else:
            # Conflict - use allocation weighting
            baseline_strength = baseline_allocation * baseline_agreement
            booster_strength = booster_allocation * booster_agreement
            
            _debug_state['conflicts'] += 1
            print(f"   ⚔️ CONFLICT #{_debug_state['conflicts']}!")
            print(f"      Baseline strength: {baseline_strength:.3f} ({baseline_allocation} × {baseline_agreement:.1%})")
            print(f"      Booster strength: {booster_strength:.3f} ({booster_allocation} × {booster_agreement:.1%})")
            
            if baseline_strength > booster_strength:
                final_signal = baseline_ensemble_signal
                print(f"   → Final: {final_signal} (baseline wins)")
            elif booster_strength > baseline_strength:
                final_signal = booster_ensemble_signal
                print(f"   → Final: {final_signal} (booster wins)")
            else:
                final_signal = 0
                print(f"   → Final: 0 (tie)")
    
    # Track signal changes
    if final_signal != _debug_state['last_signal']:
        _debug_state['signal_changes'] += 1
        _debug_state['last_signal'] = final_signal
    
    # Print summary every 100 bars
    if bar_idx % 100 == 0 and bar_idx > 0:
        print(f"\n{'='*80}")
        print(f"📊 SUMMARY at bar {bar_idx}:")
        print(f"   Total baseline calls: {_debug_state['baseline_calls']}")
        print(f"   Total booster calls: {_debug_state['booster_calls']}")
        print(f"   Regime changes: {_debug_state['regime_changes']}")
        print(f"   Signal changes: {_debug_state['signal_changes']}")
        print(f"   Conflicts resolved: {_debug_state['conflicts']}")
        print(f"{'='*80}\n")
    
    # Return ensemble signal
    timeframe = bar.get('timeframe', '1m')
    
    return {
        'signal_value': final_signal,
        'timestamp': timestamp,
        'strategy_id': 'two_layer_ensemble_enhanced_debug',
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


def reset_enhanced_debug_state():
    """Reset debug state between runs."""
    global _debug_state
    _debug_state = {
        'bar_count': 0,
        'regime_history': [],
        'signal_history': [],
        'baseline_calls': 0,
        'booster_calls': 0,
        'regime_changes': 0,
        'signal_changes': 0,
        'conflicts': 0,
        'last_regime': None,
        'last_signal': None
    }


# Export
__all__ = ['two_layer_ensemble_enhanced_debug', 'reset_enhanced_debug_state']