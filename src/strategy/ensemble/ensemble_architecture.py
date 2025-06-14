"""
Ensemble Strategy Architecture

Different approaches for combining rules and regimes into ensemble strategies.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class EnsembleType(Enum):
    """Types of ensemble architectures."""
    REGIME_SPECIFIC = "regime_specific"      # One ensemble per regime
    CLASSIFIER_SPECIFIC = "classifier_specific"  # One ensemble per classifier
    HIERARCHICAL = "hierarchical"           # Nested ensembles
    ADAPTIVE = "adaptive"                   # Dynamic weight adjustment
    CONSENSUS = "consensus"                 # Multi-classifier voting


@dataclass
class EnsembleConfig:
    """Configuration for ensemble strategies."""
    ensemble_type: EnsembleType
    min_agreement: float = 0.5  # Minimum agreement for consensus
    weight_decay: float = 0.95  # For adaptive weighting
    rebalance_frequency: int = 100  # Bars between weight updates


class RegimeSpecificEnsemble:
    """
    One ensemble per regime state.
    
    Example:
    - strong_uptrend → [sma_crossover, macd, supertrend]
    - sideways → [rsi_bands, bollinger_mean_reversion, pivot_points]
    - high_volatility → [keltner_breakout, atr_channel, vwap_deviation]
    """
    
    def __init__(self, regime_strategy_map: Dict[str, List[str]]):
        """
        Args:
            regime_strategy_map: Mapping of regime → list of strategy names
        """
        self.regime_strategy_map = regime_strategy_map
        
    def get_active_strategies(self, current_regime: str) -> List[str]:
        """Get strategies active for current regime."""
        return self.regime_strategy_map.get(current_regime, [])


class ClassifierSpecificEnsemble:
    """
    One ensemble per classifier, each with its own strategy set.
    
    Example:
    - trend_classifier → trend-following strategies
    - volatility_classifier → volatility-based strategies
    - microstructure_classifier → short-term pattern strategies
    """
    
    def __init__(self, classifier_strategy_map: Dict[str, Dict[str, List[str]]]):
        """
        Args:
            classifier_strategy_map: Mapping of classifier → regime → strategies
        """
        self.classifier_strategy_map = classifier_strategy_map
        
    def get_ensemble_signals(self, classifier_states: Dict[str, str]) -> Dict[str, List[str]]:
        """Get active strategies for each classifier's current state."""
        ensemble_signals = {}
        for classifier, regime in classifier_states.items():
            if classifier in self.classifier_strategy_map:
                regime_strategies = self.classifier_strategy_map[classifier].get(regime, [])
                ensemble_signals[classifier] = regime_strategies
        return ensemble_signals


class HierarchicalEnsemble:
    """
    Nested ensemble with master and sub-ensembles.
    
    Structure:
    - Master ensemble selects active sub-ensembles
    - Each sub-ensemble specializes in specific conditions
    - Sub-ensembles can have their own regime filters
    """
    
    def __init__(self):
        self.master_ensemble = None
        self.sub_ensembles = {}
        
    def add_sub_ensemble(self, name: str, 
                        ensemble: RegimeSpecificEnsemble,
                        activation_regimes: List[str]):
        """Add a sub-ensemble with activation conditions."""
        self.sub_ensembles[name] = {
            'ensemble': ensemble,
            'activation_regimes': activation_regimes
        }


class AdaptiveWeightEnsemble:
    """
    Dynamically adjusts strategy weights based on recent performance.
    
    Features:
    - Tracks strategy performance by regime
    - Updates weights based on rolling Sharpe ratio
    - Applies weight decay to prevent overfitting
    """
    
    def __init__(self, strategies: List[str], 
                 initial_weight: float = 1.0,
                 lookback_period: int = 100):
        self.strategies = strategies
        self.weights = {s: initial_weight for s in strategies}
        self.performance_buffer = {s: [] for s in strategies}
        self.lookback_period = lookback_period
        
    def update_weights(self, strategy_returns: Dict[str, float], 
                      current_regime: str):
        """Update strategy weights based on recent performance."""
        # Implementation would track rolling performance
        # and adjust weights accordingly
        pass


class ConsensusEnsemble:
    """
    Multi-classifier consensus approach.
    
    Rules:
    - Each classifier votes on market state
    - Strategies are activated based on consensus
    - Can require minimum agreement level
    """
    
    def __init__(self, classifiers: List[str], 
                 strategy_regime_map: Dict[str, Dict[str, List[str]]],
                 min_agreement: float = 0.6):
        self.classifiers = classifiers
        self.strategy_regime_map = strategy_regime_map
        self.min_agreement = min_agreement
        
    def get_consensus_strategies(self, classifier_states: Dict[str, str]) -> List[str]:
        """Get strategies based on classifier consensus."""
        # Count regime votes
        regime_votes = {}
        for classifier, regime in classifier_states.items():
            regime_type = self._normalize_regime(regime)
            regime_votes[regime_type] = regime_votes.get(regime_type, 0) + 1
            
        # Find consensus regimes
        consensus_strategies = set()
        total_votes = len(classifier_states)
        
        for regime_type, votes in regime_votes.items():
            if votes / total_votes >= self.min_agreement:
                # Add strategies for consensus regime
                for strategy_list in self.strategy_regime_map.get(regime_type, {}).values():
                    consensus_strategies.update(strategy_list)
                    
        return list(consensus_strategies)
    
    def _normalize_regime(self, regime: str) -> str:
        """Normalize regime names to general categories."""
        # Map specific regimes to general types
        if 'trend' in regime or 'markup' in regime or 'markdown' in regime:
            return 'trending'
        elif 'rang' in regime or 'sideways' in regime or 'consolidation' in regime:
            return 'ranging'
        elif 'vol' in regime:
            return 'volatile'
        else:
            return 'neutral'


# Example configurations for different ensemble approaches

def create_regime_specific_ensembles() -> Dict[str, RegimeSpecificEnsemble]:
    """Create ensemble for each major regime type."""
    
    ensembles = {
        'trend_ensemble': RegimeSpecificEnsemble({
            'strong_uptrend': ['sma_crossover', 'macd_crossover', 'supertrend', 'parabolic_sar'],
            'weak_uptrend': ['ema_crossover', 'adx_trend_strength', 'linear_regression_slope'],
            'strong_downtrend': ['sma_crossover', 'macd_crossover', 'supertrend', 'parabolic_sar'],
            'weak_downtrend': ['ema_crossover', 'adx_trend_strength', 'linear_regression_slope'],
            'sideways': ['rsi_bands', 'cci_bands', 'bollinger_breakout', 'pivot_points']
        }),
        
        'volatility_ensemble': RegimeSpecificEnsemble({
            'high_vol_bullish': ['keltner_breakout', 'atr_channel_breakout', 'vwap_deviation'],
            'high_vol_bearish': ['keltner_breakout', 'atr_channel_breakout', 'vwap_deviation'],
            'low_vol_bullish': ['bollinger_breakout', 'donchian_breakout', 'support_resistance_breakout'],
            'low_vol_bearish': ['bollinger_breakout', 'donchian_breakout', 'support_resistance_breakout'],
            'neutral': ['pivot_points', 'fibonacci_retracement']
        }),
        
        'microstructure_ensemble': RegimeSpecificEnsemble({
            'breakout_up': ['donchian_breakout', 'atr_channel_breakout', 'volume_breakout'],
            'breakout_down': ['donchian_breakout', 'atr_channel_breakout', 'volume_breakout'],
            'consolidation': ['pivot_points', 'bollinger_breakout', 'rsi_bands'],
            'reversal_up': ['rsi_bands', 'stochastic_rsi', 'williams_r'],
            'reversal_down': ['rsi_bands', 'stochastic_rsi', 'williams_r']
        })
    }
    
    return ensembles


def create_classifier_specific_ensemble() -> ClassifierSpecificEnsemble:
    """Create ensemble with strategies mapped to each classifier."""
    
    classifier_map = {
        'enhanced_trend_classifier': {
            'strong_uptrend': ['sma_crossover', 'macd_crossover', 'adx_trend_strength'],
            'weak_uptrend': ['ema_crossover', 'supertrend'],
            'sideways': ['rsi_bands', 'bollinger_breakout', 'pivot_points'],
            'weak_downtrend': ['ema_crossover', 'supertrend'],
            'strong_downtrend': ['sma_crossover', 'macd_crossover', 'adx_trend_strength']
        },
        
        'volatility_momentum_classifier': {
            'high_vol_bullish': ['keltner_breakout', 'atr_channel_breakout'],
            'high_vol_bearish': ['keltner_breakout', 'atr_channel_breakout'],
            'low_vol_bullish': ['donchian_breakout', 'linear_regression_slope'],
            'low_vol_bearish': ['donchian_breakout', 'linear_regression_slope'],
            'neutral': ['pivot_points', 'fibonacci_retracement']
        },
        
        'hidden_markov_classifier': {
            'accumulation': ['obv_trend', 'accumulation_distribution', 'chaikin_money_flow'],
            'markup': ['sma_crossover', 'supertrend', 'parabolic_sar'],
            'distribution': ['obv_trend', 'mfi_bands', 'chaikin_money_flow'],
            'markdown': ['sma_crossover', 'supertrend', 'parabolic_sar'],
            'uncertainty': ['pivot_points', 'vwap_deviation']
        }
    }
    
    return ClassifierSpecificEnsemble(classifier_map)


def create_adaptive_ensemble_config() -> Dict[str, Any]:
    """Configuration for adaptive weight ensemble."""
    
    return {
        'performance_metric': 'sharpe_ratio',
        'lookback_period': 100,
        'weight_update_frequency': 50,
        'min_weight': 0.0,
        'max_weight': 2.0,
        'weight_decay': 0.95,
        'regime_specific_weights': True,
        'weight_initialization': 'equal'  # or 'historical_performance'
    }