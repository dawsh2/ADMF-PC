"""
Signal Strength Filter Component

This component calculates signal strength/confidence scores for binary signals
without modifying the stored signal values. This maintains sparse storage
while providing rich information to the risk module for position sizing.
"""

from typing import Dict, Any, List, Optional
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class StrengthCalculator(ABC):
    """Base class for signal strength calculators."""
    
    @abstractmethod
    def calculate(self, signal: Dict[str, Any], features: Dict[str, Any]) -> float:
        """Calculate strength score for a signal."""
        pass


class IndicatorExtremityCalculator(StrengthCalculator):
    """
    Calculate strength based on how extreme indicator values are.
    More extreme = stronger signal.
    """
    
    def calculate(self, signal: Dict[str, Any], features: Dict[str, Any]) -> float:
        """Calculate strength based on indicator extremity."""
        if signal['signal_value'] == 0:
            return 0.0
            
        strength_scores = []
        
        # RSI extremity
        if 'rsi' in features:
            rsi = features['rsi']
            if signal['signal_value'] == 1:  # Long signal
                # Stronger as RSI gets more oversold
                if rsi < 30:
                    strength_scores.append(min(1.0, (30 - rsi) / 20))
                else:
                    strength_scores.append(0.0)
            else:  # Short signal
                # Stronger as RSI gets more overbought
                if rsi > 70:
                    strength_scores.append(min(1.0, (rsi - 70) / 20))
                else:
                    strength_scores.append(0.0)
        
        # Stochastic extremity
        for period in [14, 21]:  # Check multiple periods if available
            stoch_k = features.get(f'stochastic_{period}_3_k')
            if stoch_k is not None:
                if signal['signal_value'] == 1:  # Long
                    if stoch_k < 20:
                        strength_scores.append(min(1.0, (20 - stoch_k) / 15))
                    else:
                        strength_scores.append(0.0)
                else:  # Short
                    if stoch_k > 80:
                        strength_scores.append(min(1.0, (stoch_k - 80) / 15))
                    else:
                        strength_scores.append(0.0)
        
        return sum(strength_scores) / len(strength_scores) if strength_scores else 0.5


class TrendAlignmentCalculator(StrengthCalculator):
    """
    Calculate strength based on trend alignment across timeframes.
    Signal aligned with trend = stronger.
    """
    
    def calculate(self, signal: Dict[str, Any], features: Dict[str, Any]) -> float:
        """Calculate strength based on trend alignment."""
        if signal['signal_value'] == 0:
            return 0.0
            
        alignment_scores = []
        
        # Check multiple MA pairs for trend
        ma_pairs = [
            (10, 20),    # Short-term
            (20, 50),    # Medium-term
            (50, 200)    # Long-term
        ]
        
        for fast, slow in ma_pairs:
            fast_ma = features.get(f'sma_{fast}')
            slow_ma = features.get(f'sma_{slow}')
            
            if fast_ma is not None and slow_ma is not None:
                trend_up = fast_ma > slow_ma
                aligned = (trend_up and signal['signal_value'] == 1) or \
                         (not trend_up and signal['signal_value'] == -1)
                alignment_scores.append(1.0 if aligned else 0.0)
        
        # Check ADX for trend strength
        adx = features.get('adx_14_adx')
        if adx is not None:
            # Strong trend (ADX > 25) boosts aligned signals
            if adx > 25:
                trend_strength_multiplier = min(1.5, adx / 25)
            else:
                trend_strength_multiplier = 0.8
            
            if alignment_scores:
                alignment_scores = [s * trend_strength_multiplier for s in alignment_scores]
        
        return sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.5


class VolatilityAdjustmentCalculator(StrengthCalculator):
    """
    Adjust strength based on volatility.
    High volatility = reduce strength (risk management).
    Low volatility = increase strength (capitalize on clear signals).
    """
    
    def calculate(self, signal: Dict[str, Any], features: Dict[str, Any]) -> float:
        """Calculate volatility-based strength adjustment."""
        if signal['signal_value'] == 0:
            return 0.0
            
        # Get volatility indicators
        atr = features.get('atr_14')
        volatility = features.get('volatility_20')
        
        if atr is not None and 'close' in features:
            # ATR as percentage of price
            atr_pct = atr / features['close'] * 100
            
            if atr_pct > 3.0:  # High volatility
                return 0.3  # Reduce strength
            elif atr_pct < 1.0:  # Low volatility
                return 1.0  # Full strength
            else:  # Normal volatility
                return 0.7
        
        if volatility is not None:
            if volatility > 0.03:  # 3% daily vol
                return 0.3
            elif volatility < 0.01:  # 1% daily vol
                return 1.0
            else:
                return 0.7
                
        return 0.5  # Default neutral


class VolumeConfirmationCalculator(StrengthCalculator):
    """
    Calculate strength based on volume confirmation.
    High volume = stronger signal.
    """
    
    def calculate(self, signal: Dict[str, Any], features: Dict[str, Any]) -> float:
        """Calculate strength based on volume."""
        if signal['signal_value'] == 0:
            return 0.0
            
        volume = features.get('volume')
        volume_ma = features.get('volume_ma_20')
        
        if volume is not None and volume_ma is not None and volume_ma > 0:
            volume_ratio = volume / volume_ma
            
            if volume_ratio > 2.0:  # Very high volume
                return 1.0
            elif volume_ratio > 1.5:  # High volume
                return 0.8
            elif volume_ratio > 1.0:  # Above average
                return 0.6
            else:  # Below average
                return 0.4
                
        return 0.5  # Default neutral


class SignalStrengthFilter:
    """
    Main filter that combines multiple strength calculators
    to produce a composite strength score for binary signals.
    """
    
    def __init__(self, calculators: Optional[List[StrengthCalculator]] = None):
        """Initialize with strength calculators."""
        if calculators is None:
            # Default calculator set
            self.calculators = [
                IndicatorExtremityCalculator(),
                TrendAlignmentCalculator(),
                VolatilityAdjustmentCalculator(),
                VolumeConfirmationCalculator()
            ]
        else:
            self.calculators = calculators
            
        self.weights = {
            'IndicatorExtremityCalculator': 0.3,
            'TrendAlignmentCalculator': 0.3,
            'VolatilityAdjustmentCalculator': 0.2,
            'VolumeConfirmationCalculator': 0.2
        }
    
    def calculate_strength(self, signal: Dict[str, Any], features: Dict[str, Any]) -> float:
        """
        Calculate composite strength score for a binary signal.
        
        Args:
            signal: Binary signal dict with signal_value (-1, 0, 1)
            features: Current feature values from FeatureHub
            
        Returns:
            Strength score between 0.0 and 1.0
        """
        if signal.get('signal_value', 0) == 0:
            return 0.0
            
        scores = []
        weights = []
        
        for calculator in self.calculators:
            try:
                score = calculator.calculate(signal, features)
                calculator_name = calculator.__class__.__name__
                weight = self.weights.get(calculator_name, 1.0)
                
                scores.append(score)
                weights.append(weight)
                
                logger.debug(f"{calculator_name}: {score:.3f} (weight: {weight})")
                
            except Exception as e:
                logger.warning(f"Error in {calculator.__class__.__name__}: {e}")
                
        if not scores:
            return 0.5  # Default neutral strength
            
        # Weighted average
        total_weight = sum(weights)
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        composite_score = weighted_sum / total_weight
        
        logger.debug(f"Composite strength for {signal.get('strategy_id')}: {composite_score:.3f}")
        
        return composite_score
    
    def enrich_signal(self, signal: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a binary signal with strength information.
        
        Returns a copy of the signal with added strength data.
        Signal value remains unchanged for storage.
        """
        enriched = signal.copy()
        
        strength = self.calculate_strength(signal, features)
        
        enriched['strength_analysis'] = {
            'composite_strength': strength,
            'strength_factors': self._get_strength_factors(signal, features),
            'recommended_position_pct': self._get_position_recommendation(strength)
        }
        
        return enriched
    
    def _get_strength_factors(self, signal: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, float]:
        """Get individual strength factors for transparency."""
        factors = {}
        
        for calculator in self.calculators:
            try:
                score = calculator.calculate(signal, features)
                factors[calculator.__class__.__name__.replace('Calculator', '')] = score
            except:
                pass
                
        return factors
    
    def _get_position_recommendation(self, strength: float) -> float:
        """
        Convert strength score to position size recommendation.
        
        This is just a suggestion - actual position sizing is done by risk module.
        """
        if strength >= 0.8:
            return 1.0  # Full position
        elif strength >= 0.6:
            return 0.75
        elif strength >= 0.4:
            return 0.5
        elif strength >= 0.2:
            return 0.25
        else:
            return 0.1  # Minimum position


# Usage example in comments:
"""
# In the coordinator or risk module
signal_filter = SignalStrengthFilter()

# When processing a binary signal
signal = {
    'symbol_timeframe': 'AAPL_5m',
    'signal_value': 1,  # Binary long signal
    'strategy_id': 'rule1_ma_crossover',
    'metadata': {'fast_ma': 150.5, 'slow_ma': 149.8}
}

# Calculate strength without modifying signal
strength = signal_filter.calculate_strength(signal, current_features)

# Or get enriched signal (for risk module)
enriched_signal = signal_filter.enrich_signal(signal, current_features)
# enriched_signal now contains strength_analysis but signal_value is unchanged

# Risk module uses both binary signal and strength
position_size = risk_module.calculate_position(
    signal_value=signal['signal_value'],  # Still 1
    signal_strength=strength,              # 0.0 to 1.0
    portfolio_state=portfolio_state
)
"""