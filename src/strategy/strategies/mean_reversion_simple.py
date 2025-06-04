"""
Stateless mean reversion trading strategy using Bollinger Bands.

This strategy consumes features from FeatureHub and makes pure decisions
based on current feature values. No state is maintained.
"""

from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import logging

from ...risk.protocols import Signal as RiskSignal
from ...execution.protocols import OrderSide

logger = logging.getLogger(__name__)


class MeanReversionStrategy:
    """
    Stateless mean reversion strategy using Bollinger Bands.
    
    This strategy consumes features from FeatureHub and makes pure
    decisions based on current feature values. No state is maintained.
    """
    
    def __init__(self, entry_threshold: float = 2.0, exit_threshold: float = 0.5):
        """
        Initialize mean reversion strategy.
        
        Args:
            entry_threshold: Standard deviations for entry signals
            exit_threshold: Standard deviations for exit signals
        """
        # Configuration only - no state!
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        
        logger.info(f"Stateless MeanReversionStrategy initialized with "
                   f"entry={entry_threshold}, exit={exit_threshold}")
    
    @property
    def name(self) -> str:
        """Strategy name for identification."""
        return "mean_reversion_strategy"
    
    def get_required_features(self) -> Set[str]:
        """Get features required by this strategy."""
        return {"bollinger", "rsi"}  # Bollinger Bands and RSI
    
    def generate_signals(self, strategy_input: Dict[str, Any]) -> List[RiskSignal]:
        """
        Generate trading signals based on mean reversion.
        
        This is a STATELESS function that makes decisions based purely
        on current feature values from FeatureHub.
        """
        logger.info("Stateless MeanReversionStrategy.generate_signals() called")
        
        signals = []
        market_data = strategy_input.get('market_data', {})
        features = strategy_input.get('features', {})
        timestamp = strategy_input.get('timestamp', datetime.now())
        
        # Process each symbol - pure stateless decision logic
        for symbol, symbol_data in market_data.items():
            # Get current price
            if isinstance(symbol_data, dict):
                price = symbol_data.get('close', symbol_data.get('price', 0))
            else:
                price = float(symbol_data) if symbol_data else 0
                
            if not price:
                continue
            
            # Get features for this symbol from FeatureHub
            symbol_features = features.get(symbol, {})
            
            # Extract Bollinger Bands features
            upper_band = symbol_features.get('bollinger_upper')
            middle_band = symbol_features.get('bollinger_middle')
            lower_band = symbol_features.get('bollinger_lower')
            rsi = symbol_features.get('rsi')
            
            # Skip if required features not available
            if any(x is None for x in [upper_band, middle_band, lower_band, rsi]):
                logger.debug("Missing features for %s", symbol)
                continue
            
            # Pure stateless calculation - no position tracking!
            band_width = upper_band - middle_band
            if band_width > 0:
                z_score = (price - middle_band) / (band_width / 2)
            else:
                z_score = 0
            
            logger.debug(
                "Signal analysis for %s: z_score=%.2f, rsi=%.2f", 
                symbol, z_score, rsi
            )
            
            # Stateless signal generation logic - no position state!
            signal = None
            
            if z_score < -self.entry_threshold:
                # Price is below lower band - oversold
                signal = RiskSignal(
                    signal_id=f"mr_{symbol}_{timestamp}",
                    strategy_id="mean_reversion_strategy",
                    symbol=symbol,
                    signal_type="entry",
                    side=OrderSide.BUY,
                    strength=min(abs(z_score) / self.entry_threshold, 1.0),
                    timestamp=timestamp,
                    metadata={
                        'z_score': z_score,
                        'rsi': rsi,
                        'upper_band': upper_band,
                        'middle_band': middle_band,
                        'lower_band': lower_band,
                        'reason': 'Price below lower band - oversold'
                    }
                )
                
            elif z_score > self.entry_threshold:
                # Price is above upper band - overbought
                signal = RiskSignal(
                    signal_id=f"mr_{symbol}_{timestamp}",
                    strategy_id="mean_reversion_strategy", 
                    symbol=symbol,
                    signal_type="entry",
                    side=OrderSide.SELL,
                    strength=min(z_score / self.entry_threshold, 1.0),
                    timestamp=timestamp,
                    metadata={
                        'z_score': z_score,
                        'rsi': rsi,
                        'upper_band': upper_band,
                        'middle_band': middle_band,
                        'lower_band': lower_band,
                        'reason': 'Price above upper band - overbought'
                    }
                )
            
            if signal:
                signals.append(signal)
                logger.info("Generated %s signal for %s (z_score=%.2f)", 
                           signal.side.name, symbol, z_score)
        
        logger.info("Generated %d mean reversion signals", len(signals))
        return signals
    
    def reset(self) -> None:
        """
        Reset strategy state.
        
        Since this strategy is stateless, reset does nothing.
        All state is managed by FeatureHub.
        """
        pass  # No state to reset!


# Factory function for creating strategy
def create_mean_reversion_strategy(config: Dict[str, Any] = None) -> MeanReversionStrategy:
    """
    Factory function to create mean reversion strategy.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MeanReversionStrategy instance
    """
    default_config = {
        'entry_threshold': 2.0,
        'exit_threshold': 0.5
    }
    
    if config:
        default_config.update(config)
    
    return MeanReversionStrategy(**default_config)