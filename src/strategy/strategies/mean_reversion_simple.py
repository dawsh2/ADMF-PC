"""
Stateless mean reversion trading strategy using Bollinger Bands.

This module provides both stateful (container-based) and stateless implementations
of the mean reversion strategy to support the unified architecture transition.
"""

from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import logging

from ...risk.protocols import Signal as RiskSignal
from ...execution.protocols import OrderSide
from ...core.components.protocols import StatelessStrategy
from ...core.components.discovery import strategy

logger = logging.getLogger(__name__)


class StatelessMeanReversionStrategy:
    """
    Stateless mean reversion strategy for unified architecture.
    
    Implements the StatelessStrategy protocol for use as a lightweight
    service in the event-driven architecture. All state is passed as
    parameters - no internal state is maintained.
    """
    
    def __init__(self):
        """Initialize stateless mean reversion strategy."""
        # No configuration stored - everything comes from params
        pass
    
    def generate_signal(
        self,
        features: Dict[str, Any],
        bar: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a trading signal from features and current bar.
        
        Pure function implementation - no side effects or state changes.
        
        Args:
            features: Calculated indicators from FeatureHub
                - bollinger_upper: Upper Bollinger Band
                - bollinger_middle: Middle Bollinger Band (SMA)
                - bollinger_lower: Lower Bollinger Band
                - rsi: Relative Strength Index
            bar: Current market bar with OHLCV data
            params: Strategy parameters
                - entry_threshold: Std devs for entry (default: 2.0)
                - exit_threshold: Std devs for exit (default: 0.5)
                
        Returns:
            Signal dict with direction, strength, and metadata
        """
        # Extract parameters with defaults
        entry_threshold = params.get('entry_threshold', 2.0)
        exit_threshold = params.get('exit_threshold', 0.5)
        
        # Get current price
        price = bar.get('close', bar.get('price', 0))
        
        # Get required features
        upper_band = features.get('bollinger_upper')
        middle_band = features.get('bollinger_middle')
        lower_band = features.get('bollinger_lower')
        rsi = features.get('rsi')
        
        # Return empty signal if features missing
        if any(x is None for x in [upper_band, middle_band, lower_band]) or price <= 0:
            return {
                'direction': 'flat',
                'strength': 0.0,
                'metadata': {'reason': 'Missing required features or price'}
            }
        
        # Calculate z-score (distance from mean in standard deviations)
        band_width = upper_band - middle_band
        if band_width > 0:
            z_score = (price - middle_band) / (band_width / 2)
        else:
            z_score = 0
        
        # Generate signal based on mean reversion logic
        if z_score < -entry_threshold:
            # Price below lower band - oversold, expect reversion up
            return {
                'direction': 'long',
                'strength': min(abs(z_score) / entry_threshold, 1.0),
                'metadata': {
                    'z_score': z_score,
                    'rsi': rsi,
                    'upper_band': upper_band,
                    'middle_band': middle_band,
                    'lower_band': lower_band,
                    'price': price,
                    'reason': 'Price below lower band - oversold'
                }
            }
        elif z_score > entry_threshold:
            # Price above upper band - overbought, expect reversion down
            return {
                'direction': 'short',
                'strength': min(z_score / entry_threshold, 1.0),
                'metadata': {
                    'z_score': z_score,
                    'rsi': rsi,
                    'upper_band': upper_band,
                    'middle_band': middle_band,
                    'lower_band': lower_band,
                    'price': price,
                    'reason': 'Price above upper band - overbought'
                }
            }
        elif abs(z_score) < exit_threshold:
            # Near the mean - potential exit signal (flat)
            return {
                'direction': 'flat',
                'strength': 0.0,
                'metadata': {
                    'z_score': z_score,
                    'reason': 'Price near mean - no clear signal'
                }
            }
        else:
            # No signal
            return {
                'direction': 'flat',
                'strength': 0.0,
                'metadata': {
                    'z_score': z_score,
                    'reason': 'No mean reversion signal'
                }
            }
    
    @property
    def required_features(self) -> List[str]:
        """List of feature names this strategy requires."""
        return ['bollinger_upper', 'bollinger_middle', 'bollinger_lower', 'rsi']


# Stateless factory function
def create_stateless_mean_reversion() -> StatelessMeanReversionStrategy:
    """Create a stateless mean reversion strategy instance."""
    return StatelessMeanReversionStrategy()


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


# Pure function version for EVENT_FLOW_ARCHITECTURE
@strategy(
    name='mean_reversion',
    feature_config={
        'bollinger': {'params': ['period', 'num_std'], 'defaults': {'period': 20, 'num_std': 2}},
        'rsi': {'params': ['rsi_period'], 'default': 14}
    },
    validate_features=False  # Disable validation since features are dynamically named
)
def mean_reversion_strategy(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Pure function mean reversion strategy using Bollinger Bands.
    
    Args:
        features: Calculated indicators (bollinger_upper, bollinger_middle, bollinger_lower, rsi)
        bar: Current market bar with OHLCV data
        params: Strategy parameters (entry_threshold, exit_threshold)
        
    Returns:
        Signal dict or None
    """
    # Extract parameters
    entry_threshold = params.get('entry_threshold', 2.0)
    exit_threshold = params.get('exit_threshold', 0.5)
    
    # Get current price
    price = bar.get('close', 0)
    
    # Get required features - look for bollinger bands with any period
    upper_band = None
    middle_band = None
    lower_band = None
    
    # Find bollinger bands (could be bollinger_20_upper, bollinger_2_upper, etc.)
    for key in features.keys():
        if key.endswith('_upper') and 'bollinger' in key:
            upper_band = features[key]
        elif key.endswith('_middle') and 'bollinger' in key:
            middle_band = features[key]
        elif key.endswith('_lower') and 'bollinger' in key:
            lower_band = features[key]
    
    # Also try generic RSI
    rsi = features.get('rsi') or features.get('rsi_14')
    
    # Debug log available features (commented out for performance)
    # logger.info(f"Mean reversion features available: {list(features.keys())}")
    # logger.info(f"Looking for bollinger features, found: upper={upper_band}, middle={middle_band}, lower={lower_band}")
    
    # Check if we have required features
    if any(x is None for x in [upper_band, middle_band, lower_band]) or price <= 0:
        logger.warning(f"Missing features for mean reversion: upper={upper_band}, middle={middle_band}, lower={lower_band}, price={price}")
        return None
    
    # Calculate z-score
    band_width = upper_band - middle_band
    if band_width > 0:
        z_score = (price - middle_band) / (band_width / 2)
    else:
        z_score = 0
    
    # Generate signal
    signal = None
    
    # logger.info(f"Mean reversion z-score: {z_score:.3f}, entry_threshold: {entry_threshold}, exit_threshold: {exit_threshold}")
    
    if z_score < -entry_threshold:
        # Oversold - expect reversion up
        signal = {
            'symbol': bar.get('symbol'),
            'direction': 'long',
            'signal_type': 'entry',
            'strength': min(abs(z_score) / entry_threshold, 1.0),
            'price': price,
            'reason': f'Mean reversion long: z-score={z_score:.2f} < -{entry_threshold}',
            'indicators': {
                'price': price,
                'upper_band': upper_band,
                'middle_band': middle_band,
                'lower_band': lower_band,
                'z_score': z_score,
                'rsi': rsi
            }
        }
        logger.info(f"Generated LONG signal: price={price}, z_score={z_score:.2f}")
        
    elif z_score > entry_threshold:
        # Overbought - expect reversion down
        signal = {
            'symbol': bar.get('symbol'),
            'direction': 'short',
            'signal_type': 'entry',
            'strength': min(z_score / entry_threshold, 1.0),
            'price': price,
            'reason': f'Mean reversion short: z-score={z_score:.2f} > {entry_threshold}',
            'indicators': {
                'price': price,
                'upper_band': upper_band,
                'middle_band': middle_band,
                'lower_band': lower_band,
                'z_score': z_score,
                'rsi': rsi
            }
        }
        logger.info(f"Generated SHORT signal: price={price}, z_score={z_score:.2f}")
    
    # If no signal generated, return flat signal for tracking
    if signal is None:
        # logger.info(f"Mean reversion flat: z-score={z_score:.3f} within thresholds")
        signal = {
            'symbol': bar.get('symbol'),
            'direction': 'flat',
            'signal_type': 'entry',  # Required for signal processing
            'strength': 0.0,
            'price': price,
            'reason': f'Mean reversion flat: z-score={z_score:.2f} within thresholds',
            'indicators': {
                'upper_band': upper_band,
                'middle_band': middle_band,
                'lower_band': lower_band,
                'z_score': z_score,
                'rsi': rsi
            }
        }
    
    return signal