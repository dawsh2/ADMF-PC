"""
Simple mean reversion trading strategy using available indicators.
"""

from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import numpy as np
import logging

from ...risk.protocols import Signal as RiskSignal
from ...execution.protocols import OrderSide

logger = logging.getLogger(__name__)


class MeanReversionStrategy:
    """
    Simple mean reversion strategy using SMA and BB indicators.
    
    This version uses the indicators available from IndicatorHub.
    """
    
    def __init__(self, lookback_period: int = 20, entry_threshold: float = 2.0, 
                 exit_threshold: float = 0.5):
        """
        Initialize mean reversion strategy.
        
        Args:
            lookback_period: Period for moving average
            entry_threshold: Standard deviations for entry
            exit_threshold: Standard deviations for exit
        """
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self._position = None  # Track position
        logger.info(f"MeanReversionStrategy initialized with period={lookback_period}, "
                   f"entry={entry_threshold}, exit={exit_threshold}")
    
    def get_required_indicators(self) -> Set[str]:
        """Get indicators required by this strategy."""
        return {"BB_20", "RSI"}  # BB_20 provides Bollinger Bands
    
    def generate_signals(self, data: Dict[str, Any]) -> List[RiskSignal]:
        """
        Generate trading signals based on mean reversion.
        
        Args:
            data: Dictionary containing market_data, indicators, timestamp
            
        Returns:
            List of trading signals
        """
        logger.info("🎯 MeanReversionStrategy.generate_signals() called!")
        
        signals = []
        market_data = data.get('market_data', {})
        indicators = data.get('indicators', {})
        timestamp = data.get('timestamp', datetime.now())
        
        logger.info(f"   Market data: {list(market_data.keys())} symbols")
        logger.info(f"   Indicators: {indicators}")
        logger.info(f"   Timestamp: {timestamp}")
        
        for symbol, symbol_data in market_data.items():
            # Get current price
            if isinstance(symbol_data, dict):
                price = symbol_data.get('close', symbol_data.get('price', 0))
            else:
                price = float(symbol_data) if symbol_data else 0
                
            if not price:
                continue
                
            logger.info(f"   📊 Processing symbol: {symbol}, data: {symbol_data}")
            logger.info(f"   💰 Extracted price: {price}")
            
            # Get indicators
            symbol_indicators = indicators.get(symbol, {})
            # Use BB indicator with the lookback period from parameters
            bb_key = f'BB_{self.lookback_period}'
            bb_data = symbol_indicators.get(bb_key, {})
            rsi = symbol_indicators.get('RSI')
            
            # Extract Bollinger Bands values
            if isinstance(bb_data, dict):
                upper_band = bb_data.get('upper')
                middle_band = bb_data.get('middle')
                lower_band = bb_data.get('lower')
            else:
                # Skip if no BB data
                continue
            
            if not all([upper_band, middle_band, lower_band]):
                logger.info(f"   ⚠️ Missing Bollinger Bands data for {symbol}")
                continue
            
            logger.info(f"   📊 BB: upper={upper_band:.2f}, middle={middle_band:.2f}, lower={lower_band:.2f}")
            logger.info(f"   📊 RSI: {rsi}")
            
            # Calculate how many standard deviations away from middle
            band_width = upper_band - middle_band
            if band_width > 0:
                z_score = (price - middle_band) / (band_width / 2)
            else:
                z_score = 0
            
            logger.info(f"   📊 Z-score: {z_score:.2f}")
            
            # Entry signals
            if z_score < -self.entry_threshold:
                # Price is below lower band - potential buy
                signal = RiskSignal(
                    signal_id=f"mr_{symbol}_{timestamp}",
                    strategy_id="mean_reversion_strategy",
                    symbol=symbol,
                    signal_type="entry",
                    side=OrderSide.BUY,
                    strength=1.0,
                    timestamp=timestamp,
                    metadata={
                        'z_score': z_score,
                        'rsi': rsi,
                        'reason': 'Price below lower band - oversold'
                    }
                )
                signals.append(signal)
                self._position = 'long'
                logger.info(f"   🟢 BUY signal: z_score={z_score:.2f}, RSI={rsi}")
                
            elif z_score > self.entry_threshold:
                # Price is above upper band - potential sell
                signal = RiskSignal(
                    signal_id=f"mr_{symbol}_{timestamp}",
                    strategy_id="mean_reversion_strategy", 
                    symbol=symbol,
                    signal_type="entry",
                    side=OrderSide.SELL,
                    strength=1.0,
                    timestamp=timestamp,
                    metadata={
                        'z_score': z_score,
                        'rsi': rsi,
                        'reason': 'Price above upper band - overbought'
                    }
                )
                signals.append(signal)
                self._position = 'short'
                logger.info(f"   🔴 SELL signal: z_score={z_score:.2f}, RSI={rsi}")
                
            # Exit signals (when price returns to mean)
            elif self._position and abs(z_score) < self.exit_threshold:
                # Price has returned to mean - exit position
                exit_side = OrderSide.SELL if self._position == 'long' else OrderSide.BUY
                signal = RiskSignal(
                    signal_id=f"mr_exit_{symbol}_{timestamp}",
                    strategy_id="mean_reversion_strategy",
                    symbol=symbol,
                    signal_type="exit",
                    side=exit_side,
                    strength=0.8,
                    timestamp=timestamp,
                    metadata={
                        'z_score': z_score,
                        'reason': 'Price returned to mean'
                    }
                )
                signals.append(signal)
                self._position = None
                logger.info(f"   ⚪ EXIT signal: z_score={z_score:.2f}")
        
        logger.info(f"Generated {len(signals)} mean reversion signals")
        return signals