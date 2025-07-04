"""
Self-contained Bollinger RSI Divergence - computes its own indicators

This avoids dependency issues by computing BB and RSI internally.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class BollingerRSIDivergenceSelfContained:
    """
    Self-contained implementation that computes BB and RSI internally.
    """
    
    def __init__(self, config: Dict[str, Any] = None, name: str = 'bb_rsi_divergence_self', **kwargs):
        if config is None:
            config = kwargs
            
        self._name = name
        logger.info(f"BollingerRSIDivergenceSelfContained initialized")
        
        # Parameters
        self.bb_period = 20
        self.bb_std = 2.0
        self.rsi_period = 14
        self.lookback_bars = 20
        self.rsi_divergence_threshold = 5.0
        self.confirmation_bars = 10
        self.max_holding_bars = 50
        
        # Indicator calculation state
        self.price_history = deque(maxlen=max(self.bb_period, self.rsi_period) + 1)
        
        # Pattern tracking state
        self.stage = 'scanning'
        self.position_type = 0
        self.extremes_below_band = {}
        self.extremes_above_band = {}
        self.divergence_info = None
        self.divergence_found_idx = None
        self.entry_idx = None
        self.entry_price = None
        self.target_price = None
        
        # Debug
        self.bar_count = 0
        self.signals_generated = 0
        
    def update(self, price: float = 0, high: float = None, low: float = None, 
               volume: float = None, **kwargs) -> Dict[str, Any]:
        """Update with new price data"""
        self.bar_count += 1
        
        # Add to price history
        self.price_history.append(price)
        
        # Need enough data for indicators
        if len(self.price_history) < self.bb_period:
            return self._create_signal(0, f'Warming up: {len(self.price_history)}/{self.bb_period} bars')
        
        # Calculate indicators
        prices = np.array(self.price_history)
        
        # Bollinger Bands
        bb_middle = np.mean(prices[-self.bb_period:])
        bb_std_val = np.std(prices[-self.bb_period:])
        bb_upper = bb_middle + self.bb_std * bb_std_val
        bb_lower = bb_middle - self.bb_std * bb_std_val
        
        # RSI
        rsi = self._calculate_rsi()
        
        # Current values
        close = price
        high_val = high or price
        low_val = low or price
        idx = self.bar_count
        
        # Log every 1000 bars
        if self.bar_count % 1000 == 0:
            logger.info(f"Processed {self.bar_count} bars, signals: {self.signals_generated}, "
                       f"stage: {self.stage}, extremes: {len(self.extremes_below_band) + len(self.extremes_above_band)}")
        
        # Clean old extremes
        self.extremes_below_band = {k: v for k, v in self.extremes_below_band.items() 
                                   if idx - k <= self.lookback_bars}
        self.extremes_above_band = {k: v for k, v in self.extremes_above_band.items() 
                                   if idx - k <= self.lookback_bars}
        
        # Check for exit first
        if self.stage == 'in_position' and self.position_type != 0:
            bars_held = idx - self.entry_idx
            
            exit_signal = False
            exit_reason = None
            
            if self.position_type == 1:  # Long
                if close >= self.target_price:
                    exit_signal = True
                    exit_reason = f"Target hit: {close:.2f} >= {self.target_price:.2f}"
                elif bars_held >= self.max_holding_bars:
                    exit_signal = True
                    exit_reason = f"Max bars: {bars_held}"
            else:  # Short
                if close <= self.target_price:
                    exit_signal = True
                    exit_reason = f"Target hit: {close:.2f} <= {self.target_price:.2f}"
                elif bars_held >= self.max_holding_bars:
                    exit_signal = True
                    exit_reason = f"Max bars: {bars_held}"
            
            if exit_signal:
                self.stage = 'scanning'
                self.position_type = 0
                self.entry_idx = None
                logger.info(f"Exit at bar {idx}: {exit_reason}")
                return self._create_signal(0, exit_reason)
            else:
                return self._create_signal(self.position_type, f"Holding {bars_held} bars")
        
        # Track extremes
        if close < bb_lower:
            self.extremes_below_band[idx] = (low_val, rsi)
            if self.bar_count % 100 == 0:
                logger.debug(f"Bar {idx}: Below band, low={low_val:.2f}, rsi={rsi:.1f}")
                
        if close > bb_upper:
            self.extremes_above_band[idx] = (high_val, rsi)
        
        # Look for divergence in scanning stage
        if self.stage == 'scanning':
            # Bullish divergence
            if close < bb_lower:
                for prev_idx in range(max(idx - self.lookback_bars, 1), idx):
                    if prev_idx in self.extremes_below_band:
                        prev_low, prev_rsi = self.extremes_below_band[prev_idx]
                        
                        if low_val < prev_low and rsi > prev_rsi + self.rsi_divergence_threshold:
                            self.stage = 'waiting_confirmation'
                            self.divergence_info = {
                                'type': 'bullish',
                                'current_idx': idx,
                                'prev_idx': prev_idx,
                                'rsi_diff': rsi - prev_rsi
                            }
                            self.divergence_found_idx = idx
                            logger.info(f"Bullish divergence at bar {idx}")
                            break
            
            # Bearish divergence
            elif close > bb_upper:
                for prev_idx in range(max(idx - self.lookback_bars, 1), idx):
                    if prev_idx in self.extremes_above_band:
                        prev_high, prev_rsi = self.extremes_above_band[prev_idx]
                        
                        if high_val > prev_high and rsi < prev_rsi - self.rsi_divergence_threshold:
                            self.stage = 'waiting_confirmation'
                            self.divergence_info = {
                                'type': 'bearish',
                                'current_idx': idx,
                                'prev_idx': prev_idx,
                                'rsi_diff': prev_rsi - rsi
                            }
                            self.divergence_found_idx = idx
                            logger.info(f"Bearish divergence at bar {idx}")
                            break
        
        # Check for confirmation
        elif self.stage == 'waiting_confirmation':
            bars_since = idx - self.divergence_found_idx
            
            if bars_since > self.confirmation_bars:
                self.stage = 'scanning'
                self.divergence_info = None
                return self._create_signal(0, 'Confirmation expired')
            
            confirmed = False
            signal_value = 0
            
            if self.divergence_info['type'] == 'bullish' and close > bb_lower:
                confirmed = True
                self.position_type = 1
                signal_value = 1
            elif self.divergence_info['type'] == 'bearish' and close < bb_upper:
                confirmed = True
                self.position_type = -1
                signal_value = -1
            
            if confirmed:
                self.stage = 'in_position'
                self.entry_idx = idx
                self.entry_price = close
                self.target_price = bb_middle
                self.signals_generated += 1
                logger.info(f"Entry confirmed at bar {idx}, target={bb_middle:.2f}")
                return self._create_signal(signal_value, f"{self.divergence_info['type']} confirmed")
        
        return self._create_signal(0, f"Stage: {self.stage}")
    
    def _calculate_rsi(self) -> float:
        """Calculate RSI from price history"""
        if len(self.price_history) < self.rsi_period + 1:
            return 50.0
            
        prices = list(self.price_history)
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        gains = [d if d > 0 else 0 for d in deltas[-self.rsi_period:]]
        losses = [-d if d < 0 else 0 for d in deltas[-self.rsi_period:]]
        
        avg_gain = sum(gains) / self.rsi_period
        avg_loss = sum(losses) / self.rsi_period
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _create_signal(self, value: int, reason: str) -> Dict[str, Any]:
        """Create output signal"""
        return {
            'value': value,
            'signal': value,
            'reason': reason,
            'stage': self.stage,
            'position_type': self.position_type,
            'signals_generated': self.signals_generated,
            'bar_count': self.bar_count
        }
    
    @property
    def name(self) -> str:
        return self._name
    
    @property  
    def dependencies(self) -> List[str]:
        return []  # No dependencies - self contained