"""
Pattern-based regime classifier.

Implements market regime detection using technical patterns
for identifying breakout, range-bound, and trending markets.
"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from enum import Enum
import logging

from ...core.events import Event, EventType, EventBus
from ...data.models import MarketData
from .classifier import RegimeClassifier, RegimeState, RegimeContext


logger = logging.getLogger(__name__)


class PatternRegimeState(Enum):
    """Pattern-specific regime states."""
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    RANGE_BOUND = "range_bound"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    BREAKOUT_UP = "breakout_up"
    BREAKOUT_DOWN = "breakout_down"
    CONSOLIDATION = "consolidation"
    
    def to_regime_state(self) -> RegimeState:
        """Convert to standard regime state."""
        mapping = {
            self.STRONG_UPTREND: RegimeState.TRENDING,
            self.UPTREND: RegimeState.TRENDING,
            self.RANGE_BOUND: RegimeState.RANGE_BOUND,
            self.DOWNTREND: RegimeState.TRENDING,
            self.STRONG_DOWNTREND: RegimeState.TRENDING,
            self.BREAKOUT_UP: RegimeState.VOLATILE,
            self.BREAKOUT_DOWN: RegimeState.VOLATILE,
            self.CONSOLIDATION: RegimeState.RANGE_BOUND
        }
        return mapping[self]


@dataclass
class PatternParameters:
    """Parameters for pattern detection."""
    # Lookback periods
    short_period: int = 10
    medium_period: int = 20
    long_period: int = 50
    
    # Pattern thresholds
    trend_strength_threshold: float = 0.6
    breakout_threshold: float = 2.0  # Standard deviations
    range_threshold: float = 0.3  # Percentage of ATR
    
    # Support/Resistance
    support_resistance_lookback: int = 100
    support_resistance_touches: int = 3
    support_resistance_tolerance: float = 0.02
    
    # Volume confirmation
    volume_spike_threshold: float = 1.5
    require_volume_confirmation: bool = True


@dataclass
class PatternMetrics:
    """Metrics for pattern analysis."""
    trend_direction: float  # -1 to 1
    trend_strength: float  # 0 to 1
    volatility: float
    support_levels: List[float]
    resistance_levels: List[float]
    recent_highs: List[float]
    recent_lows: List[float]
    volume_ratio: float
    price_position: float  # Position within range (0 to 1)


class PatternClassifier(RegimeClassifier):
    """
    Pattern-based classifier for regime detection.
    
    Identifies market regimes based on:
    - Price patterns and trends
    - Support/resistance levels
    - Breakout detection
    - Volatility patterns
    - Volume confirmation
    """
    
    def __init__(
        self,
        parameters: Optional[PatternParameters] = None,
        event_bus: Optional[EventBus] = None
    ):
        """Initialize pattern classifier."""
        super().__init__(name="pattern_classifier", event_bus=event_bus)
        
        self.params = parameters or PatternParameters()
        
        # Pattern detection state
        self._current_pattern: Optional[PatternRegimeState] = None
        self._pattern_confidence: float = 0.0
        
        # Data buffers
        self._price_history: List[Tuple[datetime, float, float, float, float]] = []  # OHLC
        self._volume_history: List[Tuple[datetime, float]] = []
        self._pattern_history: List[Tuple[datetime, PatternRegimeState, float]] = []
        
        # Technical indicators
        self._moving_averages: Dict[int, float] = {}
        self._support_resistance: Dict[str, List[float]] = {'support': [], 'resistance': []}
        
    def classify_regime(
        self,
        market_data: Dict[str, MarketData],
        timestamp: datetime
    ) -> RegimeContext:
        """
        Classify current market regime using pattern analysis.
        
        Args:
            market_data: Current market data
            timestamp: Current timestamp
            
        Returns:
            RegimeContext with classification
        """
        # Update data buffers
        self._update_buffers(market_data, timestamp)
        
        # Check minimum data requirement
        if len(self._price_history) < self.params.long_period:
            return self._create_context(
                RegimeState.UNKNOWN,
                0.0,
                "Insufficient data for pattern analysis"
            )
            
        # Calculate pattern metrics
        metrics = self._calculate_pattern_metrics()
        
        # Identify primary pattern
        pattern, confidence = self._identify_pattern(metrics)
        
        # Update state
        self._current_pattern = pattern
        self._pattern_confidence = confidence
        self._pattern_history.append((timestamp, pattern, confidence))
        
        # Convert to standard regime
        regime = pattern.to_regime_state()
        
        # Create detailed context
        context = self._create_context(
            regime,
            confidence,
            f"Pattern: {pattern.value}",
            additional_data={
                'pattern': pattern.value,
                'pattern_metrics': {
                    'trend_direction': metrics.trend_direction,
                    'trend_strength': metrics.trend_strength,
                    'volatility': metrics.volatility,
                    'support_levels': metrics.support_levels,
                    'resistance_levels': metrics.resistance_levels,
                    'volume_ratio': metrics.volume_ratio,
                    'price_position': metrics.price_position
                },
                'trading_bias': self._get_trading_bias(pattern, metrics)
            }
        )
        
        return context
        
    def _update_buffers(
        self,
        market_data: Dict[str, MarketData],
        timestamp: datetime
    ) -> None:
        """Update internal data buffers."""
        if not market_data:
            return
            
        # Use first symbol as market proxy
        symbol = next(iter(market_data.keys()))
        data = market_data[symbol]
        
        # Add OHLC data
        if hasattr(data, 'open') and hasattr(data, 'high') and hasattr(data, 'low') and hasattr(data, 'close'):
            self._price_history.append((
                timestamp,
                data.open,
                data.high,
                data.low,
                data.close
            ))
        else:
            # Use price as OHLC if not available
            price = data.price
            self._price_history.append((timestamp, price, price, price, price))
            
        # Add volume data
        if hasattr(data, 'volume'):
            self._volume_history.append((timestamp, data.volume))
            
        # Limit buffer sizes
        max_size = max(self.params.support_resistance_lookback, self.params.long_period * 2)
        if len(self._price_history) > max_size:
            self._price_history.pop(0)
        if len(self._volume_history) > max_size:
            self._volume_history.pop(0)
            
    def _calculate_pattern_metrics(self) -> PatternMetrics:
        """Calculate comprehensive pattern metrics."""
        # Extract price data
        closes = [p[4] for p in self._price_history]
        highs = [p[2] for p in self._price_history]
        lows = [p[3] for p in self._price_history]
        
        # Calculate moving averages
        self._moving_averages = {
            self.params.short_period: np.mean(closes[-self.params.short_period:]),
            self.params.medium_period: np.mean(closes[-self.params.medium_period:]),
            self.params.long_period: np.mean(closes[-self.params.long_period:])
        }
        
        # Calculate trend
        trend_direction, trend_strength = self._calculate_trend(closes)
        
        # Calculate volatility
        returns = np.diff(closes[-self.params.medium_period:]) / closes[-self.params.medium_period:-1]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Find support/resistance levels
        self._update_support_resistance(highs, lows)
        
        # Calculate volume ratio
        volume_ratio = 1.0
        if len(self._volume_history) >= self.params.medium_period:
            recent_volume = self._volume_history[-1][1]
            avg_volume = np.mean([v[1] for v in self._volume_history[-self.params.medium_period:]])
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
        # Calculate price position in range
        current_price = closes[-1]
        recent_high = max(highs[-self.params.medium_period:])
        recent_low = min(lows[-self.params.medium_period:])
        price_range = recent_high - recent_low
        price_position = (current_price - recent_low) / price_range if price_range > 0 else 0.5
        
        return PatternMetrics(
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            volatility=volatility,
            support_levels=self._support_resistance['support'],
            resistance_levels=self._support_resistance['resistance'],
            recent_highs=highs[-self.params.short_period:],
            recent_lows=lows[-self.params.short_period:],
            volume_ratio=volume_ratio,
            price_position=price_position
        )
        
    def _calculate_trend(self, prices: List[float]) -> Tuple[float, float]:
        """Calculate trend direction and strength."""
        if len(prices) < 2:
            return 0.0, 0.0
            
        # Simple linear regression for trend
        x = np.arange(len(prices))
        y = np.array(prices)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize slope to price level
        avg_price = np.mean(prices)
        normalized_slope = slope / avg_price if avg_price > 0 else 0
        
        # Calculate R-squared for trend strength
        y_pred = np.polyval(np.polyfit(x, y, 1), x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Direction: -1 to 1
        direction = np.clip(normalized_slope * 100, -1, 1)
        
        # Strength: 0 to 1
        strength = abs(r_squared)
        
        return direction, strength
        
    def _update_support_resistance(
        self,
        highs: List[float],
        lows: List[float]
    ) -> None:
        """Update support and resistance levels."""
        if len(highs) < self.params.support_resistance_lookback:
            return
            
        # Find local extrema
        window = 5
        support_candidates = []
        resistance_candidates = []
        
        for i in range(window, len(lows) - window):
            # Local minimum (support)
            if all(lows[i] <= lows[j] for j in range(i - window, i + window + 1)):
                support_candidates.append(lows[i])
                
            # Local maximum (resistance)
            if all(highs[i] >= highs[j] for j in range(i - window, i + window + 1)):
                resistance_candidates.append(highs[i])
                
        # Cluster nearby levels
        self._support_resistance['support'] = self._cluster_levels(support_candidates)
        self._support_resistance['resistance'] = self._cluster_levels(resistance_candidates)
        
    def _cluster_levels(self, levels: List[float]) -> List[float]:
        """Cluster nearby price levels."""
        if not levels:
            return []
            
        # Sort levels
        sorted_levels = sorted(levels)
        
        # Cluster nearby levels
        clusters = []
        current_cluster = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            if (level - current_cluster[-1]) / current_cluster[-1] < self.params.support_resistance_tolerance:
                current_cluster.append(level)
            else:
                # New cluster
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
                
        # Add last cluster
        clusters.append(np.mean(current_cluster))
        
        # Keep only significant levels (with multiple touches)
        significant_clusters = []
        for cluster_level in clusters:
            touches = sum(
                1 for level in levels
                if abs(level - cluster_level) / cluster_level < self.params.support_resistance_tolerance
            )
            if touches >= self.params.support_resistance_touches:
                significant_clusters.append(cluster_level)
                
        return significant_clusters[:5]  # Keep top 5 levels
        
    def _identify_pattern(self, metrics: PatternMetrics) -> Tuple[PatternRegimeState, float]:
        """Identify the primary pattern from metrics."""
        patterns_scores = {}
        
        # Check for breakout patterns
        breakout_score = self._check_breakout_pattern(metrics)
        if breakout_score[0] is not None:
            patterns_scores[breakout_score[0]] = breakout_score[1]
            
        # Check for trending patterns
        trend_score = self._check_trend_pattern(metrics)
        if trend_score[0] is not None:
            patterns_scores[trend_score[0]] = trend_score[1]
            
        # Check for range-bound patterns
        range_score = self._check_range_pattern(metrics)
        if range_score[0] is not None:
            patterns_scores[range_score[0]] = range_score[1]
            
        # Select pattern with highest confidence
        if patterns_scores:
            best_pattern = max(patterns_scores.items(), key=lambda x: x[1])
            return best_pattern[0], best_pattern[1]
        else:
            return PatternRegimeState.CONSOLIDATION, 0.5
            
    def _check_breakout_pattern(
        self,
        metrics: PatternMetrics
    ) -> Tuple[Optional[PatternRegimeState], float]:
        """Check for breakout patterns."""
        if not metrics.resistance_levels or not metrics.support_levels:
            return None, 0.0
            
        current_price = metrics.recent_highs[-1]  # Use recent high for breakout
        
        # Check resistance breakout
        for resistance in metrics.resistance_levels:
            if current_price > resistance * (1 + self.params.support_resistance_tolerance):
                # Confirm with volume
                if metrics.volume_ratio > self.params.volume_spike_threshold:
                    return PatternRegimeState.BREAKOUT_UP, 0.8
                elif not self.params.require_volume_confirmation:
                    return PatternRegimeState.BREAKOUT_UP, 0.6
                    
        # Check support breakdown
        current_low = metrics.recent_lows[-1]
        for support in metrics.support_levels:
            if current_low < support * (1 - self.params.support_resistance_tolerance):
                # Confirm with volume
                if metrics.volume_ratio > self.params.volume_spike_threshold:
                    return PatternRegimeState.BREAKOUT_DOWN, 0.8
                elif not self.params.require_volume_confirmation:
                    return PatternRegimeState.BREAKOUT_DOWN, 0.6
                    
        return None, 0.0
        
    def _check_trend_pattern(
        self,
        metrics: PatternMetrics
    ) -> Tuple[Optional[PatternRegimeState], float]:
        """Check for trending patterns."""
        if metrics.trend_strength < self.params.trend_strength_threshold:
            return None, 0.0
            
        confidence = metrics.trend_strength
        
        # Strong trends
        if abs(metrics.trend_direction) > 0.7:
            if metrics.trend_direction > 0:
                return PatternRegimeState.STRONG_UPTREND, confidence
            else:
                return PatternRegimeState.STRONG_DOWNTREND, confidence
                
        # Normal trends
        elif abs(metrics.trend_direction) > 0.3:
            if metrics.trend_direction > 0:
                return PatternRegimeState.UPTREND, confidence * 0.8
            else:
                return PatternRegimeState.DOWNTREND, confidence * 0.8
                
        return None, 0.0
        
    def _check_range_pattern(
        self,
        metrics: PatternMetrics
    ) -> Tuple[Optional[PatternRegimeState], float]:
        """Check for range-bound patterns."""
        # Low trend strength indicates ranging
        if metrics.trend_strength > self.params.trend_strength_threshold:
            return None, 0.0
            
        # Check if price is oscillating between levels
        if metrics.support_levels and metrics.resistance_levels:
            # Price should be between support and resistance
            current_price = metrics.recent_highs[-1]
            nearest_support = max([s for s in metrics.support_levels if s < current_price], default=0)
            nearest_resistance = min([r for r in metrics.resistance_levels if r > current_price], default=float('inf'))
            
            if nearest_support > 0 and nearest_resistance < float('inf'):
                range_size = (nearest_resistance - nearest_support) / current_price
                if range_size < self.params.range_threshold:
                    return PatternRegimeState.RANGE_BOUND, 0.7
                    
        # Low volatility consolidation
        if metrics.volatility < 0.1:  # Less than 10% annualized volatility
            return PatternRegimeState.CONSOLIDATION, 0.6
            
        return PatternRegimeState.RANGE_BOUND, 0.5
        
    def _get_trading_bias(
        self,
        pattern: PatternRegimeState,
        metrics: PatternMetrics
    ) -> str:
        """Get trading bias based on pattern."""
        bias_map = {
            PatternRegimeState.STRONG_UPTREND: "strong_long",
            PatternRegimeState.UPTREND: "long",
            PatternRegimeState.RANGE_BOUND: "neutral",
            PatternRegimeState.DOWNTREND: "short",
            PatternRegimeState.STRONG_DOWNTREND: "strong_short",
            PatternRegimeState.BREAKOUT_UP: "long",
            PatternRegimeState.BREAKOUT_DOWN: "short",
            PatternRegimeState.CONSOLIDATION: "neutral"
        }
        
        return bias_map.get(pattern, "neutral")
        
    def process_indicator_event(self, event: Event) -> None:
        """Process indicator events for pattern enhancement."""
        if event.event_type == EventType.INDICATOR and isinstance(event.payload, dict):
            indicator_type = event.payload.get('indicator_type')
            
            # Could enhance pattern detection with additional indicators
            # For example: RSI for overbought/oversold
            # Bollinger Bands for volatility expansion
            # MACD for trend confirmation
            pass
            
    def get_pattern_history(self) -> List[Tuple[datetime, PatternRegimeState, float]]:
        """Get history of pattern classifications."""
        return self._pattern_history.copy()
        
    def get_current_levels(self) -> Dict[str, List[float]]:
        """Get current support and resistance levels."""
        return self._support_resistance.copy()