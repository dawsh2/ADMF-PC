"""
Centralized Indicator Hub for shared computation.

Implements the shared indicator architecture from BACKTEST.MD where
indicators are computed once and shared across all strategies.
"""
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd
from enum import Enum
import logging

from ...core.events import Event, EventType, EventBus
from ...data.models import MarketData


logger = logging.getLogger(__name__)


class IndicatorType(Enum):
    """Types of technical indicators."""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    SUPPORT_RESISTANCE = "support_resistance"
    CUSTOM = "custom"


@dataclass
class IndicatorConfig:
    """Configuration for an indicator."""
    name: str
    indicator_type: IndicatorType
    parameters: Dict[str, Any] = field(default_factory=dict)
    symbols: Optional[List[str]] = None  # None means all symbols
    enabled: bool = True


@dataclass
class IndicatorValue:
    """Value of an indicator at a point in time."""
    name: str
    symbol: str
    timestamp: datetime
    value: Union[float, Dict[str, float]]
    metadata: Dict[str, Any] = field(default_factory=dict)


class IndicatorHub:
    """
    Centralized hub for indicator computation and caching.
    
    Features:
    - Computes indicators once from streamed data
    - Caches results for efficiency
    - Emits indicator events to downstream consumers
    - Supports multiple symbols
    - Provides both push (events) and pull (get_value) interfaces
    """
    
    def __init__(
        self,
        indicators: List[IndicatorConfig],
        cache_size: int = 1000,
        event_bus: Optional[EventBus] = None
    ):
        """
        Initialize the indicator hub.
        
        Args:
            indicators: List of indicator configurations
            cache_size: Maximum cache size per indicator per symbol
            event_bus: Event bus for publishing indicator updates
        """
        self.indicators = {ind.name: ind for ind in indicators if ind.enabled}
        self.cache_size = cache_size
        self.event_bus = event_bus
        
        # Data storage
        self._price_history: Dict[str, List[Tuple[datetime, float, float, float, float, float]]] = {}  # symbol -> [(timestamp, O, H, L, C, V)]
        self._indicator_cache: Dict[str, Dict[str, List[IndicatorValue]]] = {}  # indicator -> symbol -> values
        
        # Indicator calculators
        self._calculators: Dict[str, Callable] = {}
        self._setup_calculators()
        
        # State tracking
        self._last_update: Dict[str, datetime] = {}
        
    def _setup_calculators(self) -> None:
        """Set up indicator calculation functions."""
        # Built-in indicators
        self._calculators.update({
            'SMA': self._calculate_sma,
            'EMA': self._calculate_ema,
            'RSI': self._calculate_rsi,
            'MACD': self._calculate_macd,
            'BB': self._calculate_bollinger_bands,
            'ATR': self._calculate_atr,
            'ADX': self._calculate_adx,
            'VWAP': self._calculate_vwap,
            'OBV': self._calculate_obv,
            'STOCH': self._calculate_stochastic
        })
        
    def process_market_data(self, event: Event) -> None:
        """
        Process market data event and update indicators.
        
        Args:
            event: Market data event (BAR or TICK)
        """
        if event.event_type not in [EventType.BAR, EventType.TICK]:
            return
            
        timestamp = event.payload.get('timestamp')
        market_data = event.payload.get('market_data', {})
        
        # Update price history
        self._update_price_history(timestamp, market_data)
        
        # Calculate indicators for each symbol
        for symbol in market_data.keys():
            self._calculate_indicators_for_symbol(symbol, timestamp)
            
    def _update_price_history(
        self,
        timestamp: datetime,
        market_data: Dict[str, MarketData]
    ) -> None:
        """Update price history with new market data."""
        for symbol, data in market_data.items():
            if symbol not in self._price_history:
                self._price_history[symbol] = []
                
            # Extract OHLCV data
            if hasattr(data, 'open'):
                ohlcv = (
                    timestamp,
                    data.open,
                    data.high,
                    data.low,
                    data.close,
                    getattr(data, 'volume', 0)
                )
            else:
                # Use price for all OHLC if not available
                price = data.price
                ohlcv = (timestamp, price, price, price, price, 0)
                
            self._price_history[symbol].append(ohlcv)
            
            # Limit history size
            if len(self._price_history[symbol]) > self.cache_size:
                self._price_history[symbol].pop(0)
                
    def _calculate_indicators_for_symbol(
        self,
        symbol: str,
        timestamp: datetime
    ) -> None:
        """Calculate all indicators for a symbol."""
        if symbol not in self._price_history or not self._price_history[symbol]:
            return
            
        for indicator_name, config in self.indicators.items():
            # Check if indicator applies to this symbol
            if config.symbols and symbol not in config.symbols:
                continue
                
            # Get calculator
            base_indicator = indicator_name.split('_')[0]  # Handle names like SMA_20
            calculator = self._calculators.get(base_indicator)
            
            if not calculator:
                logger.warning(f"No calculator for indicator {base_indicator}")
                continue
                
            # Calculate indicator value
            try:
                value = calculator(symbol, config.parameters)
                
                if value is not None:
                    # Create indicator value
                    indicator_value = IndicatorValue(
                        name=indicator_name,
                        symbol=symbol,
                        timestamp=timestamp,
                        value=value,
                        metadata={'parameters': config.parameters}
                    )
                    
                    # Cache value
                    self._cache_indicator_value(indicator_value)
                    
                    # Emit event
                    self._emit_indicator_event(indicator_value)
                    
            except Exception as e:
                logger.error(f"Error calculating {indicator_name} for {symbol}: {e}")
                
    def _cache_indicator_value(self, indicator_value: IndicatorValue) -> None:
        """Cache an indicator value."""
        indicator_name = indicator_value.name
        symbol = indicator_value.symbol
        
        if indicator_name not in self._indicator_cache:
            self._indicator_cache[indicator_name] = {}
            
        if symbol not in self._indicator_cache[indicator_name]:
            self._indicator_cache[indicator_name][symbol] = []
            
        cache = self._indicator_cache[indicator_name][symbol]
        cache.append(indicator_value)
        
        # Limit cache size
        if len(cache) > self.cache_size:
            cache.pop(0)
            
    def _emit_indicator_event(self, indicator_value: IndicatorValue) -> None:
        """Emit an indicator update event."""
        if not self.event_bus:
            return
            
        event = Event(
            event_type=EventType.INDICATOR,
            payload={
                'indicator_type': indicator_value.name,
                'symbol': indicator_value.symbol,
                'timestamp': indicator_value.timestamp,
                'value': indicator_value.value,
                'metadata': indicator_value.metadata
            },
            source_id="indicator_hub"
        )
        
        self.event_bus.publish(event)
        
    # Indicator calculation methods
    
    def _calculate_sma(self, symbol: str, params: Dict[str, Any]) -> Optional[float]:
        """Calculate Simple Moving Average."""
        period = params.get('period', 20)
        price_type = params.get('price_type', 'close')
        
        if symbol not in self._price_history:
            return None
            
        history = self._price_history[symbol]
        if len(history) < period:
            return None
            
        # Extract prices
        price_idx = {'open': 1, 'high': 2, 'low': 3, 'close': 4}[price_type]
        prices = [h[price_idx] for h in history[-period:]]
        
        return np.mean(prices)
        
    def _calculate_ema(self, symbol: str, params: Dict[str, Any]) -> Optional[float]:
        """Calculate Exponential Moving Average."""
        period = params.get('period', 20)
        price_type = params.get('price_type', 'close')
        
        if symbol not in self._price_history:
            return None
            
        history = self._price_history[symbol]
        if len(history) < period:
            return None
            
        # Extract prices
        price_idx = {'open': 1, 'high': 2, 'low': 3, 'close': 4}[price_type]
        prices = [h[price_idx] for h in history]
        
        # Calculate EMA
        ema = pd.Series(prices).ewm(span=period, adjust=False).mean()
        
        return ema.iloc[-1]
        
    def _calculate_rsi(self, symbol: str, params: Dict[str, Any]) -> Optional[float]:
        """Calculate Relative Strength Index."""
        period = params.get('period', 14)
        
        if symbol not in self._price_history:
            return None
            
        history = self._price_history[symbol]
        if len(history) < period + 1:
            return None
            
        # Extract closing prices
        closes = [h[4] for h in history]
        
        # Calculate price changes
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate average gains and losses
        avg_gain = pd.Series(gains).rolling(period).mean().iloc[-1]
        avg_loss = pd.Series(losses).rolling(period).mean().iloc[-1]
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    def _calculate_macd(self, symbol: str, params: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        fast_period = params.get('fast_period', 12)
        slow_period = params.get('slow_period', 26)
        signal_period = params.get('signal_period', 9)
        
        if symbol not in self._price_history:
            return None
            
        history = self._price_history[symbol]
        if len(history) < slow_period + signal_period:
            return None
            
        # Extract closing prices
        closes = pd.Series([h[4] for h in history])
        
        # Calculate MACD
        ema_fast = closes.ewm(span=fast_period, adjust=False).mean()
        ema_slow = closes.ewm(span=slow_period, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line.iloc[-1],
            'signal': signal_line.iloc[-1],
            'histogram': histogram.iloc[-1]
        }
        
    def _calculate_bollinger_bands(self, symbol: str, params: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Calculate Bollinger Bands."""
        period = params.get('period', 20)
        std_dev = params.get('std_dev', 2)
        
        if symbol not in self._price_history:
            return None
            
        history = self._price_history[symbol]
        if len(history) < period:
            return None
            
        # Extract closing prices
        closes = [h[4] for h in history[-period:]]
        
        # Calculate bands
        sma = np.mean(closes)
        std = np.std(closes)
        
        return {
            'upper': sma + (std_dev * std),
            'middle': sma,
            'lower': sma - (std_dev * std),
            'bandwidth': 2 * std_dev * std / sma if sma > 0 else 0
        }
        
    def _calculate_atr(self, symbol: str, params: Dict[str, Any]) -> Optional[float]:
        """Calculate Average True Range."""
        period = params.get('period', 14)
        
        if symbol not in self._price_history:
            return None
            
        history = self._price_history[symbol]
        if len(history) < period + 1:
            return None
            
        # Calculate true ranges
        true_ranges = []
        for i in range(1, len(history)):
            high = history[i][2]
            low = history[i][3]
            prev_close = history[i-1][4]
            
            true_range = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(true_range)
            
        # Calculate ATR
        atr = pd.Series(true_ranges).rolling(period).mean().iloc[-1]
        
        return atr
        
    def _calculate_adx(self, symbol: str, params: Dict[str, Any]) -> Optional[float]:
        """Calculate Average Directional Index."""
        period = params.get('period', 14)
        
        if symbol not in self._price_history:
            return None
            
        history = self._price_history[symbol]
        if len(history) < period * 2:
            return None
            
        # This is a simplified ADX calculation
        # Full implementation would include +DI and -DI
        highs = [h[2] for h in history]
        lows = [h[3] for h in history]
        closes = [h[4] for h in history]
        
        # Calculate directional movement
        plus_dm = []
        minus_dm = []
        
        for i in range(1, len(history)):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
                minus_dm.append(0)
            elif down_move > up_move and down_move > 0:
                plus_dm.append(0)
                minus_dm.append(down_move)
            else:
                plus_dm.append(0)
                minus_dm.append(0)
                
        # Calculate ATR for normalization
        atr = self._calculate_atr(symbol, params)
        if not atr:
            return None
            
        # Simplified ADX calculation
        adx = np.mean(plus_dm[-period:]) / atr * 100
        
        return min(adx, 100)
        
    def _calculate_vwap(self, symbol: str, params: Dict[str, Any]) -> Optional[float]:
        """Calculate Volume Weighted Average Price."""
        if symbol not in self._price_history:
            return None
            
        history = self._price_history[symbol]
        if not history:
            return None
            
        # VWAP is typically calculated from day start
        # For simplicity, we'll use available history
        total_volume = 0
        total_pv = 0  # price * volume
        
        for h in history:
            typical_price = (h[2] + h[3] + h[4]) / 3  # (H + L + C) / 3
            volume = h[5]
            
            total_pv += typical_price * volume
            total_volume += volume
            
        if total_volume == 0:
            return None
            
        return total_pv / total_volume
        
    def _calculate_obv(self, symbol: str, params: Dict[str, Any]) -> Optional[float]:
        """Calculate On Balance Volume."""
        if symbol not in self._price_history:
            return None
            
        history = self._price_history[symbol]
        if len(history) < 2:
            return None
            
        # Calculate OBV
        obv = 0
        for i in range(1, len(history)):
            close = history[i][4]
            prev_close = history[i-1][4]
            volume = history[i][5]
            
            if close > prev_close:
                obv += volume
            elif close < prev_close:
                obv -= volume
            # If close == prev_close, OBV doesn't change
            
        return obv
        
    def _calculate_stochastic(self, symbol: str, params: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Calculate Stochastic Oscillator."""
        k_period = params.get('k_period', 14)
        d_period = params.get('d_period', 3)
        
        if symbol not in self._price_history:
            return None
            
        history = self._price_history[symbol]
        if len(history) < k_period + d_period:
            return None
            
        # Calculate %K values
        k_values = []
        for i in range(k_period - 1, len(history)):
            period_data = history[i - k_period + 1:i + 1]
            highs = [h[2] for h in period_data]
            lows = [h[3] for h in period_data]
            close = history[i][4]
            
            highest = max(highs)
            lowest = min(lows)
            
            if highest - lowest > 0:
                k = (close - lowest) / (highest - lowest) * 100
            else:
                k = 50  # Default to middle
                
            k_values.append(k)
            
        # Calculate %D (SMA of %K)
        if len(k_values) < d_period:
            return None
            
        d_values = pd.Series(k_values).rolling(d_period).mean()
        
        return {
            'k': k_values[-1],
            'd': d_values.iloc[-1]
        }
        
    # Public interface methods
    
    def get_latest_value(
        self,
        indicator_name: str,
        symbol: str
    ) -> Optional[IndicatorValue]:
        """Get the latest cached value for an indicator."""
        if (indicator_name in self._indicator_cache and
            symbol in self._indicator_cache[indicator_name] and
            self._indicator_cache[indicator_name][symbol]):
            return self._indicator_cache[indicator_name][symbol][-1]
        return None
        
    def get_indicator_history(
        self,
        indicator_name: str,
        symbol: str,
        lookback: int = 100
    ) -> List[IndicatorValue]:
        """Get historical values for an indicator."""
        if (indicator_name in self._indicator_cache and
            symbol in self._indicator_cache[indicator_name]):
            cache = self._indicator_cache[indicator_name][symbol]
            return cache[-lookback:] if len(cache) > lookback else cache.copy()
        return []
        
    def add_custom_indicator(
        self,
        name: str,
        calculator: Callable[[str, Dict[str, Any]], Optional[Union[float, Dict[str, float]]]],
        config: IndicatorConfig
    ) -> None:
        """Add a custom indicator to the hub."""
        self.indicators[name] = config
        self._calculators[name] = calculator
        
        logger.info(f"Added custom indicator: {name}")
        
    def get_available_indicators(self) -> List[str]:
        """Get list of available indicators."""
        return list(self.indicators.keys())
        
    def clear_cache(self, indicator_name: Optional[str] = None) -> None:
        """Clear indicator cache."""
        if indicator_name:
            if indicator_name in self._indicator_cache:
                self._indicator_cache[indicator_name].clear()
        else:
            self._indicator_cache.clear()