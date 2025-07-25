"""
File: src/strategy/strategies/simple_trend.py
Status: ACTIVE
Architecture Ref: SYSTEM_ARCHITECTURE_v5.md#strategies
Step: 1 - Core Pipeline Test
Dependencies: core.events, data.models, strategy.indicators, core.logging

Simple trend following strategy for Step 1 validation.
Uses SMA crossover to demonstrate event-driven pipeline.
"""

from __future__ import annotations
from src.core.features.feature_spec import FeatureSpec
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

import logging

from ...data.models import Bar
from ..components.features import FeatureHub
from ...risk.protocols import Signal, SignalType, OrderSide
from decimal import Decimal
from ...core.components.discovery import strategy

logger = logging.getLogger(__name__)


class SimpleTrendStrategy:
    """
    SMA crossover strategy for testing Step 1 pipeline.
    
    This strategy demonstrates the Protocol + Composition pattern
    for event-driven trading systems. Uses two SMAs to generate
    buy/sell signals when they cross.
    
    Architecture Context:
        - Part of: Core Pipeline Test (step-01-core-pipeline.md)
        - Implements: Protocol-based strategy pattern
        - Enables: Event flow validation from indicator to risk manager
        - Dependencies: SimpleMovingAverage, ComponentLogger
    
    Example:
        strategy = SimpleTrendStrategy(
            fast_period=10, slow_period=20, container_id="test_001"
        )
        signal = strategy.on_bar(bar)
    """
    
    def __init__(self, fast_period: int, slow_period: int, container_id: str):
        """
        Initialize simple trend strategy.
        
        Args:
            fast_period: Fast SMA period
            slow_period: Slow SMA period
            container_id: Container ID for logging context
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.container_id = container_id
        
        # Create indicators
        self.fast_sma = SimpleMovingAverage(fast_period, container_id)
        self.slow_sma = SimpleMovingAverage(slow_period, container_id)
        
        # State tracking
        self.position = 0  # -1: short, 0: neutral, 1: long
        self.last_signal_time: Optional[datetime] = None
        self.signal_cooldown = 300  # 5 minutes in seconds
        
        # Event bus (injected by container)
        self.event_bus = None
        
        logger.info(
            f"SimpleTrendStrategy initialized: fast={fast_period}, slow={slow_period}, container={container_id}"
        )
    
    @property
    def name(self) -> str:
        """Strategy name for identification."""
        return "simple_trend_strategy"
    
    def on_bar(self, bar: Bar) -> None:
        """
        Process market data and generate signals.
        
        This method implements the event-driven pattern where strategies
        react to market data events and generate trading signals.
        
        Args:
            bar: Market data bar to process
            
        Architecture Context:
            - Called by: Data source or event bus
            - Triggers: Signal generation and event publishing
            - Enables: Event flow validation for Step 1
        """
        logger.debug(
            f"Processing bar: {bar.symbol} @ {bar.timestamp.isoformat()}, close={bar.close}"
        )
        
        # Update indicators
        self.fast_sma.on_bar(bar)
        self.slow_sma.on_bar(bar)
        
        # Check if we can generate signals
        if self._should_generate_signal(bar.timestamp):
            signal = self._create_signal(bar)
            if signal and self.event_bus:
                logger.info(
                    f"SIGNAL: strategy -> risk_manager: {signal.side.value} {bar.symbol}"
                )
                self.event_bus.publish("SIGNAL", signal)
                self.last_signal_time = bar.timestamp
    
    def _should_generate_signal(self, timestamp: datetime) -> bool:
        """
        Check if we should generate a signal.
        
        Args:
            timestamp: Current bar timestamp
            
        Returns:
            True if signal should be generated, False otherwise
        """
        # Need both SMAs ready
        if not (self.fast_sma.is_ready and self.slow_sma.is_ready):
            logger.debug(
                f"SMAs not ready: fast={self.fast_sma.is_ready}, slow={self.slow_sma.is_ready}"
            )
            return False
        
        # Check cooldown
        if self.last_signal_time:
            time_since_last = (timestamp - self.last_signal_time).total_seconds()
            if time_since_last < self.signal_cooldown:
                logger.debug(
                    f"Signal in cooldown: {time_since_last}s < {self.signal_cooldown}s"
                )
                return False
        
        return True
    
    def _create_signal(self, bar: Bar) -> Optional[Signal]:
        """
        Create trading signal based on SMA crossover.
        
        Args:
            bar: Current market data bar
            
        Returns:
            Signal object if crossover detected, None otherwise
        """
        fast_value = self.fast_sma.current_value
        slow_value = self.slow_sma.current_value
        
        if fast_value is None or slow_value is None:
            return None
        
        signal = None
        new_position = None
        
        # Detect crossovers
        if fast_value > slow_value and self.position <= 0:
            # Bullish crossover - go long
            new_position = 1
            signal = Signal(
                signal_id=str(uuid.uuid4()),
                strategy_id=self.name,
                symbol=bar.symbol,
                signal_type=SignalType.ENTRY,
                side=OrderSide.BUY,
                strength=Decimal('0.8'),  # Strong signal for clear crossover
                timestamp=bar.timestamp,
                metadata={
                    'fast_sma': float(fast_value),
                    'slow_sma': float(slow_value),
                    'crossover_type': 'bullish',
                    'reason': 'Fast SMA crossed above Slow SMA'
                }
            )
            
        elif fast_value < slow_value and self.position >= 0:
            # Bearish crossover - go short
            new_position = -1
            signal = Signal(
                signal_id=str(uuid.uuid4()),
                strategy_id=self.name,
                symbol=bar.symbol,
                signal_type=SignalType.ENTRY,
                side=OrderSide.SELL,
                strength=Decimal('0.8'),  # Strong signal for clear crossover
                timestamp=bar.timestamp,
                metadata={
                    'fast_sma': float(fast_value),
                    'slow_sma': float(slow_value),
                    'crossover_type': 'bearish',
                    'reason': 'Fast SMA crossed below Slow SMA'
                }
            )
        
        if signal:
            self.position = new_position
            logger.info(
                f"Signal generated: {signal.side.value} {bar.symbol}, "
                f"fast_sma={fast_value}, slow_sma={slow_value}, position={new_position}"
            )
        
        return signal
    
    def reset(self) -> None:
        """
        Reset strategy state for new calculation cycle.
        
        This method supports backtesting scenarios where strategies
        need to be reset between test runs.
        """
        self.fast_sma.reset()
        self.slow_sma.reset()
        self.position = 0
        self.last_signal_time = None
        
        logger.info("SimpleTrendStrategy reset")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current strategy state for debugging and validation.
        
        Returns:
            Dictionary containing strategy state information
        """
        return {
            "name": self.name,
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "position": self.position,
            "fast_sma_ready": self.fast_sma.is_ready,
            "slow_sma_ready": self.slow_sma.is_ready,
            "fast_sma_value": self.fast_sma.current_value,
            "slow_sma_value": self.slow_sma.current_value,
            "container_id": self.container_id,
            "last_signal_time": self.last_signal_time.isoformat() if self.last_signal_time else None
        }


# Pure function version for EVENT_FLOW_ARCHITECTURE
@strategy(
    name='simple_trend',
    feature_discovery=lambda params: [FeatureSpec('sma', {'period': params.get('sma_period', 20)})]  # Topology builder infers parameters from strategy logic
)
def simple_trend_strategy(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Pure function simple trend following strategy using SMA crossover.
    
    Args:
        features: Calculated indicators from FeatureHub
        bar: Current market bar with OHLCV data
        params: Strategy parameters (fast_period, slow_period)
        
    Returns:
        Signal dict or None
    """
    # Extract parameters
    fast_period = params.get('fast_period', 10)
    slow_period = params.get('slow_period', 20)
    
    # Get features
    price = bar.get('close', 0)
    fast_sma = features.get(f'sma_{fast_period}')
    slow_sma = features.get(f'sma_{slow_period}')
    
    # Check if we have required features
    if fast_sma is None or slow_sma is None:
        return None
    
    # Previous values for crossover detection (if available)
    prev_fast = features.get(f'prev_sma_{fast_period}')
    prev_slow = features.get(f'prev_sma_{slow_period}')
    
    # Generate signal based on crossover
    signal = None
    
    # Bullish crossover: fast crosses above slow
    if prev_fast and prev_slow:
        if prev_fast <= prev_slow and fast_sma > slow_sma:
            signal = {
                'symbol': bar.get('symbol'),
                'direction': 'long',
                'strength': 0.8,
                'price': price,
                'reason': f'Bullish crossover: SMA{fast_period} crossed above SMA{slow_period}',
                'indicators': {
                    'price': price,
                    'fast_sma': fast_sma,
                    'slow_sma': slow_sma
                }
            }
            logger.info(f"Generated LONG signal: fast_sma={fast_sma}, slow_sma={slow_sma}")
            
        # Bearish crossover: fast crosses below slow
        elif prev_fast >= prev_slow and fast_sma < slow_sma:
            signal = {
                'symbol': bar.get('symbol'),
                'direction': 'short',
                'strength': 0.8,
                'price': price,
                'reason': f'Bearish crossover: SMA{fast_period} crossed below SMA{slow_period}',
                'indicators': {
                    'price': price,
                    'fast_sma': fast_sma,
                    'slow_sma': slow_sma
                }
            }
            logger.info(f"Generated SHORT signal: fast_sma={fast_sma}, slow_sma={slow_sma}")
    
    # If no previous values, just check current state
    elif fast_sma > slow_sma:
        signal = {
            'symbol': bar.get('symbol'),
            'direction': 'long',
            'strength': 0.6,  # Lower strength without crossover confirmation
            'price': price,
            'reason': f'Bullish trend: SMA{fast_period} > SMA{slow_period}',
            'indicators': {
                'price': price,
                'fast_sma': fast_sma,
                'slow_sma': slow_sma
            }
        }
    elif fast_sma < slow_sma:
        signal = {
            'symbol': bar.get('symbol'),
            'direction': 'short',
            'strength': 0.6,  # Lower strength without crossover confirmation
            'price': price,
            'reason': f'Bearish trend: SMA{fast_period} < SMA{slow_period}',
            'indicators': {
                'price': price,
                'fast_sma': fast_sma,
                'slow_sma': slow_sma
            }
        }
    
    return signal