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
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from ...data.models import Bar
from ...core.logging.structured import ContainerLogger
from ..components.features import FeatureHub, sma_feature
from ...risk.protocols import Signal, SignalType, OrderSide
from decimal import Decimal


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
        
        # Logging
        self.logger = ContainerLogger("SimpleTrendStrategy", container_id, "simple_trend_strategy")
        
        self.logger.info(
            "SimpleTrendStrategy initialized",
            fast_period=fast_period,
            slow_period=slow_period,
            container_id=container_id
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
        self.logger.trace(
            "Processing bar",
            symbol=bar.symbol,
            timestamp=bar.timestamp.isoformat(),
            close_price=bar.close
        )
        
        # Update indicators
        self.fast_sma.on_bar(bar)
        self.slow_sma.on_bar(bar)
        
        # Check if we can generate signals
        if self._should_generate_signal(bar.timestamp):
            signal = self._create_signal(bar)
            if signal and self.event_bus:
                self.logger.log_event_flow(
                    "SIGNAL",
                    "strategy",
                    "risk_manager",
                    f"{signal.side.value} {bar.symbol}"
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
            self.logger.trace(
                "SMAs not ready",
                fast_ready=self.fast_sma.is_ready,
                slow_ready=self.slow_sma.is_ready
            )
            return False
        
        # Check cooldown
        if self.last_signal_time:
            time_since_last = (timestamp - self.last_signal_time).total_seconds()
            if time_since_last < self.signal_cooldown:
                self.logger.trace(
                    "Signal in cooldown",
                    time_since_last=time_since_last,
                    cooldown=self.signal_cooldown
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
            self.logger.info(
                "Signal generated",
                signal_type=signal.side.value,
                symbol=bar.symbol,
                fast_sma=fast_value,
                slow_sma=slow_value,
                new_position=new_position
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
        
        self.logger.info("SimpleTrendStrategy reset")
    
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