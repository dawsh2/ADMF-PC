"""
File: src/strategy/indicators.py
Status: ACTIVE
Architecture Ref: SYSTEM_ARCHITECTURE_v5.md#indicators
Step: 1 - Core Pipeline Test
Dependencies: core.events, data.models, core.logging

Simple Moving Average indicator implementation for Step 1 of complexity guide.
Demonstrates Protocol + Composition pattern with event-driven architecture.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from collections import deque
from datetime import datetime

from ..data.models import Bar
from ..core.logging.structured import ContainerLogger


class SimpleMovingAverage:
    """
    Basic SMA indicator for testing event-driven pipeline.
    
    This class demonstrates the Protocol + Composition pattern without inheritance.
    Designed for Step 1 requirements to validate the core event pipeline.
    
    Architecture Context:
        - Part of: Core Pipeline Test (step-01-core-pipeline.md)
        - Implements: Protocol-based indicator pattern
        - Enables: Event flow validation from data to strategy
        - Dependencies: ComponentLogger for event flow tracking
    
    Example:
        indicator = SimpleMovingAverage(period=20, container_id="test_001")
        indicator.on_bar(bar)
        if indicator.is_ready:
            print(f"SMA: {indicator.current_value}")
    """
    
    def __init__(self, period: int, container_id: str):
        """
        Initialize Simple Moving Average indicator.
        
        Args:
            period: Number of bars for SMA calculation
            container_id: Container ID for logging context
        """
        self.period = period
        self.container_id = container_id
        self.values = deque(maxlen=period)
        self.logger = ContainerLogger("SMA", container_id, "sma_indicator")
        
        # State tracking
        self.current_value: Optional[float] = None
        self._bar_count = 0
        
        self.logger.info(
            "SMA indicator initialized",
            period=period,
            container_id=container_id
        )
    
    def on_bar(self, bar: Bar) -> None:
        """
        Process new market data bar.
        
        This method implements the event-driven pattern where indicators
        react to market data events and update their calculations.
        
        Args:
            bar: Market data bar to process
            
        Architecture Context:
            - Called by: Data source or event bus
            - Triggers: SMA calculation and event logging
            - Enables: Event flow validation for Step 1
        """
        self.logger.trace(
            "Processing new bar",
            symbol=bar.symbol,
            timestamp=bar.timestamp.isoformat(),
            close_price=bar.close
        )
        
        # Add new value
        self.values.append(bar.close)
        self._bar_count += 1
        
        # Calculate SMA if we have enough data
        if len(self.values) == self.period:
            self.current_value = sum(self.values) / self.period
            
            self.logger.log_event_flow(
                "SMA_CALCULATED",
                "indicator",
                "strategy", 
                f"SMA={self.current_value:.2f}"
            )
            
            self.logger.debug(
                "SMA calculated",
                current_value=self.current_value,
                period=self.period,
                bar_count=self._bar_count
            )
        else:
            self.logger.trace(
                "Insufficient data for SMA",
                values_count=len(self.values),
                required_count=self.period
            )
    
    @property
    def is_ready(self) -> bool:
        """
        Check if indicator has enough data to provide valid values.
        
        Returns:
            True if SMA can be calculated, False otherwise
        """
        return len(self.values) == self.period and self.current_value is not None
    
    @property
    def name(self) -> str:
        """Get indicator name for identification."""
        return f"SMA_{self.period}"
    
    def reset(self) -> None:
        """
        Reset indicator state for new calculation cycle.
        
        This method supports backtesting scenarios where indicators
        need to be reset between test runs.
        """
        self.values.clear()
        self.current_value = None
        self._bar_count = 0
        
        self.logger.info("SMA indicator reset")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current indicator state for debugging and validation.
        
        Returns:
            Dictionary containing indicator state information
        """
        return {
            "name": self.name,
            "period": self.period,
            "current_value": self.current_value,
            "is_ready": self.is_ready,
            "values_count": len(self.values),
            "bar_count": self._bar_count,
            "container_id": self.container_id
        }