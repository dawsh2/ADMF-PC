"""
File: tests/unit/test_step1_components.py
Status: ACTIVE
Architecture Ref: SYSTEM_ARCHITECTURE_v5.md#testing
Step: 1 - Core Pipeline Test
Dependencies: pytest, core.events, strategy, data.models

Unit tests for Step 1 components.
Validates individual component behavior in isolation.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from src.data.models import Bar
from src.strategy.indicators import SimpleMovingAverage
from src.strategy.strategies.simple_trend import SimpleTrendStrategy
from src.risk.protocols import OrderSide, SignalType


class TestSimpleMovingAverage:
    """Unit tests for SMA indicator."""
    
    def test_sma_initialization(self):
        """Test SMA initializes correctly."""
        sma = SimpleMovingAverage(period=3, container_id="test")
        
        assert sma.period == 3
        assert sma.container_id == "test"
        assert sma.current_value is None
        assert not sma.is_ready
        assert sma.name == "SMA_3"
    
    def test_sma_calculation(self):
        """Test SMA calculates correct values."""
        sma = SimpleMovingAverage(period=3, container_id="test")
        
        # Create test bars with known values
        bars = [
            Bar(symbol="TEST", timestamp=datetime.now(), open=10, high=12, low=8, close=10, volume=100),
            Bar(symbol="TEST", timestamp=datetime.now(), open=11, high=13, low=9, close=20, volume=100),
            Bar(symbol="TEST", timestamp=datetime.now(), open=21, high=23, low=19, close=30, volume=100)
        ]
        
        # Process bars
        for bar in bars:
            sma.on_bar(bar)
        
        # Expected: (10 + 20 + 30) / 3 = 20
        assert sma.current_value == 20.0
        assert sma.is_ready
    
    def test_sma_insufficient_data(self):
        """Test SMA with insufficient data."""
        sma = SimpleMovingAverage(period=5, container_id="test")
        
        # Add only 3 bars
        bars = [
            Bar(symbol="TEST", timestamp=datetime.now(), open=10, high=12, low=8, close=10, volume=100),
            Bar(symbol="TEST", timestamp=datetime.now(), open=11, high=13, low=9, close=20, volume=100),
            Bar(symbol="TEST", timestamp=datetime.now(), open=21, high=23, low=19, close=30, volume=100)
        ]
        
        for bar in bars:
            sma.on_bar(bar)
        
        # Should not be ready with only 3 of 5 required bars
        assert not sma.is_ready
        assert sma.current_value is None
    
    def test_sma_rolling_window(self):
        """Test SMA maintains rolling window correctly."""
        sma = SimpleMovingAverage(period=3, container_id="test")
        
        # Add 5 bars to test rolling window
        values = [10, 20, 30, 40, 50]
        for i, value in enumerate(values):
            bar = Bar(
                symbol="TEST",
                timestamp=datetime.now() + timedelta(minutes=i),
                open=value,
                high=value + 2,
                low=value - 2,
                close=value,
                volume=100
            )
            sma.on_bar(bar)
        
        # Should use last 3 values: (30 + 40 + 50) / 3 = 40
        assert sma.current_value == 40.0
        assert sma.is_ready
    
    def test_sma_reset(self):
        """Test SMA reset functionality."""
        sma = SimpleMovingAverage(period=2, container_id="test")
        
        # Add bars to make it ready
        bars = [
            Bar(symbol="TEST", timestamp=datetime.now(), open=10, high=12, low=8, close=10, volume=100),
            Bar(symbol="TEST", timestamp=datetime.now(), open=11, high=13, low=9, close=20, volume=100)
        ]
        
        for bar in bars:
            sma.on_bar(bar)
        
        assert sma.is_ready
        assert sma.current_value == 15.0
        
        # Reset and verify
        sma.reset()
        assert not sma.is_ready
        assert sma.current_value is None
        assert len(sma.values) == 0


class TestSimpleTrendStrategy:
    """Unit tests for simple trend strategy."""
    
    def test_strategy_initialization(self):
        """Test strategy initializes correctly."""
        strategy = SimpleTrendStrategy(
            fast_period=2,
            slow_period=4,
            container_id="test"
        )
        
        assert strategy.fast_period == 2
        assert strategy.slow_period == 4
        assert strategy.container_id == "test"
        assert strategy.position == 0
        assert strategy.name == "simple_trend_strategy"
        assert strategy.event_bus is None
    
    def test_strategy_requires_sma_ready(self):
        """Test strategy waits for SMAs to be ready."""
        strategy = SimpleTrendStrategy(
            fast_period=2,
            slow_period=3,
            container_id="test"
        )
        
        # Mock event bus
        events_published = []
        strategy.event_bus = MockEventBus(events_published)
        
        # Add insufficient bars
        bar = Bar(
            symbol="TEST",
            timestamp=datetime.now(),
            open=100,
            high=102,
            low=98,
            close=100,
            volume=1000
        )
        
        strategy.on_bar(bar)
        
        # No signals should be generated
        assert len(events_published) == 0
    
    def test_bullish_crossover_signal(self):
        """Test strategy generates bullish signal on upward crossover."""
        strategy = SimpleTrendStrategy(
            fast_period=2,
            slow_period=3,
            container_id="test"
        )
        
        # Mock event bus
        events_published = []
        strategy.event_bus = MockEventBus(events_published)
        
        # Create upward trend data
        # Values designed to create bullish crossover
        prices = [95, 96, 98, 102, 105]  # Rising trend
        
        for i, price in enumerate(prices):
            bar = Bar(
                symbol="TEST",
                timestamp=datetime.now() + timedelta(minutes=i),
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1000
            )
            strategy.on_bar(bar)
        
        # Should generate bullish signal
        assert len(events_published) > 0
        
        # Find the signal event
        signal_event = None
        for event_type, data in events_published:
            if event_type == "SIGNAL":
                signal_event = data
                break
        
        assert signal_event is not None
        assert signal_event.side == OrderSide.BUY
        assert signal_event.signal_type == SignalType.ENTRY
        assert strategy.position == 1
    
    def test_bearish_crossover_signal(self):
        """Test strategy generates bearish signal on downward crossover."""
        strategy = SimpleTrendStrategy(
            fast_period=2,
            slow_period=3,
            container_id="test"
        )
        
        # Mock event bus
        events_published = []
        strategy.event_bus = MockEventBus(events_published)
        
        # Create trend: up then down to trigger bearish crossover
        prices = [105, 104, 102, 98, 95]  # Falling trend
        
        for i, price in enumerate(prices):
            bar = Bar(
                symbol="TEST",
                timestamp=datetime.now() + timedelta(minutes=i),
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1000
            )
            strategy.on_bar(bar)
        
        # Should generate bearish signal
        assert len(events_published) > 0
        
        # Find the signal event
        signal_event = None
        for event_type, data in events_published:
            if event_type == "SIGNAL":
                signal_event = data
                break
        
        assert signal_event is not None
        assert signal_event.side == OrderSide.SELL
        assert signal_event.signal_type == SignalType.ENTRY
        assert strategy.position == -1
    
    def test_signal_cooldown(self):
        """Test strategy respects signal cooldown period."""
        strategy = SimpleTrendStrategy(
            fast_period=2,
            slow_period=3,
            container_id="test"
        )
        strategy.signal_cooldown = 10  # 10 seconds for testing
        
        # Mock event bus
        events_published = []
        strategy.event_bus = MockEventBus(events_published)
        
        # Create first signal
        base_time = datetime.now()
        prices = [95, 96, 98, 102, 105]
        
        for i, price in enumerate(prices):
            bar = Bar(
                symbol="TEST",
                timestamp=base_time + timedelta(seconds=i),
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1000
            )
            strategy.on_bar(bar)
        
        initial_signal_count = len([e for e in events_published if e[0] == "SIGNAL"])
        
        # Try to generate another signal within cooldown
        bar = Bar(
            symbol="TEST",
            timestamp=base_time + timedelta(seconds=5),  # Within cooldown
            open=110,
            high=111,
            low=109,
            close=110,
            volume=1000
        )
        strategy.on_bar(bar)
        
        # Should not generate additional signal
        final_signal_count = len([e for e in events_published if e[0] == "SIGNAL"])
        assert final_signal_count == initial_signal_count
    
    def test_strategy_reset(self):
        """Test strategy reset functionality."""
        strategy = SimpleTrendStrategy(
            fast_period=2,
            slow_period=3,
            container_id="test"
        )
        
        # Set some state
        strategy.position = 1
        strategy.last_signal_time = datetime.now()
        
        # Add bars to indicators
        bar = Bar(
            symbol="TEST",
            timestamp=datetime.now(),
            open=100,
            high=102,
            low=98,
            close=100,
            volume=1000
        )
        strategy.on_bar(bar)
        
        # Reset and verify
        strategy.reset()
        assert strategy.position == 0
        assert strategy.last_signal_time is None
        assert not strategy.fast_sma.is_ready
        assert not strategy.slow_sma.is_ready


class MockEventBus:
    """Mock event bus for testing."""
    
    def __init__(self, events_list):
        self.events_list = events_list
    
    def publish(self, event_type, data):
        """Record published events."""
        self.events_list.append((event_type, data))