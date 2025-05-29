"""
Tests for the Risk & Portfolio module.
"""

import pytest
from decimal import Decimal
from datetime import datetime
from typing import Dict, Any

from .protocols import Signal, Order, SignalType, OrderSide, OrderType
from .risk_portfolio import RiskPortfolioContainer
from .position_sizing import FixedPositionSizer, PercentagePositionSizer
from .risk_limits import MaxPositionLimit, MaxExposureLimit
from .portfolio_state import PortfolioState
from ..core.containers import UniversalScopedContainer


class TestRiskPortfolioContainer:
    """Test the unified Risk & Portfolio container."""
    
    def test_initialization(self):
        """Test basic initialization."""
        container = UniversalScopedContainer("test_container")
        risk_portfolio = RiskPortfolioContainer(
            container_id="test_risk",
            parent_container=container,
            initial_capital=Decimal("100000")
        )
        
        assert risk_portfolio.container_id == "test_risk"
        assert risk_portfolio.portfolio_state.cash == Decimal("100000")
        assert len(risk_portfolio.position_sizers) == 0
        assert len(risk_portfolio.risk_limits) == 0
    
    def test_signal_to_order_conversion(self):
        """Test converting signals to orders."""
        container = UniversalScopedContainer("test_container")
        risk_portfolio = RiskPortfolioContainer(
            container_id="test_risk",
            parent_container=container,
            initial_capital=Decimal("100000")
        )
        
        # Add position sizer
        risk_portfolio.add_position_sizer(
            "default",
            FixedPositionSizer(position_size=100)
        )
        
        # Create signal
        signal = {
            'signal_id': 'test_001',
            'timestamp': datetime.now(),
            'strategy_id': 'test_strategy',
            'symbol': 'AAPL',
            'signal_type': SignalType.ENTRY.value,
            'side': OrderSide.BUY.value,
            'strength': 0.8,
            'metadata': {'price': 150.0}
        }
        
        # Process signal
        order = risk_portfolio.process_signal(signal)
        
        # Verify order created
        assert order is not None
        assert order['symbol'] == 'AAPL'
        assert order['quantity'] == 100
        assert order['side'] == OrderSide.BUY.value
        assert order['order_type'] == OrderType.MARKET.value
    
    def test_risk_limit_rejection(self):
        """Test that risk limits can reject signals."""
        container = UniversalScopedContainer("test_container")
        risk_portfolio = RiskPortfolioContainer(
            container_id="test_risk",
            parent_container=container,
            initial_capital=Decimal("100000")
        )
        
        # Add position sizer
        risk_portfolio.add_position_sizer(
            "default",
            FixedPositionSizer(position_size=10000)  # Large position
        )
        
        # Add strict position limit
        risk_portfolio.add_risk_limit(
            MaxPositionLimit(max_position=1000)  # Max 1000 shares
        )
        
        # Create signal
        signal = {
            'signal_id': 'test_002',
            'timestamp': datetime.now(),
            'strategy_id': 'test_strategy',
            'symbol': 'AAPL',
            'signal_type': SignalType.ENTRY.value,
            'side': OrderSide.BUY.value,
            'strength': 0.8,
            'metadata': {'price': 150.0}
        }
        
        # Process signal - should be rejected
        order = risk_portfolio.process_signal(signal)
        assert order is None  # Rejected due to position limit
    
    def test_portfolio_state_tracking(self):
        """Test portfolio state tracking after fills."""
        container = UniversalScopedContainer("test_container")
        risk_portfolio = RiskPortfolioContainer(
            container_id="test_risk",
            parent_container=container,
            initial_capital=Decimal("100000")
        )
        
        # Add position sizer
        risk_portfolio.add_position_sizer(
            "default",
            FixedPositionSizer(position_size=100)
        )
        
        # Process signal to create order
        signal = {
            'signal_id': 'test_003',
            'timestamp': datetime.now(),
            'strategy_id': 'test_strategy',
            'symbol': 'AAPL',
            'signal_type': SignalType.ENTRY.value,
            'side': OrderSide.BUY.value,
            'strength': 0.8,
            'metadata': {'price': 150.0}
        }
        
        order = risk_portfolio.process_signal(signal)
        assert order is not None
        
        # Simulate fill
        fill = {
            'order_id': order['order_id'],
            'symbol': 'AAPL',
            'quantity': 100,
            'price': 150.05,
            'commission': 1.0,
            'timestamp': datetime.now()
        }
        
        risk_portfolio.handle_fill(fill)
        
        # Check portfolio state
        state = risk_portfolio.get_portfolio_state()
        assert state['position_count'] == 1
        assert state['cash'] < 100000  # Cash reduced by purchase
        
        # Check position
        position = risk_portfolio.portfolio_state.get_position('AAPL')
        assert position is not None
        assert position.quantity == 100
        assert position.average_price == Decimal("150.05")
    
    def test_multiple_strategies(self):
        """Test handling signals from multiple strategies."""
        container = UniversalScopedContainer("test_container")
        risk_portfolio = RiskPortfolioContainer(
            container_id="test_risk",
            parent_container=container,
            initial_capital=Decimal("100000")
        )
        
        # Add position sizers for different strategies
        risk_portfolio.add_position_sizer(
            "momentum",
            FixedPositionSizer(position_size=200)
        )
        risk_portfolio.add_position_sizer(
            "mean_reversion",
            FixedPositionSizer(position_size=100)
        )
        risk_portfolio.add_position_sizer(
            "default",
            FixedPositionSizer(position_size=50)
        )
        
        # Signal from momentum strategy
        signal1 = {
            'signal_id': 'mom_001',
            'timestamp': datetime.now(),
            'strategy_id': 'momentum',
            'symbol': 'AAPL',
            'signal_type': SignalType.ENTRY.value,
            'side': OrderSide.BUY.value,
            'strength': 0.9,
            'metadata': {'price': 150.0}
        }
        
        # Signal from mean reversion strategy
        signal2 = {
            'signal_id': 'mr_001',
            'timestamp': datetime.now(),
            'strategy_id': 'mean_reversion',
            'symbol': 'SPY',
            'signal_type': SignalType.ENTRY.value,
            'side': OrderSide.BUY.value,
            'strength': 0.7,
            'metadata': {'price': 450.0}
        }
        
        # Process both signals
        order1 = risk_portfolio.process_signal(signal1)
        order2 = risk_portfolio.process_signal(signal2)
        
        # Verify different position sizes
        assert order1 is not None
        assert order1['quantity'] == 200  # Momentum sizer
        
        assert order2 is not None
        assert order2['quantity'] == 100  # Mean reversion sizer
    
    def test_exposure_limit(self):
        """Test exposure limit across portfolio."""
        container = UniversalScopedContainer("test_container")
        risk_portfolio = RiskPortfolioContainer(
            container_id="test_risk",
            parent_container=container,
            initial_capital=Decimal("100000")
        )
        
        # Add percentage position sizer
        risk_portfolio.add_position_sizer(
            "default",
            PercentagePositionSizer(percentage=Decimal("10"))  # 10% per position
        )
        
        # Add exposure limit
        risk_portfolio.add_risk_limit(
            MaxExposureLimit(max_exposure_pct=Decimal("20"))  # Max 20% exposure
        )
        
        # First signal - should use 10% of capital
        signal1 = {
            'signal_id': 'test_004',
            'timestamp': datetime.now(),
            'strategy_id': 'test_strategy',
            'symbol': 'AAPL',
            'signal_type': SignalType.ENTRY.value,
            'side': OrderSide.BUY.value,
            'strength': 0.8,
            'metadata': {'price': 150.0}
        }
        
        order1 = risk_portfolio.process_signal(signal1)
        assert order1 is not None
        # Should be ~66 shares (10% of 100k / 150)
        assert 65 <= order1['quantity'] <= 67
        
        # Simulate fill to update portfolio
        fill1 = {
            'order_id': order1['order_id'],
            'symbol': 'AAPL',
            'quantity': order1['quantity'],
            'price': 150.0,
            'commission': 1.0,
            'timestamp': datetime.now()
        }
        risk_portfolio.handle_fill(fill1)
        
        # Second signal - should still pass (under 20% limit)
        signal2 = {
            'signal_id': 'test_005',
            'timestamp': datetime.now(),
            'strategy_id': 'test_strategy',
            'symbol': 'GOOGL',
            'signal_type': SignalType.ENTRY.value,
            'side': OrderSide.BUY.value,
            'strength': 0.8,
            'metadata': {'price': 2800.0}
        }
        
        order2 = risk_portfolio.process_signal(signal2)
        assert order2 is not None
        
        # Simulate second fill
        fill2 = {
            'order_id': order2['order_id'],
            'symbol': 'GOOGL',
            'quantity': order2['quantity'],
            'price': 2800.0,
            'commission': 1.0,
            'timestamp': datetime.now()
        }
        risk_portfolio.handle_fill(fill2)
        
        # Third signal - should be rejected (would exceed 20% limit)
        signal3 = {
            'signal_id': 'test_006',
            'timestamp': datetime.now(),
            'strategy_id': 'test_strategy',
            'symbol': 'MSFT',
            'signal_type': SignalType.ENTRY.value,
            'side': OrderSide.BUY.value,
            'strength': 0.8,
            'metadata': {'price': 380.0}
        }
        
        order3 = risk_portfolio.process_signal(signal3)
        assert order3 is None  # Should be rejected due to exposure limit


class TestPositionSizing:
    """Test position sizing strategies."""
    
    def test_fixed_position_sizer(self):
        """Test fixed position sizing."""
        sizer = FixedPositionSizer(position_size=100)
        
        signal = Signal(
            signal_id="test",
            timestamp=datetime.now(),
            strategy_id="test",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=0.8
        )
        
        portfolio_state = PortfolioState(initial_capital=Decimal("100000"))
        
        size = sizer.calculate_position_size(signal, portfolio_state)
        assert size == 100
    
    def test_percentage_position_sizer(self):
        """Test percentage-based position sizing."""
        sizer = PercentagePositionSizer(percentage=Decimal("2"))  # 2%
        
        signal = Signal(
            signal_id="test",
            timestamp=datetime.now(),
            strategy_id="test",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=0.8,
            metadata={'price': 150.0}
        )
        
        portfolio_state = PortfolioState(initial_capital=Decimal("100000"))
        
        size = sizer.calculate_position_size(signal, portfolio_state)
        # 2% of 100k = 2000, at $150/share = ~13 shares
        assert 12 <= size <= 14


class TestRiskLimits:
    """Test risk limit implementations."""
    
    def test_max_position_limit(self):
        """Test maximum position size limit."""
        limit = MaxPositionLimit(max_position=1000)
        
        signal = Signal(
            signal_id="test",
            timestamp=datetime.now(),
            strategy_id="test",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=0.8
        )
        
        portfolio_state = PortfolioState(initial_capital=Decimal("100000"))
        
        # Check small position - should pass
        passed, reason = limit.check(signal, 500, portfolio_state)
        assert passed is True
        
        # Check large position - should fail
        passed, reason = limit.check(signal, 1500, portfolio_state)
        assert passed is False
        assert "position size" in reason.lower()
    
    def test_max_exposure_limit(self):
        """Test maximum exposure limit."""
        limit = MaxExposureLimit(max_exposure_pct=Decimal("20"))  # 20%
        
        signal = Signal(
            signal_id="test",
            timestamp=datetime.now(),
            strategy_id="test",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=0.8,
            metadata={'price': 150.0}
        )
        
        portfolio_state = PortfolioState(initial_capital=Decimal("100000"))
        
        # Check position that would be 15% of portfolio - should pass
        # 100 shares * $150 = $15,000 = 15% of $100k
        passed, reason = limit.check(signal, 100, portfolio_state)
        assert passed is True
        
        # Check position that would be 25% of portfolio - should fail
        # 167 shares * $150 = $25,050 = 25% of $100k
        passed, reason = limit.check(signal, 167, portfolio_state)
        assert passed is False
        assert "exposure" in reason.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])