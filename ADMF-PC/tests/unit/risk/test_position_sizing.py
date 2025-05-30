"""
Unit tests for position sizing strategies.

Tests various position sizing algorithms including fixed, percentage,
Kelly criterion, volatility-based, and ATR-based sizing.
"""

import unittest
from decimal import Decimal
from unittest.mock import Mock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.risk.position_sizing import (
    FixedPositionSizer,
    PercentagePositionSizer,
    KellyCriterionSizer,
    VolatilityBasedSizer,
    ATRBasedSizer
)
from src.risk.protocols import Signal, SignalType, OrderSide


class MockPortfolioState:
    """Mock portfolio state for testing."""
    
    def __init__(self, total_value=Decimal("100000"), cash_balance=Decimal("50000")):
        self.total_value = total_value
        self.cash_balance = cash_balance
    
    def get_total_value(self):
        return self.total_value
    
    def get_cash_balance(self):
        return self.cash_balance


class TestFixedPositionSizer(unittest.TestCase):
    """Test fixed position sizing strategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sizer = FixedPositionSizer(size=Decimal("100"))
        self.portfolio = MockPortfolioState()
        self.signal = Signal(
            signal_id="test_001",
            strategy_id="test",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.8"),
            timestamp=None,
            metadata={}
        )
        self.market_data = {"prices": {"AAPL": 150.00}}
    
    def test_fixed_size_calculation(self):
        """Test basic fixed size calculation."""
        size = self.sizer.calculate_size(
            self.signal, self.portfolio, self.market_data
        )
        
        # Should be fixed size * signal strength
        expected = Decimal("100") * Decimal("0.8")
        self.assertEqual(size, expected)
    
    def test_signal_strength_adjustment(self):
        """Test that size is adjusted by signal strength."""
        # Test with different signal strengths
        strengths = [Decimal("0.2"), Decimal("0.5"), Decimal("1.0")]
        
        for strength in strengths:
            signal = Signal(
                signal_id="test",
                strategy_id="test",
                symbol="AAPL",
                signal_type=SignalType.ENTRY,
                side=OrderSide.BUY,
                strength=strength,
                timestamp=None,
                metadata={}
            )
            
            size = self.sizer.calculate_size(signal, self.portfolio, self.market_data)
            expected = Decimal("100") * strength
            self.assertEqual(size, expected)
    
    def test_minimum_size_constraint(self):
        """Test minimum size constraint."""
        # Very weak signal
        weak_signal = Signal(
            signal_id="test",
            strategy_id="test",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.0001"),
            timestamp=None,
            metadata={}
        )
        
        size = self.sizer.calculate_size(weak_signal, self.portfolio, self.market_data)
        
        # Should be 0 due to minimum size constraint
        self.assertEqual(size, Decimal("0"))
    
    def test_size_quantization(self):
        """Test that sizes are properly quantized."""
        signal = Signal(
            signal_id="test",
            strategy_id="test",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.789"),
            timestamp=None,
            metadata={}
        )
        
        size = self.sizer.calculate_size(signal, self.portfolio, self.market_data)
        
        # Should be quantized to 2 decimal places
        self.assertEqual(size, Decimal("78.90"))


class TestPercentagePositionSizer(unittest.TestCase):
    """Test percentage-based position sizing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sizer = PercentagePositionSizer(
            percentage=Decimal("0.02"),  # 2% of portfolio
            use_leverage=False
        )
        self.portfolio = MockPortfolioState(
            total_value=Decimal("100000"),
            cash_balance=Decimal("50000")
        )
        self.signal = Signal(
            signal_id="test_001",
            strategy_id="test",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("1.0"),
            timestamp=None,
            metadata={}
        )
    
    def test_percentage_calculation(self):
        """Test basic percentage calculation."""
        market_data = {"prices": {"AAPL": 100.00}}
        
        size = self.sizer.calculate_size(self.signal, self.portfolio, market_data)
        
        # 2% of 100,000 = 2,000 / 100 = 20 shares
        expected = Decimal("20.00")
        self.assertEqual(size, expected)
    
    def test_no_leverage_constraint(self):
        """Test that position is limited by cash when leverage disabled."""
        # Large percentage that would exceed cash
        sizer = PercentagePositionSizer(
            percentage=Decimal("0.8"),  # 80% of portfolio
            use_leverage=False
        )
        
        market_data = {"prices": {"AAPL": 100.00}}
        
        size = sizer.calculate_size(self.signal, self.portfolio, market_data)
        
        # Should be limited to cash: 50,000 / 100 = 500 shares
        expected = Decimal("500.00")
        self.assertEqual(size, expected)
    
    def test_leverage_allowed(self):
        """Test position sizing when leverage is allowed."""
        sizer = PercentagePositionSizer(
            percentage=Decimal("0.8"),  # 80% of portfolio
            use_leverage=True
        )
        
        market_data = {"prices": {"AAPL": 100.00}}
        
        size = sizer.calculate_size(self.signal, self.portfolio, market_data)
        
        # Should allow full 80%: 80,000 / 100 = 800 shares
        expected = Decimal("800.00")
        self.assertEqual(size, expected)
    
    def test_missing_price_data(self):
        """Test handling of missing price data."""
        market_data = {"prices": {}}  # No price for AAPL
        
        size = self.sizer.calculate_size(self.signal, self.portfolio, market_data)
        
        # Should return 0
        self.assertEqual(size, Decimal("0"))
    
    def test_signal_strength_scaling(self):
        """Test that position scales with signal strength."""
        market_data = {"prices": {"AAPL": 100.00}}
        
        for strength in [Decimal("0.2"), Decimal("0.5"), Decimal("1.0")]:
            signal = Signal(
                signal_id="test",
                strategy_id="test",
                symbol="AAPL",
                signal_type=SignalType.ENTRY,
                side=OrderSide.BUY,
                strength=strength,
                timestamp=None,
                metadata={}
            )
            
            size = self.sizer.calculate_size(signal, self.portfolio, market_data)
            
            # Base: 2% of 100k = 2k / 100 = 20 shares
            expected = Decimal("20.00") * strength
            self.assertEqual(size, expected.quantize(Decimal("0.01")))


class TestKellyCriterionSizer(unittest.TestCase):
    """Test Kelly Criterion position sizing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sizer = KellyCriterionSizer(
            win_rate=Decimal("0.6"),      # 60% win rate
            avg_win=Decimal("100"),       # Average win $100
            avg_loss=Decimal("50"),       # Average loss $50
            kelly_fraction=Decimal("0.25"), # Use 25% of Kelly
            max_percentage=Decimal("0.25")  # Max 25% of portfolio
        )
        self.portfolio = MockPortfolioState()
        self.signal = Signal(
            signal_id="test_001",
            strategy_id="test",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("1.0"),
            timestamp=None,
            metadata={}
        )
    
    def test_kelly_calculation(self):
        """Test Kelly Criterion calculation."""
        market_data = {"prices": {"AAPL": 100.00}}
        
        size = self.sizer.calculate_size(self.signal, self.portfolio, market_data)
        
        # Kelly calculation:
        # b = 100/50 = 2
        # p = 0.6, q = 0.4
        # f = (0.6*2 - 0.4) / 2 = 0.8 / 2 = 0.4
        # With 25% Kelly fraction: 0.4 * 0.25 = 0.1
        # 10% of 100k = 10k / 100 = 100 shares
        expected = Decimal("100.00")
        self.assertEqual(size, expected)
    
    def test_max_percentage_constraint(self):
        """Test maximum percentage constraint."""
        # High Kelly with low max percentage
        sizer = KellyCriterionSizer(
            win_rate=Decimal("0.8"),      # 80% win rate
            avg_win=Decimal("200"),       # Big wins
            avg_loss=Decimal("50"),       # Small losses
            kelly_fraction=Decimal("1.0"), # Full Kelly
            max_percentage=Decimal("0.10")  # But max 10%
        )
        
        market_data = {"prices": {"AAPL": 100.00}}
        
        size = sizer.calculate_size(self.signal, self.portfolio, market_data)
        
        # Should be capped at 10%: 10k / 100 = 100 shares
        expected = Decimal("100.00")
        self.assertEqual(size, expected)
    
    def test_negative_kelly(self):
        """Test handling of negative Kelly (don't bet)."""
        # Poor odds
        sizer = KellyCriterionSizer(
            win_rate=Decimal("0.3"),      # 30% win rate
            avg_win=Decimal("100"),       # Same win
            avg_loss=Decimal("100"),      # Same loss
            kelly_fraction=Decimal("0.25"),
            max_percentage=Decimal("0.25")
        )
        
        market_data = {"prices": {"AAPL": 100.00}}
        
        size = sizer.calculate_size(self.signal, self.portfolio, market_data)
        
        # Kelly would be negative, but max(0, kelly) should apply
        self.assertEqual(size, Decimal("0"))
    
    def test_signal_strength_scaling(self):
        """Test Kelly scales with signal strength."""
        market_data = {"prices": {"AAPL": 100.00}}
        
        # Test with 50% signal strength
        weak_signal = Signal(
            signal_id="test",
            strategy_id="test",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.5"),
            timestamp=None,
            metadata={}
        )
        
        full_size = self.sizer.calculate_size(self.signal, self.portfolio, market_data)
        half_size = self.sizer.calculate_size(weak_signal, self.portfolio, market_data)
        
        # Half strength should give half size
        self.assertEqual(half_size, full_size * Decimal("0.5"))


class TestVolatilityBasedSizer(unittest.TestCase):
    """Test volatility-based position sizing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sizer = VolatilityBasedSizer(
            target_volatility=Decimal("0.15"),  # 15% target vol
            lookback_days=20,
            max_percentage=Decimal("0.25")
        )
        self.portfolio = MockPortfolioState()
        self.signal = Signal(
            signal_id="test_001",
            strategy_id="test",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("1.0"),
            timestamp=None,
            metadata={}
        )
    
    def test_volatility_sizing(self):
        """Test sizing based on volatility."""
        # Asset with 30% volatility
        market_data = {
            "prices": {"AAPL": 100.00},
            "volatility": {"AAPL": 0.30}
        }
        
        size = self.sizer.calculate_size(self.signal, self.portfolio, market_data)
        
        # Target 15% / Asset 30% = 0.5 position size
        # But capped at 25%
        # 25% of 100k = 25k / 100 = 250 shares
        expected = Decimal("250.00")
        self.assertEqual(size, expected)
    
    def test_low_volatility_asset(self):
        """Test sizing for low volatility asset."""
        # Asset with 5% volatility
        market_data = {
            "prices": {"AAPL": 100.00},
            "volatility": {"AAPL": 0.05}
        }
        
        size = self.sizer.calculate_size(self.signal, self.portfolio, market_data)
        
        # Target 15% / Asset 5% = 3.0, but capped at 25%
        # 25% of 100k = 25k / 100 = 250 shares
        expected = Decimal("250.00")
        self.assertEqual(size, expected)
    
    def test_missing_volatility_fallback(self):
        """Test fallback when volatility data missing."""
        market_data = {
            "prices": {"AAPL": 100.00},
            "volatility": {}  # No volatility data
        }
        
        size = self.sizer.calculate_size(self.signal, self.portfolio, market_data)
        
        # Should fallback to 2% sizing
        # 2% of 100k = 2k / 100 = 20 shares
        expected = Decimal("20.00")
        self.assertEqual(size, expected)
    
    def test_volatility_with_signal_strength(self):
        """Test volatility sizing with signal strength adjustment."""
        market_data = {
            "prices": {"AAPL": 100.00},
            "volatility": {"AAPL": 0.15}  # Matches target
        }
        
        # Test with 60% signal strength
        weak_signal = Signal(
            signal_id="test",
            strategy_id="test",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.6"),
            timestamp=None,
            metadata={}
        )
        
        size = self.sizer.calculate_size(weak_signal, self.portfolio, market_data)
        
        # 15% / 15% = 1.0, capped at 25%, then * 0.6 = 15%
        # 15% of 100k = 15k / 100 = 150 shares
        expected = Decimal("150.00")
        self.assertEqual(size, expected)


class TestATRBasedSizer(unittest.TestCase):
    """Test ATR-based position sizing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sizer = ATRBasedSizer(
            risk_per_trade=Decimal("0.01"),    # 1% risk per trade
            atr_multiplier=Decimal("2"),       # 2x ATR for stop
            max_percentage=Decimal("0.25")
        )
        self.portfolio = MockPortfolioState()
        self.signal = Signal(
            signal_id="test_001",
            strategy_id="test",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("1.0"),
            timestamp=None,
            metadata={}
        )
    
    def test_atr_sizing_calculation(self):
        """Test ATR-based sizing calculation."""
        market_data = {
            "prices": {"AAPL": 100.00},
            "atr": {"AAPL": 2.00}  # $2 ATR
        }
        
        size = self.sizer.calculate_size(self.signal, self.portfolio, market_data)
        
        # Risk amount: 1% of 100k = 1000
        # Stop distance: 2 * 2.00 = 4.00
        # Shares: 1000 / 4.00 = 250
        expected = Decimal("250.00")
        self.assertEqual(size, expected)
    
    def test_max_percentage_constraint(self):
        """Test maximum percentage constraint on ATR sizing."""
        market_data = {
            "prices": {"AAPL": 100.00},
            "atr": {"AAPL": 0.5}  # Very small ATR
        }
        
        size = self.sizer.calculate_size(self.signal, self.portfolio, market_data)
        
        # Risk: 1% of 100k = 1000
        # Stop: 2 * 0.5 = 1.00
        # Shares: 1000 / 1.00 = 1000
        # But position value: 1000 * 100 = 100k (100% of portfolio!)
        # Should be capped at 25%: 25k / 100 = 250 shares
        expected = Decimal("250.00")
        self.assertEqual(size, expected)
    
    def test_missing_atr_fallback(self):
        """Test fallback when ATR data missing."""
        market_data = {
            "prices": {"AAPL": 100.00},
            "atr": {}  # No ATR data
        }
        
        size = self.sizer.calculate_size(self.signal, self.portfolio, market_data)
        
        # Should fallback to 2.5% sizing (10% of max 25%)
        # 2.5% of 100k = 2.5k / 100 = 25 shares
        expected = Decimal("25.00")
        self.assertEqual(size, expected)
    
    def test_signal_strength_adjustment(self):
        """Test ATR sizing with signal strength."""
        market_data = {
            "prices": {"AAPL": 100.00},
            "atr": {"AAPL": 2.00}
        }
        
        # Test with 40% signal strength
        weak_signal = Signal(
            signal_id="test",
            strategy_id="test",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("0.4"),
            timestamp=None,
            metadata={}
        )
        
        size = self.sizer.calculate_size(weak_signal, self.portfolio, market_data)
        
        # Base size: 250 shares * 0.4 = 100 shares
        expected = Decimal("100.00")
        self.assertEqual(size, expected)
    
    def test_zero_atr_handling(self):
        """Test handling of zero ATR."""
        market_data = {
            "prices": {"AAPL": 100.00},
            "atr": {"AAPL": 0.00}  # Zero ATR
        }
        
        # Should use fallback to avoid division by zero
        size = self.sizer.calculate_size(self.signal, self.portfolio, market_data)
        
        # Fallback sizing
        expected = Decimal("25.00")
        self.assertEqual(size, expected)


class TestPositionSizerEdgeCases(unittest.TestCase):
    """Test edge cases across all position sizers."""
    
    def test_zero_portfolio_value(self):
        """Test sizing with zero portfolio value."""
        portfolio = MockPortfolioState(
            total_value=Decimal("0"),
            cash_balance=Decimal("0")
        )
        
        signal = Signal(
            signal_id="test",
            strategy_id="test",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("1.0"),
            timestamp=None,
            metadata={}
        )
        
        market_data = {"prices": {"AAPL": 100.00}}
        
        # Test all sizers with zero portfolio
        sizers = [
            FixedPositionSizer(size=Decimal("100")),
            PercentagePositionSizer(percentage=Decimal("0.02")),
            KellyCriterionSizer(
                win_rate=Decimal("0.6"),
                avg_win=Decimal("100"),
                avg_loss=Decimal("50")
            ),
            VolatilityBasedSizer(target_volatility=Decimal("0.15")),
            ATRBasedSizer(risk_per_trade=Decimal("0.01"))
        ]
        
        for sizer in sizers:
            if isinstance(sizer, FixedPositionSizer):
                # Fixed sizer doesn't depend on portfolio value
                size = sizer.calculate_size(signal, portfolio, market_data)
                self.assertEqual(size, Decimal("100.00"))
            else:
                # Others should return 0
                size = sizer.calculate_size(signal, portfolio, market_data)
                self.assertEqual(size, Decimal("0"))
    
    def test_very_high_price(self):
        """Test sizing with very high asset price."""
        portfolio = MockPortfolioState()
        
        signal = Signal(
            signal_id="test",
            strategy_id="test",
            symbol="BRK.A",  # Berkshire Hathaway A
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            strength=Decimal("1.0"),
            timestamp=None,
            metadata={}
        )
        
        # Very high price
        market_data = {"prices": {"BRK.A": 500000.00}}
        
        # Percentage sizer with 2%
        sizer = PercentagePositionSizer(percentage=Decimal("0.02"))
        size = sizer.calculate_size(signal, portfolio, market_data)
        
        # 2% of 100k = 2k / 500k = 0.004 shares
        # Should round to 0 due to minimum size
        self.assertEqual(size, Decimal("0"))
    
    def test_negative_signal_strength(self):
        """Test that negative signal strength is handled correctly."""
        portfolio = MockPortfolioState()
        
        # Signal with negative strength (shouldn't happen but test anyway)
        signal = Signal(
            signal_id="test",
            strategy_id="test",
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            side=OrderSide.SELL,
            strength=Decimal("-0.8"),
            timestamp=None,
            metadata={}
        )
        
        market_data = {"prices": {"AAPL": 100.00}}
        
        # Test with percentage sizer
        sizer = PercentagePositionSizer(percentage=Decimal("0.02"))
        size = sizer.calculate_size(signal, portfolio, market_data)
        
        # Should use absolute value of strength
        # 2% of 100k = 2k / 100 = 20 * 0.8 = 16 shares
        expected = Decimal("16.00")
        self.assertEqual(size, expected)


if __name__ == "__main__":
    unittest.main()