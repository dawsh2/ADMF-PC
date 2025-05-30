"""
Unit tests for portfolio state management.

Tests position tracking, PnL calculation, and state updates.
"""

import unittest
from datetime import datetime
from decimal import Decimal
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.risk.portfolio_state import PortfolioState
from src.risk.protocols import Position, RiskMetrics


class TestPortfolioState(unittest.TestCase):
    """Test PortfolioState functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.portfolio = PortfolioState(
            initial_capital=Decimal("100000"),
            base_currency="USD"
        )
    
    def test_initial_state(self):
        """Test initial portfolio state."""
        self.assertEqual(self.portfolio.get_cash_balance(), Decimal("100000"))
        self.assertEqual(len(self.portfolio.get_all_positions()), 0)
        self.assertEqual(self.portfolio.get_total_value(), Decimal("100000"))
    
    def test_update_position_buy(self):
        """Test updating position with buy order."""
        # Buy 100 shares at $150
        position = self.portfolio.update_position(
            symbol="AAPL",
            quantity_delta=Decimal("100"),
            price=Decimal("150.00"),
            timestamp=datetime.now()
        )
        
        self.assertEqual(position.symbol, "AAPL")
        self.assertEqual(position.quantity, Decimal("100"))
        self.assertEqual(position.average_price, Decimal("150.00"))
        self.assertEqual(position.market_value, Decimal("15000.00"))
        
        # Cash should be reduced
        expected_cash = Decimal("100000") - Decimal("15000")
        self.assertEqual(self.portfolio.get_cash_balance(), expected_cash)
    
    def test_update_position_sell(self):
        """Test updating position with sell order."""
        # First buy
        self.portfolio.update_position(
            symbol="AAPL",
            quantity_delta=Decimal("100"),
            price=Decimal("150.00"),
            timestamp=datetime.now()
        )
        
        # Then sell 50 shares at $155
        position = self.portfolio.update_position(
            symbol="AAPL",
            quantity_delta=Decimal("-50"),
            price=Decimal("155.00"),
            timestamp=datetime.now()
        )
        
        self.assertEqual(position.quantity, Decimal("50"))
        self.assertEqual(position.average_price, Decimal("150.00"))  # Unchanged
        
        # Cash should increase
        expected_cash = Decimal("85000") + Decimal("7750")  # Initial - buy + sell
        self.assertEqual(self.portfolio.get_cash_balance(), expected_cash)
        
        # Should have realized profit
        self.assertGreater(self.portfolio._realized_pnl, 0)
    
    def test_close_position(self):
        """Test closing a position completely."""
        # Buy position
        self.portfolio.update_position(
            symbol="AAPL",
            quantity_delta=Decimal("100"),
            price=Decimal("150.00"),
            timestamp=datetime.now()
        )
        
        # Close position
        position = self.portfolio.update_position(
            symbol="AAPL",
            quantity_delta=Decimal("-100"),
            price=Decimal("160.00"),
            timestamp=datetime.now()
        )
        
        # Position should be removed
        self.assertIsNone(self.portfolio.get_position("AAPL"))
        
        # Should have realized profit
        expected_pnl = Decimal("100") * (Decimal("160.00") - Decimal("150.00"))
        self.assertEqual(self.portfolio._realized_pnl, expected_pnl)
    
    def test_multiple_positions(self):
        """Test managing multiple positions."""
        # Buy multiple stocks
        self.portfolio.update_position("AAPL", Decimal("100"), Decimal("150.00"), datetime.now())
        self.portfolio.update_position("GOOGL", Decimal("50"), Decimal("2800.00"), datetime.now())
        self.portfolio.update_position("MSFT", Decimal("75"), Decimal("350.00"), datetime.now())
        
        positions = self.portfolio.get_all_positions()
        self.assertEqual(len(positions), 3)
        
        # Check individual positions
        self.assertEqual(self.portfolio.get_position("AAPL").quantity, Decimal("100"))
        self.assertEqual(self.portfolio.get_position("GOOGL").quantity, Decimal("50"))
        self.assertEqual(self.portfolio.get_position("MSFT").quantity, Decimal("75"))
    
    def test_update_market_prices(self):
        """Test updating market prices and unrealized PnL."""
        # Create positions
        self.portfolio.update_position("AAPL", Decimal("100"), Decimal("150.00"), datetime.now())
        self.portfolio.update_position("GOOGL", Decimal("50"), Decimal("2800.00"), datetime.now())
        
        # Update market prices
        new_prices = {
            "AAPL": Decimal("155.00"),
            "GOOGL": Decimal("2850.00")
        }
        self.portfolio.update_market_prices(new_prices)
        
        # Check updated values
        aapl_pos = self.portfolio.get_position("AAPL")
        self.assertEqual(aapl_pos.current_price, Decimal("155.00"))
        self.assertEqual(aapl_pos.unrealized_pnl, Decimal("500.00"))  # 100 * (155-150)
        
        googl_pos = self.portfolio.get_position("GOOGL")
        self.assertEqual(googl_pos.current_price, Decimal("2850.00"))
        self.assertEqual(googl_pos.unrealized_pnl, Decimal("2500.00"))  # 50 * (2850-2800)
    
    def test_risk_metrics(self):
        """Test risk metrics calculation."""
        # Create positions with some unrealized gains
        self.portfolio.update_position("AAPL", Decimal("100"), Decimal("150.00"), datetime.now())
        self.portfolio.update_position("GOOGL", Decimal("50"), Decimal("2800.00"), datetime.now())
        
        # Update prices for unrealized gains
        self.portfolio.update_market_prices({
            "AAPL": Decimal("155.00"),
            "GOOGL": Decimal("2850.00")
        })
        
        # Sell some for realized gains
        self.portfolio.update_position("AAPL", Decimal("-50"), Decimal("155.00"), datetime.now())
        
        metrics = self.portfolio.get_risk_metrics()
        
        self.assertIsInstance(metrics, RiskMetrics)
        self.assertGreater(metrics.total_value, Decimal("100000"))  # Should have gains
        self.assertGreater(metrics.unrealized_pnl, 0)
        self.assertGreater(metrics.realized_pnl, 0)
        self.assertEqual(metrics.position_count, 2)
    
    def test_leverage_calculation(self):
        """Test leverage calculation."""
        # Use most of capital
        self.portfolio.update_position("AAPL", Decimal("600"), Decimal("150.00"), datetime.now())
        
        metrics = self.portfolio.get_risk_metrics()
        
        # Leverage = positions_value / total_value
        # positions_value = 600 * 150 = 90000
        # cash = 100000 - 90000 = 10000
        # total_value = 10000 + 90000 = 100000
        # leverage = 90000 / 100000 = 0.9
        self.assertEqual(metrics.leverage, Decimal("0.9"))
    
    def test_drawdown_tracking(self):
        """Test drawdown calculation."""
        # Record initial high water mark
        self.portfolio._update_high_water_mark()
        
        # Make profitable trade
        self.portfolio.update_position("AAPL", Decimal("100"), Decimal("150.00"), datetime.now())
        self.portfolio.update_market_prices({"AAPL": Decimal("160.00")})
        self.portfolio._update_high_water_mark()
        
        # Now have a loss
        self.portfolio.update_market_prices({"AAPL": Decimal("140.00")})
        
        metrics = self.portfolio.get_risk_metrics()
        
        # Should have drawdown
        self.assertGreater(metrics.current_drawdown, 0)
        self.assertLess(metrics.current_drawdown, Decimal("0.1"))  # Less than 10%
    
    def test_short_positions(self):
        """Test handling short positions."""
        # Short 100 shares at $150
        position = self.portfolio.update_position(
            symbol="AAPL",
            quantity_delta=Decimal("-100"),
            price=Decimal("150.00"),
            timestamp=datetime.now()
        )
        
        self.assertEqual(position.quantity, Decimal("-100"))
        self.assertEqual(position.average_price, Decimal("150.00"))
        
        # Cash should increase from short sale
        expected_cash = Decimal("100000") + Decimal("15000")
        self.assertEqual(self.portfolio.get_cash_balance(), expected_cash)
        
        # Cover short at lower price for profit
        self.portfolio.update_position(
            symbol="AAPL",
            quantity_delta=Decimal("100"),
            price=Decimal("145.00"),
            timestamp=datetime.now()
        )
        
        # Should have realized profit
        expected_pnl = Decimal("100") * (Decimal("150.00") - Decimal("145.00"))
        self.assertEqual(self.portfolio._realized_pnl, expected_pnl)
    
    def test_position_sizing_info(self):
        """Test getting position sizing information."""
        # Create some positions
        self.portfolio.update_position("AAPL", Decimal("100"), Decimal("150.00"), datetime.now())
        self.portfolio.update_position("GOOGL", Decimal("50"), Decimal("2800.00"), datetime.now())
        
        # Get sizing info
        sizing_info = self.portfolio.get_position_sizing_info()
        
        self.assertIn("cash_available", sizing_info)
        self.assertIn("buying_power", sizing_info)
        self.assertIn("positions_value", sizing_info)
        self.assertIn("position_count", sizing_info)
        
        # Check values
        self.assertLess(sizing_info["cash_available"], Decimal("100000"))
        self.assertEqual(sizing_info["position_count"], 2)
    
    def test_thread_safety(self):
        """Test thread-safe operations."""
        import threading
        
        def buy_stock():
            for _ in range(100):
                self.portfolio.update_position(
                    "AAPL",
                    Decimal("1"),
                    Decimal("150.00"),
                    datetime.now()
                )
        
        def sell_stock():
            for _ in range(50):
                self.portfolio.update_position(
                    "AAPL",
                    Decimal("-1"),
                    Decimal("151.00"),
                    datetime.now()
                )
        
        # Run concurrent operations
        threads = []
        for _ in range(3):
            t1 = threading.Thread(target=buy_stock)
            t2 = threading.Thread(target=sell_stock)
            threads.extend([t1, t2])
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Final position should be consistent
        position = self.portfolio.get_position("AAPL")
        if position:
            # Net should be 300 buys - 150 sells = 150
            self.assertEqual(position.quantity, Decimal("150"))
    
    def test_export_state(self):
        """Test exporting portfolio state."""
        # Create some state
        self.portfolio.update_position("AAPL", Decimal("100"), Decimal("150.00"), datetime.now())
        self.portfolio.update_position("GOOGL", Decimal("50"), Decimal("2800.00"), datetime.now())
        
        # Export state
        state = self.portfolio.export_state()
        
        self.assertIn("cash_balance", state)
        self.assertIn("positions", state)
        self.assertIn("realized_pnl", state)
        self.assertIn("timestamp", state)
        
        # Check positions are exported
        self.assertEqual(len(state["positions"]), 2)
        self.assertIn("AAPL", state["positions"])
        self.assertIn("GOOGL", state["positions"])


if __name__ == "__main__":
    unittest.main()