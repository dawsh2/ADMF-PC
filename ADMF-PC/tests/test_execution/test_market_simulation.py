"""
Tests for market simulation components.

Tests cover:
- Slippage models
- Commission models
- Fill simulation
- Partial fills
- Market conditions
"""

import unittest
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, patch
import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.execution.market_simulation import (
    MarketSimulator, SlippageModel, CommissionModel,
    FixedSlippageModel, PercentageSlippageModel, VolumeBasedSlippageModel,
    FixedCommissionModel, PerShareCommissionModel, PercentageCommissionModel
)
from src.execution.protocols import Order, Fill, OrderType, OrderSide, FillType


class TestSlippageModels(unittest.TestCase):
    """Test different slippage models."""
    
    def test_fixed_slippage_model(self):
        """Test fixed slippage model."""
        model = FixedSlippageModel(amount=Decimal("0.01"))
        
        # Buy order - add slippage
        buy_slippage = model.calculate_slippage(
            OrderSide.BUY,
            Decimal("100.00"),
            Decimal("1000"),  # quantity
            1000000  # volume
        )
        self.assertEqual(buy_slippage, Decimal("0.01"))
        
        # Sell order - subtract slippage
        sell_slippage = model.calculate_slippage(
            OrderSide.SELL,
            Decimal("100.00"),
            Decimal("1000"),
            1000000
        )
        self.assertEqual(sell_slippage, Decimal("-0.01"))
    
    def test_percentage_slippage_model(self):
        """Test percentage slippage model."""
        model = PercentageSlippageModel(percentage=Decimal("0.001"))  # 0.1%
        
        # Buy order
        buy_slippage = model.calculate_slippage(
            OrderSide.BUY,
            Decimal("100.00"),
            Decimal("1000"),
            1000000
        )
        expected = Decimal("100.00") * Decimal("0.001")
        self.assertEqual(buy_slippage, expected)
        
        # Different price
        buy_slippage = model.calculate_slippage(
            OrderSide.BUY,
            Decimal("50.00"),
            Decimal("1000"),
            1000000
        )
        expected = Decimal("50.00") * Decimal("0.001")
        self.assertEqual(buy_slippage, expected)
    
    def test_volume_based_slippage_model(self):
        """Test volume-based slippage model."""
        model = VolumeBasedSlippageModel(
            base_spread=Decimal("0.01"),
            impact_factor=Decimal("0.1")
        )
        
        # Small order relative to volume
        small_slippage = model.calculate_slippage(
            OrderSide.BUY,
            Decimal("100.00"),
            Decimal("100"),  # 0.01% of volume
            1000000
        )
        # Should be close to base spread
        self.assertLess(small_slippage, Decimal("0.02"))
        
        # Large order relative to volume
        large_slippage = model.calculate_slippage(
            OrderSide.BUY,
            Decimal("100.00"),
            Decimal("50000"),  # 5% of volume
            1000000
        )
        # Should be significantly higher
        self.assertGreater(large_slippage, Decimal("0.05"))
        
        # Sell order should be negative
        sell_slippage = model.calculate_slippage(
            OrderSide.SELL,
            Decimal("100.00"),
            Decimal("1000"),
            1000000
        )
        self.assertLess(sell_slippage, Decimal("0"))


class TestCommissionModels(unittest.TestCase):
    """Test different commission models."""
    
    def test_fixed_commission_model(self):
        """Test fixed commission model."""
        model = FixedCommissionModel(amount=Decimal("5.00"))
        
        commission = model.calculate_commission(
            Decimal("100"),  # quantity
            Decimal("50.00")  # price
        )
        self.assertEqual(commission, Decimal("5.00"))
        
        # Same commission regardless of size
        commission = model.calculate_commission(
            Decimal("1000"),
            Decimal("100.00")
        )
        self.assertEqual(commission, Decimal("5.00"))
    
    def test_per_share_commission_model(self):
        """Test per-share commission model."""
        model = PerShareCommissionModel(
            rate=Decimal("0.01"),
            minimum=Decimal("1.00"),
            maximum=Decimal("100.00")
        )
        
        # Small order - hits minimum
        commission = model.calculate_commission(
            Decimal("50"),
            Decimal("20.00")
        )
        self.assertEqual(commission, Decimal("1.00"))  # minimum
        
        # Normal order
        commission = model.calculate_commission(
            Decimal("500"),
            Decimal("20.00")
        )
        self.assertEqual(commission, Decimal("5.00"))  # 500 * 0.01
        
        # Large order - hits maximum
        commission = model.calculate_commission(
            Decimal("20000"),
            Decimal("20.00")
        )
        self.assertEqual(commission, Decimal("100.00"))  # maximum
    
    def test_percentage_commission_model(self):
        """Test percentage commission model."""
        model = PercentageCommissionModel(
            percentage=Decimal("0.001"),  # 0.1%
            minimum=Decimal("5.00")
        )
        
        # Small order - hits minimum
        commission = model.calculate_commission(
            Decimal("100"),
            Decimal("10.00")  # $1000 notional
        )
        self.assertEqual(commission, Decimal("5.00"))  # minimum
        
        # Large order
        commission = model.calculate_commission(
            Decimal("1000"),
            Decimal("100.00")  # $100,000 notional
        )
        expected = Decimal("100000") * Decimal("0.001")
        self.assertEqual(commission, expected)


class TestMarketSimulator(unittest.TestCase):
    """Test the MarketSimulator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simulator = MarketSimulator(
            slippage_model="percentage",
            slippage_params={"percentage": 0.001},
            commission_model="percentage",
            commission_params={"percentage": 0.001, "minimum": 1.00}
        )
    
    def test_simulator_initialization(self):
        """Test simulator initialization with different models."""
        # Test with string model names
        sim1 = MarketSimulator(
            slippage_model="fixed",
            slippage_params={"amount": 0.01},
            commission_model="fixed",
            commission_params={"amount": 5.00}
        )
        self.assertIsInstance(sim1.slippage_model, FixedSlippageModel)
        self.assertIsInstance(sim1.commission_model, FixedCommissionModel)
        
        # Test with model instances
        slippage = PercentageSlippageModel(percentage=Decimal("0.002"))
        commission = PerShareCommissionModel(rate=Decimal("0.01"))
        sim2 = MarketSimulator(
            slippage_model=slippage,
            commission_model=commission
        )
        self.assertEqual(sim2.slippage_model, slippage)
        self.assertEqual(sim2.commission_model, commission)
    
    def test_simulate_market_fill_complete(self):
        """Test complete fill simulation for market order."""
        async def run_test():
            order = Order(
                order_id="TEST_001",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("100"),
                timestamp=datetime.now()
            )
            
            market_data = {
                "symbol": "AAPL",
                "bid": 149.95,
                "ask": 150.05,
                "last": 150.00,
                "volume": 10000000
            }
            
            fill = await self.simulator.simulate_fill(order, market_data)
            
            self.assertIsNotNone(fill)
            self.assertEqual(fill.symbol, "AAPL")
            self.assertEqual(fill.side, OrderSide.BUY)
            self.assertEqual(fill.quantity, Decimal("100"))
            self.assertEqual(fill.fill_type, FillType.COMPLETE)
            
            # Price should be ask + slippage for buy
            expected_slippage = Decimal("150.05") * Decimal("0.001")
            expected_price = Decimal("150.05") + expected_slippage
            self.assertAlmostEqual(
                float(fill.price),
                float(expected_price),
                places=4
            )
            
            # Commission should be percentage of notional
            notional = fill.quantity * fill.price
            expected_commission = notional * Decimal("0.001")
            self.assertAlmostEqual(
                float(fill.commission),
                float(expected_commission),
                places=4
            )
        
        asyncio.run(run_test())
    
    def test_simulate_limit_fill_no_fill(self):
        """Test limit order that shouldn't fill."""
        async def run_test():
            # Buy limit below market
            order = Order(
                order_id="TEST_002",
                symbol="GOOGL",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("50"),
                price=Decimal("2700.00"),
                timestamp=datetime.now()
            )
            
            market_data = {
                "symbol": "GOOGL",
                "bid": 2795.00,
                "ask": 2805.00,
                "last": 2800.00
            }
            
            fill = await self.simulator.simulate_fill(order, market_data)
            self.assertIsNone(fill)  # Should not fill
        
        asyncio.run(run_test())
    
    def test_simulate_limit_fill_marketable(self):
        """Test limit order that should fill."""
        async def run_test():
            # Sell limit at market
            order = Order(
                order_id="TEST_003",
                symbol="MSFT",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=Decimal("200"),
                price=Decimal("380.00"),
                timestamp=datetime.now()
            )
            
            market_data = {
                "symbol": "MSFT",
                "bid": 380.00,
                "ask": 380.10,
                "last": 380.05,
                "volume": 5000000
            }
            
            fill = await self.simulator.simulate_fill(order, market_data)
            
            self.assertIsNotNone(fill)
            # Should fill at limit price (no price improvement in simulation)
            self.assertEqual(fill.price, Decimal("380.00"))
        
        asyncio.run(run_test())
    
    def test_simulate_stop_fill(self):
        """Test stop order fill simulation."""
        async def run_test():
            # Stop loss order
            order = Order(
                order_id="TEST_004",
                symbol="TSLA",
                side=OrderSide.SELL,
                order_type=OrderType.STOP,
                quantity=Decimal("100"),
                stop_price=Decimal("800.00"),
                timestamp=datetime.now()
            )
            
            # Market above stop - should not trigger
            market_data = {
                "symbol": "TSLA",
                "bid": 810.00,
                "ask": 810.50,
                "last": 810.25
            }
            
            fill = await self.simulator.simulate_fill(order, market_data)
            self.assertIsNone(fill)
            
            # Market at stop - should trigger and fill
            market_data["last"] = 799.50
            market_data["bid"] = 799.50
            market_data["ask"] = 800.00
            
            fill = await self.simulator.simulate_fill(order, market_data)
            self.assertIsNotNone(fill)
            
            # Should fill at bid - slippage for sell stop
            expected_slippage = Decimal("799.50") * Decimal("0.001")
            expected_price = Decimal("799.50") - expected_slippage
            self.assertAlmostEqual(
                float(fill.price),
                float(expected_price),
                places=2
            )
        
        asyncio.run(run_test())
    
    def test_partial_fill_simulation(self):
        """Test partial fill simulation."""
        # Create simulator with partial fill enabled
        simulator = MarketSimulator(
            slippage_model="fixed",
            slippage_params={"amount": 0.01},
            commission_model="fixed",
            commission_params={"amount": 1.00},
            enable_partial_fills=True,
            partial_fill_prob=1.0  # Always partial for testing
        )
        
        async def run_test():
            order = Order(
                order_id="TEST_005",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1000"),
                timestamp=datetime.now()
            )
            
            market_data = {
                "symbol": "AAPL",
                "bid": 150.00,
                "ask": 150.10,
                "last": 150.05,
                "volume": 10000000
            }
            
            # Simulate multiple partial fills
            total_filled = Decimal("0")
            fills = []
            
            while total_filled < order.quantity:
                fill = await simulator.simulate_fill(order, market_data)
                if fill:
                    fills.append(fill)
                    total_filled += fill.quantity
                    
                    # Update order's filled quantity for next iteration
                    order.filled_quantity = total_filled
                    
                    # Should be partial until last fill
                    if total_filled < order.quantity:
                        self.assertEqual(fill.fill_type, FillType.PARTIAL)
                    else:
                        self.assertEqual(fill.fill_type, FillType.COMPLETE)
                
                # Prevent infinite loop in test
                if len(fills) > 10:
                    break
            
            # Should have multiple fills
            self.assertGreater(len(fills), 1)
            self.assertEqual(total_filled, order.quantity)
        
        asyncio.run(run_test())


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simulator = MarketSimulator()
    
    def test_missing_market_data(self):
        """Test handling of missing market data."""
        async def run_test():
            order = Order(
                order_id="TEST_001",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("100"),
                timestamp=datetime.now()
            )
            
            # Empty market data
            market_data = {}
            
            fill = await self.simulator.simulate_fill(order, market_data)
            self.assertIsNone(fill)
            
            # Market data for wrong symbol
            market_data = {
                "symbol": "GOOGL",
                "bid": 2800.00,
                "ask": 2801.00
            }
            
            fill = await self.simulator.simulate_fill(order, market_data)
            self.assertIsNone(fill)
        
        asyncio.run(run_test())
    
    def test_zero_quantity_order(self):
        """Test handling of zero quantity order."""
        async def run_test():
            order = Order(
                order_id="TEST_002",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0"),
                timestamp=datetime.now()
            )
            
            market_data = {
                "symbol": "AAPL",
                "bid": 150.00,
                "ask": 150.10
            }
            
            fill = await self.simulator.simulate_fill(order, market_data)
            self.assertIsNone(fill)
        
        asyncio.run(run_test())
    
    def test_invalid_order_type(self):
        """Test handling of unsupported order type."""
        async def run_test():
            # Create order with invalid type
            order = Order(
                order_id="TEST_003",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type="INVALID_TYPE",  # Invalid
                quantity=Decimal("100"),
                timestamp=datetime.now()
            )
            
            market_data = {
                "symbol": "AAPL",
                "bid": 150.00,
                "ask": 150.10
            }
            
            # Should handle gracefully
            fill = await self.simulator.simulate_fill(order, market_data)
            self.assertIsNone(fill)
        
        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()