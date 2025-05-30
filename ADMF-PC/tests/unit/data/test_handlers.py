"""
Unit tests for data handlers.

Tests data loading, transformation, and management.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from decimal import Decimal
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.data.handlers import (
    DataHandler,
    CSVDataHandler,
    DatabaseDataHandler,
    RealtimeDataHandler,
    DataCache
)
from src.data.models import (
    MarketData,
    BarData,
    TickData,
    OrderBookData,
    DataQuality
)


class TestMarketData(unittest.TestCase):
    """Test market data models."""
    
    def test_bar_data_creation(self):
        """Test creating bar data."""
        bar = BarData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=150.00,
            high=152.50,
            low=149.50,
            close=151.00,
            volume=1000000,
            metadata={"source": "test"}
        )
        
        self.assertEqual(bar.symbol, "AAPL")
        self.assertEqual(bar.open, 150.00)
        self.assertEqual(bar.high, 152.50)
        self.assertEqual(bar.low, 149.50)
        self.assertEqual(bar.close, 151.00)
        self.assertEqual(bar.volume, 1000000)
    
    def test_bar_data_validation(self):
        """Test bar data validation."""
        # Valid bar
        bar = BarData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=150.00,
            high=152.50,
            low=149.50,
            close=151.00,
            volume=1000000
        )
        
        self.assertTrue(bar.is_valid())
        
        # Invalid bar - high < low
        invalid_bar = BarData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=150.00,
            high=149.00,  # Less than low
            low=151.00,
            close=150.50,
            volume=1000000
        )
        
        self.assertFalse(invalid_bar.is_valid())
    
    def test_tick_data(self):
        """Test tick data creation."""
        tick = TickData(
            symbol="AAPL",
            timestamp=datetime.now(),
            price=150.25,
            size=100,
            side="bid",
            metadata={"exchange": "NASDAQ"}
        )
        
        self.assertEqual(tick.symbol, "AAPL")
        self.assertEqual(tick.price, 150.25)
        self.assertEqual(tick.size, 100)
        self.assertEqual(tick.side, "bid")
    
    def test_orderbook_data(self):
        """Test order book data."""
        orderbook = OrderBookData(
            symbol="AAPL",
            timestamp=datetime.now(),
            bids=[(150.00, 1000), (149.95, 2000), (149.90, 1500)],
            asks=[(150.05, 1000), (150.10, 1500), (150.15, 2000)],
            metadata={"depth": 3}
        )
        
        self.assertEqual(len(orderbook.bids), 3)
        self.assertEqual(len(orderbook.asks), 3)
        self.assertEqual(orderbook.get_best_bid(), (150.00, 1000))
        self.assertEqual(orderbook.get_best_ask(), (150.05, 1000))
        self.assertEqual(orderbook.get_spread(), 0.05)
        self.assertEqual(orderbook.get_mid_price(), 150.025)


class TestDataCache(unittest.TestCase):
    """Test data caching functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache = DataCache(max_size=100, ttl_seconds=300)
    
    def test_cache_operations(self):
        """Test basic cache operations."""
        # Add data to cache
        data = BarData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=150.00,
            high=151.00,
            low=149.00,
            close=150.50,
            volume=1000000
        )
        
        self.cache.put("AAPL_latest", data)
        
        # Retrieve from cache
        cached = self.cache.get("AAPL_latest")
        self.assertEqual(cached, data)
        
        # Non-existent key
        self.assertIsNone(self.cache.get("nonexistent"))
    
    def test_cache_expiration(self):
        """Test cache TTL expiration."""
        # Create cache with short TTL
        cache = DataCache(max_size=10, ttl_seconds=1)
        
        data = {"test": "data"}
        cache.put("key", data)
        
        # Should be in cache
        self.assertEqual(cache.get("key"), data)
        
        # Wait for expiration
        import time
        time.sleep(1.1)
        
        # Should be expired
        self.assertIsNone(cache.get("key"))
    
    def test_cache_size_limit(self):
        """Test cache size limiting."""
        cache = DataCache(max_size=3)
        
        # Add more items than max size
        for i in range(5):
            cache.put(f"key_{i}", f"value_{i}")
        
        # Should only have last 3 items
        self.assertEqual(cache.size(), 3)
        self.assertIsNone(cache.get("key_0"))  # Evicted
        self.assertIsNone(cache.get("key_1"))  # Evicted
        self.assertEqual(cache.get("key_2"), "value_2")
        self.assertEqual(cache.get("key_3"), "value_3")
        self.assertEqual(cache.get("key_4"), "value_4")
    
    def test_cache_clear(self):
        """Test clearing cache."""
        for i in range(5):
            self.cache.put(f"key_{i}", f"value_{i}")
        
        self.assertEqual(self.cache.size(), 5)
        
        self.cache.clear()
        
        self.assertEqual(self.cache.size(), 0)
        self.assertIsNone(self.cache.get("key_0"))


class TestCSVDataHandler(unittest.TestCase):
    """Test CSV data handler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.handler = CSVDataHandler()
    
    @patch('pandas.read_csv')
    def test_load_csv_data(self, mock_read_csv):
        """Test loading data from CSV."""
        # Mock CSV data
        mock_df = pd.DataFrame({
            'timestamp': ['2024-01-01 09:30:00', '2024-01-01 09:31:00'],
            'open': [150.00, 150.50],
            'high': [151.00, 151.50],
            'low': [149.50, 150.00],
            'close': [150.50, 151.00],
            'volume': [1000000, 1100000]
        })
        mock_read_csv.return_value = mock_df
        
        # Load data
        bars = self.handler.load_bars("AAPL", "test.csv")
        
        self.assertEqual(len(bars), 2)
        self.assertEqual(bars[0].symbol, "AAPL")
        self.assertEqual(bars[0].open, 150.00)
        self.assertEqual(bars[1].close, 151.00)
    
    def test_parse_csv_row(self):
        """Test parsing CSV row to bar data."""
        row = {
            'timestamp': '2024-01-01 09:30:00',
            'open': 150.00,
            'high': 151.00,
            'low': 149.50,
            'close': 150.50,
            'volume': 1000000
        }
        
        bar = self.handler._parse_row("AAPL", row)
        
        self.assertEqual(bar.symbol, "AAPL")
        self.assertEqual(bar.open, 150.00)
        self.assertEqual(bar.volume, 1000000)
    
    @patch('pandas.DataFrame.to_csv')
    def test_save_csv_data(self, mock_to_csv):
        """Test saving data to CSV."""
        bars = [
            BarData("AAPL", datetime(2024, 1, 1, 9, 30), 150.00, 151.00, 149.50, 150.50, 1000000),
            BarData("AAPL", datetime(2024, 1, 1, 9, 31), 150.50, 151.50, 150.00, 151.00, 1100000)
        ]
        
        self.handler.save_bars(bars, "output.csv")
        
        # Check that to_csv was called
        mock_to_csv.assert_called_once()
        call_args = mock_to_csv.call_args
        self.assertEqual(call_args[0][0], "output.csv")


class TestDatabaseDataHandler(unittest.TestCase):
    """Test database data handler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_connection = Mock()
        self.handler = DatabaseDataHandler(self.mock_connection)
    
    def test_load_from_database(self):
        """Test loading data from database."""
        # Mock cursor and results
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            ('AAPL', '2024-01-01 09:30:00', 150.00, 151.00, 149.50, 150.50, 1000000),
            ('AAPL', '2024-01-01 09:31:00', 150.50, 151.50, 150.00, 151.00, 1100000)
        ]
        self.mock_connection.cursor.return_value = mock_cursor
        
        # Load data
        bars = self.handler.load_bars(
            "AAPL",
            start_time=datetime(2024, 1, 1, 9, 30),
            end_time=datetime(2024, 1, 1, 10, 0)
        )
        
        self.assertEqual(len(bars), 2)
        self.assertEqual(bars[0].symbol, "AAPL")
        
        # Check query was executed
        mock_cursor.execute.assert_called_once()
        query = mock_cursor.execute.call_args[0][0]
        self.assertIn("SELECT", query)
        self.assertIn("symbol", query)
    
    def test_save_to_database(self):
        """Test saving data to database."""
        mock_cursor = Mock()
        self.mock_connection.cursor.return_value = mock_cursor
        
        bars = [
            BarData("AAPL", datetime(2024, 1, 1, 9, 30), 150.00, 151.00, 149.50, 150.50, 1000000)
        ]
        
        self.handler.save_bars(bars)
        
        # Check insert was called
        mock_cursor.executemany.assert_called_once()
        self.mock_connection.commit.assert_called_once()
    
    def test_database_error_handling(self):
        """Test database error handling."""
        # Mock database error
        mock_cursor = Mock()
        mock_cursor.execute.side_effect = Exception("Database error")
        self.mock_connection.cursor.return_value = mock_cursor
        
        # Should handle error gracefully
        with self.assertRaises(Exception):
            self.handler.load_bars("AAPL")
        
        # Should rollback on error
        self.mock_connection.rollback.assert_called()


class TestRealtimeDataHandler(unittest.TestCase):
    """Test realtime data handler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.handler = RealtimeDataHandler(self.mock_client)
    
    def test_subscribe_to_symbol(self):
        """Test subscribing to realtime data."""
        callback = Mock()
        
        self.handler.subscribe("AAPL", callback)
        
        # Should call client subscribe
        self.mock_client.subscribe.assert_called_with("AAPL", unittest.mock.ANY)
        
        # Test callback is registered
        self.assertIn("AAPL", self.handler._callbacks)
        self.assertEqual(self.handler._callbacks["AAPL"], callback)
    
    def test_unsubscribe_from_symbol(self):
        """Test unsubscribing from realtime data."""
        callback = Mock()
        
        self.handler.subscribe("AAPL", callback)
        self.handler.unsubscribe("AAPL")
        
        # Should call client unsubscribe
        self.mock_client.unsubscribe.assert_called_with("AAPL")
        
        # Callback should be removed
        self.assertNotIn("AAPL", self.handler._callbacks)
    
    def test_data_callback_routing(self):
        """Test routing data to callbacks."""
        callback1 = Mock()
        callback2 = Mock()
        
        self.handler.subscribe("AAPL", callback1)
        self.handler.subscribe("GOOGL", callback2)
        
        # Simulate data arrival
        aapl_data = TickData("AAPL", datetime.now(), 150.25, 100, "bid")
        googl_data = TickData("GOOGL", datetime.now(), 2800.50, 50, "ask")
        
        self.handler._on_data("AAPL", aapl_data)
        self.handler._on_data("GOOGL", googl_data)
        
        # Check callbacks were called correctly
        callback1.assert_called_once_with(aapl_data)
        callback2.assert_called_once_with(googl_data)
    
    def test_connection_management(self):
        """Test connection lifecycle."""
        # Connect
        self.handler.connect()
        self.mock_client.connect.assert_called_once()
        
        # Disconnect
        self.handler.disconnect()
        self.mock_client.disconnect.assert_called_once()
    
    def test_reconnection_on_error(self):
        """Test automatic reconnection on error."""
        # Mock connection error
        self.mock_client.connect.side_effect = [
            Exception("Connection failed"),
            None  # Success on second attempt
        ]
        
        # Should retry connection
        self.handler.connect(retry=True, max_retries=2)
        
        # Should have been called twice
        self.assertEqual(self.mock_client.connect.call_count, 2)


class TestDataQuality(unittest.TestCase):
    """Test data quality checks."""
    
    def test_data_quality_validation(self):
        """Test data quality validation."""
        quality = DataQuality()
        
        # Good quality bar
        good_bar = BarData(
            "AAPL",
            datetime.now(),
            150.00, 151.00, 149.50, 150.50,
            1000000
        )
        
        issues = quality.check_bar(good_bar)
        self.assertEqual(len(issues), 0)
        
        # Bad quality bar - negative volume
        bad_bar = BarData(
            "AAPL",
            datetime.now(),
            150.00, 151.00, 149.50, 150.50,
            -1000
        )
        
        issues = quality.check_bar(bad_bar)
        self.assertGreater(len(issues), 0)
        self.assertIn("volume", str(issues[0]).lower())
    
    def test_gap_detection(self):
        """Test gap detection in data."""
        quality = DataQuality()
        
        bars = [
            BarData("AAPL", datetime(2024, 1, 1, 9, 30), 150.00, 151.00, 149.50, 150.50, 1000000),
            BarData("AAPL", datetime(2024, 1, 1, 9, 31), 150.50, 151.50, 150.00, 151.00, 1100000),
            # Gap here - missing 9:32
            BarData("AAPL", datetime(2024, 1, 1, 9, 33), 151.00, 152.00, 150.50, 151.50, 1200000)
        ]
        
        gaps = quality.find_gaps(bars, expected_interval=timedelta(minutes=1))
        
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["start"], datetime(2024, 1, 1, 9, 31))
        self.assertEqual(gaps[0]["end"], datetime(2024, 1, 1, 9, 33))
    
    def test_outlier_detection(self):
        """Test outlier detection."""
        quality = DataQuality()
        
        bars = [
            BarData("AAPL", datetime.now(), 150.00, 151.00, 149.50, 150.50, 1000000),
            BarData("AAPL", datetime.now(), 150.50, 151.50, 150.00, 151.00, 1100000),
            BarData("AAPL", datetime.now(), 200.00, 201.00, 199.00, 200.50, 1200000),  # Outlier
            BarData("AAPL", datetime.now(), 151.00, 152.00, 150.50, 151.50, 1150000)
        ]
        
        outliers = quality.find_outliers(bars, threshold=3.0)
        
        self.assertEqual(len(outliers), 1)
        self.assertEqual(outliers[0].close, 200.50)


if __name__ == "__main__":
    unittest.main()