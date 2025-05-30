"""
Integration tests for data pipeline.

Tests data flow from sources through handlers to consumers.
"""

import unittest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.handlers import CSVDataHandler, RealtimeDataHandler, DataCache
from src.data.models import BarData, TickData, MarketData
from src.core.events.event_bus import EventBus
from src.core.events.types import Event, EventType
from src.core.containers.universal import UniversalScopedContainer


class DataConsumer:
    """Mock data consumer for testing."""
    
    def __init__(self, name: str, event_bus: EventBus = None):
        self.name = name
        self.event_bus = event_bus
        self.received_data = []
        self.processed_count = 0
        
        if event_bus:
            event_bus.subscribe(EventType.DATA, self.handle_data)
    
    def handle_data(self, event: Event):
        """Handle incoming data event."""
        self.received_data.append(event.payload)
        self.processed_count += 1
        
        # Simulate processing and emit result
        if self.event_bus and event.payload.get("symbol"):
            self.event_bus.publish(Event(
                event_type=EventType.SIGNAL,
                source_id=self.name,
                payload={
                    "symbol": event.payload["symbol"],
                    "signal": "processed",
                    "consumer": self.name
                }
            ))


class TestHistoricalDataPipeline(unittest.TestCase):
    """Test historical data pipeline integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.event_bus = EventBus()
        self.csv_handler = CSVDataHandler()
        self.cache = DataCache(max_size=1000)
    
    @patch('pandas.read_csv')
    def test_csv_to_consumer_pipeline(self, mock_read_csv):
        """Test data flow from CSV to consumers."""
        # Mock CSV data
        mock_df = pd.DataFrame({
            'timestamp': [
                '2024-01-01 09:30:00',
                '2024-01-01 09:31:00',
                '2024-01-01 09:32:00'
            ],
            'open': [150.00, 150.50, 151.00],
            'high': [150.50, 151.00, 151.50],
            'low': [149.50, 150.00, 150.50],
            'close': [150.25, 150.75, 151.25],
            'volume': [100000, 110000, 120000]
        })
        mock_read_csv.return_value = mock_df
        
        # Create consumers
        consumer1 = DataConsumer("strategy1", self.event_bus)
        consumer2 = DataConsumer("strategy2", self.event_bus)
        
        # Load and process data
        bars = self.csv_handler.load_bars("AAPL", "test.csv")
        
        # Publish bars as events
        for bar in bars:
            self.event_bus.publish(Event(
                event_type=EventType.DATA,
                source_id="csv_loader",
                payload={
                    "symbol": bar.symbol,
                    "timestamp": bar.timestamp,
                    "ohlcv": {
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume
                    }
                }
            ))
        
        # Check consumers received data
        self.assertEqual(len(consumer1.received_data), 3)
        self.assertEqual(len(consumer2.received_data), 3)
        
        # Check data integrity
        first_data = consumer1.received_data[0]
        self.assertEqual(first_data["symbol"], "AAPL")
        self.assertEqual(first_data["ohlcv"]["open"], 150.00)
    
    def test_data_caching_pipeline(self):
        """Test data caching in pipeline."""
        # Create data
        bars = []
        base_time = datetime.now()
        
        for i in range(10):
            bar = BarData(
                symbol="AAPL",
                timestamp=base_time + timedelta(minutes=i),
                open=150.00 + i * 0.1,
                high=150.50 + i * 0.1,
                low=149.50 + i * 0.1,
                close=150.25 + i * 0.1,
                volume=100000 + i * 1000
            )
            bars.append(bar)
        
        # Cache bars
        cache_key = "AAPL_bars_20240101"
        self.cache.put(cache_key, bars)
        
        # Consumer that uses cache
        class CachingConsumer:
            def __init__(self, cache):
                self.cache = cache
                self.cache_hits = 0
                self.cache_misses = 0
            
            def get_data(self, key):
                data = self.cache.get(key)
                if data:
                    self.cache_hits += 1
                else:
                    self.cache_misses += 1
                return data
        
        consumer = CachingConsumer(self.cache)
        
        # First access - should hit cache
        data = consumer.get_data(cache_key)
        self.assertIsNotNone(data)
        self.assertEqual(len(data), 10)
        self.assertEqual(consumer.cache_hits, 1)
        
        # Access non-existent - should miss
        data = consumer.get_data("nonexistent")
        self.assertIsNone(data)
        self.assertEqual(consumer.cache_misses, 1)


class TestRealtimeDataPipeline(unittest.TestCase):
    """Test realtime data pipeline integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.event_bus = EventBus()
        self.mock_client = Mock()
        self.realtime_handler = RealtimeDataHandler(self.mock_client)
    
    def test_realtime_data_flow(self):
        """Test realtime data flow to consumers."""
        # Create consumers
        consumers = [
            DataConsumer(f"consumer_{i}", self.event_bus)
            for i in range(3)
        ]
        
        # Track all signals
        signals_received = []
        self.event_bus.subscribe(
            EventType.SIGNAL,
            lambda e: signals_received.append(e)
        )
        
        # Subscribe to realtime data
        def data_callback(tick):
            # Convert tick to event
            self.event_bus.publish(Event(
                event_type=EventType.DATA,
                source_id="realtime",
                payload={
                    "symbol": tick.symbol,
                    "price": tick.price,
                    "size": tick.size,
                    "timestamp": tick.timestamp
                }
            ))
        
        self.realtime_handler.subscribe("AAPL", data_callback)
        
        # Simulate incoming ticks
        ticks = [
            TickData("AAPL", datetime.now(), 150.25, 100, "bid"),
            TickData("AAPL", datetime.now(), 150.26, 200, "ask"),
            TickData("AAPL", datetime.now(), 150.24, 150, "bid")
        ]
        
        for tick in ticks:
            self.realtime_handler._on_data("AAPL", tick)
        
        # Each consumer should process each tick
        for consumer in consumers:
            self.assertEqual(consumer.processed_count, 3)
        
        # Should have signals from all consumers
        self.assertEqual(len(signals_received), 9)  # 3 consumers * 3 ticks


class TestDataQualityPipeline(unittest.TestCase):
    """Test data quality checks in pipeline."""
    
    def test_data_validation_pipeline(self):
        """Test data validation and filtering."""
        event_bus = EventBus()
        
        # Create quality checker
        class DataQualityChecker:
            def __init__(self, event_bus):
                self.event_bus = event_bus
                self.valid_data = []
                self.invalid_data = []
                
                event_bus.subscribe(EventType.DATA, self.check_data)
            
            def check_data(self, event):
                data = event.payload
                
                # Validate OHLC relationships
                if "ohlcv" in data:
                    ohlcv = data["ohlcv"]
                    if (ohlcv["high"] >= ohlcv["low"] and
                        ohlcv["high"] >= ohlcv["open"] and
                        ohlcv["high"] >= ohlcv["close"] and
                        ohlcv["low"] <= ohlcv["open"] and
                        ohlcv["low"] <= ohlcv["close"] and
                        ohlcv["volume"] > 0):
                        
                        self.valid_data.append(data)
                        # Forward valid data
                        self.event_bus.publish(Event(
                            event_type=EventType.DATA,
                            source_id="quality_checker",
                            payload=data,
                            metadata={"validated": True}
                        ))
                    else:
                        self.invalid_data.append(data)
        
        checker = DataQualityChecker(event_bus)
        
        # Consumer that only accepts validated data
        validated_consumer = DataConsumer("validated_only", event_bus)
        
        # Send mixed quality data
        test_data = [
            # Valid
            {
                "symbol": "AAPL",
                "ohlcv": {
                    "open": 150.00,
                    "high": 151.00,
                    "low": 149.00,
                    "close": 150.50,
                    "volume": 100000
                }
            },
            # Invalid - high < low
            {
                "symbol": "AAPL",
                "ohlcv": {
                    "open": 150.00,
                    "high": 149.00,
                    "low": 151.00,
                    "close": 150.00,
                    "volume": 100000
                }
            },
            # Valid
            {
                "symbol": "AAPL",
                "ohlcv": {
                    "open": 151.00,
                    "high": 152.00,
                    "low": 150.50,
                    "close": 151.50,
                    "volume": 110000
                }
            }
        ]
        
        for data in test_data:
            event_bus.publish(Event(
                event_type=EventType.DATA,
                source_id="raw_data",
                payload=data
            ))
        
        # Check results
        self.assertEqual(len(checker.valid_data), 2)
        self.assertEqual(len(checker.invalid_data), 1)
        
        # Consumer should only see validated data
        # Each valid data point triggers quality_checker which consumer sees
        self.assertEqual(validated_consumer.processed_count, 5)  # 3 original + 2 validated


class TestMultiSourceDataPipeline(unittest.TestCase):
    """Test integrating multiple data sources."""
    
    def test_multi_source_aggregation(self):
        """Test aggregating data from multiple sources."""
        event_bus = EventBus()
        
        # Data aggregator
        class DataAggregator:
            def __init__(self, event_bus):
                self.event_bus = event_bus
                self.data_by_symbol = {}
                self.source_counts = {}
                
                event_bus.subscribe(EventType.DATA, self.aggregate_data)
            
            def aggregate_data(self, event):
                source = event.source_id
                symbol = event.payload.get("symbol")
                
                if not symbol:
                    return
                
                # Track data by symbol
                if symbol not in self.data_by_symbol:
                    self.data_by_symbol[symbol] = []
                
                self.data_by_symbol[symbol].append({
                    "source": source,
                    "data": event.payload,
                    "timestamp": event.timestamp
                })
                
                # Track source counts
                if source not in self.source_counts:
                    self.source_counts[source] = 0
                self.source_counts[source] += 1
                
                # Emit aggregated view periodically
                if len(self.data_by_symbol[symbol]) % 5 == 0:
                    self.emit_aggregated(symbol)
            
            def emit_aggregated(self, symbol):
                recent_data = self.data_by_symbol[symbol][-5:]
                
                self.event_bus.publish(Event(
                    event_type=EventType.DATA,
                    source_id="aggregator",
                    payload={
                        "symbol": symbol,
                        "aggregated": True,
                        "source_count": len(set(d["source"] for d in recent_data)),
                        "data_points": len(recent_data)
                    }
                ))
        
        aggregator = DataAggregator(event_bus)
        
        # Simulate multiple sources
        sources = ["csv_source", "database_source", "api_source"]
        
        for i in range(15):
            source = sources[i % len(sources)]
            event_bus.publish(Event(
                event_type=EventType.DATA,
                source_id=source,
                payload={
                    "symbol": "AAPL",
                    "price": 150.00 + i * 0.1,
                    "source_seq": i
                }
            ))
        
        # Check aggregation
        self.assertIn("AAPL", aggregator.data_by_symbol)
        self.assertEqual(len(aggregator.data_by_symbol["AAPL"]), 15)
        
        # Check source distribution
        for source in sources:
            self.assertEqual(aggregator.source_counts[source], 5)


class TestDataPipelineContainerIntegration(unittest.TestCase):
    """Test data pipeline with container architecture."""
    
    def test_containerized_data_pipeline(self):
        """Test data pipeline using containers."""
        # Create containers
        data_container = UniversalScopedContainer("data", "data_management")
        processing_container = UniversalScopedContainer("processing", "data_processing")
        
        # Shared event bus
        event_bus = EventBus()
        
        # Data source component
        class DataSourceComponent:
            def __init__(self, event_bus):
                self.event_bus = event_bus
            
            def emit_data(self, symbol, price):
                self.event_bus.publish(Event(
                    event_type=EventType.DATA,
                    source_id="data_source",
                    payload={"symbol": symbol, "price": price}
                ))
        
        # Processing component
        class ProcessingComponent:
            def __init__(self, event_bus):
                self.event_bus = event_bus
                self.processed = []
                event_bus.subscribe(EventType.DATA, self.process)
            
            def process(self, event):
                # Simple transformation
                data = event.payload
                processed = {
                    "symbol": data["symbol"],
                    "price": data["price"],
                    "ma": data["price"] * 1.01  # Simple moving average simulation
                }
                self.processed.append(processed)
        
        # Add components
        data_source = DataSourceComponent(event_bus)
        processor = ProcessingComponent(event_bus)
        
        data_container.add_component("source", data_source)
        processing_container.add_component("processor", processor)
        
        # Run pipeline
        async def run_pipeline():
            await data_container.start()
            await processing_container.start()
            
            # Emit data
            for i in range(5):
                data_source.emit_data("AAPL", 150.00 + i)
            
            # Check processing
            self.assertEqual(len(processor.processed), 5)
            self.assertEqual(processor.processed[0]["symbol"], "AAPL")
            
            await data_container.stop()
            await processing_container.stop()
        
        asyncio.run(run_pipeline())


if __name__ == "__main__":
    unittest.main()