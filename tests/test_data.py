"""
Tests for data module.
"""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.data import (
    Bar, Timeframe, DataView, TimeSeriesData,
    DataHandler, HistoricalDataHandler,
    CSVLoader, MemoryOptimizedCSVLoader
)
from src.core.events import EventBus, EventType


class TestBar:
    """Test Bar data model."""
    
    def test_bar_creation(self):
        """Test creating a bar."""
        bar = Bar(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000000.0,
            timeframe=Timeframe.D1
        )
        
        assert bar.symbol == "AAPL"
        assert bar.range == 6.0  # high - low
        assert bar.body == 3.0   # abs(close - open)
        assert bar.is_bullish    # close > open
        assert not bar.is_bearish
    
    def test_bar_validation(self):
        """Test bar validation."""
        # Invalid high/low
        with pytest.raises(ValueError, match="High.*cannot be less than Low"):
            Bar(
                symbol="AAPL",
                timestamp=datetime.now(),
                open=100.0,
                high=99.0,   # Invalid: high < low
                low=101.0,
                close=100.0,
                volume=1000.0
            )
        
        # Invalid high
        with pytest.raises(ValueError, match="High must be >= Open and Close"):
            Bar(
                symbol="AAPL",
                timestamp=datetime.now(),
                open=105.0,
                high=100.0,  # Invalid: high < open
                low=95.0,
                close=100.0,
                volume=1000.0
            )
        
        # Negative volume
        with pytest.raises(ValueError, match="Volume cannot be negative"):
            Bar(
                symbol="AAPL",
                timestamp=datetime.now(),
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=-1000.0
            )
    
    def test_bar_serialization(self):
        """Test bar to/from dict."""
        original = Bar(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000000.0,
            timeframe=Timeframe.D1,
            metadata={"source": "test"}
        )
        
        # To dict
        data = original.to_dict()
        assert data["symbol"] == "AAPL"
        assert data["open"] == 100.0
        assert data["timeframe"] == "1d"
        assert data["metadata"]["source"] == "test"
        
        # From dict
        restored = Bar.from_dict(data)
        assert restored.symbol == original.symbol
        assert restored.open == original.open
        assert restored.timeframe == original.timeframe


class TestDataView:
    """Test DataView for memory-efficient access."""
    
    def test_data_view_basic(self):
        """Test basic DataView functionality."""
        # Create sample data
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        data = pd.DataFrame({
            "open": np.random.randn(100) + 100,
            "high": np.random.randn(100) + 101,
            "low": np.random.randn(100) + 99,
            "close": np.random.randn(100) + 100,
            "volume": np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Create view
        view = DataView(data, start_idx=10, end_idx=50)
        
        # Test current
        current = view.get_current()
        assert current is not None
        assert current.name == dates[10]
        
        # Test advance
        assert view.advance()
        next_current = view.get_current()
        assert next_current.name == dates[11]
        
        # Test window
        window = view.get_window(5)
        assert len(window) == 5
        assert window.index[-1] == next_current.name
        
        # Test progress
        assert 0 < view.progress < 1
        
        # Test reset
        view.reset()
        assert view.get_current().name == dates[10]
    
    def test_data_view_bounds(self):
        """Test DataView boundary conditions."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        data = pd.DataFrame({
            "value": range(10)
        }, index=dates)
        
        view = DataView(data, start_idx=5, end_idx=10)
        
        # Advance to end
        advances = 0
        while view.advance():
            advances += 1
        
        assert advances == 4  # 5 data points, 4 advances
        assert not view.has_data
        assert view.progress == 1.0


class TestTimeSeriesData:
    """Test efficient time series storage."""
    
    def test_time_series_creation(self):
        """Test creating TimeSeriesData."""
        timestamps = pd.date_range("2023-01-01", periods=100, freq="D")
        data = {
            "open": np.random.randn(100),
            "close": np.random.randn(100),
            "volume": np.random.randint(1000, 10000, 100)
        }
        
        ts = TimeSeriesData(timestamps, data)
        
        assert len(ts) == 100
        assert ts[0]["timestamp"] == timestamps[0]
        assert "open" in ts[0]
        assert "close" in ts[0]
        assert "volume" in ts[0]
    
    def test_time_series_dataframe_conversion(self):
        """Test conversion to/from DataFrame."""
        # Create DataFrame
        df = pd.DataFrame({
            "open": np.random.randn(50),
            "close": np.random.randn(50)
        }, index=pd.date_range("2023-01-01", periods=50, freq="D"))
        
        # Convert to TimeSeriesData
        ts = TimeSeriesData.from_dataframe(df)
        assert len(ts) == 50
        
        # Convert back
        df2 = ts.to_dataframe()
        # Check that data is preserved (frequency info is lost in conversion)
        assert df.index.equals(df2.index)
        pd.testing.assert_frame_equal(df, df2, check_freq=False)
    
    def test_time_series_view(self):
        """Test creating views of time series."""
        timestamps = pd.date_range("2023-01-01", periods=100, freq="D")
        data = {"value": np.arange(100)}
        
        ts = TimeSeriesData(timestamps, data)
        
        # Create view
        view = ts.get_view(10, 20)
        assert len(view) == 10
        assert view[0]["value"] == 10
        assert view[9]["value"] == 19
    
    def test_time_series_validation(self):
        """Test data validation."""
        timestamps = pd.date_range("2023-01-01", periods=10, freq="D")
        
        # Mismatched lengths
        with pytest.raises(ValueError, match="expected 10"):
            TimeSeriesData(timestamps, {"value": np.arange(5)})


class TestCSVLoader:
    """Test CSV data loading."""
    
    def setup_method(self):
        """Create temporary directory with test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)
        
        # Create sample CSV
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        self.sample_data = pd.DataFrame({
            "Date": dates,
            "Open": np.random.uniform(95, 105, 100),
            "High": np.random.uniform(100, 110, 100),
            "Low": np.random.uniform(90, 100, 100),
            "Close": np.random.uniform(95, 105, 100),
            "Volume": np.random.randint(100000, 1000000, 100)
        })
        
        # Ensure OHLC relationships are valid
        for i in range(len(self.sample_data)):
            row = self.sample_data.iloc[i]
            high = max(row["Open"], row["Close"]) + abs(np.random.randn())
            low = min(row["Open"], row["Close"]) - abs(np.random.randn())
            self.sample_data.iloc[i, 2] = high  # High
            self.sample_data.iloc[i, 3] = low   # Low
        
        # Save CSV
        self.sample_data.to_csv(
            self.data_dir / "AAPL.csv",
            index=False
        )
    
    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_csv_loader_basic(self):
        """Test basic CSV loading."""
        loader = CSVLoader(data_dir=str(self.data_dir))
        
        df = loader.load("AAPL")
        
        # Check shape
        assert len(df) == 100
        assert all(col in df.columns for col in ["open", "high", "low", "close", "volume"])
        
        # Check index
        assert isinstance(df.index, pd.DatetimeIndex)
        
        # Check validation passed
        assert loader.validate(df)
    
    def test_csv_loader_normalization(self):
        """Test column name normalization."""
        # Create CSV with different column names
        weird_data = self.sample_data.copy()
        weird_data.columns = ["Date", "o", "h", "l", "c", "v"]
        weird_data.to_csv(self.data_dir / "WEIRD.csv", index=False)
        
        loader = CSVLoader(data_dir=str(self.data_dir))
        df = loader.load("WEIRD")
        
        # Should have normalized columns
        assert all(col in df.columns for col in ["open", "high", "low", "close", "volume"])
    
    def test_csv_loader_missing_file(self):
        """Test handling missing files."""
        loader = CSVLoader(data_dir=str(self.data_dir))
        
        with pytest.raises(FileNotFoundError):
            loader.load("MISSING")
    
    def test_csv_loader_invalid_data(self):
        """Test handling invalid data."""
        # Create invalid CSV (high < low)
        invalid_data = self.sample_data.copy()
        invalid_data["High"] = invalid_data["Low"] - 1
        invalid_data.to_csv(self.data_dir / "INVALID.csv", index=False)
        
        loader = CSVLoader(data_dir=str(self.data_dir))
        
        with pytest.raises(ValueError, match="Invalid data"):
            loader.load("INVALID")


class TestHistoricalDataHandler:
    """Test historical data handler."""
    
    def setup_method(self):
        """Set up test data handler."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)
        
        # Create test data for two symbols
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        
        for symbol in ["AAPL", "MSFT"]:
            data = pd.DataFrame({
                "Date": dates,
                "open": np.random.uniform(95, 105, 100),
                "high": np.random.uniform(100, 110, 100),
                "low": np.random.uniform(90, 100, 100),
                "close": np.random.uniform(95, 105, 100),
                "volume": np.random.randint(100000, 1000000, 100)
            })
            
            # Fix OHLC relationships
            data["high"] = data[["open", "high", "close"]].max(axis=1) + 0.1
            data["low"] = data[["open", "low", "close"]].min(axis=1) - 0.1
            
            data.to_csv(self.data_dir / f"{symbol}.csv", index=False)
        
        # Create handler
        self.handler = HistoricalDataHandler(
            handler_id="test_handler",
            data_dir=str(self.data_dir)
        )
        
        # Create event bus
        self.event_bus = EventBus()
        self.events = []
        
        def capture_event(event):
            self.events.append(event)
        
        self.event_bus.subscribe(EventType.BAR, capture_event)
        
        # Initialize handler
        self.handler.initialize({
            'event_bus': self.event_bus,
            'container_id': 'test_container'
        })
    
    def teardown_method(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)
    
    def test_load_data(self):
        """Test loading data."""
        self.handler.load_data(["AAPL", "MSFT"])
        
        assert "AAPL" in self.handler.data
        assert "MSFT" in self.handler.data
        assert len(self.handler.data["AAPL"]) == 100
        assert len(self.handler.data["MSFT"]) == 100
    
    def test_update_bars(self):
        """Test emitting bars."""
        self.handler.load_data(["AAPL", "MSFT"])
        self.handler.start()
        
        # Emit first bar
        has_more = self.handler.update_bars()
        assert has_more
        assert len(self.events) == 1
        
        event = self.events[0]
        assert event.event_type == EventType.BAR
        assert event.container_id == "test_container"
        assert "symbol" in event.payload
        assert "data" in event.payload
        
        # Count total bars emitted
        bar_count = 1
        while self.handler.update_bars():
            bar_count += 1
        
        # Should emit 200 bars total (100 for each symbol)
        assert bar_count == 200
    
    def test_get_latest_bars(self):
        """Test getting latest bars."""
        self.handler.load_data(["AAPL"])
        self.handler.start()
        
        # No bars yet
        assert self.handler.get_latest_bar("AAPL") is None
        
        # Emit some bars
        for _ in range(5):
            self.handler.update_bars()
        
        # Get latest bar
        latest = self.handler.get_latest_bar("AAPL")
        assert latest is not None
        assert isinstance(latest, Bar)
        assert latest.symbol == "AAPL"
        
        # Get multiple bars
        bars = self.handler.get_latest_bars("AAPL", 3)
        assert len(bars) == 3
        assert all(isinstance(b, Bar) for b in bars)
        assert bars[-1].timestamp == latest.timestamp
    
    def test_train_test_split(self):
        """Test train/test data splitting."""
        self.handler.load_data(["AAPL"])
        
        # Set up split
        self.handler.setup_train_test_split(method="ratio", train_ratio=0.7)
        
        assert "train" in self.handler.splits
        assert "test" in self.handler.splits
        
        train_split = self.handler.splits["train"]
        test_split = self.handler.splits["test"]
        
        # Check sizes
        assert len(train_split.data["AAPL"]) == 70
        assert len(test_split.data["AAPL"]) == 30
        
        # Check no overlap
        train_end = train_split.data["AAPL"].index[-1]
        test_start = test_split.data["AAPL"].index[0]
        assert train_end < test_start
    
    def test_active_split_switching(self):
        """Test switching between train/test splits."""
        self.handler.load_data(["AAPL"])
        self.handler.setup_train_test_split(method="ratio", train_ratio=0.7)
        self.handler.start()
        
        # Use training data
        self.handler.set_active_split("train")
        
        train_bars = 0
        while self.handler.update_bars():
            train_bars += 1
        
        assert train_bars == 70
        
        # Switch to test data
        self.handler.set_active_split("test")
        
        test_bars = 0
        while self.handler.update_bars():
            test_bars += 1
        
        assert test_bars == 30
        
        # Total events should be train + test
        assert len(self.events) == 100
    
    def test_multi_symbol_synchronization(self):
        """Test synchronized emission across multiple symbols."""
        self.handler.load_data(["AAPL", "MSFT"])
        self.handler.start()
        
        # Collect all emitted bars
        bars = []
        while self.handler.update_bars():
            event = self.events[-1]
            bar_data = event.payload["data"]
            bars.append((
                bar_data["timestamp"],
                bar_data["symbol"]
            ))
        
        # Check chronological order
        timestamps = [b[0] for b in bars]
        assert timestamps == sorted(timestamps)
        
        # Check both symbols are represented
        symbols = set(b[1] for b in bars)
        assert symbols == {"AAPL", "MSFT"}
    
    def test_reset(self):
        """Test resetting handler."""
        self.handler.load_data(["AAPL"])
        self.handler.start()
        
        # Emit some bars
        for _ in range(10):
            self.handler.update_bars()
        
        # Reset
        self.handler.reset()
        
        # Should be able to emit all bars again
        bar_count = 0
        while self.handler.update_bars():
            bar_count += 1
        
        assert bar_count == 100