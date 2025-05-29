# ADMF-PC Data Module Documentation

## Overview

The `src/data` module provides comprehensive data loading, management, and distribution capabilities for the ADMF-PC system. It is designed to handle market data efficiently while maintaining complete isolation between different execution containers (backtests, optimizations, etc.).

## Key Features

- **Multiple Data Sources**: Extensible loader system (currently CSV, easily extendable)
- **Memory Optimization**: Efficient data structures and chunked loading for large datasets
- **Event-Driven Distribution**: Seamless integration with the container event system
- **Multi-Symbol Synchronization**: Accurate chronological ordering across multiple instruments
- **Train/Test Splitting**: Built-in support for ML-style data splitting
- **Container Isolation**: Each data handler operates within its own container context

## Architecture

```
src/data/
├── __init__.py      # Module exports and documentation
├── models.py        # Data structures (Bar, Tick, Timeframe, etc.)
├── loaders.py       # Data loading implementations
└── handlers.py      # Data management and event distribution
```

## Data Models

### Timeframe
Enumeration of standard market timeframes with utility methods:

```python
from src.data import Timeframe

# Available timeframes
Timeframe.TICK  # Tick data
Timeframe.M1    # 1 minute
Timeframe.M5    # 5 minutes
Timeframe.H1    # 1 hour
Timeframe.D1    # 1 day
# ... and more

# Get duration in seconds
seconds = Timeframe.D1.seconds  # 86400
```

### Bar
Core OHLCV (Open, High, Low, Close, Volume) data structure:

```python
from src.data import Bar
from datetime import datetime

bar = Bar(
    symbol="AAPL",
    timestamp=datetime(2023, 1, 1),
    open=150.0,
    high=155.0,
    low=149.0,
    close=154.0,
    volume=1000000,
    timeframe=Timeframe.D1
)

# Utility properties
bar.range        # 6.0 (high - low)
bar.body         # 4.0 (abs(close - open))
bar.is_bullish   # True (close > open)

# Serialization
bar_dict = bar.to_dict()
bar_restored = Bar.from_dict(bar_dict)
```

### Tick
Represents individual trades or quotes:

```python
from src.data import Tick

tick = Tick(
    symbol="AAPL",
    timestamp=datetime.now(),
    price=150.25,
    volume=100,
    bid=150.20,
    ask=150.30
)

spread = tick.spread  # 0.10
```

### DataView
Memory-efficient windowed access to data:

```python
from src.data import DataView

# Create a view of a DataFrame
view = DataView(df, start_idx=0, end_idx=1000)

# Navigate through data
while view.has_data:
    current = view.get_current()
    window = view.get_window(size=20)  # Last 20 bars
    view.advance()

progress = view.progress  # 0.0 to 1.0
```

### TimeSeriesData
Optimized storage for time series with separate timestamp/value arrays:

```python
from src.data import TimeSeriesData

# Create from DataFrame
ts_data = TimeSeriesData.from_dataframe(df)

# Get a view
subset = ts_data.get_view(start_idx=100, end_idx=200)

# Convert back to DataFrame
df = ts_data.to_dataframe()
```

## Data Loaders

### CSVLoader
Standard CSV file loader with intelligent parsing:

```python
from src.data import CSVLoader

loader = CSVLoader(
    data_dir="data/stocks",
    date_column="Date",
    date_format="%Y-%m-%d"  # Optional, auto-detected if None
)

# Automatically handles:
# - Multiple delimiters (comma, semicolon, tab)
# - Various column name formats (Open/open/OPEN)
# - Date parsing
# - Missing data
# - OHLCV validation
df = loader.load("AAPL")
```

### MemoryOptimizedCSVLoader
For large datasets with memory constraints:

```python
from src.data import MemoryOptimizedCSVLoader

loader = MemoryOptimizedCSVLoader(
    data_dir="data/stocks",
    chunk_size=10000,     # Read in chunks
    optimize_types=True   # Use float32, optimize integers
)

# Loads data in chunks and optimizes data types
df = loader.load("AAPL")  # Uses ~50% less memory
```

### MultiFileLoader
For data split across multiple files:

```python
from src.data import MultiFileLoader, CSVLoader

base_loader = CSVLoader(data_dir="data/stocks")
multi_loader = MultiFileLoader(base_loader)

# Loads and concatenates files matching pattern
# e.g., AAPL_2020.csv, AAPL_2021.csv, AAPL_2022.csv
df = multi_loader.load("AAPL", file_pattern="{symbol}_{year}.csv")
```

## Data Handlers

### HistoricalDataHandler
Main data handler for backtesting with event emission:

```python
from src.data import HistoricalDataHandler, Timeframe

# Create handler
handler = HistoricalDataHandler(
    handler_id="historical_data",
    data_dir="data/stocks",
    timeframe=Timeframe.D1
)

# Initialize with container context (required)
handler.initialize({
    'event_bus': container.event_bus,
    'container_id': container.container_id
})

# Load data for multiple symbols
handler.load_data(['AAPL', 'GOOGL', 'MSFT'])

# Set up train/test split
handler.setup_train_test_split(
    method='ratio',      # or 'date'
    train_ratio=0.8,     # 80% train, 20% test
    # split_date=datetime(2022, 1, 1)  # for date-based split
)

# Use training data
handler.set_active_split('train')  # 'train', 'test', or None for full

# Start data emission
handler.start()

# Emit bars in chronological order across all symbols
while handler.update_bars():
    # Each bar is published as an event to the event bus
    # Strategies receive them via event subscriptions
    pass

# Access latest data
latest_bar = handler.get_latest_bar('AAPL')
last_20_bars = handler.get_latest_bars('AAPL', n=20)

# Switch to test data
handler.set_active_split('test')
handler.reset()  # Reset indices
```

## Event Integration

Data handlers publish market events to the container's event bus:

```python
# In a strategy component
class MyStrategy:
    def initialize_events(self):
        self.event_bus.subscribe(EventType.BAR, self.on_bar)
    
    def on_bar(self, event: Event):
        bar_data = event.payload['data']
        symbol = event.payload['symbol']
        
        # Bar data is a dictionary
        bar = Bar.from_dict(bar_data)
        
        # Process the bar
        signal = self.generate_signal(bar)
```

## Usage Patterns

### Basic Backtesting Setup

```python
# In a backtest container
from src.data import HistoricalDataHandler

# 1. Create and initialize handler
handler = HistoricalDataHandler("hist_data", "data/stocks")
handler.initialize(context)

# 2. Load data
symbols = ['AAPL', 'GOOGL', 'MSFT']
handler.load_data(symbols)

# 3. Start emitting data
handler.start()
while handler.update_bars():
    # Strategies process bars via events
    pass
```

### Walk-Forward Analysis

```python
# Set up expanding window for walk-forward testing
handler.setup_train_test_split(method='ratio', train_ratio=0.7)

# Train on first 70%
handler.set_active_split('train')
handler.start()
# ... run training/optimization ...

# Test on remaining 30%
handler.set_active_split('test')
handler.reset()
# ... run validation ...
```

### Custom Data Loading

```python
# Extend DataLoader for custom sources
from src.data import DataLoader

class DatabaseLoader(DataLoader):
    def __init__(self, connection_string: str):
        self.conn = create_connection(connection_string)
    
    def load(self, symbol: str, **kwargs) -> pd.DataFrame:
        query = f"SELECT * FROM prices WHERE symbol = '{symbol}'"
        df = pd.read_sql(query, self.conn)
        return df
    
    def validate(self, df: pd.DataFrame) -> bool:
        # Implement validation logic
        return True
```

## Performance Considerations

### Memory Optimization
- Use `MemoryOptimizedCSVLoader` for large datasets
- Data types optimized (float32 for prices, appropriate int types for volume)
- `DataView` provides windowed access without copying
- `TimeSeriesData` stores values in numpy arrays

### Loading Performance
- Chunked reading for large files
- Parallel loading possible for multiple symbols
- Pre-sorted timeline for efficient emission

### Event System Overhead
- Only active bar is converted to event (lazy evaluation)
- Events use dictionaries for flexibility
- Container isolation prevents cross-talk

## Data Quality and Validation

The module performs several validation checks:

1. **OHLC Relationships**
   - High ≥ Low
   - High ≥ Open, Close
   - Low ≤ Open, Close

2. **Volume Validation**
   - Volume ≥ 0

3. **Duplicate Detection**
   - No duplicate timestamps

4. **Missing Data Handling**
   - Forward fill for prices
   - Zero fill for volume
   - Rows with remaining NaN are dropped

## Current Limitations and Missing Features

### Data Source Limitations
Currently, the module only supports CSV files. Notable missing data sources include:
- **Databases**: No native PostgreSQL, MySQL, MongoDB, or TimescaleDB support
- **APIs**: No REST or WebSocket integrations for real-time data
- **Binary Formats**: No support for HDF5, Parquet, Arrow, or other efficient formats
- **Cloud Storage**: No direct S3, GCS, or Azure Blob support
- **Market Data Vendors**: No built-in integrations with Bloomberg, Reuters, etc.

### Data Type Limitations
The module currently supports only OHLCV bars and basic tick data. Missing types include:
- **Order Book Data**: No Level 2/market depth structures
- **Options Data**: No options chains, greeks, or implied volatility
- **Fundamental Data**: No earnings, financials, or economic indicators
- **Alternative Data**: No sentiment, news, or social media data structures
- **Futures/Derivatives**: No futures-specific fields (open interest, settlement)

### Processing Capabilities Not Yet Implemented
- **Corporate Actions**: No automatic adjustment for splits, dividends
- **Data Alignment**: No built-in alignment across different data sources/frequencies
- **Resampling**: No automatic timeframe conversion (e.g., 1min to 5min bars)
- **Data Quality**: No outlier detection, gap filling, or quality scoring
- **Normalization**: No cross-sectional normalization or standardization

### Performance and Scalability Gaps
- **Caching**: No intelligent caching layer for frequently accessed data
- **Parallel Loading**: Single-threaded loading (no parallel symbol loading)
- **Streaming**: No support for continuous data streaming
- **Compression**: No data compression for storage efficiency
- **Indexing**: No advanced indexing for fast data retrieval

### Real-time and Live Trading Limitations
- **Live Data**: No real-time data handling infrastructure
- **Update Mechanisms**: No support for incremental updates
- **Replay**: No tick-by-tick replay capabilities
- **Latency Tracking**: No timestamps for latency measurement

## Future Enhancements

### Planned Features
1. **Additional Data Sources**
   - Database loaders (PostgreSQL, MySQL, MongoDB)
   - API loaders (REST, WebSocket)
   - Binary formats (HDF5, Parquet, Arrow)

2. **Advanced Data Features**
   - Real-time data handling
   - Corporate action adjustments
   - Currency conversion
   - Data quality metrics

3. **Performance Improvements**
   - Caching layer for frequently accessed data
   - Parallel symbol loading
   - Memory-mapped file support

### Extension Points

The module is designed for easy extension:

1. **Custom Loaders**: Inherit from `DataLoader`
2. **Custom Models**: Create new data structures
3. **Custom Handlers**: Inherit from `DataHandler`
4. **Event Types**: Add new market event types

## Best Practices

### 1. Data Organization
```
data/
├── stocks/
│   ├── AAPL.csv
│   ├── GOOGL.csv
│   └── MSFT.csv
├── forex/
│   ├── EURUSD.csv
│   └── GBPUSD.csv
└── crypto/
    ├── BTCUSD.csv
    └── ETHUSD.csv
```

### 2. CSV Format
```csv
Date,Open,High,Low,Close,Volume
2023-01-01,150.00,155.00,149.00,154.00,1000000
2023-01-02,154.00,156.00,153.00,155.50,1100000
```

### 3. Memory Management
- Use appropriate loader for dataset size
- Consider train/test splits for large datasets
- Monitor memory usage with system tools

### 4. Error Handling
```python
try:
    handler.load_data(symbols)
except FileNotFoundError as e:
    logger.error(f"Data file missing: {e}")
except ValueError as e:
    logger.error(f"Invalid data: {e}")
```

## Testing

Run tests with:
```bash
pytest tests/test_data.py -v
```

Test coverage includes:
- Data model validation
- CSV parsing edge cases
- Multi-symbol synchronization
- Train/test splitting
- Event emission

## API Reference

### Core Classes
- `Bar` - OHLCV data point
- `Tick` - Single trade/quote
- `DataView` - Windowed data access
- `TimeSeriesData` - Efficient time series storage
- `CSVLoader` - Standard CSV loading
- `MemoryOptimizedCSVLoader` - Memory-efficient loading
- `MultiFileLoader` - Multi-file aggregation
- `HistoricalDataHandler` - Main data handler

### Enumerations
- `Timeframe` - Standard market timeframes

### Key Methods
- `load_data(symbols)` - Load market data
- `update_bars()` - Emit next chronological bar
- `get_latest_bar(symbol)` - Get most recent bar
- `setup_train_test_split()` - Configure data splitting
- `set_active_split(name)` - Switch between splits

## Contributing

When extending the data module:

1. Follow existing patterns for consistency
2. Maintain backward compatibility
3. Add comprehensive docstrings
4. Include unit tests
5. Update this documentation

## License

Part of the ADMF-PC system. See project license for details.