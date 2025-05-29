# Data Module with Protocol + Composition Architecture

## Overview

The Data module integrates seamlessly with our Protocol + Composition architecture. Instead of inheriting from abstract base classes, data components implement protocols and gain capabilities through composition.

## 1. Data Module Protocols

```python
from typing import Protocol, runtime_checkable, List, Optional, Dict, Any, Iterator
from abc import abstractmethod
from datetime import datetime
import pandas as pd

# === Core Data Protocols ===
@runtime_checkable
class DataProvider(Protocol):
    """Protocol for components that provide market data"""
    
    @abstractmethod
    def load_data(self, symbols: List[str]) -> bool:
        """Load data for specified symbols"""
        ...
    
    @abstractmethod
    def get_next_bar(self) -> Optional['Bar']:
        """Get next available bar across all symbols"""
        ...
    
    @abstractmethod
    def has_more_data(self) -> bool:
        """Check if more data is available"""
        ...

@runtime_checkable
class DataSplitter(Protocol):
    """Protocol for train/test splitting"""
    
    @abstractmethod
    def setup_split(self, method: str = 'ratio', **kwargs) -> None:
        """Set up train/test split"""
        ...
    
    @abstractmethod
    def set_active_split(self, split_name: Optional[str]) -> None:
        """Set active data split (train/test/None)"""
        ...
    
    @abstractmethod
    def get_split_info(self) -> Dict[str, Any]:
        """Get information about current splits"""
        ...

@runtime_checkable
class BarEmitter(Protocol):
    """Protocol for components that emit bar events"""
    
    @abstractmethod
    def update_bars(self) -> bool:
        """Update to next bar and emit event"""
        ...

@runtime_checkable
class DataAccessor(Protocol):
    """Protocol for accessing historical data"""
    
    @abstractmethod
    def get_latest_bar(self, symbol: str) -> Optional['Bar']:
        """Get the latest bar for a symbol"""
        ...
    
    @abstractmethod
    def get_latest_bars(self, symbol: str, N: int = 1) -> List['Bar']:
        """Get the last N bars for a symbol"""
        ...

@runtime_checkable
class DataValidator(Protocol):
    """Protocol for data validation"""
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> 'ValidationResult':
        """Validate data integrity"""
        ...
    
    @abstractmethod
    def get_validation_rules(self) -> List['ValidationRule']:
        """Get validation rules"""
        ...

@runtime_checkable
class DataTransformer(Protocol):
    """Protocol for data transformation"""
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data"""
        ...
    
    @abstractmethod
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform data"""
        ...
```

## 2. Data Module Capabilities

### 2.1 Data Loading Capability

```python
class DataLoadingCapability(Capability):
    """Adds data loading functionality to components"""
    
    def get_name(self) -> str:
        return "data_loading"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        # Initialize data containers
        if not hasattr(component, 'data'):
            component.data = {}
            component.symbols = []
            component.data_loaded = False
        
        # Add data loader
        loader_type = spec.get('loader_type', 'csv')
        if loader_type == 'csv':
            component.data_loader = CSVLoader(
                data_dir=spec.get('data_dir', 'data'),
                date_column=spec.get('date_column', 'Date'),
                parse_dates=True
            )
        elif loader_type == 'parquet':
            component.data_loader = ParquetLoader(
                data_dir=spec.get('data_dir', 'data')
            )
        # ... other loader types
        
        # Add load_data method
        def load_data(symbols: List[str]) -> bool:
            """Load data for specified symbols"""
            component.symbols = symbols
            success = True
            
            for symbol in symbols:
                try:
                    data = component.data_loader.load(symbol)
                    component.data[symbol] = data
                    if hasattr(component, 'logger'):
                        component.logger.info(f"Loaded data for {symbol}", 
                                            rows=len(data))
                except Exception as e:
                    success = False
                    if hasattr(component, 'logger'):
                        component.logger.error(f"Failed to load {symbol}", 
                                             error=str(e))
            
            component.data_loaded = success
            return success
        
        component.load_data = load_data
        
        # Add data access methods
        component.get_symbol_data = lambda symbol: component.data.get(symbol)
        component.get_loaded_symbols = lambda: list(component.data.keys())
        
        return component
```

### 2.2 Data Splitting Capability

```python
class DataSplittingCapability(Capability):
    """Adds train/test splitting functionality"""
    
    def get_name(self) -> str:
        return "data_splitting"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        # Initialize split storage
        if not hasattr(component, 'splits'):
            component.splits = {}
            component.active_split = None
        
        # Create splitter
        component.splitter = TimeSeriesSplitter()
        
        # Add setup_split method
        def setup_split(method: str = 'ratio', train_ratio: float = 0.7, 
                       split_date: Optional[str] = None, **kwargs) -> None:
            """Set up train/test split"""
            if not hasattr(component, 'data') or not component.data:
                raise ValueError("No data loaded")
            
            for symbol, data in component.data.items():
                if method == 'ratio':
                    train_data, test_data = component.splitter.split_by_ratio(
                        data, train_ratio
                    )
                elif method == 'date':
                    train_data, test_data = component.splitter.split_by_date(
                        data, split_date
                    )
                else:
                    raise ValueError(f"Unknown split method: {method}")
                
                component.splits[symbol] = {
                    'train': train_data,
                    'test': test_data,
                    'full': data
                }
            
            if hasattr(component, 'logger'):
                component.logger.info(f"Set up {method} split", 
                                    train_ratio=train_ratio if method == 'ratio' else None,
                                    split_date=split_date if method == 'date' else None)
        
        component.setup_split = setup_split
        
        # Add set_active_split method
        def set_active_split(split_name: Optional[str]) -> None:
            """Set active data split"""
            if split_name and split_name not in ['train', 'test', None]:
                raise ValueError(f"Invalid split name: {split_name}")
            
            component.active_split = split_name
            
            # Reset any indices if component tracks them
            if hasattr(component, 'current_indices'):
                for symbol in component.symbols:
                    component.current_indices[symbol] = 0
            
            if hasattr(component, 'logger'):
                component.logger.info(f"Active split set to: {split_name or 'full'}")
        
        component.set_active_split = set_active_split
        
        # Add split info method
        def get_split_info() -> Dict[str, Any]:
            """Get information about splits"""
            info = {
                'active_split': component.active_split,
                'splits_configured': len(component.splits) > 0,
                'symbols': {}
            }
            
            for symbol, splits in component.splits.items():
                info['symbols'][symbol] = {
                    'train_size': len(splits.get('train', [])),
                    'test_size': len(splits.get('test', [])),
                    'full_size': len(splits.get('full', []))
                }
            
            return info
        
        component.get_split_info = get_split_info
        
        # Helper to get active data
        def _get_active_data(symbol: str) -> pd.DataFrame:
            """Get currently active dataset for symbol"""
            if symbol not in component.splits:
                return component.data.get(symbol, pd.DataFrame())
            
            if component.active_split:
                return component.splits[symbol].get(component.active_split, pd.DataFrame())
            else:
                return component.splits[symbol].get('full', pd.DataFrame())
        
        component._get_active_data = _get_active_data
        
        return component
```

### 2.3 Bar Emission Capability

```python
class BarEmissionCapability(Capability):
    """Adds bar event emission functionality"""
    
    def get_name(self) -> str:
        return "bar_emission"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        # Initialize tracking
        if not hasattr(component, 'current_indices'):
            component.current_indices = {}
        
        # Add update_bars method
        def update_bars() -> bool:
            """Update to next bar and emit event"""
            if not hasattr(component, '_events') or not component._events.event_bus:
                raise ValueError("Component must have event capability for bar emission")
            
            # Find next bar across all symbols
            next_bar = None
            next_symbol = None
            next_timestamp = None
            
            for symbol in component.symbols:
                data = component._get_active_data(symbol) if hasattr(component, '_get_active_data') else component.data.get(symbol)
                if data is None or data.empty:
                    continue
                
                idx = component.current_indices.get(symbol, 0)
                if idx < len(data):
                    timestamp = data.index[idx]
                    if next_timestamp is None or timestamp < next_timestamp:
                        next_timestamp = timestamp
                        next_symbol = symbol
                        next_bar = data.iloc[idx]
            
            if next_bar is None:
                return False  # No more data
            
            # Create Bar object
            bar = Bar(
                symbol=next_symbol,
                timestamp=next_timestamp,
                open_price=next_bar['Open'],
                high=next_bar['High'],
                low=next_bar['Low'],
                close=next_bar['Close'],
                volume=next_bar.get('Volume', 0)
            )
            
            # Emit event
            event = Event(EventType.BAR, bar.to_dict())
            component._events.event_bus.publish(event)
            
            # Update index
            component.current_indices[next_symbol] += 1
            
            # Check if more data available
            return any(
                component.current_indices.get(sym, 0) < len(
                    component._get_active_data(sym) if hasattr(component, '_get_active_data') 
                    else component.data.get(sym, [])
                )
                for sym in component.symbols
            )
        
        component.update_bars = update_bars
        
        # Add convenience method
        def has_more_data() -> bool:
            """Check if more data is available"""
            return any(
                component.current_indices.get(sym, 0) < len(
                    component._get_active_data(sym) if hasattr(component, '_get_active_data')
                    else component.data.get(sym, [])
                )
                for sym in component.symbols
            )
        
        component.has_more_data = has_more_data
        
        return component
```

### 2.4 Data Access Capability

```python
class DataAccessCapability(Capability):
    """Adds historical data access functionality"""
    
    def get_name(self) -> str:
        return "data_access"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        # Initialize bar cache
        if not hasattr(component, 'bar_cache'):
            component.bar_cache = {}
            component.max_cache_size = spec.get('max_cache_size', 1000)
        
        # Add get_latest_bar method
        def get_latest_bar(symbol: str) -> Optional[Bar]:
            """Get the latest bar for a symbol"""
            data = component._get_active_data(symbol) if hasattr(component, '_get_active_data') else component.data.get(symbol)
            if data is None or data.empty:
                return None
            
            idx = component.current_indices.get(symbol, 0)
            if idx == 0:
                return None
            
            # Get the last processed bar
            bar_data = data.iloc[idx - 1]
            return Bar(
                symbol=symbol,
                timestamp=data.index[idx - 1],
                open_price=bar_data['Open'],
                high=bar_data['High'],
                low=bar_data['Low'],
                close=bar_data['Close'],
                volume=bar_data.get('Volume', 0)
            )
        
        component.get_latest_bar = get_latest_bar
        
        # Add get_latest_bars method
        def get_latest_bars(symbol: str, N: int = 1) -> List[Bar]:
            """Get the last N bars for a symbol"""
            data = component._get_active_data(symbol) if hasattr(component, '_get_active_data') else component.data.get(symbol)
            if data is None or data.empty:
                return []
            
            idx = component.current_indices.get(symbol, 0)
            start_idx = max(0, idx - N)
            
            bars = []
            for i in range(start_idx, idx):
                bar_data = data.iloc[i]
                bar = Bar(
                    symbol=symbol,
                    timestamp=data.index[i],
                    open_price=bar_data['Open'],
                    high=bar_data['High'],
                    low=bar_data['Low'],
                    close=bar_data['Close'],
                    volume=bar_data.get('Volume', 0)
                )
                bars.append(bar)
            
            return bars
        
        component.get_latest_bars = get_latest_bars
        
        # Add direct data access
        def get_data_slice(symbol: str, start_idx: int, end_idx: int) -> pd.DataFrame:
            """Get a slice of data"""
            data = component._get_active_data(symbol) if hasattr(component, '_get_active_data') else component.data.get(symbol)
            if data is None:
                return pd.DataFrame()
            return data.iloc[start_idx:end_idx]
        
        component.get_data_slice = get_data_slice
        
        return component
```

### 2.5 Data Validation Capability

```python
class DataValidationCapability(Capability):
    """Adds data validation functionality"""
    
    def get_name(self) -> str:
        return "data_validation"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        # Create validators
        validators = []
        
        # Add default validators
        validators.append(ContinuityValidator(
            timestamp_col=spec.get('timestamp_col', 'index'),
            expected_frequency=spec.get('expected_frequency', '1D')
        ))
        
        validators.append(OHLCValidator())
        
        if spec.get('detect_outliers', True):
            validators.append(OutlierDetector(
                method=spec.get('outlier_method', 'iqr'),
                threshold=spec.get('outlier_threshold', 3.0)
            ))
        
        component.validators = validators
        
        # Add validation method
        def validate_data(data: pd.DataFrame = None, symbol: str = None) -> ValidationResult:
            """Validate data integrity"""
            if data is None and symbol:
                data = component.data.get(symbol)
            
            if data is None:
                return ValidationResult(
                    passed=False,
                    errors=["No data to validate"],
                    warnings=[],
                    metadata={}
                )
            
            # Run all validators
            all_errors = []
            all_warnings = []
            all_metadata = {}
            
            for validator in component.validators:
                result = validator.validate(data)
                all_errors.extend(result.errors)
                all_warnings.extend(result.warnings)
                all_metadata.update(result.metadata)
            
            passed = len(all_errors) == 0
            
            if hasattr(component, 'logger'):
                if passed:
                    component.logger.info("Data validation passed", 
                                        warnings=len(all_warnings))
                else:
                    component.logger.error("Data validation failed", 
                                         errors=len(all_errors))
            
            return ValidationResult(
                passed=passed,
                errors=all_errors,
                warnings=all_warnings,
                metadata=all_metadata
            )
        
        component.validate_data = validate_data
        
        # Add validation rules getter
        component.get_validation_rules = lambda: [v.__class__.__name__ for v in component.validators]
        
        # Auto-validate on data load if configured
        if spec.get('validate_on_load', True):
            original_load = component.load_data
            
            def validated_load(symbols: List[str]) -> bool:
                success = original_load(symbols)
                if success:
                    for symbol in symbols:
                        result = component.validate_data(symbol=symbol)
                        if not result.passed:
                            if hasattr(component, 'logger'):
                                component.logger.error(f"Validation failed for {symbol}")
                            success = False
                return success
            
            component.load_data = validated_load
        
        return component
```

## 3. Memory-Efficient Data Components

### 3.1 Memory Optimization Capability

```python
class MemoryOptimizationCapability(Capability):
    """Adds memory optimization features to data components"""
    
    def get_name(self) -> str:
        return "memory_optimization"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        # Add memory tracking
        component.memory_tracker = MemoryTracker()
        
        # Add optimization method
        def optimize_memory() -> Dict[str, Any]:
            """Optimize memory usage of loaded data"""
            before_memory = component.memory_tracker.get_process_memory()
            optimizations = {}
            
            for symbol, data in component.data.items():
                if isinstance(data, pd.DataFrame):
                    # Optimize dtypes
                    original_memory = data.memory_usage(deep=True).sum()
                    optimized_data = component.memory_tracker.optimize_dtypes(data)
                    optimized_memory = optimized_data.memory_usage(deep=True).sum()
                    
                    component.data[symbol] = optimized_data
                    
                    optimizations[symbol] = {
                        'original_mb': original_memory / (1024**2),
                        'optimized_mb': optimized_memory / (1024**2),
                        'reduction_pct': (1 - optimized_memory/original_memory) * 100
                    }
            
            # Force garbage collection
            import gc
            gc.collect()
            
            after_memory = component.memory_tracker.get_process_memory()
            
            return {
                'process_memory_before_mb': before_memory['rss_mb'],
                'process_memory_after_mb': after_memory['rss_mb'],
                'per_symbol_optimization': optimizations
            }
        
        component.optimize_memory = optimize_memory
        
        # Add memory usage reporting
        def get_memory_usage() -> Dict[str, Any]:
            """Get current memory usage"""
            process_memory = component.memory_tracker.get_process_memory()
            data_memory = {}
            
            for symbol, data in component.data.items():
                if isinstance(data, pd.DataFrame):
                    data_memory[symbol] = component.memory_tracker.get_dataframe_memory(data)
            
            return {
                'process': process_memory,
                'data': data_memory
            }
        
        component.get_memory_usage = get_memory_usage
        
        # Auto-optimize if configured
        if spec.get('auto_optimize', True):
            original_load = component.load_data
            
            def optimized_load(symbols: List[str]) -> bool:
                success = original_load(symbols)
                if success:
                    component.optimize_memory()
                return success
            
            component.load_data = optimized_load
        
        return component
```

### 3.2 Isolation Strategy Capability

```python
class IsolationStrategyCapability(Capability):
    """Adds configurable data isolation strategies"""
    
    def get_name(self) -> str:
        return "isolation_strategy"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        # Determine isolation mode
        isolation_mode = spec.get('isolation_mode', 'auto')
        
        if isolation_mode == 'auto':
            # Auto-select based on data size
            component.isolation_factory = DataIsolationFactory()
        else:
            component.isolation_mode = IsolationMode[isolation_mode.upper()]
        
        # Override setup_split to use isolation strategy
        if hasattr(component, 'setup_split'):
            original_setup_split = component.setup_split
            
            def isolated_setup_split(method: str = 'ratio', **kwargs) -> None:
                """Set up split with isolation strategy"""
                # First do the normal split
                original_setup_split(method, **kwargs)
                
                # Then apply isolation strategy
                for symbol in component.symbols:
                    if symbol in component.splits:
                        data = component.data[symbol]
                        
                        if hasattr(component, 'isolation_factory'):
                            isolation = component.isolation_factory.create_isolation(
                                data, mode=None  # Auto-select
                            )
                        else:
                            isolation = component.isolation_factory.create_isolation(
                                data, mode=component.isolation_mode
                            )
                        
                        # Replace splits with isolated versions
                        train_view, test_view = isolation.get_train_test_split(
                            train_ratio=kwargs.get('train_ratio', 0.7)
                        )
                        
                        component.splits[symbol]['train'] = train_view
                        component.splits[symbol]['test'] = test_view
                        component.splits[symbol]['isolation'] = isolation
                
                if hasattr(component, 'logger'):
                    component.logger.info("Applied isolation strategy to splits")
            
            component.setup_split = isolated_setup_split
        
        return component
```

## 4. Complete Data Handler Implementation

```python
class HistoricalDataHandler:
    """Data handler using composition - no inheritance needed"""
    
    def __init__(self, name: str = "historical_data", 
                 data_dir: str = "data",
                 symbols: List[str] = None):
        self.name = name
        self.data_dir = data_dir
        self.symbols = symbols or []
        
        # No base class, just the data we need
        self.data = {}
        self.current_indices = {}
        
        # Capabilities will be added by the factory
        # based on configuration

# Usage with capabilities
data_handler = ComponentFactory().create_component({
    'name': 'data_handler',
    'class': 'HistoricalDataHandler',
    'params': {
        'data_dir': 'data',
        'symbols': ['EURUSD', 'GBPUSD']
    },
    'capabilities': [
        'lifecycle',
        'events', 
        'data_loading',
        'data_splitting',
        'bar_emission',
        'data_access',
        'data_validation',
        'memory_optimization',
        'logging',
        'monitoring'
    ],
    
    # Data-specific configuration
    'loader_type': 'csv',
    'expected_frequency': '1D',
    'validate_on_load': True,
    'auto_optimize': True,
    'isolation_mode': 'auto',
    
    # Infrastructure configuration
    'logger_name': 'data.historical',
    'track_performance': ['update_bars', 'load_data']
})
```

## 5. Specialized Data Components

### 5.1 Streaming Data Handler

```python
class StreamingDataHandler:
    """Real-time data handler with composition"""
    
    def __init__(self, name: str = "streaming_data",
                 websocket_url: str = None):
        self.name = name
        self.websocket_url = websocket_url
        self.buffer = asyncio.Queue(maxsize=1000)
        self.connected = False

# Configuration
streaming_config = {
    'class': 'StreamingDataHandler',
    'params': {
        'websocket_url': 'wss://market-data.example.com'
    },
    'capabilities': [
        'lifecycle',
        'events',
        'bar_emission',
        'data_access',
        'logging',
        'monitoring',
        'error_handling'
    ],
    'error_handling': {
        'retry': {
            'max_attempts': 5,
            'backoff': 'exponential'
        },
        'critical_methods': ['connect', 'process_message']
    }
}
```

### 5.2 Multi-Resolution Data Handler

```python
class MultiResolutionDataHandler:
    """Handles multiple timeframes with composition"""
    
    def __init__(self, name: str = "multi_resolution",
                 base_timeframe: str = "1min"):
        self.name = name
        self.base_timeframe = base_timeframe
        self.resolutions = {}
        self.aggregators = {}

# Configuration with multiple capabilities
multi_res_config = {
    'class': 'MultiResolutionDataHandler',
    'params': {
        'base_timeframe': '1min'
    },
    'capabilities': [
        'lifecycle',
        'events',
        'data_loading',
        'data_transformation',  # Custom capability for aggregation
        'bar_emission',
        'data_access',
        'memory_optimization'
    ],
    'resolutions': ['1min', '5min', '15min', '1H', '1D'],
    'aggregation_method': 'ohlc'
}
```

## 6. Data Module Configuration Examples

### 6.1 Development Configuration

```yaml
data:
  historical:
    class: "HistoricalDataHandler"
    capabilities: 
      - "lifecycle"
      - "events"
      - "data_loading"
      - "data_splitting"
      - "bar_emission"
      - "data_access"
      - "logging"
      - "debugging"
    
    params:
      data_dir: "test_data"
    
    # Simple CSV loading
    loader_type: "csv"
    
    # Full logging for development
    logger_name: "data.dev"
    log_level: "DEBUG"
    
    # Enable debugging
    debugging:
      trace_enabled: true
      capture_state_on_error: true
```

### 6.2 Production Configuration

```yaml
data:
  market_data:
    class: "OptimizedDataHandler"
    profile: "production_data"  # Uses predefined profile
    
    params:
      data_dir: "market_data"
      cache_size: 10000
    
    # Production optimizations
    loader_type: "parquet"
    memory_optimization:
      auto_optimize: true
      compression: "snappy"
    
    isolation_mode: "shared_memory"  # For multi-process access
    
    # Validation
    validation:
      validate_on_load: true
      outlier_detection: true
      continuity_check: true
    
    # Monitoring
    monitoring:
      track_performance: ["load_data", "update_bars"]
      alert_on_gaps: true
```

### 6.3 Backtesting Configuration

```yaml
data:
  backtest_data:
    class: "HistoricalDataHandler"
    capabilities: 
      - "lifecycle"
      - "events"
      - "data_loading"
      - "data_splitting"
      - "bar_emission"
      - "data_access"
      - "data_validation"
      - "memory_optimization"
      - "logging"
      - "monitoring"
    
    params:
      data_dir: "historical"
    
    # Backtesting specific
    train_test_split:
      method: "ratio"
      train_ratio: 0.7
      validate_no_overlap: true
    
    # Memory efficient for large backtests
    memory_optimization:
      auto_optimize: true
      use_float32: true
      categorize_strings: true
    
    isolation_mode: "copy_on_write"  # Balance memory and flexibility
```

## 7. Benefits of Protocol + Composition for Data Module

### 7.1 Flexibility in Data Sources

```python
# Any data source can be used as long as it implements the protocol
class CustomDataSource:
    """Custom data source - no inheritance needed"""
    
    def load_data(self, symbols: List[str]) -> bool:
        # Custom implementation
        pass
    
    def get_next_bar(self) -> Optional[Bar]:
        # Custom implementation
        pass

# Can be enhanced with standard capabilities
data_source = add_capabilities(CustomDataSource(), [
    'events',
    'logging',
    'monitoring'
])
```

### 7.2 Testability

```python
def test_data_handler():
    # Create data handler with test configuration
    handler = create_component({
        'class': 'HistoricalDataHandler',
        'capabilities': ['data_loading', 'data_splitting'],
        'params': {'data_dir': 'test_data'}
    })
    
    # Test data loading
    assert handler.load_data(['TEST'])
    
    # Test splitting
    handler.setup_split(method='ratio', train_ratio=0.8)
    split_info = handler.get_split_info()
    
    assert split_info['active_split'] is None
    assert split_info['symbols']['TEST']['train_size'] > 0
```

### 7.3 Performance Optimization

```python
# Start with basic data handler
basic_handler = HistoricalDataHandler()

# Add only needed capabilities
if config.needs_splitting:
    basic_handler = add_capability(basic_handler, DataSplittingCapability())

if config.large_dataset:
    basic_handler = add_capability(basic_handler, MemoryOptimizationCapability())

# No overhead from unused features!
```

## 8. Integration with Container Architecture

The data module works seamlessly with our container architecture:

```python
class DataContainer(UniversalScopedContainer):
    """Specialized container for data components"""
    
    def create_data_handler(self, spec: Dict[str, Any]) -> Any:
        """Create data handler with appropriate isolation"""
        
        # Shared read-only data across containers
        if spec.get('shared_data', False):
            # Use shared memory or memory-mapped files
            spec['isolation_mode'] = 'shared_memory'
        
        # Create handler with capabilities
        handler = self.create_component(spec)
        
        # Register as shared service if needed
        if spec.get('shared_data', False):
            self.register_shared_service('market_data', handler)
        
        return handler
```

## Summary

The Data module with Protocol + Composition architecture provides:

1. **Flexible data sources** - Any component can provide data by implementing protocols
2. **Composable capabilities** - Add only the features you need
3. **Memory efficiency** - Multiple isolation strategies without inheritance complexity
4. **Easy testing** - Mock or real data sources with the same interface
5. **Production ready** - Full monitoring, validation, and error handling when needed
6. **Container compatible** - Works perfectly with our isolation architecture

This approach maintains the simplicity of data handling while providing enterprise-grade features through composition rather than inheritance.
