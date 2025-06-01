# Step 7: Signal Capture

**Status**: Signal Capture & Replay Step
**Complexity**: Medium-High
**Prerequisites**: [Step 6: Multiple Classifiers](../02-container-architecture/step-06-multiple-classifiers.md) completed
**Architecture Ref**: [EVENT-DRIVEN-ARCHITECTURE.md](../../architecture/01-EVENT-DRIVEN-ARCHITECTURE.md#event-sourcing-pattern)

## 🎯 Objective

Implement comprehensive signal capture system:
- Capture all trading signals with full context
- Store indicator values at signal generation time
- Record market conditions and portfolio state
- Enable signal analysis and debugging
- Prepare for fast replay-based optimization

## 📋 Required Reading

Before starting:
1. [Event Sourcing Pattern](../../architecture/01-EVENT-DRIVEN-ARCHITECTURE.md#event-sourcing-pattern)
2. [Storage Patterns](../references/storage-patterns.md)
3. [Signal Analysis Guide](../references/signal-analysis.md)

## 🏗️ Implementation Tasks

### 1. Signal Capture Infrastructure

```python
# src/execution/signal_capture.py
from dataclasses import dataclass, asdict
import pandas as pd
import pyarrow.parquet as pq
import h5py
from typing import Dict, List, Optional, Any
import json
import gzip

@dataclass
class CapturedSignal:
    """
    Complete signal snapshot with all context needed for replay.
    """
    # Signal identification
    signal_id: str
    timestamp: datetime
    strategy_id: str
    
    # Core signal data
    symbol: str
    direction: Direction
    strength: float
    signal_type: str  # 'entry', 'exit', 'adjustment'
    
    # Market context
    market_data: Dict[str, float]  # OHLCV + extras
    spread: float
    liquidity_score: float
    
    # Indicator snapshot
    indicators: Dict[str, float]
    indicator_metadata: Dict[str, Dict]  # periods, params
    
    # Portfolio context
    portfolio_state: Dict[str, Any]
    position_before: Optional[float]
    capital_available: float
    
    # Risk context
    risk_metrics: Dict[str, float]
    active_limits: List[str]
    
    # Regime context
    market_regime: Optional[MarketRegime]
    regime_confidence: Optional[float]
    
    # Strategy parameters
    strategy_params: Dict[str, Any]
    
    # Metadata
    capture_version: str = "1.0"
    processing_time_ms: float = 0.0

class SignalCaptureEngine:
    """
    Captures and stores all signals with comprehensive context.
    Supports multiple storage backends.
    """
    
    def __init__(self, config: SignalCaptureConfig):
        self.config = config
        self.storage_backend = self._create_storage_backend()
        self.capture_buffer: List[CapturedSignal] = []
        self.buffer_size = config.buffer_size
        self.compression = config.compression
        
        # Performance tracking
        self.capture_count = 0
        self.capture_time_total = 0.0
        
        # Setup logging
        self.logger = ComponentLogger("SignalCapture", "global")
    
    def _create_storage_backend(self) -> StorageBackend:
        """Create appropriate storage backend"""
        if self.config.backend == "parquet":
            return ParquetStorageBackend(self.config.storage_path)
        elif self.config.backend == "hdf5":
            return HDF5StorageBackend(self.config.storage_path)
        elif self.config.backend == "json":
            return JSONStorageBackend(self.config.storage_path)
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")
    
    def capture_signal(self, signal: TradingSignal, context: SignalContext) -> None:
        """
        Capture signal with full context.
        
        Args:
            signal: The trading signal to capture
            context: Complete context at signal generation time
        """
        start_time = time.time()
        
        # Create captured signal
        captured = CapturedSignal(
            signal_id=self._generate_signal_id(),
            timestamp=context.timestamp,
            strategy_id=context.strategy_id,
            
            # Core signal
            symbol=signal.symbol,
            direction=signal.direction,
            strength=signal.strength,
            signal_type=signal.signal_type,
            
            # Market context
            market_data=self._extract_market_data(context),
            spread=context.current_spread,
            liquidity_score=context.liquidity_score,
            
            # Indicators
            indicators=self._extract_indicators(context),
            indicator_metadata=context.indicator_metadata,
            
            # Portfolio
            portfolio_state=self._extract_portfolio_state(context),
            position_before=context.current_position,
            capital_available=context.available_capital,
            
            # Risk
            risk_metrics=self._extract_risk_metrics(context),
            active_limits=context.active_risk_limits,
            
            # Regime
            market_regime=context.current_regime,
            regime_confidence=context.regime_confidence,
            
            # Strategy
            strategy_params=context.strategy_params,
            
            # Metadata
            processing_time_ms=(time.time() - start_time) * 1000
        )
        
        # Add to buffer
        self.capture_buffer.append(captured)
        self.capture_count += 1
        self.capture_time_total += captured.processing_time_ms
        
        # Flush if buffer full
        if len(self.capture_buffer) >= self.buffer_size:
            self.flush()
        
        # Log capture
        self.logger.debug(
            f"Captured signal {captured.signal_id}: "
            f"{signal.symbol} {signal.direction} strength={signal.strength:.2f}"
        )
    
    def _extract_market_data(self, context: SignalContext) -> Dict[str, float]:
        """Extract relevant market data"""
        bar = context.current_bar
        return {
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume,
            'vwap': bar.vwap if hasattr(bar, 'vwap') else None,
            'bid': context.best_bid,
            'ask': context.best_ask,
            'mid': (context.best_bid + context.best_ask) / 2
        }
    
    def _extract_indicators(self, context: SignalContext) -> Dict[str, float]:
        """Extract all indicator values"""
        indicators = {}
        
        for name, indicator in context.indicators.items():
            if hasattr(indicator, 'value'):
                indicators[name] = indicator.value
            elif hasattr(indicator, 'get_value'):
                indicators[name] = indicator.get_value()
            
            # Handle complex indicators
            if hasattr(indicator, 'get_all_values'):
                values = indicator.get_all_values()
                for sub_name, value in values.items():
                    indicators[f"{name}_{sub_name}"] = value
        
        return indicators
    
    def flush(self) -> None:
        """Flush capture buffer to storage"""
        if not self.capture_buffer:
            return
        
        try:
            self.storage_backend.store_signals(self.capture_buffer)
            buffer_size = len(self.capture_buffer)
            self.capture_buffer.clear()
            
            self.logger.info(
                f"Flushed {buffer_size} signals to storage. "
                f"Total captured: {self.capture_count}"
            )
        except Exception as e:
            self.logger.error(f"Failed to flush signals: {e}")
            raise
```

### 2. Storage Backends

```python
# src/execution/storage_backends.py
class ParquetStorageBackend(StorageBackend):
    """
    High-performance columnar storage using Parquet.
    Excellent for analytical queries.
    """
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def store_signals(self, signals: List[CapturedSignal]) -> None:
        """Store signals in Parquet format"""
        # Convert to DataFrame
        records = []
        for signal in signals:
            record = self._flatten_signal(signal)
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Partition by date for efficient querying
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        
        # Write partitioned dataset
        table = pa.Table.from_pandas(df)
        pq.write_to_dataset(
            table,
            root_path=str(self.base_path),
            partition_cols=['date', 'symbol'],
            compression='snappy',
            use_legacy_dataset=False
        )
    
    def _flatten_signal(self, signal: CapturedSignal) -> Dict:
        """Flatten nested signal structure for DataFrame"""
        flat = {
            'signal_id': signal.signal_id,
            'timestamp': signal.timestamp,
            'strategy_id': signal.strategy_id,
            'symbol': signal.symbol,
            'direction': signal.direction.value,
            'strength': signal.strength,
            'signal_type': signal.signal_type
        }
        
        # Flatten market data
        for key, value in signal.market_data.items():
            flat[f'market_{key}'] = value
        
        # Flatten indicators
        for key, value in signal.indicators.items():
            flat[f'indicator_{key}'] = value
        
        # Flatten other fields
        flat.update({
            'spread': signal.spread,
            'position_before': signal.position_before,
            'capital_available': signal.capital_available,
            'regime': signal.market_regime.value if signal.market_regime else None,
            'regime_confidence': signal.regime_confidence
        })
        
        # Store complex fields as JSON
        flat['portfolio_state_json'] = json.dumps(signal.portfolio_state)
        flat['risk_metrics_json'] = json.dumps(signal.risk_metrics)
        flat['strategy_params_json'] = json.dumps(signal.strategy_params)
        
        return flat
    
    def load_signals(self, filters: Optional[Dict] = None) -> pd.DataFrame:
        """Load signals with optional filtering"""
        # Read partitioned dataset
        dataset = pq.ParquetDataset(
            self.base_path,
            filters=self._build_filters(filters)
        )
        
        df = dataset.read().to_pandas()
        
        # Deserialize JSON fields
        if 'portfolio_state_json' in df.columns:
            df['portfolio_state'] = df['portfolio_state_json'].apply(json.loads)
        
        return df

class HDF5StorageBackend(StorageBackend):
    """
    Hierarchical storage with good compression.
    Supports partial loading and streaming.
    """
    
    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
    
    def store_signals(self, signals: List[CapturedSignal]) -> None:
        """Store signals in HDF5 format"""
        with h5py.File(self.file_path, 'a') as f:
            # Group by date
            for signal in signals:
                date_key = signal.timestamp.strftime('%Y-%m-%d')
                
                if date_key not in f:
                    date_group = f.create_group(date_key)
                else:
                    date_group = f[date_key]
                
                # Store signal
                signal_group = date_group.create_group(signal.signal_id)
                self._store_signal_data(signal_group, signal)
    
    def _store_signal_data(self, group: h5py.Group, signal: CapturedSignal):
        """Store individual signal data"""
        # Store scalar attributes
        group.attrs['timestamp'] = signal.timestamp.isoformat()
        group.attrs['symbol'] = signal.symbol
        group.attrs['direction'] = signal.direction.value
        group.attrs['strength'] = signal.strength
        
        # Store arrays
        if signal.indicators:
            indicator_names = list(signal.indicators.keys())
            indicator_values = list(signal.indicators.values())
            group.create_dataset('indicator_names', data=indicator_names)
            group.create_dataset('indicator_values', data=indicator_values)
        
        # Store complex data as JSON
        group.attrs['market_data'] = json.dumps(signal.market_data)
        group.attrs['portfolio_state'] = json.dumps(signal.portfolio_state)
```

### 3. Signal Analysis Tools

```python
# src/analysis/signal_analyzer.py
class SignalAnalyzer:
    """
    Analyzes captured signals for patterns and insights.
    """
    
    def __init__(self, storage_backend: StorageBackend):
        self.storage = storage_backend
        self.signals_df = None
    
    def load_signals(self, start_date: datetime, end_date: datetime,
                    symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Load signals for analysis"""
        filters = {
            'timestamp': {'>=': start_date, '<=': end_date}
        }
        if symbols:
            filters['symbol'] = {'in': symbols}
        
        self.signals_df = self.storage.load_signals(filters)
        return self.signals_df
    
    def analyze_signal_quality(self) -> SignalQualityReport:
        """Analyze signal quality metrics"""
        if self.signals_df is None:
            raise ValueError("No signals loaded")
        
        # Signal frequency analysis
        signal_freq = self.signals_df.groupby(
            ['symbol', pd.Grouper(key='timestamp', freq='D')]
        ).size()
        
        # Direction distribution
        direction_dist = self.signals_df['direction'].value_counts(normalize=True)
        
        # Strength distribution
        strength_stats = self.signals_df['strength'].describe()
        
        # Regime analysis
        if 'regime' in self.signals_df.columns:
            regime_dist = self.signals_df.groupby('regime')['signal_id'].count()
        else:
            regime_dist = None
        
        return SignalQualityReport(
            total_signals=len(self.signals_df),
            avg_signals_per_day=signal_freq.mean(),
            direction_distribution=direction_dist.to_dict(),
            strength_statistics=strength_stats.to_dict(),
            regime_distribution=regime_dist.to_dict() if regime_dist is not None else None
        )
    
    def find_signal_patterns(self, min_support: float = 0.1) -> List[SignalPattern]:
        """Find frequent patterns in signals"""
        # Extract features for pattern mining
        features = self._extract_pattern_features()
        
        # Use FP-Growth or similar algorithm
        patterns = self._mine_patterns(features, min_support)
        
        return patterns
    
    def validate_signal_consistency(self) -> ConsistencyReport:
        """Validate signals for consistency issues"""
        issues = []
        
        # Check for duplicate signals
        duplicates = self.signals_df[
            self.signals_df.duplicated(['timestamp', 'symbol', 'strategy_id'])
        ]
        if not duplicates.empty:
            issues.append(f"Found {len(duplicates)} duplicate signals")
        
        # Check for conflicting signals
        conflicts = self._find_conflicting_signals()
        if conflicts:
            issues.append(f"Found {len(conflicts)} conflicting signals")
        
        # Check for missing data
        null_counts = self.signals_df.isnull().sum()
        missing_critical = null_counts[
            ['symbol', 'direction', 'timestamp', 'strength']
        ].sum()
        if missing_critical > 0:
            issues.append(f"Missing critical data in {missing_critical} signals")
        
        return ConsistencyReport(
            is_consistent=len(issues) == 0,
            issues=issues,
            total_signals_checked=len(self.signals_df)
        )
```

### 4. Integration with Trading System

```python
# src/execution/capture_integration.py
class CaptureAwareStrategy(BaseStrategy):
    """
    Strategy wrapper that automatically captures signals.
    """
    
    def __init__(self, strategy: BaseStrategy, capture_engine: SignalCaptureEngine):
        self.strategy = strategy
        self.capture_engine = capture_engine
        self.container_id = strategy.container_id
    
    def generate_signal(self, market_data: MarketData) -> Optional[TradingSignal]:
        """Generate signal with automatic capture"""
        # Collect context before signal generation
        context = self._build_signal_context(market_data)
        
        # Generate signal
        signal = self.strategy.generate_signal(market_data)
        
        # Capture if signal generated
        if signal:
            self.capture_engine.capture_signal(signal, context)
        
        return signal
    
    def _build_signal_context(self, market_data: MarketData) -> SignalContext:
        """Build comprehensive context for signal capture"""
        return SignalContext(
            timestamp=market_data.timestamp,
            strategy_id=self.container_id,
            current_bar=market_data.current_bar,
            best_bid=market_data.best_bid,
            best_ask=market_data.best_ask,
            current_spread=market_data.best_ask - market_data.best_bid,
            liquidity_score=self._calculate_liquidity_score(market_data),
            indicators=self.strategy.get_indicators(),
            indicator_metadata=self.strategy.get_indicator_metadata(),
            portfolio_state=self.get_portfolio_state(),
            current_position=self.get_current_position(market_data.symbol),
            available_capital=self.get_available_capital(),
            risk_metrics=self.get_risk_metrics(),
            active_risk_limits=self.get_active_limits(),
            current_regime=self.get_current_regime(),
            regime_confidence=self.get_regime_confidence(),
            strategy_params=self.strategy.get_params()
        )
```

## 🧪 Testing Requirements

### Unit Tests

Create `tests/unit/test_step7_signal_capture.py`:

```python
class TestSignalCapture:
    """Test signal capture functionality"""
    
    def test_signal_capture_completeness(self):
        """Test all required fields are captured"""
        # Create test signal and context
        signal = create_test_signal()
        context = create_test_context()
        
        # Capture signal
        capture_engine = SignalCaptureEngine(test_config())
        capture_engine.capture_signal(signal, context)
        
        # Verify captured data
        captured = capture_engine.capture_buffer[0]
        
        assert captured.symbol == signal.symbol
        assert captured.direction == signal.direction
        assert captured.market_data['close'] == context.current_bar.close
        assert 'SMA' in captured.indicators
        assert captured.portfolio_state is not None
    
    def test_storage_backend_roundtrip(self):
        """Test storage and retrieval"""
        # Create and store signals
        signals = [create_test_captured_signal() for _ in range(10)]
        
        backend = ParquetStorageBackend(Path('/tmp/test_signals'))
        backend.store_signals(signals)
        
        # Load and verify
        loaded_df = backend.load_signals()
        assert len(loaded_df) == 10
        assert all(col in loaded_df.columns for col in 
                  ['signal_id', 'timestamp', 'symbol', 'direction'])
```

### Integration Tests

Create `tests/integration/test_step7_capture_integration.py`:

```python
def test_capture_during_backtest():
    """Test signal capture during full backtest"""
    # Setup capture engine
    capture_engine = SignalCaptureEngine(SignalCaptureConfig(
        backend='parquet',
        storage_path='/tmp/test_capture',
        buffer_size=100
    ))
    
    # Create capture-aware strategy
    base_strategy = MomentumStrategy()
    strategy = CaptureAwareStrategy(base_strategy, capture_engine)
    
    # Run backtest
    backtest_engine = BacktestEngine()
    results = backtest_engine.run(
        strategy=strategy,
        data=create_test_market_data(),
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31)
    )
    
    # Verify signals captured
    capture_engine.flush()
    
    # Load and analyze
    analyzer = SignalAnalyzer(capture_engine.storage_backend)
    signals_df = analyzer.load_signals(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31)
    )
    
    assert len(signals_df) > 0
    assert len(signals_df) == results['total_signals']

def test_capture_performance_overhead():
    """Test capture doesn't significantly impact performance"""
    # Run backtest without capture
    strategy_no_capture = MomentumStrategy()
    start_time = time.time()
    results_no_capture = run_backtest(strategy_no_capture, test_data)
    time_no_capture = time.time() - start_time
    
    # Run with capture
    capture_engine = SignalCaptureEngine(test_config())
    strategy_with_capture = CaptureAwareStrategy(strategy_no_capture, capture_engine)
    start_time = time.time()
    results_with_capture = run_backtest(strategy_with_capture, test_data)
    time_with_capture = time.time() - start_time
    
    # Verify overhead is acceptable
    overhead = (time_with_capture - time_no_capture) / time_no_capture
    assert overhead < 0.05  # Less than 5% overhead
    
    # Verify results unchanged
    assert results_no_capture['total_return'] == results_with_capture['total_return']
```

### System Tests

Create `tests/system/test_step7_capture_analysis.py`:

```python
def test_end_to_end_capture_and_analysis():
    """Test complete capture and analysis workflow"""
    # Run multi-strategy backtest with capture
    system = create_multi_strategy_system_with_capture()
    
    # Run for extended period
    results = system.run_backtest(
        data=load_historical_data('SPY', '2023-01-01', '2023-12-31'),
        capture=True
    )
    
    # Analyze captured signals
    analyzer = SignalAnalyzer(system.capture_engine.storage_backend)
    signals_df = analyzer.load_signals(
        datetime(2023, 1, 1), datetime(2023, 12, 31)
    )
    
    # Generate quality report
    quality_report = analyzer.analyze_signal_quality()
    
    # Verify signal quality
    assert quality_report.total_signals > 1000
    assert 0.4 < quality_report.direction_distribution['BUY'] < 0.6
    assert quality_report.avg_signals_per_day > 0
    
    # Check consistency
    consistency_report = analyzer.validate_signal_consistency()
    assert consistency_report.is_consistent
    
    # Find patterns
    patterns = analyzer.find_signal_patterns(min_support=0.05)
    assert len(patterns) > 0
```

## ✅ Validation Checklist

### Capture Completeness
- [ ] All signal fields captured
- [ ] Market context preserved
- [ ] Indicator snapshots accurate
- [ ] Portfolio state recorded
- [ ] Risk metrics included

### Storage Efficiency
- [ ] Compression working
- [ ] Partitioning optimal
- [ ] Query performance acceptable
- [ ] Storage growth manageable

### Analysis Capabilities
- [ ] Signal loading fast
- [ ] Quality metrics accurate
- [ ] Pattern detection working
- [ ] Consistency validation complete

## 📊 Memory & Performance

### Performance Optimization
```python
class OptimizedCaptureEngine:
    """Optimized capture with minimal overhead"""
    
    def __init__(self, config):
        # Pre-allocate buffers
        self.signal_pool = [CapturedSignal() for _ in range(1000)]
        self.pool_index = 0
        
        # Use binary serialization
        self.use_msgpack = config.use_msgpack
        
        # Async write option
        self.async_writes = config.async_writes
        if self.async_writes:
            self.write_queue = asyncio.Queue()
            self.write_task = asyncio.create_task(self._async_writer())
```

### Memory Targets
- Capture overhead: < 100 bytes per signal
- Buffer memory: < 10MB for 100k signals
- Storage efficiency: < 1KB per signal compressed

## 🐛 Common Issues

1. **Missing Context**
   - Always capture complete context
   - Test with edge cases
   - Validate required fields

2. **Storage Growth**
   - Implement data retention policies
   - Use appropriate compression
   - Monitor disk usage

3. **Performance Impact**
   - Profile capture overhead
   - Use buffering appropriately
   - Consider async writes

## 🎯 Success Criteria

Step 7 is complete when:
1. ✅ Signal capture system operational
2. ✅ All context preserved accurately
3. ✅ Storage efficient and queryable
4. ✅ Analysis tools functional
5. ✅ Performance overhead < 5%

## 🚀 Next Steps

Once all validations pass, proceed to:
[Step 8: Signal Replay](step-08-signal-replay.md)

## 📚 Additional Resources

- [Event Sourcing Best Practices](../../patterns/event-sourcing.md)
- [Time Series Storage Guide](../references/timeseries-storage.md)
- [Signal Analysis Techniques](../references/signal-analysis.md)