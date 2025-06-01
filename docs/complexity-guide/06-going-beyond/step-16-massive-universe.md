# Step 16: Scale to 1000+ Symbols

## 📋 Status: Advanced (96% Complexity)
**Estimated Time**: 3-4 weeks
**Difficulty**: Very High
**Prerequisites**: Steps 1-15 completed, high-performance infrastructure

## 🎯 Objectives

Scale the system to handle institutional-sized universes with thousands of symbols while maintaining sub-second latency and complete market coverage.

## 🔗 Architecture References

- **Data Architecture**: [src/data/README.md](../../../src/data/README.md)
- **Event System**: [Event-Driven Architecture](../../architecture/01-EVENT-DRIVEN-ARCHITECTURE.md)
- **Container Design**: [Container Hierarchy](../../architecture/02-CONTAINER-HIERARCHY.md)
- **Performance**: [Optimization Guide](../../optimization/README.md)

## 📚 Required Reading

1. **Data Streaming**: Understand efficient data handling patterns
2. **Event Bus**: Learn high-throughput event routing
3. **Memory Management**: Study optimization techniques
4. **Parallel Processing**: Review concurrent execution patterns

## 🏗️ Implementation Tasks

### 1. Massive Universe Data Architecture

```python
# src/data/massive_universe_handler.py
from src.core.protocols import DataProtocol
from src.core.events import Event, EventType
import numpy as np
from typing import Dict, List, Set, Iterator
import pyarrow as pa
import pyarrow.compute as pc

class MassiveUniverseDataHandler(DataProtocol):
    """
    Optimized data handler for 1000+ symbols.
    
    Architecture:
    - Memory-mapped data files
    - Columnar storage (Apache Arrow)
    - Parallel data loading
    - Smart caching
    - Batch processing
    """
    
    def __init__(self, config: Dict):
        self.universe_size = config['universe_size']
        self.batch_size = config.get('batch_size', 100)
        self.cache_size_gb = config.get('cache_size_gb', 16)
        
        # Memory-mapped storage
        self.data_store = ArrowDataStore(
            path=config['data_path'],
            cache_size=self.cache_size_gb * 1024 * 1024 * 1024
        )
        
        # Parallel processors
        self.num_workers = config.get('num_workers', 8)
        self.data_pipeline = ParallelDataPipeline(self.num_workers)
        
    def stream_bars(self, timestamp: pd.Timestamp) -> Iterator[List[Event]]:
        """
        Stream bars in optimized batches.
        
        Strategy:
        1. Load data in parallel chunks
        2. Process in batches to maintain cache locality
        3. Emit events in priority order (liquid names first)
        """
        # Get symbols sorted by liquidity
        symbols_by_priority = self._get_priority_sorted_symbols()
        
        # Process in batches
        for batch in self._batch_symbols(symbols_by_priority, self.batch_size):
            # Parallel load for batch
            batch_data = self.data_pipeline.load_batch(
                symbols=batch,
                timestamp=timestamp
            )
            
            # Convert to events
            events = self._create_bar_events(batch_data, timestamp)
            
            # Yield batch of events
            yield events
            
    def _get_priority_sorted_symbols(self) -> List[str]:
        """Sort symbols by trading priority (liquidity, volatility, etc)"""
        symbol_metrics = self.data_store.load_symbol_metrics()
        
        # Score based on multiple factors
        scores = {}
        for symbol, metrics in symbol_metrics.items():
            scores[symbol] = (
                metrics['avg_volume'] * 0.4 +
                metrics['volatility'] * 0.3 +
                metrics['spread'] * -0.3  # Lower spread is better
            )
            
        # Return sorted by score
        return sorted(scores.keys(), key=lambda s: scores[s], reverse=True)
```

### 2. Optimized Indicator Computation

```python
# src/strategy/indicators/vectorized_indicators.py
import numpy as np
import numba
from numba import cuda
import cupy as cp  # GPU arrays

class VectorizedIndicatorHub:
    """
    Massively parallel indicator computation for 1000+ symbols.
    
    Features:
    - Vectorized operations across all symbols
    - GPU acceleration for heavy computations
    - Incremental updates (not full recalc)
    - Smart caching of intermediate results
    """
    
    def __init__(self, symbols: List[str], config: Dict):
        self.symbols = symbols
        self.num_symbols = len(symbols)
        
        # Pre-allocate arrays for all symbols
        self.price_matrix = np.zeros((config['lookback'], self.num_symbols))
        self.volume_matrix = np.zeros((config['lookback'], self.num_symbols))
        
        # GPU arrays if available
        if config.get('use_gpu', False) and cuda.is_available():
            self.gpu_prices = cp.zeros_like(self.price_matrix)
            self.gpu_enabled = True
        else:
            self.gpu_enabled = False
            
        # Indicator caches
        self.sma_cache = {}
        self.ema_cache = {}
        self.rsi_cache = {}
        
    @numba.jit(nopython=True, parallel=True)
    def calculate_sma_vectorized(self, periods: List[int]) -> Dict[int, np.ndarray]:
        """Calculate SMA for all symbols and periods in parallel"""
        results = {}
        
        for period in periods:
            # Vectorized SMA across all symbols
            sma_matrix = np.zeros((self.price_matrix.shape[0], self.num_symbols))
            
            # Parallel computation
            for i in numba.prange(self.num_symbols):
                prices = self.price_matrix[:, i]
                sma_matrix[:, i] = self._rolling_mean(prices, period)
                
            results[period] = sma_matrix
            
        return results
        
    def calculate_complex_indicators_gpu(self):
        """Use GPU for complex indicator calculations"""
        if not self.gpu_enabled:
            return self.calculate_complex_indicators_cpu()
            
        # Transfer to GPU
        gpu_prices = cp.asarray(self.price_matrix)
        
        # GPU kernel for RSI calculation
        rsi_values = self._gpu_rsi_kernel(gpu_prices)
        
        # Transfer back
        return cp.asnumpy(rsi_values)
```

### 3. Distributed Signal Generation

```python
# src/strategy/distributed_signal_generator.py
from multiprocessing import Pool, Queue, Manager
import asyncio
from concurrent.futures import ProcessPoolExecutor

class DistributedSignalGenerator:
    """
    Distributes signal generation across multiple processes.
    
    Architecture:
    - Symbol partitioning across processes
    - Lock-free signal aggregation
    - Async I/O for result collection
    - Fault tolerance
    """
    
    def __init__(self, num_processes: int = None):
        self.num_processes = num_processes or os.cpu_count()
        self.executor = ProcessPoolExecutor(max_workers=self.num_processes)
        
        # Shared memory for results
        self.manager = Manager()
        self.signal_queue = self.manager.Queue()
        
    async def generate_signals_parallel(self, 
                                      market_data: Dict,
                                      strategies: List[Strategy]) -> List[Signal]:
        """Generate signals in parallel across symbol partitions"""
        
        # Partition symbols across processes
        symbol_partitions = self._partition_symbols(
            list(market_data.keys()),
            self.num_processes
        )
        
        # Submit tasks
        futures = []
        for partition in symbol_partitions:
            future = self.executor.submit(
                self._process_partition,
                partition,
                market_data,
                strategies
            )
            futures.append(future)
            
        # Collect results asynchronously
        signals = []
        for future in asyncio.as_completed(futures):
            partition_signals = await future
            signals.extend(partition_signals)
            
        return signals
        
    def _process_partition(self, 
                          symbols: List[str],
                          market_data: Dict,
                          strategies: List[Strategy]) -> List[Signal]:
        """Process a partition of symbols"""
        signals = []
        
        # Create local indicator cache
        indicator_cache = LocalIndicatorCache()
        
        for symbol in symbols:
            symbol_data = market_data[symbol]
            
            # Calculate indicators once
            indicators = indicator_cache.get_or_calculate(symbol, symbol_data)
            
            # Generate signals from all strategies
            for strategy in strategies:
                signal = strategy.evaluate(symbol, indicators)
                if signal.strength > strategy.threshold:
                    signals.append(signal)
                    
        return signals
```

### 4. High-Performance Event Bus

```python
# src/core/events/high_performance_bus.py
from collections import defaultdict
import threading
from queue import Queue, Empty
import time

class HighPerformanceEventBus:
    """
    Event bus optimized for massive message throughput.
    
    Features:
    - Lock-free queues where possible
    - Batched event delivery
    - Priority-based routing
    - Backpressure handling
    """
    
    def __init__(self, config: Dict):
        self.batch_size = config.get('batch_size', 1000)
        self.queue_size = config.get('queue_size', 100000)
        
        # Multiple queues by priority
        self.priority_queues = {
            'critical': Queue(maxsize=self.queue_size),
            'high': Queue(maxsize=self.queue_size),
            'normal': Queue(maxsize=self.queue_size),
            'low': Queue(maxsize=self.queue_size)
        }
        
        # Subscriber routing table
        self.routing_table = defaultdict(lambda: defaultdict(list))
        
        # Performance metrics
        self.metrics = EventBusMetrics()
        
    def publish_batch(self, events: List[Event], priority: str = 'normal'):
        """Publish batch of events efficiently"""
        start_time = time.perf_counter()
        
        queue = self.priority_queues[priority]
        
        # Try to put all events
        failed_events = []
        for event in events:
            try:
                queue.put_nowait(event)
            except:
                failed_events.append(event)
                
        # Handle backpressure
        if failed_events:
            self._handle_backpressure(failed_events, priority)
            
        # Update metrics
        self.metrics.record_publish(
            count=len(events) - len(failed_events),
            duration=time.perf_counter() - start_time
        )
        
    def process_events(self):
        """Process events with priority"""
        batch = []
        
        # Check queues in priority order
        for priority in ['critical', 'high', 'normal', 'low']:
            queue = self.priority_queues[priority]
            
            # Collect batch
            while len(batch) < self.batch_size:
                try:
                    event = queue.get_nowait()
                    batch.append(event)
                except Empty:
                    break
                    
            # Process batch if full
            if len(batch) >= self.batch_size:
                self._deliver_batch(batch)
                batch = []
                
        # Process remaining
        if batch:
            self._deliver_batch(batch)
```

### 5. Memory-Efficient Portfolio Tracking

```python
# src/risk/massive_portfolio_tracker.py
class MassivePortfolioTracker:
    """
    Track positions across thousands of symbols efficiently.
    
    Optimizations:
    - Sparse matrix for positions (most are zero)
    - Compressed position history
    - Lazy calculation of analytics
    - Memory-mapped persistence
    """
    
    def __init__(self, universe_size: int):
        # Use sparse matrix for positions
        from scipy.sparse import dok_matrix
        self.positions = dok_matrix((1, universe_size), dtype=np.float32)
        
        # Symbol to index mapping
        self.symbol_index = {}
        self.index_symbol = {}
        
        # Compressed history
        self.position_history = CompressedHistory(
            compression='zstd',
            chunk_size=1000
        )
        
    def update_position(self, symbol: str, quantity: float):
        """Update position efficiently"""
        idx = self.symbol_index.get(symbol)
        if idx is None:
            idx = self._add_symbol(symbol)
            
        # Update sparse matrix
        old_quantity = self.positions[0, idx]
        self.positions[0, idx] = quantity
        
        # Record change in compressed history
        if old_quantity != quantity:
            self.position_history.record(
                timestamp=time.time(),
                symbol_idx=idx,
                old_qty=old_quantity,
                new_qty=quantity
            )
            
    def get_active_positions(self) -> Dict[str, float]:
        """Get only non-zero positions"""
        active = {}
        
        # Iterate only non-zero elements
        for idx, qty in self.positions.items():
            if qty != 0:
                symbol = self.index_symbol[idx[1]]
                active[symbol] = qty
                
        return active
```

## 🧪 Testing Requirements

### Unit Tests

```python
# tests/test_massive_universe.py
def test_vectorized_indicators():
    """Test indicator calculation across 1000+ symbols"""
    # Create test data
    symbols = [f"SYM_{i:04d}" for i in range(1500)]
    hub = VectorizedIndicatorHub(symbols, config)
    
    # Generate random price data
    hub.price_matrix = np.random.randn(100, 1500) * 100 + 100
    
    # Calculate indicators
    start_time = time.time()
    sma_results = hub.calculate_sma_vectorized([10, 20, 50])
    duration = time.time() - start_time
    
    # Should complete in under 1 second
    assert duration < 1.0
    
    # Verify results shape
    assert sma_results[10].shape == (100, 1500)

def test_distributed_signals():
    """Test parallel signal generation"""
    generator = DistributedSignalGenerator(num_processes=4)
    
    # Create test data for many symbols
    market_data = create_test_market_data(num_symbols=2000)
    strategies = [MomentumStrategy(), MeanReversionStrategy()]
    
    # Generate signals
    signals = asyncio.run(
        generator.generate_signals_parallel(market_data, strategies)
    )
    
    # Verify signal coverage
    unique_symbols = {s.symbol for s in signals}
    assert len(unique_symbols) > 1000
```

### Integration Tests

```python
def test_full_universe_pipeline():
    """Test complete pipeline with massive universe"""
    # Initialize components
    data_handler = MassiveUniverseDataHandler(config)
    indicator_hub = VectorizedIndicatorHub(symbols, config)
    signal_generator = DistributedSignalGenerator()
    portfolio_tracker = MassivePortfolioTracker(len(symbols))
    
    # Process one timestamp
    timestamp = pd.Timestamp('2023-01-01 09:30:00')
    
    # Stream data in batches
    for event_batch in data_handler.stream_bars(timestamp):
        # Update indicators
        indicator_hub.update_batch(event_batch)
        
        # Generate signals
        signals = signal_generator.process_batch(event_batch)
        
        # Update portfolio
        for signal in signals:
            if signal.action == 'BUY':
                portfolio_tracker.update_position(signal.symbol, 100)
                
    # Verify performance
    assert portfolio_tracker.get_active_positions()
```

### System Tests

```python
def test_sustained_throughput():
    """Test sustained operation with 1000+ symbols"""
    system = create_massive_universe_system(num_symbols=2000)
    
    # Run for extended period
    start_time = time.time()
    events_processed = 0
    
    for minute in range(60):  # 1 hour
        events = system.process_minute()
        events_processed += len(events)
        
    duration = time.time() - start_time
    throughput = events_processed / duration
    
    # Should maintain high throughput
    assert throughput > 10000  # events per second
    
    # Check memory usage stayed stable
    assert system.get_memory_usage() < 32 * 1024 * 1024 * 1024  # 32GB
```

## 🎮 Validation Checklist

### Performance Validation
- [ ] Process 1000+ symbols in real-time
- [ ] Maintain sub-second latency
- [ ] Indicator calculation under 100ms
- [ ] Signal generation parallelized

### Memory Validation
- [ ] Memory usage scales linearly
- [ ] No memory leaks over time
- [ ] Efficient position tracking
- [ ] Compressed history storage

### Accuracy Validation
- [ ] All symbols processed
- [ ] No data loss
- [ ] Correct indicator values
- [ ] Accurate position tracking

### Scalability Validation
- [ ] Works with 5000+ symbols
- [ ] Graceful degradation
- [ ] Adjustable batch sizes
- [ ] Dynamic resource allocation

## 💾 Memory Optimization

```python
# Memory optimization strategies
class MemoryOptimizer:
    def __init__(self):
        self.symbol_cache_size = 100  # Keep hot symbols in memory
        self.indicator_retention = 60  # seconds
        self.position_precision = np.float32  # vs float64
        
    def optimize_data_structures(self):
        """Use optimal data structures for scale"""
        recommendations = {
            'positions': 'scipy.sparse for mostly zero positions',
            'price_data': 'numpy arrays with float32',
            'indicators': 'circular buffers for rolling windows',
            'signals': 'object pooling to reduce GC',
            'events': 'pre-allocated event pools'
        }
        return recommendations
        
    def estimate_memory_usage(self, num_symbols: int) -> float:
        """Estimate memory requirements in GB"""
        # Base requirements
        price_history = num_symbols * 390 * 20 * 4  # mins * days * bytes
        indicators = num_symbols * 10 * 100 * 4     # types * values * bytes
        positions = num_symbols * 8                  # sparse representation
        overhead = 2 * 1024 * 1024 * 1024           # 2GB system overhead
        
        total_bytes = price_history + indicators + positions + overhead
        return total_bytes / (1024 * 1024 * 1024)
```

## 🔧 Common Issues

1. **Memory Pressure**: Use memory-mapped files and compression
2. **CPU Bottlenecks**: Distribute computation across cores/machines
3. **Event Bus Overload**: Implement backpressure and batching
4. **Database I/O**: Use column stores and parallel reads
5. **Network Latency**: Batch operations and use compression

## ✅ Success Criteria

- [ ] Handle 1000+ symbols in production
- [ ] Maintain real-time processing
- [ ] Scale to 5000+ symbols
- [ ] Memory usage under control
- [ ] Performance metrics acceptable
- [ ] System remains stable

## 📚 Next Steps

Once massive universe scale is achieved:
1. Proceed to [Step 17: Scale to Institutional AUM](step-17-institutional-aum.md)
2. Implement distributed computing
3. Add more optimization techniques
4. Enhance monitoring capabilities