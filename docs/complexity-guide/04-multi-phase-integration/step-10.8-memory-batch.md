# Step 10.8: Memory & Batch Processing

**Status**: Multi-Phase Integration Step
**Complexity**: Very High
**Prerequisites**: [Step 10: End-to-End Workflow](step-10-end-to-end-workflow.md) completed
**Architecture Ref**: [Performance Optimization Guide](../optimization/performance-guide.md)

## 🎯 Objective

Implement memory-efficient batch processing infrastructure:
- Monitor and manage memory usage in real-time
- Process thousands of configurations in parallel
- Implement intelligent batch sizing
- Support distributed processing
- Enable spillover to disk for large datasets

## 📋 Required Reading

Before starting:
1. [Memory Management Patterns](../references/memory-management.md)
2. [Batch Processing Best Practices](../references/batch-processing.md)
3. [Distributed Computing Guide](../references/distributed-computing.md)

## 🏗️ Implementation Tasks

### 1. Memory Monitoring Infrastructure

```python
# src/core/monitoring/memory_monitor.py
import psutil
import gc
import tracemalloc
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
import numpy as np

@dataclass
class MemorySnapshot:
    """Point-in-time memory usage snapshot"""
    timestamp: float
    process_memory_mb: float
    system_memory_mb: float
    system_memory_percent: float
    gc_stats: Dict[str, int]
    top_allocations: List[Tuple[str, int]]  # (traceback, size)
    
    @property
    def available_memory_mb(self) -> float:
        """Calculate available system memory"""
        return psutil.virtual_memory().available / 1024 / 1024

class MemoryMonitor:
    """
    Real-time memory monitoring with alerting and profiling.
    Tracks memory usage patterns and detects leaks.
    """
    
    def __init__(self, 
                 sampling_interval: float = 1.0,
                 history_size: int = 3600,  # 1 hour at 1s intervals
                 alert_threshold_percent: float = 80.0):
        self.sampling_interval = sampling_interval
        self.history = deque(maxlen=history_size)
        self.alert_threshold = alert_threshold_percent
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.process = psutil.Process()
        
        # Memory profiling
        self.profiling_enabled = False
        self.allocation_tracking = {}
        
        # Alerts and callbacks
        self.alert_callbacks: List[Callable] = []
        self.memory_pressure_detected = False
        
        self.logger = ComponentLogger("MemoryMonitor", "monitoring")
    
    def start(self, enable_profiling: bool = False) -> None:
        """Start memory monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.profiling_enabled = enable_profiling
        
        if enable_profiling:
            tracemalloc.start()
        
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info("Memory monitoring started")
    
    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return summary statistics"""
        self.is_monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        if self.profiling_enabled:
            tracemalloc.stop()
        
        # Calculate summary statistics
        if self.history:
            memory_values = [s.process_memory_mb for s in self.history]
            summary = {
                'peak_memory_mb': max(memory_values),
                'avg_memory_mb': np.mean(memory_values),
                'min_memory_mb': min(memory_values),
                'memory_growth_mb': memory_values[-1] - memory_values[0],
                'total_snapshots': len(self.history),
                'pressure_events': sum(1 for s in self.history 
                                     if s.system_memory_percent > self.alert_threshold)
            }
        else:
            summary = {}
        
        self.logger.info(f"Memory monitoring stopped. Peak: {summary.get('peak_memory_mb', 0):.1f}MB")
        
        return summary
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                snapshot = self._take_snapshot()
                self.history.append(snapshot)
                
                # Check for memory pressure
                if snapshot.system_memory_percent > self.alert_threshold:
                    self._handle_memory_pressure(snapshot)
                
                # Detect memory leaks
                if len(self.history) > 10:
                    if self._detect_memory_leak():
                        self.logger.warning("Potential memory leak detected")
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(self.sampling_interval)
    
    def _take_snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot"""
        # Process memory
        process_info = self.process.memory_info()
        process_memory_mb = process_info.rss / 1024 / 1024
        
        # System memory
        system_memory = psutil.virtual_memory()
        system_memory_mb = system_memory.used / 1024 / 1024
        system_memory_percent = system_memory.percent
        
        # GC stats
        gc_stats = {
            f'gen{i}_collections': gc.get_count()[i]
            for i in range(gc.get_count().__len__())
        }
        gc_stats['objects'] = len(gc.get_objects())
        
        # Top allocations (if profiling)
        top_allocations = []
        if self.profiling_enabled:
            snapshot_trace = tracemalloc.take_snapshot()
            top_stats = snapshot_trace.statistics('traceback')[:10]
            top_allocations = [
                (str(stat.traceback), stat.size) 
                for stat in top_stats
            ]
        
        return MemorySnapshot(
            timestamp=time.time(),
            process_memory_mb=process_memory_mb,
            system_memory_mb=system_memory_mb,
            system_memory_percent=system_memory_percent,
            gc_stats=gc_stats,
            top_allocations=top_allocations
        )
    
    def _detect_memory_leak(self, window: int = 60) -> bool:
        """Detect potential memory leaks using trend analysis"""
        if len(self.history) < window:
            return False
        
        recent_memory = [s.process_memory_mb for s in list(self.history)[-window:]]
        
        # Linear regression to detect trend
        x = np.arange(len(recent_memory))
        slope, _ = np.polyfit(x, recent_memory, 1)
        
        # Leak if consistent growth > 1MB per minute
        return slope > (1.0 / 60.0) * self.sampling_interval
    
    def _handle_memory_pressure(self, snapshot: MemorySnapshot) -> None:
        """Handle high memory usage"""
        if not self.memory_pressure_detected:
            self.memory_pressure_detected = True
            
            self.logger.warning(
                f"Memory pressure detected: {snapshot.system_memory_percent:.1f}% "
                f"(process: {snapshot.process_memory_mb:.1f}MB)"
            )
            
            # Trigger callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(snapshot)
                except Exception as e:
                    self.logger.error(f"Alert callback error: {e}")
            
            # Force garbage collection
            gc.collect()
            
            # Log top allocations if available
            if snapshot.top_allocations:
                self.logger.info("Top memory allocations:")
                for trace, size in snapshot.top_allocations[:5]:
                    self.logger.info(f"  {size / 1024 / 1024:.1f}MB: {trace}")
    
    def add_alert_callback(self, callback: Callable[[MemorySnapshot], None]) -> None:
        """Add callback for memory alerts"""
        self.alert_callbacks.append(callback)
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        snapshot = self._take_snapshot()
        return {
            'process_mb': snapshot.process_memory_mb,
            'system_percent': snapshot.system_memory_percent,
            'available_mb': snapshot.available_memory_mb
        }
```

### 2. Intelligent Batch Processing

```python
# src/core/batch/batch_processor.py
class AdaptiveBatchProcessor:
    """
    Processes large numbers of configurations in batches.
    Dynamically adjusts batch size based on memory usage.
    """
    
    def __init__(self, 
                 memory_monitor: MemoryMonitor,
                 target_memory_usage_mb: float = 8192,  # 8GB target
                 min_batch_size: int = 10,
                 max_batch_size: int = 1000):
        self.memory_monitor = memory_monitor
        self.target_memory_mb = target_memory_usage_mb
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        
        # Adaptive sizing state
        self.current_batch_size = min_batch_size
        self.memory_per_item_estimate = 10.0  # Initial estimate in MB
        self.batch_history: List[BatchExecutionStats] = []
        
        # Processing state
        self.total_items_processed = 0
        self.total_batches_processed = 0
        
        self.logger = ComponentLogger("AdaptiveBatchProcessor", "batch")
    
    def process_items(self, 
                     items: List[Any],
                     process_func: Callable[[List[Any]], List[Any]],
                     progress_callback: Optional[Callable[[float], None]] = None) -> List[Any]:
        """Process items in adaptive batches"""
        results = []
        total_items = len(items)
        processed = 0
        
        self.logger.info(f"Starting batch processing of {total_items} items")
        
        # Start memory monitoring
        self.memory_monitor.start()
        
        try:
            while processed < total_items:
                # Determine batch size
                batch_size = self._calculate_optimal_batch_size()
                batch_end = min(processed + batch_size, total_items)
                batch = items[processed:batch_end]
                
                self.logger.info(
                    f"Processing batch {self.total_batches_processed + 1}: "
                    f"items {processed}-{batch_end} (size: {len(batch)})"
                )
                
                # Process batch
                batch_stats = self._process_batch(batch, process_func)
                results.extend(batch_stats.results)
                
                # Update statistics
                self._update_batch_statistics(batch_stats)
                processed = batch_end
                self.total_items_processed += len(batch)
                self.total_batches_processed += 1
                
                # Progress callback
                if progress_callback:
                    progress = processed / total_items
                    progress_callback(progress)
                
                # Check memory pressure
                if self._should_pause_for_memory():
                    self._handle_memory_pressure()
        
        finally:
            # Stop monitoring and get summary
            monitor_summary = self.memory_monitor.stop()
            self._log_processing_summary(monitor_summary)
        
        return results
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on memory usage"""
        current_memory = self.memory_monitor.get_current_usage()
        available_memory = current_memory['available_mb']
        
        # Estimate how many items we can process
        memory_buffer = 0.8  # Use 80% of available memory
        usable_memory = min(
            available_memory * memory_buffer,
            self.target_memory_mb - current_memory['process_mb']
        )
        
        estimated_items = int(usable_memory / self.memory_per_item_estimate)
        
        # Apply bounds and smoothing
        optimal_size = max(
            self.min_batch_size,
            min(estimated_items, self.max_batch_size)
        )
        
        # Smooth changes to avoid oscillation
        if self.current_batch_size > 0:
            change_factor = 0.5  # Allow 50% change per batch
            max_change = int(self.current_batch_size * change_factor)
            
            if optimal_size > self.current_batch_size:
                optimal_size = min(
                    optimal_size, 
                    self.current_batch_size + max_change
                )
            else:
                optimal_size = max(
                    optimal_size,
                    self.current_batch_size - max_change
                )
        
        self.current_batch_size = optimal_size
        
        return optimal_size
    
    def _process_batch(self, batch: List[Any], 
                      process_func: Callable) -> 'BatchExecutionStats':
        """Process a single batch with monitoring"""
        start_time = time.time()
        start_memory = self.memory_monitor.get_current_usage()['process_mb']
        
        # Process batch
        try:
            results = process_func(batch)
            success = True
            error = None
        except Exception as e:
            self.logger.error(f"Batch processing error: {e}")
            results = []
            success = False
            error = str(e)
        
        # Collect statistics
        end_time = time.time()
        end_memory = self.memory_monitor.get_current_usage()['process_mb']
        
        stats = BatchExecutionStats(
            batch_size=len(batch),
            execution_time=end_time - start_time,
            memory_used_mb=max(0, end_memory - start_memory),
            success=success,
            error=error,
            results=results
        )
        
        return stats
    
    def _update_batch_statistics(self, stats: BatchExecutionStats) -> None:
        """Update batch processing statistics"""
        self.batch_history.append(stats)
        
        # Update memory per item estimate
        if stats.batch_size > 0 and stats.memory_used_mb > 0:
            new_estimate = stats.memory_used_mb / stats.batch_size
            
            # Exponential moving average
            alpha = 0.3
            self.memory_per_item_estimate = (
                alpha * new_estimate + 
                (1 - alpha) * self.memory_per_item_estimate
            )
        
        # Log performance
        if stats.batch_size > 0:
            items_per_second = stats.batch_size / stats.execution_time
            self.logger.info(
                f"Batch completed: {stats.batch_size} items in "
                f"{stats.execution_time:.1f}s ({items_per_second:.1f} items/s), "
                f"memory: {stats.memory_used_mb:.1f}MB"
            )
```

### 3. Distributed Batch Processing

```python
# src/core/batch/distributed_processor.py
import ray
from ray import serve
import dask
from dask.distributed import Client, as_completed

class DistributedBatchProcessor:
    """
    Distributed batch processing using Ray or Dask.
    Scales across multiple machines.
    """
    
    def __init__(self, 
                 backend: str = 'ray',  # 'ray' or 'dask'
                 n_workers: Optional[int] = None,
                 memory_per_worker_gb: float = 4.0):
        self.backend = backend
        self.n_workers = n_workers or psutil.cpu_count()
        self.memory_per_worker_gb = memory_per_worker_gb
        
        self.client = None
        self.is_initialized = False
        
        self.logger = ComponentLogger("DistributedBatchProcessor", "distributed")
    
    def initialize(self, cluster_address: Optional[str] = None) -> None:
        """Initialize distributed backend"""
        if self.backend == 'ray':
            self._initialize_ray(cluster_address)
        elif self.backend == 'dask':
            self._initialize_dask(cluster_address)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        
        self.is_initialized = True
    
    def _initialize_ray(self, address: Optional[str]) -> None:
        """Initialize Ray cluster"""
        if address:
            ray.init(address=address)
        else:
            ray.init(
                num_cpus=self.n_workers,
                object_store_memory=int(self.memory_per_worker_gb * 1e9)
            )
        
        self.logger.info(f"Ray initialized with {ray.cluster_resources()}")
    
    def _initialize_dask(self, address: Optional[str]) -> None:
        """Initialize Dask cluster"""
        if address:
            self.client = Client(address)
        else:
            from dask.distributed import LocalCluster
            cluster = LocalCluster(
                n_workers=self.n_workers,
                threads_per_worker=1,
                memory_limit=f"{self.memory_per_worker_gb}GB"
            )
            self.client = Client(cluster)
        
        self.logger.info(f"Dask initialized with {self.client}")
    
    def process_batch_distributed(self,
                                items: List[Any],
                                process_func: Callable,
                                batch_size: int = 100,
                                progress_callback: Optional[Callable] = None) -> List[Any]:
        """Process items in distributed batches"""
        if not self.is_initialized:
            raise RuntimeError("Distributed backend not initialized")
        
        # Create batches
        batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
        total_batches = len(batches)
        
        self.logger.info(
            f"Processing {len(items)} items in {total_batches} batches "
            f"using {self.backend}"
        )
        
        if self.backend == 'ray':
            return self._process_with_ray(batches, process_func, progress_callback)
        else:
            return self._process_with_dask(batches, process_func, progress_callback)
    
    def _process_with_ray(self, batches: List[List[Any]], 
                         process_func: Callable,
                         progress_callback: Optional[Callable]) -> List[Any]:
        """Process using Ray"""
        # Create remote function
        @ray.remote
        def process_batch_remote(batch):
            return process_func(batch)
        
        # Submit all batches
        futures = [process_batch_remote.remote(batch) for batch in batches]
        
        # Collect results with progress
        results = []
        completed = 0
        
        while futures:
            ready, futures = ray.wait(futures, num_returns=1)
            
            for future in ready:
                batch_results = ray.get(future)
                results.extend(batch_results)
                
                completed += 1
                if progress_callback:
                    progress_callback(completed / len(batches))
        
        return results
    
    def _process_with_dask(self, batches: List[List[Any]],
                          process_func: Callable,
                          progress_callback: Optional[Callable]) -> List[Any]:
        """Process using Dask"""
        # Submit batches
        futures = []
        for batch in batches:
            future = self.client.submit(process_func, batch)
            futures.append(future)
        
        # Collect results
        results = []
        completed = 0
        
        for future in as_completed(futures):
            batch_results = future.result()
            results.extend(batch_results)
            
            completed += 1
            if progress_callback:
                progress_callback(completed / len(batches))
        
        return results
    
    def shutdown(self) -> None:
        """Shutdown distributed backend"""
        if self.backend == 'ray' and ray.is_initialized():
            ray.shutdown()
        elif self.backend == 'dask' and self.client:
            self.client.close()
        
        self.is_initialized = False
```

### 4. Memory-Efficient Data Handling

```python
# src/core/batch/memory_efficient_data.py
class ChunkedDataProcessor:
    """
    Processes large datasets in chunks without loading entire dataset.
    Supports spillover to disk when memory is constrained.
    """
    
    def __init__(self,
                 chunk_size_mb: float = 100,
                 spillover_threshold_mb: float = 1024,
                 spillover_path: str = "./spillover"):
        self.chunk_size_mb = chunk_size_mb
        self.spillover_threshold_mb = spillover_threshold_mb
        self.spillover_path = Path(spillover_path)
        self.spillover_path.mkdir(exist_ok=True)
        
        # Spillover tracking
        self.spilled_chunks: List[Path] = []
        self.in_memory_chunks: List[pd.DataFrame] = []
        self.total_memory_used = 0
        
        self.logger = ComponentLogger("ChunkedDataProcessor", "data")
    
    def process_large_dataset(self,
                            data_source: Union[str, pd.DataFrame, Iterator],
                            process_func: Callable[[pd.DataFrame], pd.DataFrame],
                            output_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Process large dataset in memory-efficient chunks"""
        # Setup data iterator
        if isinstance(data_source, str):
            data_iter = self._create_file_iterator(data_source)
        elif isinstance(data_source, pd.DataFrame):
            data_iter = self._create_dataframe_iterator(data_source)
        else:
            data_iter = data_source
        
        # Process chunks
        processed_chunks = []
        chunk_num = 0
        
        for chunk in data_iter:
            self.logger.info(f"Processing chunk {chunk_num}")
            
            # Process chunk
            processed = process_func(chunk)
            
            # Handle memory management
            chunk_memory = processed.memory_usage(deep=True).sum() / 1024 / 1024
            self.total_memory_used += chunk_memory
            
            if self.total_memory_used > self.spillover_threshold_mb:
                # Spill to disk
                self._spillover_chunk(processed, chunk_num)
            else:
                # Keep in memory
                self.in_memory_chunks.append(processed)
            
            chunk_num += 1
            
            # Force garbage collection periodically
            if chunk_num % 10 == 0:
                gc.collect()
        
        # Combine results
        if output_path:
            self._save_results_to_file(output_path)
            return None
        else:
            return self._combine_all_chunks()
    
    def _create_file_iterator(self, filepath: str, 
                            chunksize: Optional[int] = None) -> Iterator[pd.DataFrame]:
        """Create iterator for large files"""
        file_size_mb = Path(filepath).stat().st_size / 1024 / 1024
        
        # Estimate rows per chunk
        if chunksize is None:
            # Sample to estimate row size
            sample = pd.read_csv(filepath, nrows=1000)
            memory_per_row = sample.memory_usage(deep=True).sum() / 1000 / 1024 / 1024
            chunksize = int(self.chunk_size_mb / memory_per_row)
        
        self.logger.info(
            f"Reading {file_size_mb:.1f}MB file in chunks of {chunksize} rows"
        )
        
        # Return iterator based on file type
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath, chunksize=chunksize)
        elif filepath.endswith('.parquet'):
            # For parquet, we need to implement manual chunking
            return self._parquet_iterator(filepath, chunksize)
        else:
            raise ValueError(f"Unsupported file type: {filepath}")
    
    def _spillover_chunk(self, chunk: pd.DataFrame, chunk_id: int) -> None:
        """Spill chunk to disk"""
        spillover_file = self.spillover_path / f"chunk_{chunk_id}.parquet"
        
        chunk.to_parquet(spillover_file, compression='snappy')
        self.spilled_chunks.append(spillover_file)
        
        self.logger.info(
            f"Spilled chunk {chunk_id} to disk ({chunk.shape[0]} rows)"
        )
        
        # Reduce memory tracking
        chunk_memory = chunk.memory_usage(deep=True).sum() / 1024 / 1024
        self.total_memory_used -= chunk_memory
    
    def _combine_all_chunks(self) -> pd.DataFrame:
        """Combine all chunks (memory and disk)"""
        all_chunks = []
        
        # Add in-memory chunks
        all_chunks.extend(self.in_memory_chunks)
        
        # Load spilled chunks
        for spillover_file in self.spilled_chunks:
            chunk = pd.read_parquet(spillover_file)
            all_chunks.append(chunk)
        
        # Combine
        if all_chunks:
            result = pd.concat(all_chunks, ignore_index=True)
            
            # Cleanup spillover files
            for spillover_file in self.spilled_chunks:
                spillover_file.unlink()
            
            return result
        else:
            return pd.DataFrame()
```

### 5. Batch Optimization Pipeline

```python
# src/optimization/batch_optimization.py
class BatchOptimizationPipeline:
    """
    Complete pipeline for batch parameter optimization.
    Integrates memory monitoring, adaptive batching, and distributed processing.
    """
    
    def __init__(self, 
                 replay_engine: SignalReplayEngine,
                 distributed: bool = False,
                 max_memory_gb: float = 16.0):
        self.replay_engine = replay_engine
        self.distributed = distributed
        self.max_memory_gb = max_memory_gb
        
        # Initialize components
        self.memory_monitor = MemoryMonitor()
        self.batch_processor = AdaptiveBatchProcessor(
            self.memory_monitor,
            target_memory_usage_mb=max_memory_gb * 1024
        )
        
        if distributed:
            self.distributed_processor = DistributedBatchProcessor()
        
        self.logger = ComponentLogger("BatchOptimizationPipeline", "optimization")
    
    def optimize_parameters(self,
                          param_configs: List[Dict[str, Any]],
                          objective: str = 'sharpe_ratio',
                          save_results: bool = True) -> OptimizationResult:
        """Run batch optimization on parameter configurations"""
        self.logger.info(
            f"Starting batch optimization of {len(param_configs)} configurations"
        )
        
        # Define processing function
        def process_configs(configs: List[Dict]) -> List[Dict]:
            results = []
            
            for config in configs:
                try:
                    # Create replay config
                    replay_config = ReplayConfig(**config)
                    
                    # Run replay
                    result = self.replay_engine.replay(replay_config)
                    
                    # Extract metrics
                    results.append({
                        'config': config,
                        'sharpe_ratio': result.sharpe_ratio,
                        'total_return': result.total_return,
                        'max_drawdown': result.max_drawdown,
                        'trades': len(result.trades)
                    })
                    
                except Exception as e:
                    self.logger.error(f"Failed to process config {config}: {e}")
                    results.append({
                        'config': config,
                        'error': str(e)
                    })
            
            return results
        
        # Progress tracking
        progress_bar = tqdm(total=len(param_configs), desc="Optimization progress")
        
        def update_progress(progress: float):
            progress_bar.update(int(progress * len(param_configs)) - progress_bar.n)
        
        # Process based on mode
        if self.distributed:
            self.distributed_processor.initialize()
            try:
                results = self.distributed_processor.process_batch_distributed(
                    param_configs,
                    process_configs,
                    batch_size=50,
                    progress_callback=update_progress
                )
            finally:
                self.distributed_processor.shutdown()
        else:
            # Use adaptive batch processing
            results = self.batch_processor.process_items(
                param_configs,
                process_configs,
                progress_callback=update_progress
            )
        
        progress_bar.close()
        
        # Analyze results
        optimization_result = self._analyze_results(results, objective)
        
        # Save if requested
        if save_results:
            self._save_optimization_results(optimization_result)
        
        return optimization_result
    
    def _analyze_results(self, results: List[Dict], 
                        objective: str) -> OptimizationResult:
        """Analyze batch optimization results"""
        # Filter successful results
        successful_results = [r for r in results if 'error' not in r]
        failed_results = [r for r in results if 'error' in r]
        
        if not successful_results:
            raise RuntimeError("All configurations failed")
        
        # Find best configuration
        if objective == 'sharpe_ratio':
            best_idx = np.argmax([r['sharpe_ratio'] for r in successful_results])
        elif objective == 'calmar_ratio':
            best_idx = np.argmax([
                r['total_return'] / r['max_drawdown'] 
                if r['max_drawdown'] > 0 else 0
                for r in successful_results
            ])
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        best_result = successful_results[best_idx]
        
        # Calculate statistics
        sharpe_values = [r['sharpe_ratio'] for r in successful_results]
        
        return OptimizationResult(
            best_config=best_result['config'],
            best_value=best_result[objective],
            all_results=successful_results,
            failed_configs=failed_results,
            statistics={
                'mean_sharpe': np.mean(sharpe_values),
                'std_sharpe': np.std(sharpe_values),
                'success_rate': len(successful_results) / len(results),
                'total_configurations': len(results)
            },
            optimization_time=self.batch_processor.total_processing_time
        )
```

## 🧪 Testing Requirements

### Unit Tests

Create `tests/unit/test_step10_8_memory_batch.py`:

```python
class TestMemoryMonitor:
    """Test memory monitoring functionality"""
    
    def test_memory_monitoring(self):
        """Test basic memory monitoring"""
        monitor = MemoryMonitor(sampling_interval=0.1)
        
        # Start monitoring
        monitor.start()
        
        # Allocate some memory
        data = np.random.randn(1000000)  # ~8MB
        
        time.sleep(0.5)
        
        # Stop and check
        summary = monitor.stop()
        
        assert summary['total_snapshots'] > 0
        assert summary['peak_memory_mb'] > 0
        assert summary['avg_memory_mb'] > 0
    
    def test_memory_leak_detection(self):
        """Test memory leak detection"""
        monitor = MemoryMonitor()
        
        # Simulate memory leak
        leak_list = []
        for i in range(100):
            snapshot = MemorySnapshot(
                timestamp=time.time() + i,
                process_memory_mb=100 + i * 2,  # Growing memory
                system_memory_mb=1000,
                system_memory_percent=50,
                gc_stats={},
                top_allocations=[]
            )
            monitor.history.append(snapshot)
        
        # Should detect leak
        assert monitor._detect_memory_leak(window=50)

class TestAdaptiveBatchProcessor:
    """Test adaptive batch processing"""
    
    def test_batch_size_adaptation(self):
        """Test batch size adapts to memory"""
        monitor = MemoryMonitor()
        processor = AdaptiveBatchProcessor(
            monitor,
            target_memory_usage_mb=1000,
            min_batch_size=10,
            max_batch_size=100
        )
        
        # Test with different memory scenarios
        # High memory available
        monitor.get_current_usage = lambda: {
            'process_mb': 100,
            'available_mb': 5000,
            'system_percent': 20
        }
        
        size1 = processor._calculate_optimal_batch_size()
        assert size1 > processor.min_batch_size
        
        # Low memory available
        monitor.get_current_usage = lambda: {
            'process_mb': 800,
            'available_mb': 500,
            'system_percent': 80
        }
        
        size2 = processor._calculate_optimal_batch_size()
        assert size2 < size1
```

### Integration Tests

Create `tests/integration/test_step10_8_batch_integration.py`:

```python
def test_batch_optimization_pipeline():
    """Test complete batch optimization pipeline"""
    # Setup
    replay_engine = create_test_replay_engine()
    
    # Generate parameter configurations
    param_configs = []
    for pos_size in np.linspace(0.05, 0.20, 10):
        for risk_level in np.linspace(0.01, 0.05, 10):
            param_configs.append({
                'max_position_size': pos_size,
                'risk_per_trade': risk_level,
                'position_sizing_method': 'risk_based'
            })
    
    # Create pipeline
    pipeline = BatchOptimizationPipeline(
        replay_engine,
        distributed=False,
        max_memory_gb=4.0
    )
    
    # Run optimization
    result = pipeline.optimize_parameters(
        param_configs,
        objective='sharpe_ratio'
    )
    
    # Verify results
    assert result.best_value > 0
    assert len(result.all_results) > 0
    assert result.statistics['success_rate'] > 0.9

def test_distributed_processing():
    """Test distributed batch processing"""
    # Initialize distributed processor
    processor = DistributedBatchProcessor(
        backend='ray',
        n_workers=2,
        memory_per_worker_gb=2.0
    )
    
    processor.initialize()
    
    try:
        # Create test items
        items = list(range(1000))
        
        # Process function
        def square_items(batch):
            return [x**2 for x in batch]
        
        # Process distributed
        results = processor.process_batch_distributed(
            items,
            square_items,
            batch_size=100
        )
        
        # Verify
        assert len(results) == len(items)
        assert results[10] == 100  # 10^2
        
    finally:
        processor.shutdown()
```

### System Tests

Create `tests/system/test_step10_8_large_scale.py`:

```python
def test_large_scale_optimization():
    """Test optimization at scale with memory management"""
    # Generate large parameter space
    param_grid = {
        'max_position_size': np.linspace(0.05, 0.25, 20),
        'risk_per_trade': np.linspace(0.01, 0.05, 20),
        'stop_loss': np.linspace(0.02, 0.10, 10),
        'take_profit': np.linspace(0.05, 0.20, 10),
        'position_sizing_method': ['fixed', 'risk_based', 'volatility_adjusted']
    }
    
    # Generate all combinations (12,000 configs)
    from itertools import product
    param_configs = [
        dict(zip(param_grid.keys(), values))
        for values in product(*param_grid.values())
    ]
    
    # Setup pipeline with memory constraints
    pipeline = BatchOptimizationPipeline(
        replay_engine=create_production_replay_engine(),
        distributed=True,
        max_memory_gb=16.0
    )
    
    # Monitor system resources
    resource_monitor = SystemResourceMonitor()
    resource_monitor.start()
    
    # Run optimization
    start_time = time.time()
    
    result = pipeline.optimize_parameters(
        param_configs,
        objective='sharpe_ratio',
        save_results=True
    )
    
    execution_time = time.time() - start_time
    resource_stats = resource_monitor.stop()
    
    # Performance assertions
    assert execution_time < 3600  # Complete within 1 hour
    assert resource_stats['peak_memory_gb'] < 20  # Stay within memory limit
    assert result.statistics['success_rate'] > 0.95
    
    # Quality assertions
    assert result.best_value > 1.5  # Good Sharpe ratio
    assert len(result.all_results) > 10000  # Most configs processed
    
    # Log performance metrics
    configs_per_second = len(param_configs) / execution_time
    print(f"\nPerformance Metrics:")
    print(f"Total configurations: {len(param_configs)}")
    print(f"Execution time: {execution_time:.1f}s")
    print(f"Throughput: {configs_per_second:.1f} configs/s")
    print(f"Peak memory: {resource_stats['peak_memory_gb']:.1f}GB")
    print(f"Best Sharpe: {result.best_value:.3f}")

def test_memory_spillover():
    """Test handling of datasets larger than memory"""
    # Create large dataset that exceeds memory
    processor = ChunkedDataProcessor(
        chunk_size_mb=100,
        spillover_threshold_mb=500,
        spillover_path="./test_spillover"
    )
    
    # Process large dataset
    def process_chunk(chunk):
        # Simulate processing
        chunk['processed'] = chunk['value'] * 2
        return chunk
    
    # Create data iterator that generates 1GB of data
    def large_data_generator():
        for i in range(100):  # 100 chunks of 10MB each
            yield pd.DataFrame({
                'id': range(i * 100000, (i + 1) * 100000),
                'value': np.random.randn(100000)
            })
    
    # Process
    result = processor.process_large_dataset(
        large_data_generator(),
        process_chunk
    )
    
    # Verify spillover occurred
    assert len(processor.spilled_chunks) > 0
    
    # Verify result
    assert len(result) == 10000000  # All rows processed
    assert 'processed' in result.columns
    
    # Cleanup
    shutil.rmtree("./test_spillover")
```

## ✅ Validation Checklist

### Memory Management
- [ ] Real-time monitoring working
- [ ] Memory leak detection accurate
- [ ] Alert system functional
- [ ] Memory pressure handled

### Batch Processing
- [ ] Adaptive sizing works
- [ ] Batch statistics tracked
- [ ] Progress reporting accurate
- [ ] Error handling robust

### Distributed Processing
- [ ] Ray backend functional
- [ ] Dask backend functional
- [ ] Work distribution even
- [ ] Fault tolerance working

### Large Scale Testing
- [ ] 10k+ configurations handled
- [ ] Memory limits respected
- [ ] Performance acceptable
- [ ] Results quality maintained

## 📊 Performance Benchmarks

### Single Machine
- 1,000 configs: < 5 minutes
- 10,000 configs: < 30 minutes
- 100,000 configs: < 4 hours
- Memory usage: < configured limit

### Distributed (8 workers)
- 10,000 configs: < 10 minutes
- 100,000 configs: < 1 hour
- 1,000,000 configs: < 8 hours
- Linear scaling with workers

## 🐛 Common Issues

1. **Memory Fragmentation**
   - Use memory pools
   - Force periodic GC
   - Restart workers periodically

2. **Uneven Work Distribution**
   - Balance batch sizes
   - Consider work stealing
   - Monitor worker utilization

3. **Spillover Performance**
   - Use fast local storage
   - Compress spilled data
   - Tune spillover threshold

## 🎯 Success Criteria

Step 10.8 is complete when:
1. ✅ Memory monitoring prevents OOM
2. ✅ Batch processing adapts to resources
3. ✅ Distributed processing scales linearly
4. ✅ Large datasets handled efficiently
5. ✅ Production-scale tests pass

## 🚀 Next Steps

Once all validations pass, proceed to:
[Step 10.1: Advanced Analytics](../05-intermediate-complexity/step-10.1-advanced-analytics.md)

## 📚 Additional Resources

- [Memory Management Best Practices](../references/memory-best-practices.md)
- [Distributed Computing Patterns](../references/distributed-patterns.md)
- [Performance Tuning Guide](../optimization/performance-tuning.md)