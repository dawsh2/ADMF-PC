# Analytics Module

The analytics module provides post-optimization analysis capabilities for ADMF-PC, acting as a bridge between the event tracing system and sophisticated data mining features.

## Overview

This module implements a **minimal yet extensible** analytics framework that:
- Extracts metrics from containers without duplicating calculations
- Detects patterns using SQL-like operations
- Generates simple reports for immediate insights
- Provides foundation for sophisticated data mining

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Analytics Module                    │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────┐  │
│  │  Metrics    │   │  Patterns    │   │ Reports │  │
│  │ Extractor   │→→→│  Detector    │→→→│ Generator│  │
│  └──────┬──────┘   └──────────────┘   └─────────┘  │
│         │                                            │
│         ↓ extract_batch_metrics()                    │
│  ┌─────────────────────────────────────────────┐    │
│  │         Container Pool (Parallel)            │    │
│  │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐       │    │
│  │  │ C1  │  │ C2  │  │ C3  │  │ C4  │  ...  │    │
│  │  └─────┘  └─────┘  └─────┘  └─────┘       │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
│  Correlation IDs → Event Traces (Deep Analysis)     │
└─────────────────────────────────────────────────────┘
```

## Components

### 1. Metrics Extractor (`metrics.py`)

Extracts metrics that containers already calculate via their event observers:

```python
extractor = ContainerMetricsExtractor()

# Extract from single container
result = extractor.extract_metrics(container, "portfolio_1")

# Extract from multiple containers (parallelization-ready)
results = extractor.extract_batch_metrics({
    "portfolio_1": container1,
    "portfolio_2": container2,
    "portfolio_3": container3
})

# Convert to DataFrame for analysis
df = extractor.to_dataframe(results)
```

**Key Features:**
- Multiple fallback methods to find metrics
- Container-level isolation for parallel processing
- Correlation ID extraction for event trace linking
- Error isolation prevents cascade failures

### 2. Pattern Detector (`patterns.py`)

Simple SQL-like pattern detection for immediate insights:

```python
detector = SimplePatternDetector(
    min_sample_size=5,
    min_success_rate=0.6
)

patterns = detector.detect_patterns(metrics_df)

# Detected patterns include:
# - High Sharpe ratio strategies
# - Low drawdown strategies
# - Activity level patterns
# - Consistent winners
```

### 3. Report Generator (`reports.py`)

Minimal text-based reporting for quick insights:

```python
generator = MinimalReportGenerator(output_dir="./analytics_reports")

# Generate summary report
report_path = generator.generate_summary_report(
    metrics_results=results,
    patterns=patterns
)

# Export correlation IDs for deep analysis
correlation_ids_path = generator.export_correlation_ids(
    metrics_results=results,
    criteria={'min_sharpe': 1.5, 'min_return': 10.0}
)
```

### 4. Data Mining (`mining/`)

Advanced two-layer mining architecture:

```python
# Layer 1: SQL database for high-level metrics
mining = TwoLayerMiningArchitecture(
    db_path=Path("./optimization_results.db"),
    event_storage=event_storage
)

# Record optimization runs
mining.record_optimization_run(run)

# Find best parameters
best_params = mining.find_best_parameters(top_n=10)

# Discover patterns
patterns = mining.discover_patterns(min_frequency=5)
```

## Integration with Event Tracing

The analytics module is designed to work seamlessly with the event tracing system:

### 1. **Correlation ID Bridge**

Every container has a correlation ID that links:
- High-level metrics (in analytics)
- Detailed event traces (in event system)

```python
# From analytics, get promising containers
best_performers = df.nlargest(5, 'sharpe_ratio')

# Use correlation IDs to dive into event traces
for _, row in best_performers.iterrows():
    correlation_id = row['correlation_id']
    
    # Load event traces for deep analysis
    events = event_storage.query({'correlation_id': correlation_id})
    
    # Analyze why this strategy performed well
    analysis = analyze_event_sequence(events)
```

### 2. **Event Observer Integration**

The module extracts metrics from existing event observers:

```python
# Containers already have observers calculating metrics
container.event_bus.attach_observer(MetricsObserver(calculator))

# Analytics module just extracts the results
metrics = container.get_metrics()  # From observer
```

### 3. **Memory Efficiency**

Leverages event system's retention policies:
- `trade_complete`: Only keeps events for open trades
- `sliding_window`: Keeps last N events
- `minimal`: Maximum memory efficiency

## Parallelization Design

The module is designed for parallel execution from day one:

### Container Isolation

Each container is processed independently:

```python
def extract_batch_metrics(self, containers: Dict[str, Any]) -> List[MetricsResult]:
    # Each container can be processed in parallel
    for container_id, container in containers.items():
        try:
            metrics = self.extract_metrics(container, container_id)
            if metrics:
                results.append(metrics)
        except Exception as e:
            # Error in one container doesn't affect others
            logger.error(f"Failed on {container_id}: {e}")
```

### Future Parallel Implementation

The design allows easy parallelization without architecture changes:

```python
from concurrent.futures import ProcessPoolExecutor

def parallel_extract(containers):
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(extract_metrics, c, cid): cid 
            for cid, c in containers.items()
        }
        # Gather results...
```

## Data Mining Ambitions

The module provides a foundation for sophisticated data mining:

### Two-Layer Architecture

1. **Layer 1: SQL Database**
   - Fast queries for high-level patterns
   - Cross-run analysis
   - Parameter optimization insights

2. **Layer 2: Event Traces**
   - Deep behavioral analysis
   - Causal chain investigation
   - Micro-pattern discovery

### Pattern Evolution

```python
# Current: Simple statistical patterns
patterns = SimplePatternDetector().detect_patterns(df)

# Future: ML-based pattern discovery
patterns = MLPatternMiner().discover_complex_patterns(df, events)

# Future: Real-time pattern matching
live_matcher = LivePatternMatcher(pattern_library)
live_matcher.monitor_live_trading(event_stream)
```

### Scientific Method Framework

The architecture enables systematic discovery:

1. **Hypothesis** (SQL): "High Sharpe strategies exist in volatile markets"
2. **Investigation** (Events): Analyze exact decision chains
3. **Validation** (Backtest): Test patterns out-of-sample
4. **Production** (Monitoring): Deploy patterns with alerts

## Usage Example

```python
from src.analytics import (
    ContainerMetricsExtractor,
    SimplePatternDetector,
    MinimalReportGenerator
)

# 1. Extract metrics from optimization run
extractor = ContainerMetricsExtractor()
metrics_results = extractor.extract_batch_metrics(containers)

# 2. Detect patterns
detector = SimplePatternDetector()
df = extractor.to_dataframe(metrics_results)
patterns = detector.detect_patterns(df)

# 3. Generate report
reporter = MinimalReportGenerator()
report_path = reporter.generate_summary_report(
    metrics_results=metrics_results,
    patterns=patterns
)

# 4. Export correlation IDs for deep analysis
correlation_ids = reporter.export_correlation_ids(
    metrics_results=metrics_results,
    criteria={'min_sharpe': 1.5}
)

# 5. Use correlation IDs with event system
for cid in correlation_ids:
    events = event_storage.query({'correlation_id': cid})
    # Perform deep analysis...
```

## Future Enhancements

The module is designed to grow:

1. **Storage Backends**: Currently pandas DataFrames → DuckDB/Polars
2. **Pattern Detection**: Statistical → Machine Learning
3. **Reports**: Text → HTML/PDF with visualizations
4. **Real-time**: Batch analysis → Streaming analytics
5. **Distributed**: Single machine → Cluster computing

## Key Design Principles

1. **Bridge, Don't Replace**: Works with existing systems
2. **Start Simple**: Immediate value with minimal complexity
3. **Parallel-Ready**: Container isolation from the start
4. **Protocol-Based**: Easy to extend without breaking changes
5. **Correlation ID Centric**: Links all analysis layers

## Dependencies

- `pandas`: DataFrame operations
- `numpy`: Statistical calculations
- `sqlite3`: Lightweight SQL storage
- Standard library only otherwise

## See Also

- `docs/architecture/data-mining-architecture.md` - Detailed mining architecture
- `src/core/events/` - Event tracing system
- `src/core/containers/` - Container architecture