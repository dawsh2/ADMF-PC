# Hierarchical Parquet Storage Implementation Summary

## Overview

Successfully implemented a clean hierarchical storage structure for signals and classifiers, replacing the messy JSON files in `tmp/` directory.

## New Structure

```
analytics_storage/
├── signals/
│   ├── momentum/
│   │   ├── momentum_fast_lb_10_a1b2c3d4.json
│   │   └── index.json
│   ├── ma_crossover/
│   │   ├── ma_crossover_5_20_e5f6g7h8.json
│   │   └── index.json
│   └── mean_reversion/
│       ├── mean_reversion_99914b93.json
│       └── index.json
└── classifiers/
    ├── regime/
    │   ├── volatility_regime_99914b93.json
    │   └── index.json
    └── trend/
        ├── trend_classifier_99914b93.json
        └── index.json
```

## Implementation Details

### 1. New Storage Manager
- **File**: `src/core/events/storage/simple_parquet_storage.py`
- **Class**: `MinimalHierarchicalStorage`
- Creates hierarchical directory structure
- Generates hash-based filenames from strategy/classifier parameters
- Maintains index.json files for each directory
- Currently saves as JSON (can be upgraded to Parquet later without duckdb dependency)

### 2. Hierarchical Portfolio Tracer
- **File**: `src/core/events/observers/hierarchical_portfolio_tracer.py`
- **Class**: `HierarchicalPortfolioTracer`
- Replaces `SparsePortfolioTracer` for clean storage
- Buffers signals and classifiers by source
- Saves to hierarchical structure on cleanup
- Tracks compression ratios and statistics

### 3. Container Integration
- **File**: `src/core/containers/container.py`
- Updated to recognize `storage.type: hierarchical` in trace settings
- Automatically uses `HierarchicalPortfolioTracer` when hierarchical storage is configured
- Works for both portfolio and strategy containers

### 4. Configuration
- Set `storage.type: hierarchical` in trace settings
- Set `storage.base_dir: ./analytics_storage` to specify location
- Example configuration in `config/test_hierarchical_storage.yaml`

## Benefits

1. **Clean Organization**: Strategies and classifiers organized by type
2. **Unique Filenames**: Hash-based names prevent collisions
3. **Indexing**: Easy to find and list all files of a type
4. **Compression**: Only stores signal changes (sparse storage)
5. **Extensible**: Can upgrade to Parquet format later
6. **No Dependencies**: Works without duckdb or other heavy dependencies

## Usage

### Enable Hierarchical Storage

```yaml
execution:
  enable_event_tracing: true
  trace_settings:
    storage:
      type: hierarchical
      base_dir: ./analytics_storage
```

### Migration from Old Storage

Use the migration script to convert existing JSON files:

```bash
python migrate_to_hierarchical_storage.py
```

## Testing

Run the test to verify hierarchical storage:

```bash
python test_hierarchical_simple.py
```

## Future Enhancements

1. **Parquet Format**: Upgrade from JSON to Parquet for better compression
2. **Analytics Integration**: Connect to SQL analytics database
3. **Metadata Enhancement**: Add more metadata like data source, date range
4. **Performance Metrics**: Include backtest performance in storage
5. **Visualization**: Add tools to visualize signal patterns from storage