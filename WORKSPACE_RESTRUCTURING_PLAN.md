# Workspace Restructuring Plan for Grid Search & Sparse Storage

## Overview

This plan restructures the workspace storage to handle large-scale grid searches (e.g., 15,470 strategy-classifier combinations) while maintaining sparse storage efficiency and enabling fast metadata scanning.

## New Directory Structure

```
workspaces/
├── 20250611_180000_grid_search_SPY_expansive_v1/
│   ├── manifest.json                           # Master metadata
│   ├── signals/
│   │   ├── ma_crossover/                      # Group by strategy type
│   │   │   ├── ma_5_20_sl_1.0_a3f4b2c1.parquet
│   │   │   ├── ma_5_20_sl_2.0_b4c5d2e3.parquet
│   │   │   └── ...
│   │   ├── momentum/
│   │   │   ├── mom_10_30_70_5_c5d6e3f4.parquet
│   │   │   └── ...
│   │   └── index.json                         # Strategy catalog with params
│   ├── classifiers/
│   │   ├── momentum_regime/
│   │   │   ├── mr_65_25_0.05_d7e8f9a1.parquet
│   │   │   └── ...
│   │   ├── volatility/
│   │   │   ├── vol_1.5_0.3_20_e9f0a1b2.parquet
│   │   │   └── ...
│   │   └── index.json                         # Classifier catalog
│   ├── events/
│   │   ├── traces_{batch_id}.parquet          # Batched event traces
│   │   └── index.json                         # Event index
│   └── analytics/
│       ├── performance_matrix.parquet         # Strategy × Metrics
│       ├── regime_performance.parquet         # Strategy × Classifier × Metrics
│       ├── correlation_matrix.parquet         # Strategy correlations
│       └── summary_stats.json                 # Quick-access statistics
```

## Sparse Storage Format

### Signal Storage (Parquet Schema)
```python
# signals/ma_crossover/ma_5_20_sl_1.0_a3f4b2c1.parquet
Schema:
- bar_idx: int32          # Index into base data
- timestamp: timestamp    # For verification
- signal: int8           # -1, 0, 1
- confidence: float32    # Optional signal strength
- metadata: binary       # Optional additional data

# Example data (only changes stored):
bar_idx | timestamp           | signal | confidence
0       | 2023-01-01 09:30:00 | 1      | 0.85
47      | 2023-01-01 10:17:00 | -1     | 0.92
156     | 2023-01-01 12:06:00 | 1      | 0.78
```

### Classifier Storage (Parquet Schema)
```python
# classifiers/momentum_regime/mr_65_25_0.05_d7e8f9a1.parquet
Schema:
- bar_idx: int32          # Index into base data
- timestamp: timestamp    # For verification
- regime: string         # Regime name
- confidence: float32    # Classification confidence
- metadata: binary       # Optional additional data

# Example data (only regime changes stored):
bar_idx | timestamp           | regime      | confidence
0       | 2023-01-01 09:30:00 | NEUTRAL     | 0.95
234     | 2023-01-01 13:24:00 | TRENDING    | 0.88
567     | 2023-01-02 10:45:00 | VOLATILE    | 0.91
```

## Enhanced Metadata Structure

### manifest.json
```json
{
  "run_id": "20250611_180000_grid_search_SPY_expansive_v1",
  "created_at": "2025-06-11T18:00:00Z",
  "workflow": {
    "type": "grid_search",
    "config_file": "expansive_grid_search.yaml",
    "config_hash": "a3f4b2c1d5e6f7g8"
  },
  "data": {
    "symbols": ["SPY"],
    "timeframes": ["1m"],
    "start_date": "2023-01-01",
    "end_date": "2023-02-01",
    "total_bars": 19500,
    "data_source": "file"
  },
  "grid_search": {
    "strategy_types": 6,
    "total_strategies": 238,
    "classifier_types": 4,
    "total_classifiers": 65,
    "total_combinations": 15470
  },
  "performance_summary": {
    "best_strategy": {
      "id": "momentum_20_30_70_10",
      "sharpe": 2.14,
      "max_drawdown": -0.082
    },
    "best_classifier": {
      "id": "volatility_2.0_0.5_20",
      "regime_stability": 0.89
    },
    "best_combination": {
      "strategy": "momentum_20_30_70_10",
      "classifier": "volatility_2.0_0.5_20",
      "regime_aware_sharpe": 2.47
    }
  },
  "storage": {
    "format_version": "2.0",
    "compression": "snappy",
    "total_size_mb": 127.3,
    "signal_files": 238,
    "classifier_files": 65,
    "avg_compression_ratio": 0.0087
  }
}
```

### signals/index.json
```json
{
  "strategies": {
    "ma_crossover": {
      "total": 18,
      "parameter_grid": {
        "fast_period": [5, 10, 20],
        "slow_period": [20, 50, 100],
        "stop_loss_pct": [1.0, 2.0]
      },
      "files": {
        "ma_5_20_sl_1.0_a3f4b2c1": {
          "params": {"fast_period": 5, "slow_period": 20, "stop_loss_pct": 1.0},
          "file": "ma_crossover/ma_5_20_sl_1.0_a3f4b2c1.parquet",
          "signal_changes": 127,
          "compression_ratio": 0.0065,
          "performance": {
            "sharpe": 1.82,
            "max_drawdown": -0.124,
            "win_rate": 0.583
          }
        }
        // ... more strategies
      }
    },
    "momentum": {
      "total": 81,
      // ... momentum strategies
    }
    // ... other strategy types
  },
  "param_encoding": {
    "ma_crossover": "ma_{fast}_{slow}_sl_{stop_loss}",
    "momentum": "mom_{sma}_{long}_{short}_{exit}"
  }
}
```

## Migration Implementation

### Phase 1: Create Migration Tool
```python
# src/analytics/storage/workspace_migrator.py

class WorkspaceMigrator:
    """Migrate from old format to new grid-search optimized format"""
    
    def __init__(self, old_root: Path, new_root: Path):
        self.old_root = old_root
        self.new_root = new_root
        self.strategy_grouper = StrategyGrouper()
        
    def migrate_workspace(self, workspace_id: str):
        """Migrate a single workspace"""
        old_path = self.old_root / workspace_id
        
        # Analyze workspace type
        workspace_info = self._analyze_workspace(old_path)
        
        # Generate new directory name
        new_name = self._generate_directory_name(workspace_info)
        new_path = self.new_root / new_name
        
        # Migrate components
        self._migrate_signals(old_path, new_path, workspace_info)
        self._migrate_classifiers(old_path, new_path, workspace_info)
        self._consolidate_events(old_path, new_path)
        self._create_analytics(new_path)
        
        return new_path
```

### Phase 2: Sparse Storage Handler
```python
# src/analytics/storage/sparse_storage.py

class SparseSignalStorage:
    """Handle sparse signal storage with Parquet"""
    
    @staticmethod
    def from_json_changes(changes: List[Dict], total_bars: int) -> pd.DataFrame:
        """Convert JSON sparse format to Parquet DataFrame"""
        df = pd.DataFrame(changes)
        df['bar_idx'] = df['idx']
        df['timestamp'] = pd.to_datetime(df['ts'])
        df['signal'] = df['val'].astype('int8')
        
        # Add metadata
        df.attrs['total_bars'] = total_bars
        df.attrs['compression_ratio'] = len(changes) / total_bars
        
        return df[['bar_idx', 'timestamp', 'signal']]
    
    @staticmethod
    def to_parquet(df: pd.DataFrame, path: Path, compression='snappy'):
        """Save sparse signals to Parquet with metadata"""
        # Convert metadata to schema metadata
        metadata = {
            b'total_bars': str(df.attrs.get('total_bars', 0)).encode(),
            b'compression_ratio': str(df.attrs.get('compression_ratio', 0)).encode(),
            b'sparse_format': b'true'
        }
        
        # Create table with metadata
        table = pa.Table.from_pandas(df)
        table = table.replace_schema_metadata(metadata)
        
        # Write with compression
        pq.write_table(table, path, compression=compression)
```

### Phase 3: Fast Metadata Scanner
```python
# src/analytics/storage/metadata_scanner.py

class GridSearchScanner:
    """Fast scanning for grid search workspaces"""
    
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self._cache = {}
        
    def scan_performance(self, min_sharpe: float = None) -> pd.DataFrame:
        """Scan all strategies across all workspaces"""
        results = []
        
        for workspace_dir in self.workspace_root.iterdir():
            if not workspace_dir.is_dir():
                continue
                
            # Quick manifest check
            manifest_path = workspace_dir / "manifest.json"
            if not manifest_path.exists():
                continue
                
            # Check if grid search
            manifest = self._load_manifest(manifest_path)
            if manifest.get('workflow', {}).get('type') != 'grid_search':
                continue
                
            # Load strategy index
            strategy_index = workspace_dir / "signals" / "index.json"
            if strategy_index.exists():
                with open(strategy_index) as f:
                    index = json.load(f)
                    
                # Extract all strategy performances
                for strategy_type, type_data in index['strategies'].items():
                    for strategy_id, strategy_info in type_data['files'].items():
                        perf = strategy_info.get('performance', {})
                        
                        if min_sharpe and perf.get('sharpe', 0) < min_sharpe:
                            continue
                            
                        results.append({
                            'workspace': workspace_dir.name,
                            'strategy_type': strategy_type,
                            'strategy_id': strategy_id,
                            'params': strategy_info['params'],
                            'sharpe': perf.get('sharpe'),
                            'max_drawdown': perf.get('max_drawdown'),
                            'win_rate': perf.get('win_rate'),
                            'signal_changes': strategy_info['signal_changes'],
                            'compression_ratio': strategy_info['compression_ratio']
                        })
        
        return pd.DataFrame(results)
```

### Phase 4: Analytics Pre-computation
```python
# src/analytics/storage/analytics_builder.py

class AnalyticsBuilder:
    """Pre-compute analytics for fast querying"""
    
    def build_performance_matrix(self, workspace_path: Path):
        """Build strategy × metrics matrix"""
        # Load all strategy performances
        strategy_perfs = self._load_all_strategies(workspace_path)
        
        # Create performance matrix
        metrics = ['sharpe', 'sortino', 'max_drawdown', 'win_rate', 
                  'profit_factor', 'avg_trade_duration']
        
        matrix = pd.DataFrame(
            index=strategy_perfs.keys(),
            columns=metrics
        )
        
        for strategy_id, perf in strategy_perfs.items():
            for metric in metrics:
                matrix.loc[strategy_id, metric] = perf.get(metric)
        
        # Save as Parquet
        matrix.to_parquet(
            workspace_path / "analytics" / "performance_matrix.parquet"
        )
        
    def build_regime_performance(self, workspace_path: Path):
        """Build strategy × classifier × metrics tensor"""
        # This creates a 3D performance cube for regime-aware analysis
        pass
```

## CLI Integration

### New Commands
```bash
# Migrate existing workspaces
admf analytics migrate --source ./workspaces --format sparse-v2

# Scan grid search results
admf analytics scan --type grid_search --min-sharpe 1.5

# Quick performance overview
admf analytics summary 20250611_180000_grid_search_SPY_expansive_v1

# Find best strategy-classifier pairs
admf analytics analyze-grid 20250611_180000_grid_search_SPY_expansive_v1 \
    --top 10 \
    --metric regime_aware_sharpe
```

## Benefits

1. **Organized by Type**: Strategies grouped by type (ma_crossover, momentum, etc.)
2. **Fast Scanning**: Index files enable scanning without loading Parquet files
3. **Sparse Efficiency**: Maintains your existing sparse storage approach
4. **Grid Search Optimized**: Handles thousands of combinations efficiently
5. **Pre-computed Analytics**: Performance matrices ready for instant queries
6. **Human-Readable Names**: Directory names indicate date, type, symbol, and identifier
7. **Backwards Compatible**: Migration tool handles old format

## Implementation Priority

1. **Phase 1**: Implement sparse storage handlers (Parquet conversion)
2. **Phase 2**: Create migration tool for existing workspaces
3. **Phase 3**: Build metadata scanner for fast queries
4. **Phase 4**: Integrate with CLI commands
5. **Phase 5**: Add analytics pre-computation

This structure scales to handle your expansive grid searches while maintaining fast access to results.