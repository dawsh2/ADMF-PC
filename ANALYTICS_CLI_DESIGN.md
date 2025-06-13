# Analytics CLI Design & Storage Architecture

## Overview

The Analytics CLI provides a comprehensive interface for analyzing trading system performance, managing signal/classifier storage, and orchestrating signal replay workflows. It follows a hierarchical command structure with intelligent defaults and progressive disclosure of complexity.

## CLI Architecture

### Command Structure
```bash
# Main entry point
admf analytics [subcommand] [options]

# Subcommands
admf analytics scan       # Scan workspaces for runs
admf analytics analyze    # Analyze specific runs  
admf analytics compare    # Compare multiple runs
admf analytics replay     # Generate replay configs
admf analytics report     # Generate reports
admf analytics clean      # Manage storage
```

### Core Commands

#### 1. Scan Command - Discovery & Overview
```bash
# Scan all workspaces with summary
admf analytics scan

# Output:
Found 47 optimization runs in ./workspaces/
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Run ID                                  | Date       | Strategies | Best Sharpe | Status
fc4bb91c-2cea-441b-85e4-10d83a0e1580  | 2025-06-10 | 8          | 1.82       | ✓ Complete
a3d5f892-1234-5678-90ab-cdef12345678  | 2025-06-09 | 12         | 2.14       | ✓ Complete  
b7e9c0a1-9876-5432-10fe-dcba98765432  | 2025-06-08 | 6          | 0.94       | ⚠ Incomplete

# Filter by criteria
admf analytics scan --min-sharpe 1.5 --after 2025-06-01

# Detailed scan of specific run
admf analytics scan fc4bb91c-2cea-441b-85e4-10d83a0e1580 --detailed
```

#### 2. Analyze Command - Deep Analysis
```bash
# Analyze single run with execution costs
admf analytics analyze fc4bb91c-2cea-441b-85e4-10d83a0e1580 \
    --commission 0.001 \
    --slippage sqrt:0.0005

# Output:
Analyzing run fc4bb91c-2cea-441b-85e4-10d83a0e1580...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PERFORMANCE SUMMARY (After Costs)
Strategy               | Gross Sharpe | Net Sharpe | Win Rate | Max DD  | Cost Impact
ma_crossover_5_20     | 1.82         | 1.54       | 58.3%    | -12.4%  | -15.4%
momentum_rsi_14       | 2.14         | 1.89       | 62.1%    | -9.8%   | -11.7%
mean_reversion_bb     | 1.45         | 1.12       | 55.2%    | -15.2%  | -22.8%

REGIME PERFORMANCE
                      | TRENDING     | VOLATILE    | NEUTRAL
ma_crossover_5_20    | 2.31         | 0.84       | 1.45
momentum_rsi_14      | 2.89         | 1.23       | 1.67
mean_reversion_bb    | 0.45         | 2.14       | 1.38

DISCOVERED PATTERNS
1. Strong momentum during trending regimes (78% win rate)
2. Mean reversion excels in volatile regimes after 10am
3. Correlation breakdown between strategies during regime transitions

# Analyze with pattern mining
admf analytics analyze fc4bb91c --mine-patterns --min-confidence 0.7

# Analyze multiple runs for ensemble
admf analytics analyze run1 run2 run3 --ensemble --max-correlation 0.6
```

#### 3. Compare Command - Cross-Run Analysis
```bash
# Compare multiple runs
admf analytics compare run1 run2 run3 \
    --metrics sharpe,drawdown,win_rate \
    --output comparison_report.html

# Compare parameter sensitivity
admf analytics compare --parameter-sweep ma_period \
    --runs "fc4bb91c-*" \
    --visualize
```

#### 4. Replay Command - Signal Replay Configuration
```bash
# Generate replay config from analysis
admf analytics replay fc4bb91c \
    --strategies top:3 \
    --regimes TRENDING,VOLATILE \
    --weights optimal

# Output:
Generated replay configuration:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SELECTED STRATEGIES
1. momentum_rsi_14    (weight: 0.45 in TRENDING, 0.20 in VOLATILE)
2. ma_crossover_5_20  (weight: 0.35 in TRENDING, 0.30 in VOLATILE)  
3. mean_reversion_bb  (weight: 0.20 in TRENDING, 0.50 in VOLATILE)

EXECUTION PARAMETERS
- Position sizing: Kelly Criterion (capped at 0.25)
- Risk limits: 2% per trade, 6% total
- Rebalance frequency: On regime change

Config saved to: ./replay_configs/ensemble_20250611_141523.yaml

# Generate and execute replay
admf analytics replay fc4bb91c --execute --live-test
```

#### 5. Report Command - Visualization & Export
```bash
# Generate comprehensive HTML report
admf analytics report fc4bb91c \
    --format html \
    --include performance,patterns,regimes,correlations

# Generate PDF report for multiple runs
admf analytics report run1 run2 run3 \
    --format pdf \
    --template institutional \
    --output quarterly_analysis.pdf

# Quick performance summary
admf analytics report fc4bb91c --format terminal --summary
```

#### 6. Clean Command - Storage Management
```bash
# Show storage usage
admf analytics clean --dry-run

# Output:
Storage Analysis:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total runs: 47
Total size: 2.3 GB
Average run size: 49 MB

By age:
< 7 days:    12 runs (580 MB)
7-30 days:   23 runs (1.1 GB)  
> 30 days:   12 runs (620 MB)

By performance:
Sharpe > 1.5: 18 runs (890 MB) [KEEP]
Sharpe < 0.5: 8 runs (390 MB)  [REMOVE?]

# Clean old, poor-performing runs
admf analytics clean \
    --older-than 30d \
    --max-sharpe 0.5 \
    --keep-top 10

# Archive to cold storage
admf analytics clean \
    --archive s3://my-bucket/admf-archive \
    --older-than 90d
```

### Interactive Mode

```bash
# Enter interactive analytics shell
admf analytics interactive

ADMF Analytics> scan --last 10
[Shows last 10 runs]

ADMF Analytics> select fc4bb91c
Selected run: fc4bb91c-2cea-441b-85e4-10d83a0e1580

ADMF Analytics [fc4bb91c]> analyze --quick
[Shows quick analysis]

ADMF Analytics [fc4bb91c]> strategies
[Lists all strategies with performance]

ADMF Analytics [fc4bb91c]> regime TRENDING
[Shows performance in TRENDING regime]

ADMF Analytics [fc4bb91c]> correlations --threshold 0.5
[Shows strategy correlations above 0.5]

ADMF Analytics [fc4bb91c]> replay --dry-run
[Shows what replay config would be generated]
```

## Improved Storage Architecture

### 1. Standardized Naming Convention
```
workspaces/
├── {timestamp}_{workflow_type}_{symbol}_{identifier}/
│   Example: 20250611_141523_optimization_SPY_momentum_sweep
│   
│   ├── manifest.json                    # Run manifest with all metadata
│   ├── signals/
│   │   ├── {strategy_type}_{params_hash}.parquet
│   │   │   Example: ma_crossover_5_20_a3f4b2c1.parquet
│   │   └── index.json                  # Strategy index with param mapping
│   ├── classifiers/
│   │   ├── {classifier_type}_{params_hash}.parquet
│   │   │   Example: regime_hmm_3state_d5e6f7g8.parquet
│   │   └── index.json                  # Classifier index
│   └── analytics/
│       ├── performance_matrix.parquet  # Pre-computed performance
│       ├── correlation_matrix.parquet  # Strategy correlations
│       └── patterns.json               # Discovered patterns
```

### 2. Enhanced Metadata Schema
```python
# manifest.json structure
{
    "run_id": "fc4bb91c-2cea-441b-85e4-10d83a0e1580",
    "created_at": "2025-06-11T14:15:23Z",
    "workflow": {
        "type": "optimization",
        "version": "1.2.0",
        "config_hash": "a3f4b2c1d5e6f7g8"
    },
    "data": {
        "symbols": ["SPY"],
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "frequency": "1m",
        "total_bars": 98550
    },
    "execution": {
        "duration_seconds": 324.5,
        "peak_memory_mb": 1250,
        "cpu_cores": 8,
        "completed": true
    },
    "summary": {
        "total_strategies": 24,
        "total_classifiers": 3,
        "best_sharpe": 2.14,
        "best_strategy": "momentum_rsi_14_30",
        "parameter_ranges": {
            "ma_period": [5, 10, 20, 30, 50],
            "rsi_period": [14, 21, 30],
            "bb_period": [20],
            "bb_std": [2.0, 2.5]
        }
    },
    "storage": {
        "compression": "snappy",
        "signal_format": "sparse_parquet",
        "total_size_mb": 48.3,
        "signal_compression_ratio": 0.0124
    }
}
```

### 3. Strategy Index Schema
```python
# signals/index.json
{
    "strategies": {
        "ma_crossover_5_20_a3f4b2c1": {
            "type": "ma_crossover",
            "params": {"fast_period": 5, "slow_period": 20},
            "file": "ma_crossover_5_20_a3f4b2c1.parquet",
            "summary": {
                "total_signals": 245,
                "compression_ratio": 0.0025,
                "sharpe_ratio": 1.82,
                "max_drawdown": -0.124
            }
        }
    },
    "param_hash_map": {
        "a3f4b2c1": "ma_crossover_fast:5_slow:20",
        "b4c5d2e3": "ma_crossover_fast:5_slow:30"
    }
}
```

### 4. Fast Metadata Scanner
```python
# File: src/analytics/storage/scanner.py

class WorkspaceScanner:
    """Fast metadata scanning without loading signal data"""
    
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self._manifest_cache = {}
        self._index_cache = {}
    
    def scan_all(self, filters: Optional[ScanFilters] = None) -> List[RunSummary]:
        """Scan all runs with optional filtering"""
        summaries = []
        
        for run_dir in self.workspace_root.iterdir():
            if not run_dir.is_dir():
                continue
                
            # Fast manifest check
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            
            manifest = self._load_manifest(manifest_path)
            
            # Apply filters
            if filters and not filters.matches(manifest):
                continue
            
            summary = RunSummary(
                run_id=manifest["run_id"],
                timestamp=manifest["created_at"],
                workflow_type=manifest["workflow"]["type"],
                symbols=manifest["data"]["symbols"],
                total_strategies=manifest["summary"]["total_strategies"],
                best_sharpe=manifest["summary"]["best_sharpe"],
                best_strategy=manifest["summary"]["best_strategy"],
                path=run_dir
            )
            summaries.append(summary)
        
        return sorted(summaries, key=lambda x: x.timestamp, reverse=True)
    
    def get_strategy_metadata(self, run_id: str) -> Dict[str, StrategyMetadata]:
        """Get all strategy metadata for a run"""
        run_path = self._find_run_path(run_id)
        index_path = run_path / "signals" / "index.json"
        
        if index_path in self._index_cache:
            return self._index_cache[index_path]
        
        with open(index_path) as f:
            index = json.load(f)
        
        metadata = {}
        for strategy_id, strategy_info in index["strategies"].items():
            metadata[strategy_id] = StrategyMetadata(
                strategy_id=strategy_id,
                strategy_type=strategy_info["type"],
                parameters=strategy_info["params"],
                file_path=run_path / "signals" / strategy_info["file"],
                **strategy_info["summary"]
            )
        
        self._index_cache[index_path] = metadata
        return metadata
```

## Implementation Examples

### 1. Quick Performance Check
```python
# File: src/analytics/cli/commands/analyze.py

@click.command()
@click.argument('run_id')
@click.option('--commission', default=0.001, help='Commission rate')
@click.option('--slippage', default='linear:0.0005', help='Slippage model')
def analyze(run_id: str, commission: float, slippage: str):
    """Analyze optimization run with execution costs"""
    
    # Load metadata
    scanner = WorkspaceScanner(Path('./workspaces'))
    metadata = scanner.get_run_metadata(run_id)
    
    if not metadata:
        click.echo(f"Run {run_id} not found", err=True)
        return
    
    # Create analyzer
    analyzer = RunAnalyzer(
        run_path=metadata.path,
        execution_costs=ExecutionCosts(
            commission_rate=commission,
            slippage_model=slippage
        )
    )
    
    # Perform analysis
    with click.progressbar(
        analyzer.analyze_strategies(),
        label='Analyzing strategies'
    ) as strategies:
        results = []
        for strategy in strategies:
            result = analyzer.calculate_performance(strategy)
            results.append(result)
    
    # Display results
    display_performance_table(results)
```

### 2. Ensemble Generation
```python
@click.command()
@click.argument('run_ids', nargs=-1, required=True)
@click.option('--max-strategies', default=5)
@click.option('--max-correlation', default=0.7)
@click.option('--target-regimes', multiple=True)
def ensemble(run_ids: List[str], max_strategies: int, 
             max_correlation: float, target_regimes: List[str]):
    """Generate optimal ensemble from multiple runs"""
    
    # Load all strategies
    all_strategies = []
    for run_id in run_ids:
        strategies = scanner.get_strategy_metadata(run_id)
        all_strategies.extend(strategies.values())
    
    # Filter by performance
    top_strategies = sorted(
        all_strategies, 
        key=lambda x: x.sharpe_ratio,
        reverse=True
    )[:max_strategies * 3]  # Pre-filter
    
    # Calculate correlations
    correlation_matrix = calculate_correlation_matrix(top_strategies)
    
    # Select uncorrelated strategies
    selected = select_uncorrelated_strategies(
        top_strategies,
        correlation_matrix,
        max_correlation,
        max_strategies
    )
    
    # Optimize weights
    if target_regimes:
        weights = optimize_regime_weights(selected, target_regimes)
    else:
        weights = optimize_global_weights(selected)
    
    # Generate config
    config = generate_replay_config(selected, weights)
    
    # Save and display
    config_path = save_replay_config(config)
    click.echo(f"Ensemble configuration saved to: {config_path}")
    display_ensemble_summary(selected, weights)
```

### 3. Pattern Discovery
```python
@click.command()
@click.option('--min-support', default=0.05)
@click.option('--min-confidence', default=0.6)
def discover_patterns(run_id: str, min_support: float, min_confidence: float):
    """Discover trading patterns in run data"""
    
    # Load event traces
    events = load_event_traces(run_id)
    
    # Mine patterns
    miner = PatternMiner(
        min_support=min_support,
        min_confidence=min_confidence
    )
    
    patterns = miner.mine_patterns(events)
    
    # Display patterns
    for pattern in patterns:
        click.echo(f"\nPattern: {pattern.pattern_type}")
        click.echo(f"  Win Rate: {pattern.win_rate:.1%}")
        click.echo(f"  Avg Duration: {pattern.avg_duration:.1f} bars")
        click.echo(f"  Best in: {', '.join(pattern.dominant_regimes)}")
```

## Benefits of This Design

1. **Progressive Disclosure**: Simple commands for basic use, advanced options for power users
2. **Fast Scanning**: Metadata-first approach enables quick overview without loading GB of data
3. **Standardized Storage**: Consistent naming and structure improves maintainability
4. **Execution Cost Integration**: Built-in support for realistic performance assessment
5. **Workflow Automation**: From analysis to replay config generation in one command
6. **Interactive Exploration**: REPL mode for iterative analysis
7. **Extensible**: Easy to add new commands and analysis types

## Next Steps

1. Implement core scanner and metadata system
2. Build basic CLI commands (scan, analyze)
3. Add performance calculation with costs
4. Implement ensemble optimization
5. Add visualization/reporting
6. Build interactive mode