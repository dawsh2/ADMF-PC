# Signal Storage and Replay Architecture

## Overview

Signal generation and replay is a critical optimization technique in systematic trading systems. The core idea is simple: computing trading signals from market data and features is computationally expensive, especially when running parameter sweeps across hundreds or thousands of combinations. By storing the generated signals and replaying them later, we can test different portfolio configurations, risk parameters, and execution models without recomputing the underlying signals.

However, implementing this efficiently presents several challenges. The naive approach of storing complete dataset copies for each parameter combination quickly becomes untenable - both in terms of storage space and memory usage during parallel execution. This document presents an elegant solution using indexed sparse storage that maintains the mathematical purity of data transformation while achieving practical efficiency.

## The Problem

When running a parameter optimization across 500 different strategy configurations, we face a fundamental trade-off. We need to store enough information to perfectly reconstruct the trading decisions, but we want to avoid massive data duplication. The challenge becomes even more complex when we introduce regime-aware strategies, where we need to filter signals based on market regimes, handle trades that span regime boundaries, and maintain proper event ordering.

The traditional approach of storing full datasets for each parameter combination fails because it creates storage requirements that grow linearly with the number of parameters. For a modest 100MB dataset with 500 parameter combinations, we would need 50GB of storage just for a single optimization run. This approach also makes parallelization difficult due to memory constraints.

## The Key Insight: Indexed Sparse Storage

The breakthrough comes from recognizing that all derived data (signals, regimes, filters) can be represented as sparse indices into an immutable base dataset. Instead of storing complete datasets, we store only the points where something interesting happens - a signal is generated, a regime changes, a threshold is crossed. Everything else can be efficiently reconstructed by combining these sparse indices with the original data.

This approach achieves several critical properties. First, it maintains a perfect isomorphism between the stored representation and the full dataset - we can always reconstruct the complete trading sequence without loss of information. Second, it scales with the number of events rather than the size of the dataset, making it extremely efficient for typical trading strategies that generate signals infrequently. Third, it naturally supports filtering and composition, allowing us to efficiently query for specific conditions like "all buy signals during bull regimes."

## Topology Considerations

An important architectural decision is the ordering of components in the event flow. The system uses a refined topology where the Feature Container serves as the central processing hub for all market data, feature calculation, and strategy execution.

The recommended topology is:
```
Symbol_Timeframe → BARS → Feature Container → SIGNALS → Portfolio → ORDERS → Execution
                              ↓
                   (calls Classifiers → Strategies)
                   (includes bar data in signals)
```

Where:
- **Symbol_Timeframe containers** publish BAR events to the root bus
- **Feature Container** receives all bars, calculates features, and calls stateless classifier and strategy functions
- **Portfolio containers** receive SIGNAL events and call stateless risk validators before generating orders
- **Execution container** processes approved orders

This design achieves several critical properties:
- **No race conditions**: Bars are synchronized in the Feature container before processing
- **Consistency**: All features/classifications/strategies see the same synchronized data
- **Simplicity**: Signals contain all necessary context (including bar data)
- **Efficiency**: Single Feature container eliminates redundant calculations

### Multi-Symbol Synchronization with TimeAlignmentBuffer

The Feature Container uses TimeAlignmentBuffer to handle complex multi-symbol, multi-timeframe strategies:

```python
# Strategy declares its data requirements
strategy_requirements = [
    {
        'strategy_id': 'correlation_momentum',
        'required_data': [('SPY', '1m'), ('QQQ', '1m'), ('NVDA', '5m')],
        'strategy_function': correlation_momentum_strategy
    }
]

# Feature container configuration with TimeAlignmentBuffer
feature_config = {
    'container_type': 'feature',
    'components': {
        'synchronizer': {
            'type': 'TimeAlignmentBuffer',
            'strategy_requirements': strategy_requirements,
            'max_buffer_size': 1000
        },
        'feature_calc': {
            'type': 'MultiSymbolFeatureCalculator',
            'indicators': ['sma', 'rsi', 'correlation']
        }
    },
    'stateless_services': {
        'classifiers': ['trend', 'volatility'],
        'strategies': ['momentum', 'mean_reversion']
    }
}
```

During signal generation, the buffer handles synchronization:
- Waits for all required bars at each timestamp
- Aligns different timeframes (e.g., 5m bars with 1m bars)
- Only calls strategies when complete data is available
- Prevents race conditions and ensures consistent feature calculation

**Important Event Flow Adjustment for Signal Storage:**
When implementing signal generation and replay topologies, the event flow must be adjusted:

1. **Signal Generation Mode**: Events flow through Feature container, then signals are captured
   - Symbol_Timeframe containers stream bars
   - Feature container processes all bars, runs classifiers and strategies
   - Signal events are captured via event tracing before reaching portfolios
   
2. **Signal Replay Mode**: Events start from stored signals, bypassing the Feature container
   - Load signals from storage and emit them directly to portfolios
   - Skip the entire Symbol_Timeframe → Feature container pipeline
   - Signals already contain necessary bar context from generation phase

This requires the topology builder to construct different container graphs based on the mode.

### Simplified Event Flow

With the removal of the routing module, the architecture is cleaner:

```python
# Signal Generation Mode
root_container = Container(role=ContainerRole.BACKTEST)

# Children publish to parent, parent's bus distributes
symbol_tf_containers → root (via publish_event(target_scope="parent"))
                    ↓
    root.event_bus distributes to all children
                    ↓
feature_container (subscribed to BAR events on root bus)
                    ↓
    publishes SIGNAL events to root
                    ↓
portfolio_containers (filtered subscriptions on root bus)

# No complex routing - just parent/child + filtered subscriptions
```

## Architecture

### Event Bus Signal Filtering

The EventBus requires filtering for SIGNAL events to ensure efficient routing in multi-portfolio topologies:

```python
# Portfolio containers MUST specify which strategies they handle
root_bus.subscribe(
    EventType.SIGNAL.value,
    portfolio.receive_event,
    filter_func=lambda e: e.payload.get('strategy_id') in ['momentum_1', 'pairs_1']
)

# This prevents the N×M delivery problem:
# Without filtering: 10 portfolios × 20 strategies = 200 deliveries
# With filtering: 20 deliveries (each signal to exactly one portfolio)
```

### Integration with ADMF-PC Containers

The signal replay functionality leverages ADMF-PC's container architecture with a key architectural component - the Feature container:

```python
# Feature container configuration
feature_container_config = {
    'role': ContainerRole.FEATURE,
    'components': {
        'synchronizer': {
            'type': 'TimeAlignmentBuffer',
            'strategy_requirements': [
                {
                    'strategy_id': 'momentum_multi',
                    'required_data': [('SPY', '1m'), ('QQQ', '1m'), ('NVDA', '5m')],
                    'strategy_function': momentum_strategy
                }
            ]
        },
        'feature_calculator': {...}
    },
    'stateless_services': {
        'classifiers': ['trend', 'volatility'],
        'strategies': ['momentum', 'mean_reversion']
    }
}

# Signal generation topology
generation_topology = {
    'symbol_timeframe_containers': {
        'SPY_1m': {...},
        'QQQ_1m': {...},
        'NVDA_5m': {...}
    },
    'feature_container': feature_container_config,
    # No portfolio/execution needed for generation
}

# Signal replay topology  
replay_topology = {
    'signal_source': {
        'type': 'signal_replay',
        'indices': {...}
    },
    'portfolio_containers': {...},
    'execution_container': {...}
    # No symbol_timeframe or feature containers needed
}
```

### Integrated Optimization Approach

The optimization functionality is now integrated into the core architecture:

```python
# TopologyBuilder handles parameter expansion directly
topology_builder.build_topology({
    'portfolios': [
        {
            'id': f'portfolio_{i}',
            'strategy_assignments': strategy_subset,
            'risk_params': risk_variant,
            # Event tracing configured per portfolio
            'event_tracing': {
                'events_to_trace': ['POSITION_OPEN', 'POSITION_CLOSE', 'FILL'],
                'retention_policy': 'trade_complete'  # Sparse storage!
            }
        }
        for i, (strategy_subset, risk_variant) in enumerate(parameter_grid)
    ]
})

# Objective function calculated from event traces
# No separate optimization module needed
```

This leverages the sparse event storage - only storing events that matter for objective function calculation.

**Old Approach:**
- Dedicated optimization module handled parameter expansion and result collection
- Complex coordination between optimization module and topology builder
- Separate result extraction logic

**New Approach:**
1. **TopologyBuilder handles parameter expansion** - The topology builder directly expands parameter combinations when building topologies
2. **Objective function observers** - Each portfolio container gets an observer that calculates the objective function from event traces
3. **Sequencer orchestrates results** - The sequencer collects results from containers and feeds them back to the topology builder for the next phase

```python
# Example: New optimization flow
class TopologyBuilder:
    def build_topology(self, config, parameter_expansion=None):
        # Expand parameters inline during topology creation
        if parameter_expansion:
            portfolios = self._expand_portfolio_parameters(config, parameter_expansion)
        
        # Add objective function observer to each portfolio
        for portfolio in portfolios:
            objective_observer = ObjectiveFunctionObserver(
                function=config.get('objective_function', 'sharpe_ratio')
            )
            portfolio.event_bus.attach_observer(objective_observer)

class Sequencer:
    def execute_optimization_sequence(self, phases):
        for phase in phases:
            # Build topology with parameter expansion
            topology = self.topology_builder.build_topology(
                phase.config, 
                phase.parameter_expansion
            )
            
            # Execute phase
            results = self.execute_topology(topology)
            
            # Extract objective values from observers
            objective_values = self._extract_objective_values(topology)
            
            # Feed results to next phase
            next_phase.parameter_selection = self._select_best_parameters(
                objective_values
            )
```

This integrated approach:
- Eliminates the need for a separate optimization module
- Leverages the event tracing system for objective function calculation
- Simplifies the data flow between phases
- Makes optimization a natural part of the topology/sequencer pattern

### Core Data Structure

The system uses a simple but powerful data model where everything is an indexed reference to the base dataset:

```python
from dataclasses import dataclass
from typing import Dict, Any, Protocol, runtime_checkable

@dataclass
class IndexedSignal:
    """A trading signal at a specific point in time."""
    bar_idx: int      # Index into base dataset
    value: float      # Signal value (-1, 0, 1 or continuous)
    metadata: Dict    # Optional metadata (confidence, features used, etc.)

@dataclass
class IndexedRegime:
    """A regime classification at a specific point in time."""
    bar_idx: int      # Index into base dataset  
    regime: str       # Regime identifier
    confidence: float # Classification confidence

@dataclass
class IndexedClassifierChange:
    """Tracks when a classifier changes its regime classification."""
    bar_idx: int      # Index in the base dataset
    classifier: str   # Classifier name
    old_regime: str   # Previous regime (None for first)
    new_regime: str   # New regime classification

@dataclass
class IndexedEvent:
    """Generic indexed event for extensibility."""
    bar_idx: int      # Index into base dataset
    event_type: str   # Event type identifier
    payload: Any      # Event-specific data

    
@dataclass
class MultiSymbolSignal:
    """Signal that references multiple symbols/timeframes."""
    primary_symbol: str
    primary_timeframe: str
    primary_bar_idx: int
    
    # Additional symbol references
    symbol_refs: Dict[str, Dict[str, Any]]  # {symbol: {timeframe: tf, bar_idx: idx}}
    
    signal_value: float
    metadata: Dict[str, Any]
    
    def to_sparse_format(self) -> Dict[str, Any]:
        """Convert to sparse storage format."""
        return {
            'p': f"{self.primary_symbol}|{self.primary_timeframe}|{self.primary_bar_idx}",
            'refs': [(s, d['timeframe'], d['bar_idx']) for s, d in self.symbol_refs.items()],
            'v': self.signal_value,
            'm': self.metadata
        }
    
@runtime_checkable
class IndexedEventProtocol(Protocol):
    """Event that references a specific data index."""
    @property
    def bar_idx(self) -> int:
        """Index in the base dataset."""
        ...
```

### Sparse Regime Change Storage

Instead of storing regime classifications at every bar, the system only stores regime changes:

```python
# Storage format for classifier changes
classifier_changes.idx:
    bar_idx | classifier | old_regime | new_regime
    1000    | trend      | NEUTRAL    | TRENDING
    1523    | trend      | TRENDING   | CHOPPY
    2001    | volatility | LOW        | HIGH
```

This achieves extreme sparsity - if a regime lasts for 1000 bars, we store 1 entry instead of 1000.

### Multi-Symbol, Multi-Timeframe Storage Layout

The indexed approach naturally handles multiple symbols and timeframes without duplication:

```
data/
├── base/
│   ├── SPY_1m.parquet          # Base 1-minute data
│   ├── QQQ_1m.parquet
│   └── NVDA_5m.parquet         # Different timeframe base
└── indices/
    ├── signals/
    │   ├── multi_symbol_momentum.idx    # References multiple bases
    │   │   # Format: symbol|timeframe|bar_idx|value|metadata
    │   ├── pairs_SPY_QQQ.idx           # Cross-symbol strategies
    │   └── SPY_multi_tf.idx            # Multi-timeframe signals
    ├── classifier_changes/
    │   ├── market_regime_changes.idx    # Market-wide classifier
    │   └── SPY_trend_changes.idx       # Symbol-specific classifier
    └── cross_references/
        └── symbol_alignment.idx         # Maps time alignment across symbols
```

### Event Flow for Signal Generation

During signal generation, the Feature container orchestrates the entire process:

1. **Symbol_Timeframe containers publish BAR events** - Each source publishes to root bus
2. **Feature container synchronizes bars** - TimeAlignmentBuffer waits for all required sources
3. **Features are calculated on synchronized data** - Ensures consistency across symbols
4. **Classifiers are called as stateless functions** - Feature container tracks state changes
5. **Strategies are called with features and classifier states** - Generate signals sparsely
6. **Signal events include bar context** - Portfolios don't need separate BAR subscriptions

Example flow inside Feature container:
```python
# Inside Feature container with TimeAlignmentBuffer
def on_synchronized_bars(self, bars: Dict[str, Bar]):
    # Calculate features for all symbols
    features = {}
    for symbol, bar in bars.items():
        features[symbol] = self.calculate_features(bar)
    
    # Call classifiers (stateless functions)
    classifier_states = {}
    for classifier_name in self.classifiers:
        classifier_func = self.get_classifier(classifier_name)
        new_state = classifier_func(features)
        
        # Track changes for sparse storage
        if new_state != self.current_states.get(classifier_name):
            self.store_classifier_change(bar_idx, classifier_name, 
                                       self.current_states.get(classifier_name), 
                                       new_state)
            self.current_states[classifier_name] = new_state
        
        classifier_states[classifier_name] = new_state
    
    # Call strategies with full context
    for strategy_name in self.strategies:
        strategy_func = self.get_strategy(strategy_name)
        signal = strategy_func(features, classifier_states)
        
        if signal.value != 0:
            # Emit signal with bar data included
            self.emit_signal(SignalEvent(
                strategy=strategy_name,
                signal=signal,
                bars=bars,  # Include bar data
                features=features,
                classifier_states=classifier_states
            ))
```

### Event Flow for Signal Replay

During replay, the system reconstructs full events on demand:

1. **Load sparse indices** - Signal and classifier change indices are loaded
2. **Reconstruct classifier state** - Apply changes up to current bar to get state
3. **Merge signals with context** - When a signal index is hit, emit with full context
4. **Event-driven replay** - System remains reactive, not pre-computed

```python
def reconstruct_classifier_state(bar_idx: int, changes: List[ClassifierChange]) -> Dict[str, str]:
    """Reconstruct classifier state at any bar by replaying changes."""
    state = {}
    for change in changes:
        if change.bar_idx <= bar_idx:
            state[change.classifier] = change.new_regime
        else:
            break  # Changes are ordered
    return state

# During replay
for bar in base_data:
    current_classifier_state = reconstruct_classifier_state(bar.idx, classifier_changes)
    
    if bar.idx in signal_indices:
        signal_data = signal_indices[bar.idx]
        # Emit signal with reconstructed context
        emit_signal_event(bar, signal_data, current_classifier_state)
```

## Implementation Details

### Multi-Timeframe Signal Efficiency

Longer timeframes naturally produce sparser signals, making indexed storage extremely efficient:

```python
# Signal frequency by timeframe (typical momentum strategy)
timeframe_sparsity = {
    '1m': 20/390,    # ~5% of bars have signals
    '5m': 5/78,      # ~6% of bars have signals  
    '1h': 1/6.5,     # ~15% of bars have signals
    '1d': 0.2/1,     # ~20% of bars have signals
}

# Storage comparison for 1 year of data
traditional_storage = {
    '1m': 97500 * 4,  # Store signal value for every bar
    '5m': 19500 * 4,  # Even if 95% are zero
    '1h': 1625 * 4,   # Wasteful!
    '1d': 252 * 4
}  # Total: ~470KB

indexed_storage = {
    '1m': 4875 * 8,   # Only non-zero signals + index
    '5m': 1170 * 8,   # Much more efficient
    '1h': 244 * 8,    
    '1d': 50 * 8
}  # Total: ~50KB (90% reduction!)
```

### Sparse Storage Format

Each index file contains minimal information needed for reconstruction:

```python
class ClassifierChangeIndex:
    """Storage for classifier state changes."""
    
    def __init__(self, classifier_name: str):
        self.classifier = classifier_name
        self.changes = []  # List of (bar_idx, old_regime, new_regime)
        
    def append_change(self, bar_idx: int, old_regime: str, new_regime: str):
        """Record a regime change."""
        self.changes.append({
            'bar_idx': bar_idx,
            'old': old_regime,
            'new': new_regime
        })
        
    def save(self, filepath: str):
        """Save to Parquet format."""
        df = pd.DataFrame(self.changes)
        df.to_parquet(filepath, compression='snappy')
    
    def get_state_at_bar(self, bar_idx: int) -> Optional[str]:
        """Get classifier state at specific bar."""
        state = None
        for change in self.changes:
            if change['bar_idx'] <= bar_idx:
                state = change['new']
            else:
                break
        return state

class SignalIndex:
    """Storage for sparse signals with metadata."""
    
    def __init__(self, strategy_name: str):
        self.strategy = strategy_name
        self.signals = []
        
    def append_signal(self, bar_idx: int, signal_value: float, 
                     classifier_states: Dict[str, str]):
        """Record a signal with context."""
        self.signals.append({
            'bar_idx': bar_idx,
            'value': signal_value,
            'classifiers': classifier_states  # Snapshot at signal time
        })
```

### Regime-Aware Filtering

The indexed approach makes regime filtering elegant and efficient:

```python
def get_regime_filtered_signals(
    signals: List[IndexedSignal],
    regimes: List[IndexedRegime], 
    target_regime: str
) -> List[IndexedSignal]:
    """Get signals that occur during specific regime."""
    
    # Build regime ranges
    regime_ranges = []
    for i, regime in enumerate(regimes):
        if regime.regime == target_regime:
            start = regime.bar_idx
            end = regimes[i+1].bar_idx if i+1 < len(regimes) else float('inf')
            regime_ranges.append((start, end))
    
    # Filter signals by regime ranges
    return [
        s for s in signals
        if any(start <= s.bar_idx < end for start, end in regime_ranges)
    ]
```

### Boundary Trade Handling

Trades that span regime changes require careful handling to ensure positions opened in the target regime are properly closed even if the regime changes:

```python
# NOTE: This is reference code and should be reviewed for production use
class BoundaryAwareReplay:
    """Handles trades that cross regime boundaries."""
    
    def __init__(self, target_regime: str):
        self.target_regime = target_regime
        self.in_position = False
        self.position_open_regime = None
        self.position_open_idx = None
    
    def should_emit_signal(self, signal_idx: int, signal_value: float, 
                          current_regime: str) -> bool:
        """Determine if signal should be emitted based on regime rules."""
        
        # Case 1: Opening position
        if signal_value != 0 and not self.in_position:
            if current_regime == self.target_regime:
                # Open position in target regime
                self.in_position = True
                self.position_open_regime = current_regime
                self.position_open_idx = signal_idx
                return True
            else:
                # Skip signal - not in target regime
                return False
        
        # Case 2: Closing position
        elif signal_value != 0 and self.in_position:
            # ALWAYS emit close signal if we have open position
            # This handles boundary trades correctly
            self.in_position = False
            
            # Log if this was a boundary trade
            if current_regime != self.position_open_regime:
                print(f"Boundary trade: opened in {self.position_open_regime}, "
                      f"closed in {current_regime}")
            
            return True
        
        # Case 3: No signal
        return False
```

The key insight: We always honor close signals if we have an open position, regardless of current regime. This ensures we don't have hanging positions when replaying regime-filtered signals.

## Benefits

### Storage Efficiency
- **Ultra-sparse representation** - Store classifier changes, not states (100x fewer entries)
- **Single base dataset** - No duplication across parameter combinations  
- **Compressed format** - Parquet provides excellent compression for sparse data
- **Change-based storage** - If regime lasts 1000 bars, store 1 entry not 1000

### Computational Efficiency
- **Fast replay** - Simple index lookups and merges
- **Parallel friendly** - Each parameter set has independent indices
- **Memory efficient** - Can stream base data without loading everything

### Architectural Elegance
- **Clean separation** - Base data is immutable, indices are derived
- **Composable** - Can layer multiple indices (signals + regimes + filters)
- **Extensible** - New event types just add new index sets

### Mathematical Properties
- **Isomorphic** - Perfect mapping between stored and full representation
- **Filterable** - Can efficiently query any subset
- **Traceable** - Clear lineage from base data to final signals

## Future Extensions

The indexed approach naturally supports advanced features:

1. **Multi-timeframe analysis** - Indices can reference different timeframes
2. **Cross-asset signals** - Indices can reference multiple base datasets
3. **Feature importance** - Can store which features influenced each signal
4. **Online learning** - Can append to indices in real-time

## Event System Integration and Race Conditions

### Current State of Event System Refactor

The event system refactor (as described in `refactor.md`) provides a clean architecture with observer patterns and proper separation of concerns. However, it's important to note that the refactor focuses on code organization and does not inherently solve potential race conditions in multi-container systems.

### Potential Race Conditions

In the current architecture, race conditions could theoretically occur in scenarios such as:

1. **Multiple Portfolio Containers** - Different portfolios processing the same bar at different speeds
2. **Signal vs Risk Timing** - Risk decisions based on stale state while portfolios are updating  
3. **Event Bus Isolation** - Each container has its own EventBus with no global ordering

### Why Linear Topology Helps

The linear topology (Bars → Features → Classifiers → Strategies → Portfolio) significantly reduces race condition risks by:

1. **Sequential Processing** - Each stage completes before the next begins
2. **Atomic Updates** - Portfolios receive bar data and signals together from strategies
3. **No Broadcast Splitting** - Avoiding parallel paths that could diverge in timing

### Coordinator's Role

The Coordinator already enforces sequential processing of bars in the current implementation. Each bar is fully processed through all containers before the next bar begins. This provides natural phase barriers that prevent most race conditions.

For signal replay, the same coordination ensures that:
- All sparse indices are loaded before replay begins
- Each bar's signals are fully merged before emission
- Portfolio state remains consistent throughout replay

## Storage Example

Here's a concrete example showing the efficiency gains:

```python
# Traditional approach: Store regime at every bar
traditional_storage = {
    0: {'trend': 'NEUTRAL', 'volatility': 'LOW'},      # 2 values
    1: {'trend': 'NEUTRAL', 'volatility': 'LOW'},      # 2 values
    # ... 998 more identical entries ...
    1000: {'trend': 'TRENDING', 'volatility': 'LOW'},  # 2 values
    # ... continues for all 97,500 bars
}
# Storage: 97,500 bars × 2 classifiers = 195,000 values

# Sparse change approach: Store only changes
sparse_storage = {
    'trend_changes': [
        {'bar_idx': 0, 'old': None, 'new': 'NEUTRAL'},
        {'bar_idx': 1000, 'old': 'NEUTRAL', 'new': 'TRENDING'},
        {'bar_idx': 2500, 'old': 'TRENDING', 'new': 'CHOPPY'}
    ],
    'volatility_changes': [
        {'bar_idx': 0, 'old': None, 'new': 'LOW'},
        {'bar_idx': 3000, 'old': 'LOW', 'new': 'HIGH'}
    ]
}
# Storage: 5 change events total!

# Reconstruction is simple:
def get_state_at_bar(bar_idx):
    state = {}
    for change in sparse_storage['trend_changes']:
        if change['bar_idx'] <= bar_idx:
            state['trend'] = change['new']
    for change in sparse_storage['volatility_changes']:
        if change['bar_idx'] <= bar_idx:
            state['volatility'] = change['new']
    return state
```

## Conclusion

The indexed sparse storage approach provides an elegant solution to the signal storage and replay challenge. By recognizing that trading events are inherently sparse and that all derived data can be represented as indices into base data, we achieve both mathematical purity and practical efficiency. 

The key improvement of storing only classifier changes rather than states reduces storage requirements by 100-1000x for typical trading systems where regimes are stable for extended periods. This architecture scales naturally with the complexity of trading strategies while maintaining the simplicity needed for robust production systems.

The integration with ADMF-PC's event-driven architecture is natural - sparse indices are essentially stored events that can be replayed through the same container topology. The linear event flow and coordinator-enforced sequencing provide sufficient protection against race conditions for backtesting scenarios.

## Results Storage Architecture Options

### Storage Pattern Options

**Option 1: Parquet + Metadata JSON**
```
./results/
└── workflow_id/
    ├── metadata.json          # Workflow metadata, parameters
    ├── phases/
    │   ├── phase1/
    │   │   ├── metrics.parquet    # Columnar storage for metrics
    │   │   ├── trades.parquet     # Trade records
    │   │   └── events/            # Event traces (if enabled)
    │   │       └── *.jsonl.gz
    │   └── phase2/
    └── summary.json           # Final aggregate results
```

**Option 2: Time-Series Optimized**
```python
# Use DuckDB or TimescaleDB for time-series data
class TimeSeriesResultStore:
    def __init__(self, db_path: str):
        self.conn = duckdb.connect(db_path)
        self._init_schema()
    
    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                workflow_id VARCHAR,
                phase VARCHAR,
                container_id VARCHAR,
                timestamp TIMESTAMP,
                metric_name VARCHAR,
                metric_value DOUBLE,
                PRIMARY KEY (workflow_id, container_id, timestamp, metric_name)
            )
        """)
```

**Option 3: Hybrid Approach (Recommended)**
- Use Parquet for analytical queries (metrics, trades)
- Keep JSONL for event traces (better for streaming writes)
- SQLite for metadata and indexing
- Configure retention per data type:

```python
class HierarchicalResultStore:
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.retention_policies = {
            'metrics': timedelta(days=365),    # Keep metrics for 1 year
            'trades': timedelta(days=90),      # Keep trades for 90 days
            'events': timedelta(days=7),       # Keep detailed events for 7 days
            'summary': None                    # Keep summaries forever
        }
```

### Benefits of Each Approach

**Option 1 (Parquet + JSON)**
- Simple file-based structure
- Easy to browse and debug
- Good compression with Parquet
- No database dependencies

**Option 2 (Time-Series DB)**
- Optimized for time-series queries
- Built-in aggregation functions
- Efficient storage for high-frequency data
- Better for real-time analytics

**Option 3 (Hybrid)**
- Best of both worlds
- Flexible retention policies
- Optimized for different access patterns
- Scalable to production needs

### Retention Policy Considerations

1. **Event Traces**: Short retention (7-30 days) due to size
2. **Metrics**: Medium retention (90-365 days) for analysis
3. **Trade Records**: Long retention (1-5 years) for compliance
4. **Summaries**: Permanent retention for historical reference

### Data Migration Strategy

```python
class ResultMigrator:
    """Handle migrations between storage formats."""
    
    def migrate_v1_to_v2(self, old_path: Path, new_path: Path):
        """Migrate from JSON to Parquet format."""
        # Load old format
        with open(old_path / 'results.json') as f:
            old_data = json.load(f)
        
        # Convert to new format
        metrics_df = pd.DataFrame(old_data['metrics'])
        trades_df = pd.DataFrame(old_data['trades'])
        
        # Save in new format
        metrics_df.to_parquet(new_path / 'metrics.parquet')
        trades_df.to_parquet(new_path / 'trades.parquet')
        
        # Preserve metadata
        metadata = {
            'version': 2,
            'migrated_from': str(old_path),
            'migration_date': datetime.now().isoformat(),
            'original_metadata': old_data.get('metadata', {})
        }
        with open(new_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
```
