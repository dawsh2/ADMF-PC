# Multi-Process Fan-Out Architecture for Quantitative Trading System

## Current Bottleneck
Your single-threaded root container processes all portfolio containers sequentially, creating a bottleneck when running multiple portfolio configurations.

## Staged Execution Architecture for Heterogeneous Strategies

The key insight: **Don't wait for the slowest component at each time step**. Instead, separate execution into stages that run at their natural speeds, then replay signals for parallelized portfolio execution.

### The Problem with Synchronous Multi-Process Execution

Even with multi-process parallelization, synchronous execution forces all components to wait for the slowest one at each bar:

```
Bar 1: Fast strategies (10ms) → Wait for ML models (500ms) → Next bar
Bar 2: Fast strategies (10ms) → Wait for ML models (500ms) → Next bar
...
Total time: n_bars × slowest_component_time
```

### The Solution: Staged Asynchronous Execution

```
Stage 1: Fast Strategies (Parallel)
├── Group by shared features
├── Process all bars quickly
├── Save signals to storage
└── Time: ~10ms per bar

Stage 2: ML Strategy Models (Sequential with full resources)
├── Load each model once
├── Process all bars in batch
├── Save predictions as signals
└── Time: ~5ms per bar (after model load)

Stage 3: Risk Assessment (Parallel ML Models)
├── Portfolio Optimization ML
├── Correlation Risk ML
├── Custom Risk Ensembles
├── Save risk decisions to storage
└── Time: ~20ms per portfolio batch

Stage 4: Signal + Risk Fusion & Execution (Massively Parallel)
├── Load all signals from storage
├── Load all risk decisions from storage
├── Fuse signals with risk assessments
├── Run thousands of portfolios in parallel
├── Generate risk-adjusted orders
└── Track performance with full lineage
```

### Performance Comparison

#### Synchronous Approach (Traditional)
- 1000 bars × 500ms (waiting for ML) = **500 seconds**
- ML models running at 10% efficiency (waiting for next bar)
- Fast strategies idle 98% of the time
- Risk models constrained by portfolio processing bottleneck

#### Staged Approach (Optimized)
- Stage 1: 1000 bars × 10ms = **10 seconds** (fast strategies)
- Stage 2: Model load (2s) + 1000 bars × 5ms = **7 seconds** (ML strategy batch)
- Stage 3: Risk model load (3s) + 1000 portfolios × 20ms = **23 seconds** (risk assessment)
- Stage 4: Signal + risk fusion replay = **30 seconds** (parallel execution)
- **Total: 70 seconds (7.1x faster)**

#### Additional Benefits with Risk Workers
- **Resource Isolation**: Strategy ML and Risk ML don't compete for GPU/CPU
- **Independent Optimization**: Each risk model type optimized separately
- **Parallel Risk Assessment**: Portfolio optimization and correlation analysis run simultaneously
- **Complete Audit Trail**: Full lineage from signal → risk assessment → final order

### Implementation Benefits

1. **Natural Speed Processing**: Each component runs at its optimal speed
2. **Resource Optimization**: ML models use 100% GPU during their stage
3. **Debugging**: Can inspect saved signals between stages
4. **Flexibility**: Can re-run portfolio stage with different parameters
5. **Scalability**: Add more portfolio workers without re-running strategies

## Multi-Process Fan-Out Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MAIN PRODUCER PROCESS                                │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────────────────┐ │
│  │ Data        │  │ Shared       │  │ Feature Cache & Distribution        │ │
│  │ Streamers   │→ │ Feature      │→ │ • OHLCV, Volume, Technical Indicators│ │
│  │             │  │ Computer     │  │ • Market Microstructure              │ │
│  └─────────────┘  └──────────────┘  │ • Cross-Asset Correlations          │ │
│                                      └─────────────────────────────────────┘ │
│                           ↓                          ↓                       │
│                  ┌─────────────────┐       ┌─────────────────┐              │
│                  │ IPC Queue       │       │ IPC Queue       │              │
│                  │ (Features)      │       │ (Market Data)   │              │
│                  └─────────────────┘       └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
                             ↓                          ↓
        ┌────────────────────┼──────────────────────────┼────────────────────┐
        ↓                    ↓                          ↓                    ↓
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ STRATEGY WORKER │ │ STRATEGY WORKER │ │ STRATEGY WORKER │ │ STRATEGY WORKER │
│    PROCESS 1    │ │    PROCESS 2    │ │    PROCESS 3    │ │    PROCESS N    │
│                 │ │                 │ │                 │ │                 │
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │
│ │ML Model A   │ │ │ │ML Model B   │ │ │ │Traditional  │ │ │ │ML Model Z   │ │
│ │(LSTM)       │ │ │ │(XGBoost)    │ │ │ │Strategies   │ │ │ │(Transformer)│ │
│ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │
│ │Inference    │ │ │ │Inference    │ │ │ │Rule-based  │ │ │ │Inference    │ │
│ │Engine       │ │ │ │Engine       │ │ │ │Logic        │ │ │ │Engine       │ │
│ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │
└─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘
        ↓                    ↓                    ↓                    ↓
        └────────────────────┼────────────────────┼────────────────────┘
                             ↓                    ↓
                    ┌─────────────────┐  ┌─────────────────┐
                    │ IPC Queue       │  │ Signal Router & │
                    │ (Raw Signals)   │→ │ Aggregator      │
                    └─────────────────┘  └─────────────────┘
                                                  ↓
                                         ┌─────────────────┐
                                         │ IPC Queue       │
                                         │ (Final Signals) │
                                         └─────────────────┘
                                                  ↓

        ┌────────────────────┼────────────────────┼────────────────────┐
        ↓                    ↓                    ↓                    ↓
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│PORTFOLIO WORKER │ │PORTFOLIO WORKER │ │PORTFOLIO WORKER │ │PORTFOLIO WORKER │
│   PROCESS 1     │ │   PROCESS 2     │ │   PROCESS 3     │ │   PROCESS N     │
│                 │ │                 │ │                 │ │                 │
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │
│ │Portfolio    │ │ │ │Portfolio    │ │ │ │Portfolio    │ │ │ │Portfolio    │ │
│ │Manager      │ │ │ │Manager      │ │ │ │Manager      │ │ │ │Manager      │ │
│ │(Signal +    │ │ │ │(Signal +    │ │ │ │(Signal +    │ │ │ │(Signal +    │ │
│ │Risk Fusion) │ │ │ │Risk Fusion) │ │ │ │Risk Fusion) │ │ │ │Risk Fusion) │ │
│ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │
│ │Order        │ │ │ │Order        │ │ │ │Order        │ │ │ │Order        │ │
│ │Generation   │ │ │ │Generation   │ │ │ │Generation   │ │ │ │Generation   │ │
│ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │
└─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘
        ↓                    ↓                    ↓                    ↓
        └────────────────────┼────────────────────┼────────────────────┘
                             ↓                    ↓
        ┌────────────────────┼────────────────────┼────────────────────┐
        ↓                    ↓                    ↓                    ↓
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ RISK WORKER     │ │ RISK WORKER     │ │ RISK WORKER     │ │ RISK WORKER     │
│    PROCESS 1    │ │    PROCESS 2    │ │    PROCESS 3    │ │    PROCESS N    │
│                 │ │                 │ │                 │ │                 │
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │
│ │Portfolio    │ │ │ │Correlation  │ │ │ │Stateless    │ │ │ │Custom ML    │ │
│ │Optimization │ │ │ │Risk ML      │ │ │ │Risk Group   │ │ │ │Risk Ensemble│ │
│ │ML Model     │ │ │ │Model        │ │ │ │(VaR, Limits)│ │ │ │             │ │
│ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │
│ │Risk Feature │ │ │ │Risk Feature │ │ │ │Risk Feature │ │ │ │Risk Feature │ │
│ │Engineering  │ │ │ │Engineering  │ │ │ │Engineering  │ │ │ │Engineering  │ │
│ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │
└─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘
        ↓                    ↓                    ↓                    ↓
        └────────────────────┼────────────────────┼────────────────────┘
                             ↓                    ↓
                    ┌─────────────────┐  ┌─────────────────┐
                    │ IPC Queue       │  │ Risk Assessment │
                    │ (Risk Signals)  │→ │ Aggregator      │
                    └─────────────────┘  └─────────────────┘
                                                  ↓
                                         ┌─────────────────┐
                                         │ IPC Queue       │
                                         │ (Risk Decisions)│
                                         └─────────────────┘
                                                  ↓							 
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EXECUTION PROCESS                                      │
│  ┌─────────────────┐                    ┌─────────────────┐                │
│  │ IPC Queue       │ ←─ Receives ORDERs │ IPC Queue       │                │
│  │ (Orders)        │                    │ (Fills)         │ ←─ Publishes   │
│  └─────────────────┘                    └─────────────────┘    FILLs        │
│           ↓                                       ↑                         │
│  ┌─────────────────┐                             │                         │
│  │ Execution       │─────────────────────────────┘                         │
│  │ Engine          │                                                       │
│  └─────────────────┘                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Architectural Enhancements

### 1. **Computational Separation**
- **Shared Feature Computer**: Computes expensive features once (OHLCV, technical indicators, correlations)
- **Strategy Worker Processes**: Each runs different ML models or strategy types in parallel
- **Feature Distribution**: Cached features distributed via IPC to avoid recomputation

### 2. **ML Model Isolation**
- Each ML model runs in its own process with dedicated resources
- GPU allocation can be managed per process
- Model loading and inference isolated from each other
- Different model types (PyTorch, TensorFlow, XGBoost) don't interfere

### 3. **Signal Aggregation Layer**
- **Signal Router**: Collects raw signals from all strategy workers
- **Signal Aggregation**: Combines multiple signals (ensemble methods, voting, etc.)
- **Final Signal Distribution**: Processes and routes final signals to portfolio workers

### 4. **Resource Optimization**
- Heavy feature computation done once in main producer
- Strategy workers focus on model inference only
- Portfolio workers receive pre-processed, aggregated signals
- Clear separation of computational concerns

## Scaling to Thousands of Portfolios on Limited Cores

### **Problem**: 8 CPU cores can only run 8 truly parallel processes

### **Solution**: Batch Processing Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     8-CORE MACHINE OPTIMIZATION                             │
│                                                                             │
│ CORE 1: Producer Process          CORE 2: Strategy Worker 1                │
│ • Data Streaming                  • ML Model A (LSTM)                      │
│ • Shared Features                 • Serves all portfolios                  │
│                                                                             │
│ CORE 3: Strategy Worker 2         CORE 4: Strategy Worker 3                │
│ • ML Model B (XGBoost)           • Traditional Strategies                  │
│ • Serves all portfolios          • Serves all portfolios                   │
│                                                                             │
│ CORE 5: Portfolio Batch 1        CORE 6: Portfolio Batch 2                │
│ • Portfolios 1-500               • Portfolios 501-1000                    │
│ • Vectorized Processing          • Vectorized Processing                   │
│                                                                             │
│ CORE 7: Portfolio Batch 3        CORE 8: Execution + Cleanup              │
│ • Portfolios 1001-1500           • Order Processing                        │
│ • Vectorized Processing          • Fill Distribution                       │
│                                  • System Management                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### **Batch Processing Strategies**

#### **1. Portfolio Batching (Primary Strategy)**
- **Batch Size**: 500-1000 portfolios per worker process
- **Processing Mode**: Vectorized operations across portfolio batch
- **Memory Efficiency**: Shared data structures, copy-on-write
- **Performance**: ~95% efficiency vs individual processes

**Example**: 3000 portfolios = 6 batch workers (500 each) + 2 cores for other tasks

#### **2. Time-Based Batching** 
- **Signal Batching**: Collect 100 signals, process across all portfolios simultaneously
- **Vectorized Risk**: Single risk calculation across entire batch
- **Batch Orders**: Generate orders for multiple portfolios in single operation
- **Throughput**: 10-100x faster than individual processing

#### **3. Hierarchical Processing**
```
Level 1: Market Data Processing (1 core)
    ↓
Level 2: Feature Computation (1 core) 
    ↓
Level 3: Strategy Execution (2-3 cores)
    ↓
Level 4: Portfolio Batches (3-4 cores)
    ↓
Level 5: Execution & Management (1 core)
```

### **Scaling Techniques**

#### **Memory Optimization**
- **Shared Signal Cache**: All portfolios share same signal data
- **Copy-on-Write**: Portfolio states only duplicated when modified
- **Lazy Evaluation**: Only compute what's needed for active portfolios
- **Memory Pooling**: Reuse allocated memory between batches

#### **Computational Optimization**
- **Vectorized Operations**: NumPy/Pandas operations across portfolio arrays
- **Pre-compiled Filters**: Strategy filters compiled once, reused thousands of times
- **Batch Risk Checks**: Single risk calculation validates entire portfolio batch
- **SIMD Instructions**: CPU vector instructions for parallel calculations

#### **I/O Optimization**
- **Signal Routing**: Pre-filter signals before distribution to batches
- **Order Batching**: Combine multiple portfolio orders into single messages
- **Asynchronous Processing**: Non-blocking I/O for order submission
- **Memory Mapping**: Zero-copy data sharing between processes

### **Performance Characteristics**

#### **Traditional Approach (1 Portfolio/Process)**
- **3000 portfolios**: Requires 3000 processes (impossible on 8 cores)
- **Performance**: Limited by context switching overhead
- **Memory**: 3000x duplication of shared data

#### **Batch Processing Approach**
- **3000 portfolios**: 6 batch workers (500 portfolios each)
- **Throughput**: 2-3 million portfolio operations/second
- **Memory**: 95% reduction vs individual processes
- **Latency**: <1ms per portfolio operation

### **Real-World Scaling Examples**

#### **Parameter Sweeps**
- **Scenario**: Test 10,000 parameter combinations
- **Implementation**: 20 batches × 500 portfolios per batch
- **Cores Used**: 6 (batches) + 2 (strategy/data)
- **Total Time**: ~2-3x single-threaded performance

#### **Walk-Forward Analysis** 
- **Scenario**: 1000 portfolios × 100 time windows
- **Implementation**: Sequential batches through time windows
- **Memory**: Constant (reuse same batch workers)
- **Scalability**: Linear with available time

#### **Multi-Strategy Optimization**
- **Scenario**: 50 strategies × 1000 parameter sets each
- **Implementation**: Strategy batching + portfolio batching
- **Resource Usage**: All 8 cores fully utilized
- **Efficiency**: 90%+ CPU utilization

### **When to Use Each Approach**

#### **Use Individual Processes When:**
- **Small Scale**: <100 portfolios
- **High Isolation**: Portfolios must be completely independent
- **Different Resources**: Portfolios need different CPU/memory allocations

#### **Use Batch Processing When:**
- **Large Scale**: >500 portfolios  
- **Similar Logic**: Portfolios use same strategies/risk models
- **Performance Critical**: Need maximum throughput
- **Resource Constrained**: Limited cores/memory

### **Implementation Complexity**

#### **Low Complexity**: Use existing multi-process architecture as-is
- **Scale**: Up to ~50-100 portfolios efficiently
- **Development**: Minimal changes to existing code

#### **Medium Complexity**: Add portfolio batching
- **Scale**: Up to ~5,000 portfolios efficiently  
- **Development**: Batch processing layer + vectorized operations

#### **High Complexity**: Full optimization with SIMD/GPU
- **Scale**: 10,000+ portfolios efficiently
- **Development**: Custom optimized batch processors + GPU acceleration

## Implementation Strategy

### Stage-Based Implementation

#### Stage 1: Fast Strategy Signal Generation
```python
# Run all fast strategies in parallel, grouped by shared features
def run_fast_strategies():
    # Group strategies by their feature requirements
    strategy_groups = group_by_features(fast_strategies)
    
    # Process each group in parallel
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for group_id, strategies in strategy_groups.items():
            future = executor.submit(process_strategy_group, group_id, strategies)
            futures.append(future)
    
    # Collect and save signals
    for future in futures:
        signals = future.result()
        signal_storage.save_batch(signals)
```

#### Stage 2: ML Strategy Model Signal Generation
```python
# Run ML strategy models sequentially with full resources
def run_ml_strategy_models():
    for model_config in ml_strategy_models:
        # Load model once
        model = load_model(model_config)
        
        # Process all historical data in batches
        for batch in data_batches:
            predictions = model.predict_batch(batch)
            signals = convert_predictions_to_signals(predictions)
            signal_storage.save_batch(signals)
        
        # Free resources before next model
        del model
        torch.cuda.empty_cache()
```

#### Stage 3: Risk Assessment Generation
```python
# Run risk models in parallel, specialized by type
def run_risk_assessment():
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        
        # Portfolio optimization models
        future = executor.submit(run_portfolio_optimization_models)
        futures.append(future)
        
        # Correlation risk models  
        future = executor.submit(run_correlation_risk_models)
        futures.append(future)
        
        # Stateless risk validators (VaR, limits)
        future = executor.submit(run_stateless_risk_validators)
        futures.append(future)
        
        # Custom ML risk ensembles
        future = executor.submit(run_custom_risk_ensembles)
        futures.append(future)
        
        # Collect all risk assessments
        for future in futures:
            risk_decisions = future.result()
            risk_storage.save_batch(risk_decisions)
```

#### Stage 4: Signal + Risk Fusion and Portfolio Execution
```python
# Fuse signals with risk assessments and execute portfolios in parallel
def fuse_and_execute_portfolios():
    # Load all generated signals and risk decisions
    all_signals = signal_storage.load_all()
    all_risk_decisions = risk_storage.load_all()
    
    # Split portfolios into batches for workers
    portfolio_batches = split_portfolios(all_portfolios, n_workers=8)
    
    # Process portfolio batches in parallel
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for batch in portfolio_batches:
            future = executor.submit(
                process_portfolio_batch_with_risk_fusion,
                batch,
                all_signals,
                all_risk_decisions,
                vectorized=True  # Use vectorized operations
            )
            futures.append(future)
        
        # Collect results
        for future in futures:
            batch_results = future.result()
            save_portfolio_results(batch_results)

def process_portfolio_batch_with_risk_fusion(batch, signals, risk_decisions, vectorized=True):
    """Process portfolio batch with signal-risk fusion."""
    results = []
    
    for portfolio in batch:
        # Filter relevant signals and risk decisions
        portfolio_signals = filter_signals_for_portfolio(signals, portfolio.config)
        portfolio_risk = filter_risk_for_portfolio(risk_decisions, portfolio.config)
        
        # Fuse signals with risk assessments
        risk_adjusted_signals = fuse_signals_with_risk(
            portfolio_signals,
            portfolio_risk,
            portfolio.risk_tolerance
        )
        
        # Generate orders based on fused signals
        orders = portfolio.generate_orders(risk_adjusted_signals)
        results.append({
            'portfolio_id': portfolio.id,
            'orders': orders,
            'risk_metrics': portfolio_risk,
            'signal_count': len(portfolio_signals)
        })
    
    return results
```

### Traditional Process Structure (Still Supported)
- **Main Producer Process**: Runs data streamers + strategy container + signal generation
- **Portfolio Worker Processes**: One process per portfolio configuration
- **Execution Process**: Dedicated process for order handling and fills

### Signal Storage and Replay Architecture

The staged approach relies on efficient signal storage and replay:

```python
# Signal and risk storage during generation stages
class SignalStorage:
    def __init__(self, storage_path: str):
        self.path = Path(storage_path)
        self.index = {}  # Fast lookup by strategy/time
    
    def save_signal(self, signal: Signal):
        # Store with indexing for fast replay
        key = f"{signal.strategy_id}_{signal.timestamp}"
        self.index[key] = signal.to_dict()
    
    def get_signals_for_portfolio(self, portfolio_config: Dict) -> List[Signal]:
        # Efficiently retrieve only relevant signals
        strategy_ids = portfolio_config['subscribed_strategies']
        return [s for s in self.load_all() if s.strategy_id in strategy_ids]

class RiskStorage:
    def __init__(self, storage_path: str):
        self.path = Path(storage_path)
        self.risk_index = {}  # Fast lookup by risk type/portfolio
    
    def save_risk_decision(self, risk_decision: RiskDecision):
        # Store with indexing for fast fusion
        key = f"{risk_decision.risk_type}_{risk_decision.portfolio_pattern}"
        self.risk_index[key] = risk_decision.to_dict()
    
    def get_risk_for_portfolio(self, portfolio_config: Dict) -> List[RiskDecision]:
        # Efficiently retrieve applicable risk decisions
        portfolio_pattern = portfolio_config.get('risk_pattern', 'default')
        return [r for r in self.load_all() if r.applies_to_portfolio(portfolio_pattern)]
```

### Traditional IPC for Real-Time Systems

For live trading or traditional multi-process setups:

```python
# Use ZeroMQ, Redis Pub/Sub, or Python multiprocessing.Queue

# Main producer publishes to signal queue
signal_queue.publish({
    'event_type': 'SIGNAL',
    'symbol': 'AAPL',
    'direction': 'BUY',
    'strategy_id': 'momentum_1',
    'timestamp': datetime.now(),
    'container_id': 'strategy_container'
})

# Portfolio workers subscribe with filtering
portfolio_worker.subscribe(
    queue='signals',
    filter_func=lambda event: event['strategy_id'] in assigned_strategies
)
```

### Event Flow Comparison

#### Staged Approach (Research/Backtesting)
1. **Stage 1**: Fast strategies → Save signals to storage
2. **Stage 2**: ML strategy models → Append signals to storage
3. **Stage 3**: Risk models → Save risk assessments to storage
4. **Stage 4**: Load signals + risk → Fuse and execute through portfolios in parallel

#### Traditional Approach (Live Trading)
1. **Data Flow**: Data streamers publish BAR events to main process event bus
2. **Feature Computation**: Strategy container computes features/signals once (shared efficiency)
3. **Signal Distribution**: Main process publishes SIGNAL events to IPC signal queue
4. **Portfolio Processing**: Each portfolio worker receives relevant signals in parallel
5. **Order Routing**: Portfolio workers send ORDER events to execution queue
6. **Fill Distribution**: Execution process publishes FILL events back to portfolio workers

### Container ID Management
```python
# Orders must include originating container ID
order_event = {
    'event_type': 'ORDER',
    'container_id': 'portfolio_worker_1',  # Critical for routing fills back
    'order_id': 'order_12345',
    'symbol': 'AAPL',
    'quantity': 100,
    'side': 'BUY'
}

# Fills include original container ID for filtering
fill_event = {
    'event_type': 'FILL',
    'container_id': 'portfolio_worker_1',  # Same as original order
    'order_id': 'order_12345',
    'fill_price': 150.25,
    'fill_quantity': 100
}
```

### Benefits of Multi-Process Architecture
- **True Parallelism**: Portfolio logic runs concurrently on all CPU cores
- **Shared Computation**: Feature/signal generation happens only once
- **Process Isolation**: Bug in one portfolio can't crash others
- **Scalability**: Add more portfolio workers as needed

### Trade-offs
- **Serialization Overhead**: Events must be serialized for IPC (usually negligible)
- **Process Management**: Need to start/manage multiple OS processes
- **Determinism**: Non-deterministic order arrival at execution (acceptable for research)

## Implementation with Your Current Code

Your existing `barriers.py` system will work perfectly within each worker process to prevent duplicate orders from that specific portfolio. The event bus filtering you've already implemented can be adapted for IPC message filtering.

The key insight is that you keep the efficient shared computation (data + strategy) while parallelizing the bottleneck (portfolio processing).

### Choosing the Right Approach

#### Use Staged Execution When:
- Running backtests with heterogeneous strategies (fast + ML)
- Using complex ML risk models (portfolio optimization, correlation analysis)
- Optimizing thousands of parameter combinations
- ML models have high latency (>100ms per prediction)
- You want to analyze signals and risk assessments between generation and execution
- Research and development workflows requiring complete audit trails

#### Use Traditional Multi-Process When:
- Live trading with low-latency requirements
- All strategies have similar performance characteristics
- Real-time signal generation is critical
- Simpler system architecture is preferred

### Next Steps

1. **For Staged Execution**: 
   - Implement signal and risk storage layers
   - Create risk worker process framework
   - Modify topology to support 4-stage workflows
   - Add signal-risk fusion logic
2. **For Traditional Multi-Process**: 
   - Add IPC layer to existing event bus
   - Implement process management for strategy and risk workers
   - Create risk assessment communication protocols
3. **Hybrid Approach**: 
   - Use staged for research and parameter optimization
   - Use traditional for production trading with real-time risk assessment
