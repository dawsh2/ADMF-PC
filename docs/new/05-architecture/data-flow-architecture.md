# Data Flow Architecture

Complete analysis of how data moves through ADMF-PC from ingestion to execution, including transformation patterns, performance optimizations, and production data pipelines.

## 🎯 Overview

ADMF-PC's data flow architecture is designed around three core principles: **semantic data transformation**, **container isolation**, and **performance optimization**. Data moves through the system via strongly-typed events that preserve lineage while enabling massive parallelization and optimization.

The architecture supports multiple execution modes optimized for different use cases - from rapid research iteration to production trading execution.

## 📊 Core Data Flow Patterns

### Primary Data Flow: Traditional Backtest

The fundamental data transformation pattern through ADMF-PC containers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PRIMARY DATA FLOW                                 │
│                                                                             │
│  Raw Market Data → Enriched Data → Signals → Risk Assessment → Orders      │
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │    DATA     │───▶│ INDICATORS  │───▶│  STRATEGY   │───▶│ EXECUTION   │  │
│  │ CONTAINER   │    │ CONTAINER   │    │ CONTAINER   │    │ CONTAINER   │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                   │                   │                   │       │
│         ▼                   ▼                   ▼                   ▼       │
│   Raw OHLCV Data    Technical Indicators   Trading Signals     Order Events │
│   Volume Data       • SMA, EMA, RSI       • BUY/SELL/HOLD     • Market     │
│   Timestamp Info    • MACD, ATR, BB       • Signal Strength   • Limit      │
│   Symbol Metadata   • Custom Indicators   • Confidence Score  • Quantities │
│                                                                             │
│  Data Characteristics:                                                      │
│  • Type-safe semantic events with full schema validation                   │
│  • Correlation IDs linking related events across containers                │
│  • Performance tiers (fast/standard/reliable) based on event importance    │
│  • Automatic lineage tracking for debugging and compliance                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Event-Driven Data Transformation

ADMF-PC uses semantic events to ensure type safety and traceability:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SEMANTIC EVENT TRANSFORMATION                          │
│                                                                             │
│  Event Type         Schema                    Purpose                       │
│  ═══════════════════════════════════════════════════════════════════════    │
│                                                                             │
│  BarEvent          Raw market data structure                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ symbol: str                    # "SPY", "QQQ"                       │   │
│  │ timestamp: datetime            # Market data timestamp              │   │
│  │ open, high, low, close: float  # OHLC prices                        │   │
│  │ volume: int                    # Trading volume                     │   │
│  │ correlation_id: str            # Links related events               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                   │                                         │
│                         TRANSFORM │                                         │
│                                   ▼                                         │
│  IndicatorEvent    Enriched technical analysis data                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ indicator_name: str            # "RSI_14", "SMA_20"                 │   │
│  │ value: float                   # Calculated indicator value         │   │
│  │ confidence: float              # Calculation confidence             │   │
│  │ parameters: Dict               # Indicator parameters used          │   │
│  │ causation_id: str              # Links to source BarEvent           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                   │                                         │
│                         TRANSFORM │                                         │
│                                   ▼                                         │
│  TradingSignal     Strategy decision with context                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ action: "BUY"|"SELL"|"HOLD"    # Trading decision                   │   │
│  │ strength: float                # Signal strength (0.0-1.0)          │   │
│  │ confidence: float              # Strategy confidence                │   │
│  │ regime_context: str            # Market regime context              │   │
│  │ strategy_id: str               # Strategy that generated signal     │   │
│  │ causation_id: str              # Links to source IndicatorEvent     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                   │                                         │
│                         TRANSFORM │                                         │
│                                   ▼                                         │
│  OrderEvent        Executable trading instruction                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ symbol: str                    # Trading symbol                     │   │
│  │ side: "BUY"|"SELL"             # Order side                         │   │
│  │ quantity: int                  # Share quantity                     │   │
│  │ order_type: str                # "MARKET", "LIMIT", etc.            │   │
│  │ risk_validated: bool           # Passed risk checks                 │   │
│  │ causation_id: str              # Links to source TradingSignal      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Key Benefits:                                                              │
│  • Complete data lineage from market data to execution                     │
│  • Type safety prevents runtime errors                                     │
│  • Schema evolution supports system upgrades                               │
│  • Correlation tracking enables end-to-end debugging                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Multi-Phase Data Flow Patterns

ADMF-PC's power comes from coordinating data flow across multiple execution phases:

### Phase 1: Signal Generation

Data flow optimized for signal capture and analysis:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SIGNAL GENERATION DATA FLOW                          │
│                                                                             │
│  Purpose: Capture high-quality signals for later replay optimization       │
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │ Historical  │───▶│ Indicator   │───▶│  Strategy   │───▶│  Signal     │  │
│  │    Data     │    │ Calculation │    │ Evaluation  │    │ Capture     │  │
│  │   Loader    │    │   Engine    │    │   Engine    │    │  Storage    │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                   │                   │                   │       │
│         ▼                   ▼                   ▼                   ▼       │
│                                                                             │
│  Raw OHLCV        Technical Analysis    Trading Decisions    Signal Database │
│  • CSV files     • RSI, MACD, ATR      • Signal strength    • Compressed    │
│  • Databases     • Moving averages     • Confidence scores  • Parquet files │
│  • API feeds     • Custom indicators   • Regime context     • Fast replay   │
│                                                                             │
│  Data Optimizations:                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ • Indicator results cached for multiple strategy evaluation        │   │
│  │ • Signals compressed with metadata for efficient storage           │   │
│  │ • Parallel indicator calculation across multiple CPU cores         │   │
│  │ • Memory-mapped data access for large datasets                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Output: Signal files containing enriched trading signals with full        │
│          context for 10-100x faster optimization in subsequent phases      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 2: Signal Replay Optimization

Ultra-fast data flow that skips expensive computations:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SIGNAL REPLAY DATA FLOW                              │
│                                                                             │
│  Purpose: 10-100x faster optimization by replaying pre-computed signals    │
│                                                                             │
│  ┌─────────────┐                        ┌─────────────┐    ┌─────────────┐  │
│  │   Signal    │                       │    Risk     │───▶│ Portfolio   │  │
│  │  Database   │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ▶│ Management  │    │ Simulation  │  │
│  │   Files     │     (Skip data &      │  Engine     │    │   Engine    │  │
│  └─────────────┘     indicator calc)   └─────────────┘    └─────────────┘  │
│         │                                      │                   │       │
│         ▼                                      ▼                   ▼       │
│                                                                             │
│  Pre-computed Signals    NO COMPUTATION      Risk Assessment   Performance  │
│  • Trading decisions    • No data loading    • Position sizing  • Returns  │
│  • Signal strength      • No indicators      • Stop losses      • Sharpe   │
│  • Market context       • No strategy eval   • Risk limits      • Drawdown │
│                                                                             │
│  Performance Comparison:                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Execution Time                              │   │
│  │                                                                     │   │
│  │  Traditional Backtest:  ████████████████████████████ 100% (baseline) │   │
│  │  Signal Replay:         ██ 2-10% (10-50x faster)                    │   │
│  │                                                                     │   │
│  │  Memory Usage:                                                      │   │
│  │  Traditional Backtest:  ████████████████ 100% (baseline)           │   │
│  │  Signal Replay:         ████ 25% (4x less memory)                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Key Optimization: Focus computational resources only on components        │
│                    being optimized (risk parameters, position sizing)      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 3: Multi-Strategy Ensemble

Data flow for combining multiple strategy signals:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ENSEMBLE DATA FLOW PATTERN                             │
│                                                                             │
│  Purpose: Combine signals from multiple strategies with optimal weighting   │
│                                                                             │
│                               Market Data                                   │
│                                    │                                        │
│                    ┌───────────────┼───────────────┐                        │
│                    │               │               │                        │
│                    ▼               ▼               ▼                        │
│           ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│           │ Momentum    │  │ Mean Rev    │  │ Breakout    │                │
│           │ Strategy    │  │ Strategy    │  │ Strategy    │                │
│           │ Container   │  │ Container   │  │ Container   │                │
│           └─────────────┘  └─────────────┘  └─────────────┘                │
│                    │               │               │                        │
│                    ▼               ▼               ▼                        │
│                Signal A         Signal B       Signal C                    │
│                strength=0.8     strength=0.6   strength=0.4                │
│                confidence=0.9   confidence=0.7 confidence=0.8               │
│                    │               │               │                        │
│                    └───────────────┼───────────────┘                        │
│                                    │                                        │
│                                    ▼                                        │
│                    ┌─────────────────────────────────┐                      │
│                    │      ENSEMBLE OPTIMIZER        │                      │
│                    │                                 │                      │
│                    │ • Weight allocation algorithm   │                      │
│                    │ • Signal correlation analysis   │                      │
│                    │ • Performance-based weighting   │                      │
│                    │ • Risk-adjusted combination     │                      │
│                    └─────────────────────────────────┘                      │
│                                    │                                        │
│                                    ▼                                        │
│                        Combined Ensemble Signal                             │
│                        strength = Σ(weight_i × signal_i)                   │
│                        confidence = f(individual_confidences)               │
│                                                                             │
│  Ensemble Weighting Strategies:                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ • Equal Weight:     w₁ = w₂ = w₃ = 1/3                             │   │
│  │ • Performance:      wᵢ ∝ sharpe_ratio_i                            │   │
│  │ • Volatility:       wᵢ ∝ 1/volatility_i                            │   │
│  │ • Kelly Criterion:  wᵢ ∝ kelly_fraction_i                          │   │
│  │ • Adaptive:         wᵢ = f(recent_performance_i, correlation_i)     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Data Characteristics:                                                      │
│  • Signal correlation matrix updated continuously                          │
│  • Performance attribution tracks individual strategy contributions        │
│  • Dynamic weight adjustment based on changing market conditions           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🏗️ Container Data Boundaries

Clear data interfaces maintain isolation while enabling communication:

### Container Internal Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CONTAINER INTERNAL DATA FLOW                          │
│                                                                             │
│  Each container maintains complete data isolation with clean interfaces     │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    STRATEGY CONTAINER                                 │ │
│  │                                                                       │ │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐               │ │
│  │  │   Market    │───▶│  Strategy   │───▶│   Signal    │               │ │
│  │  │    Data     │    │   Logic     │    │ Generator   │               │ │
│  │  │   Buffer    │    │   Engine    │    │             │               │ │
│  │  └─────────────┘    └─────────────┘    └─────────────┘               │ │
│  │         │                   │                   │                     │ │
│  │         │                   │                   │                     │ │
│  │  Internal Event Bus (Isolated)                   │                     │ │
│  │  ┌─────────────────────────────────────────────┐ │                     │ │
│  │  │ • BarEvent subscriptions                    │ │                     │ │
│  │  │ • IndicatorEvent processing                 │ │                     │ │
│  │  │ • Internal state management                 │ │                     │ │
│  │  │ • Component lifecycle events                │ │                     │ │
│  │  └─────────────────────────────────────────────┘ │                     │ │
│  │                                                   │                     │ │
│  └───────────────────────────────────────────────────┼─────────────────────┘ │
│                                                      │                       │
│                                          Cross-Container Event               │
│                                                      │                       │
│                                                      ▼                       │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                     RISK CONTAINER                                    │ │
│  │                                                                       │ │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐               │ │
│  │  │   Signal    │───▶│    Risk     │───▶│   Order     │               │ │
│  │  │ Validation  │    │ Assessment  │    │ Generation  │               │ │
│  │  │   Engine    │    │   Engine    │    │   Engine    │               │ │
│  │  └─────────────┘    └─────────────┘    └─────────────┘               │ │
│  │                                                                       │ │
│  │  Container State (Isolated):                                          │ │
│  │  • Current positions and exposures                                    │ │
│  │  • Risk limits and thresholds                                         │ │
│  │  • Performance tracking metrics                                       │ │
│  │  • Internal calculation cache                                         │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  Data Flow Characteristics:                                                 │
│  • No shared memory between containers                                     │
│  • All communication via strongly-typed events                             │
│  • Complete state encapsulation within containers                          │
│  • Event routing handles cross-container data flow                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Cross-Container Data Communication

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CROSS-CONTAINER DATA FLOW                               │
│                                                                             │
│  Event Router manages all data flow between isolated containers             │
│                                                                             │
│  ┌─────────────┐                                          ┌─────────────┐  │
│  │ Container A │                EVENT ROUTER              │ Container B │  │
│  │             │                                          │             │  │
│  │ ┌─────────┐ │    ┌─────────────────────────────────┐    │ ┌─────────┐ │  │
│  │ │Internal │ │───▶│                                 │───▶│ │Internal │ │  │
│  │ │Event Bus│ │    │ • Scope-based routing           │    │ │Event Bus│ │  │
│  │ └─────────┘ │    │ • Type-safe event validation    │    │ └─────────┘ │  │
│  │             │    │ • Performance tier optimization │    │             │  │
│  │             │    │ • Correlation ID tracking       │    │             │  │
│  │             │    │ • Dead letter queue handling    │    │             │  │
│  │             │    └─────────────────────────────────┘    │             │  │
│  └─────────────┘                                          └─────────────┘  │
│                                                                             │
│  Event Routing Patterns:                                                    │
│                                                                             │
│  1. Pipeline: A → B → C (sequential processing)                           │
│     Data flows through ordered sequence of containers                      │
│                                                                             │
│  2. Broadcast: A → [B, C, D] (one-to-many distribution)                   │
│     Single data source distributes to multiple consumers                   │
│                                                                             │
│  3. Hierarchical: Parent ↔ Children (tree-based communication)            │
│     Context flows down, results aggregate up                               │
│                                                                             │
│  4. Selective: A → router → [B|C|D] (content-based routing)               │
│     Data routed based on content or business rules                         │
│                                                                             │
│  Performance Optimization by Event Type:                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Fast Tier (Market Data):     < 1ms latency, batched delivery       │   │
│  │ Standard Tier (Signals):     < 10ms latency, reliable delivery      │   │
│  │ Reliable Tier (Orders):      < 100ms latency, guaranteed delivery   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 📈 Performance-Optimized Data Flows

### Memory-Efficient Data Handling

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MEMORY-OPTIMIZED DATA PATTERNS                           │
│                                                                             │
│  Large Dataset Optimization (> 1M bars):                                   │
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│  │   Memory    │───▶│   Chunked   │───▶│  Streaming  │                     │
│  │   Mapping   │    │ Processing  │    │   Results   │                     │
│  │             │    │             │    │             │                     │
│  │ • mmap()    │    │ • 10K chunks│    │ • Disk-based│                     │
│  │ • Zero-copy │    │ • Parallel  │    │ • Top-N only│                     │
│  │ • Lazy load │    │ • Memory    │    │ • Compressed│                     │
│  │            │    │   recycling │    │             │                     │
│  └─────────────┘    └─────────────┘    └─────────────┘                     │
│                                                                             │
│  Memory Usage Pattern:                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Without Optimization:                                              │   │
│  │  Memory Usage ████████████████████████████████ 8GB (all in RAM)    │   │
│  │                                                                     │   │
│  │  With Memory Mapping:                                               │   │
│  │  Memory Usage ████████ 2GB (lazy loading)                          │   │
│  │                                                                     │   │
│  │  With Chunked Processing:                                           │   │
│  │  Memory Usage ████ 1GB (streaming)                                  │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Implementation Strategy:                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. Data Assessment:                                                 │   │
│  │    if data_size > 500MB: enable_memory_mapping()                    │   │
│  │                                                                     │   │
│  │ 2. Processing Strategy:                                             │   │
│  │    if memory_available < data_size * 2: enable_chunked_processing() │   │
│  │                                                                     │   │
│  │ 3. Result Management:                                               │   │
│  │    if optimization_trials > 1000: enable_streaming_results()        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Parallel Data Processing

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PARALLEL DATA PROCESSING                               │
│                                                                             │
│  Indicator Calculation Parallelization:                                    │
│                                                                             │
│  Single-Threaded:                                                          │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐                 │
│  │ SMA(10) │───▶│ SMA(20) │───▶│ RSI(14) │───▶│ MACD    │ = 28 seconds    │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘                 │
│                                                                             │
│  Multi-Threaded:                                                           │
│  ┌─────────┐    ┌─────────┐                                                │
│  │ SMA(10) │    │ SMA(20) │                                                │
│  └─────────┘    └─────────┘                                                │
│  ┌─────────┐    ┌─────────┐  = 7 seconds (4x speedup)                     │
│  │ RSI(14) │    │  MACD   │                                                │
│  └─────────┘    └─────────┘                                                │
│                                                                             │
│  Dependency-Aware Parallelization:                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Level 1 (No Dependencies):    SMA, EMA, RSI                       │   │
│  │         │                         │                                │   │
│  │         ▼                         ▼                                │   │
│  │  Level 2 (Depends on Level 1):   MACD, Bollinger Bands            │   │
│  │         │                                                          │   │
│  │         ▼                                                          │   │
│  │  Level 3 (Depends on Level 2):   Custom Composite Indicators      │   │
│  │                                                                     │   │
│  │  Execution: All indicators at same level calculated in parallel    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Container-Level Parallelization:                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Parameter Optimization (1000 trials):                             │   │
│  │                                                                     │   │
│  │  Single Container:     ████████████████████████████ 60 minutes     │   │
│  │  8 Containers:         ████████ 8 minutes (7.5x speedup)           │   │
│  │  16 Containers:        ████ 4 minutes (15x speedup)                │   │
│  │  32 Containers:        ██ 2 minutes (30x speedup)                  │   │
│  │                                                                     │   │
│  │  Signal Replay (1000 trials):                                      │   │
│  │  8 Containers:         ██ 2 minutes (baseline for signal replay)   │   │
│  │  16 Containers:        █ 1 minute (2x speedup)                     │   │
│  │  32 Containers:        █ 30 seconds (4x speedup)                   │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Production Data Pipeline Patterns

### Real-Time Data Integration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        REAL-TIME DATA PIPELINE                              │
│                                                                             │
│  Production trading requires millisecond-latency data processing            │
│                                                                             │
│  External Data Sources                 ADMF-PC Processing                   │
│  ┌─────────────────┐                 ┌─────────────────────────────────┐   │
│  │ Market Data     │────────────────▶│ Data Ingestion Container        │   │
│  │ • Bloomberg     │  TCP/Websocket  │ • Real-time validation          │   │
│  │ • Refinitiv     │  < 1ms latency  │ • Format normalization          │   │
│  │ • Polygon       │                 │ • Missing data handling         │   │
│  │ • IEX Cloud     │                 │ • Timestamp synchronization     │   │
│  └─────────────────┘                 └─────────────────────────────────┘   │
│                                                      │                       │
│  ┌─────────────────┐                                │                       │
│  │ Alternative     │────────────────────────────────┘                       │
│  │ Data Sources    │  HTTP/REST APIs                                        │
│  │ • News feeds    │  < 100ms latency                                       │
│  │ • Social data   │                                                        │
│  │ • Economic data │                                                        │
│  └─────────────────┘                                                        │
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   REAL-TIME PROCESSING PIPELINE                     │   │
│  │                                                                     │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │   │
│  │  │   Market    │───▶│ Technical   │───▶│  Strategy   │              │   │
│  │  │    Data     │    │ Indicators  │    │ Evaluation  │              │   │
│  │  │ Streaming   │    │ (Real-time) │    │(Sub-second) │              │   │
│  │  └─────────────┘    └─────────────┘    └─────────────┘              │   │
│  │                                                                     │   │
│  │  Performance Requirements:                                          │   │
│  │  • Bar processing: < 5ms                                           │   │
│  │  • Indicator calculation: < 10ms                                   │   │
│  │  • Signal generation: < 25ms                                       │   │
│  │  • Risk validation: < 15ms                                         │   │
│  │  • Order generation: < 10ms                                        │   │
│  │  • Total latency: < 65ms (market data to order)                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Data Quality Assurance:                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ • Duplicate detection and removal                                  │   │
│  │ • Out-of-sequence data handling                                     │   │
│  │ • Price validation (circuit breaker detection)                     │   │
│  │ • Volume validation (unusual activity detection)                   │   │
│  │ • Gap detection and interpolation                                   │   │
│  │ • Market hours validation                                           │   │
│  │ • Holiday and halted stock handling                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### High-Availability Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     HIGH-AVAILABILITY DATA ARCHITECTURE                     │
│                                                                             │
│  Production systems require fault-tolerant data flow with failover          │
│                                                                             │
│  Primary Data Path:                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│  │ Primary     │───▶│ Primary     │───▶│ Primary     │                     │
│  │ Data Feed   │    │ Processing  │    │ Strategy    │                     │
│  │ (Bloomberg) │    │ Container   │    │ Container   │                     │
│  └─────────────┘    └─────────────┘    └─────────────┘                     │
│                                                                             │
│  Backup Data Path (Automatic Failover):                                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│  │ Secondary   │───▶│ Secondary   │───▶│ Standby     │                     │
│  │ Data Feed   │    │ Processing  │    │ Strategy    │                     │
│  │ (Refinitiv) │    │ Container   │    │ Container   │                     │
│  └─────────────┘    └─────────────┘    └─────────────┘                     │
│                                                                             │
│  Data Validation and Reconciliation:                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  ┌─────────────┐              ┌─────────────┐                       │   │
│  │  │ Data Source │             │ Data Source │                       │   │
│  │  │      A      │             │      B      │                       │   │
│  │  └──────┬──────┘             └──────┬──────┘                       │   │
│  │         │                           │                              │   │
│  │         ▼                           ▼                              │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │            RECONCILIATION ENGINE                            │   │   │
│  │  │                                                             │   │   │
│  │  │ • Price consistency validation (tolerance: ±0.01%)         │   │   │
│  │  │ • Volume correlation analysis                               │   │   │
│  │  │ • Timestamp synchronization                                │   │   │
│  │  │ • Missing data detection and alerts                        │   │   │
│  │  │ • Automatic source switching on quality degradation        │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Failover Decision Logic:                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. Data Quality Metrics:                                            │   │
│  │    • Latency > 100ms for 30 seconds → Switch                       │   │
│  │    • Missing bars > 5% → Switch                                     │   │
│  │    • Price discrepancy > 0.1% → Alert and validate                 │   │
│  │                                                                     │   │
│  │ 2. Connection Health:                                               │   │
│  │    • TCP connection drops → Immediate reconnect attempt             │   │
│  │    • 3 consecutive reconnect failures → Switch to backup           │   │
│  │                                                                     │   │
│  │ 3. Processing Health:                                               │   │
│  │    • Container health check failure → Restart container            │   │
│  │    • Processing delay > 500ms → Scale up resources                 │   │
│  │    • Memory usage > 90% → Enable memory optimization               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🔍 Data Flow Debugging and Monitoring

### Event Lineage Tracking

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EVENT LINEAGE AND DEBUGGING                          │
│                                                                             │
│  Complete traceability of data flow for debugging and compliance            │
│                                                                             │
│  Event Correlation Example:                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Trade Execution ID: "trade_20231201_001"                          │   │
│  │                                                                     │   │
│  │  Event Chain:                                                       │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ 1. BarEvent                                                 │   │   │
│  │  │    event_id: "bar_spy_20231201_093000"                     │   │   │
│  │  │    correlation_id: "trade_20231201_001"                    │   │   │
│  │  │    data: {symbol: "SPY", close: 451.25, volume: 1500000}   │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                              ↓ caused                              │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ 2. IndicatorEvent                                           │   │   │
│  │  │    event_id: "ind_rsi_20231201_093000"                     │   │   │
│  │  │    causation_id: "bar_spy_20231201_093000"                 │   │   │
│  │  │    correlation_id: "trade_20231201_001"                    │   │   │
│  │  │    data: {indicator: "RSI_14", value: 65.4, confidence: 0.85} │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                              ↓ caused                              │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ 3. TradingSignal                                            │   │   │
│  │  │    event_id: "sig_momentum_20231201_093000"                 │   │   │
│  │  │    causation_id: "ind_rsi_20231201_093000"                 │   │   │
│  │  │    correlation_id: "trade_20231201_001"                    │   │   │
│  │  │    data: {action: "BUY", strength: 0.75, confidence: 0.8}  │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                              ↓ caused                              │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ 4. OrderEvent                                               │   │   │
│  │  │    event_id: "ord_spy_20231201_093000"                     │   │   │
│  │  │    causation_id: "sig_momentum_20231201_093000"            │   │   │
│  │  │    correlation_id: "trade_20231201_001"                    │   │   │
│  │  │    data: {symbol: "SPY", side: "BUY", quantity: 100}       │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                              ↓ caused                              │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ 5. FillEvent                                                │   │   │
│  │  │    event_id: "fill_spy_20231201_093015"                    │   │   │
│  │  │    causation_id: "ord_spy_20231201_093000"                 │   │   │
│  │  │    correlation_id: "trade_20231201_001"                    │   │   │
│  │  │    data: {price: 451.30, quantity: 100, commission: 0.50}  │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Debugging Capabilities:                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ • Full event history reconstruction                                 │   │
│  │ • Performance analysis at each transformation step                  │   │
│  │ • Container-level timing and resource usage                        │   │
│  │ • Signal quality metrics and confidence tracking                   │   │
│  │ • Risk decision audit trail                                        │   │
│  │ • Execution quality analysis (slippage, timing)                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🎯 Key Takeaways

### Data Flow Design Principles

1. **Semantic Transformation**: Data moves through strongly-typed events that preserve meaning and enable validation
2. **Container Isolation**: Clean data boundaries prevent contamination while enabling parallel processing  
3. **Performance Optimization**: Different flow patterns optimized for different use cases (signal replay, real-time, batch)
4. **Lineage Tracking**: Complete traceability from market data to execution for debugging and compliance
5. **Scalable Architecture**: Patterns that scale from research (single machine) to production (distributed systems)

### Production Considerations

- **Latency Requirements**: Real-time flows require <100ms end-to-end latency
- **Reliability**: High-availability patterns with automatic failover and data validation
- **Monitoring**: Comprehensive event tracking and performance monitoring
- **Resource Management**: Memory-efficient patterns for large-scale data processing
- **Integration**: Clean interfaces for external system integration

The data flow architecture enables ADMF-PC to scale from rapid research iteration to production trading execution while maintaining data integrity, performance, and operational excellence.

---

Continue to [System Integration Architecture](system-integration-architecture.md) for external system integration patterns →