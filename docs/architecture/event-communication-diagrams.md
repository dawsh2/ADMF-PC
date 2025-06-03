# Pluggable Event Communication Adapters - Visual Problem Analysis

## The Core Problem: Organizational Patterns vs Communication Needs

### Problem 1: Same Components, Different Organizations

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           THE COMPONENT LIBRARY                         │
│  Data Container │ Indicator Container │ Strategy Container │ Risk │ Exec │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    Same Components, Different Organizations
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        STRATEGY-FIRST PATTERN                           │
│                                                                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │ Strategy A      │    │ Strategy B      │    │ Strategy C      │     │
│  │ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │     │
│  │ │ Data        │ │    │ │ Data        │ │    │ │ Data        │ │     │
│  │ │ Indicators  │ │    │ │ Indicators  │ │    │ │ Indicators  │ │     │
│  │ │ Risk        │ │    │ │ Risk        │ │    │ │ Risk        │ │     │
│  │ │ Execution   │ │    │ │ Execution   │ │    │ │ Execution   │ │     │
│  │ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │     │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘     │
│         │                       │                       │              │
│         ▼                       ▼                       ▼              │
│    How do these communicate? Different patterns needed!                │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                       CLASSIFIER-FIRST PATTERN                          │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    HMM Classifier                               │   │
│  │  ┌─────────────────┐    ┌─────────────────┐                    │   │
│  │  │ Conservative    │    │ Aggressive      │                    │   │
│  │  │ Risk Profile    │    │ Risk Profile    │                    │   │
│  │  │ ┌─────────────┐ │    │ ┌─────────────┐ │                    │   │
│  │  │ │ Portfolio A │ │    │ │ Portfolio B │ │                    │   │
│  │  │ │ Strategies  │ │    │ │ Strategies  │ │                    │   │
│  │  │ └─────────────┘ │    │ └─────────────┘ │                    │   │
│  │  └─────────────────┘    └─────────────────┘                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│                               ▼                                         │
│         Completely different communication patterns needed!             │
└─────────────────────────────────────────────────────────────────────────┘
```

### Problem 2: Without Adapters - Hardcoded Communication

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    NIGHTMARE: HARDCODED COMMUNICATION                   │
│                                                                         │
│  if organization == "strategy_first":                                   │
│      for strategy in strategies:                                        │
│          strategy.connect_to_execution(execution_container)             │
│          strategy.on_signal(lambda s: execution.execute(s))             │
│                                                                         │
│  elif organization == "classifier_first":                              │
│      classifier.on_regime_change(lambda r: update_all_risk_containers)  │
│      for risk_container in classifier.risk_containers:                  │
│          risk_container.on_order(lambda o: execution.execute(o))        │
│                                                                         │
│  elif organization == "risk_first":                                     │
│      for risk_profile in risk_profiles:                                 │
│          risk_profile.aggregate_signals_from_strategies()               │
│          risk_profile.on_order(lambda o: execution.execute(o))          │
│                                                                         │
│  elif organization == "portfolio_first":                               │
│      # ... yet another completely different pattern                     │
│                                                                         │
│  ❌ PROBLEMS:                                                           │
│  • Code changes needed for each organizational pattern                  │
│  • Testing nightmare - must test all combinations                       │
│  • Cannot switch patterns without code changes                          │
│  • No logging integration                                               │
│  • Cannot A/B test communication patterns                               │
└─────────────────────────────────────────────────────────────────────────┘
```

## The Solution: Pluggable Communication Adapters

### Solution Overview: Adapters Decouple Organization from Communication

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ADAPTER SOLUTION                                │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    ORGANIZATIONAL PATTERNS                      │   │
│  │         (How components are grouped and configured)             │   │
│  │                                                                 │   │
│  │  Strategy-First │ Classifier-First │ Risk-First │ Portfolio    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                   │
│                                    │ DECOUPLED                         │
│                                    │                                   │
│  ┌─────────────────────────────────▼───────────────────────────────┐   │
│  │                    COMMUNICATION ADAPTERS                       │   │
│  │              (How data flows between components)                 │   │
│  │                                                                 │   │
│  │  Pipeline │ Broadcast │ Hierarchical │ Selective │ Custom       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                   │
│                                    ▼                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    ISOLATED EVENT BUSES                         │   │
│  │          (Maintains all isolation and logging benefits)         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ✅ BENEFITS:                                                          │
│  • Same code works with ANY organizational pattern                     │
│  • Communication patterns configurable via YAML                        │
│  • Easy A/B testing of different communication approaches              │
│  • Full logging integration maintained                                 │
│  • Future-proof for distributed deployment                             │
└─────────────────────────────────────────────────────────────────────────┘
```

## Adapter Types and Their Use Cases

### Adapter Type 1: Pipeline Adapter

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE ADAPTER                                 │
│           For: Linear data transformation workflows                     │
│                                                                         │
│  ┌─────────┐    ┌─────────────┐    ┌─────────┐    ┌─────────────┐      │
│  │  Data   │───▶│ Indicators  │───▶│Strategy │───▶│ Execution   │      │
│  │Container│    │ Container   │    │Container│    │ Container   │      │
│  └─────────┘    └─────────────┘    └─────────┘    └─────────────┘      │
│       │               │                │               │               │
│       │               │                │               │               │
│       ▼               ▼                ▼               ▼               │
│   BAR Events    INDICATOR Events   SIGNAL Events   ORDER Events        │
│                                                                         │
│  Configuration:                                                         │
│  adapters:                                                              │
│    - type: "pipeline"                                                   │
│      containers: ["data", "indicators", "strategy", "execution"]       │
│      tier: "standard"  # Performance optimization                      │
│                                                                         │
│  Use Cases:                                                             │
│  • Simple strategy backtests                                           │
│  • Single-strategy workflows                                           │
│  • Basic data → analysis → execution flows                             │
│  • Works with ANY organizational pattern!                              │
└─────────────────────────────────────────────────────────────────────────┘
```

### Adapter Type 2: Broadcast Adapter

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        BROADCAST ADAPTER                                │
│           For: One-to-many data distribution                            │
│                                                                         │
│  ┌─────────────────┐                                                    │
│  │ Indicator Hub   │                                                    │
│  │ (Single Source) │                                                    │
│  └─────────┬───────┘                                                    │
│            │                                                            │
│            │ BROADCAST                                                  │
│    ┌───────┼───────┬───────┬───────┐                                    │
│    │       │       │       │       │                                    │
│    ▼       ▼       ▼       ▼       ▼                                    │
│  ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐                                  │
│  │St1│   │St2│   │St3│   │St4│   │St5│                                  │
│  └───┘   └───┘   └───┘   └───┘   └───┘                                  │
│ Strategy Strategy Strategy Strategy Strategy                             │
│    1       2       3       4       5                                    │
│                                                                         │
│  Configuration:                                                         │
│  adapters:                                                              │
│    - type: "broadcast"                                                  │
│      source: "indicator_hub"                                            │
│      targets: ["strategy_001", "strategy_002", "strategy_003", ...]     │
│      tier: "fast"  # High-frequency data distribution                   │
│                                                                         │
│  Use Cases:                                                             │
│  • Market data distribution to multiple strategies                     │
│  • Indicator updates to strategy ensemble                              │
│  • Risk alerts to all portfolios                                       │
│  • Perfect for Strategy-First organizational pattern                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Adapter Type 3: Hierarchical Adapter

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       HIERARCHICAL ADAPTER                              │
│        For: Parent-child relationships with context flow               │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    HMM CLASSIFIER                               │   │
│  │                 (Market Regime Context)                         │   │
│  └─────────────────────────┬───────────────────────────────────────┘   │
│                            │                                           │
│                            │ HIERARCHICAL FLOW                         │
│              ┌─────────────┼─────────────┐                             │
│              │             │             │                             │
│              ▼             ▼             ▼                             │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐           │
│  │ Conservative    │ │ Balanced        │ │ Aggressive      │           │
│  │ Risk Profile    │ │ Risk Profile    │ │ Risk Profile    │           │
│  └─────────┬───────┘ └─────────┬───────┘ └─────────┬───────┘           │
│            │                   │                   │                   │
│            ▼                   ▼                   ▼                   │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐           │
│  │ Portfolio A     │ │ Portfolio B     │ │ Portfolio C     │           │
│  │ (Tech Focus)    │ │ (Broad Market)  │ │ (High Beta)     │           │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘           │
│                                                                         │
│  Context Flow:                                                          │
│  • Regime changes flow DOWN to all risk profiles                       │
│  • Portfolio results aggregate UP to risk profiles                     │
│  • Risk metrics aggregate UP to classifier                             │
│                                                                         │
│  Configuration:                                                         │
│  adapters:                                                              │
│    - type: "hierarchical"                                              │
│      parent: "hmm_classifier"                                          │
│      children: ["conservative_risk", "balanced_risk", "aggressive"]    │
│                                                                         │
│  Use Cases:                                                             │
│  • Perfect for Classifier-First organizational pattern                 │
│  • Regime-based parameter adjustment                                   │
│  • Risk profile coordination                                           │
│  • Portfolio performance aggregation                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Adapter Type 4: Selective Adapter

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SELECTIVE ADAPTER                                │
│        For: Conditional routing based on content/rules                 │
│                                                                         │
│  ┌─────────────────┐                                                    │
│  │ Strategy Signal │                                                    │
│  │ Generator       │                                                    │
│  └─────────┬───────┘                                                    │
│            │                                                            │
│            ▼                                                            │
│  ┌─────────────────┐                                                    │
│  │ SELECTIVE       │                                                    │
│  │ ROUTING RULES   │                                                    │
│  └─────────┬───────┘                                                    │
│            │                                                            │
│    ┌───────┼───────┬───────┬───────┐                                    │
│    │       │       │       │       │                                    │
│    ▼       ▼       ▼       ▼       ▼                                    │
│                                                                         │
│  Signal.confidence > 0.8  →  Aggressive Portfolio                      │
│  Signal.confidence < 0.3  →  Paper Trading                             │
│  Signal.regime == "BULL"  →  Growth Portfolio                          │
│  Signal.regime == "BEAR"  →  Defensive Portfolio                       │
│  Default                  →  Balanced Portfolio                        │
│                                                                         │
│  Configuration:                                                         │
│  adapters:                                                              │
│    - type: "selective"                                                  │
│      source: "strategy_signals"                                        │
│      rules:                                                             │
│        - condition: "signal.confidence > 0.8"                          │
│          target: "aggressive_portfolio"                                 │
│        - condition: "signal.regime == 'BULL'"                          │
│          target: "growth_portfolio"                                     │
│        - condition: "default"                                           │
│          target: "balanced_portfolio"                                   │
│      tier: "reliable"  # Important routing decisions                    │
│                                                                         │
│  Use Cases:                                                             │
│  • Complex ensemble signal routing                                     │
│  • Performance-based allocation                                        │
│  • Risk-based signal filtering                                         │
│  • A/B testing of different routing strategies                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Real-World Example: Same Trading System, Different Patterns

### Scenario: Momentum + Mean Reversion Strategy System

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    THE SAME TRADING COMPONENTS                          │
│                                                                         │
│  Data    │ Indicators │ Momentum   │ Mean Rev   │ Risk │ Execution      │
│  Reader  │ (RSI,MACD) │ Strategy   │ Strategy   │ Mgmt │ Engine         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Pattern 1: Strategy-First Organization

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        STRATEGY-FIRST APPROACH                          │
│                                                                         │
│  ┌─────────────────────────┐    ┌─────────────────────────────────────┐ │
│  │  MOMENTUM STRATEGY      │    │  MEAN REVERSION STRATEGY            │ │
│  │  CONTAINER              │    │  CONTAINER                          │ │
│  │ ┌─────────────────────┐ │    │ ┌─────────────────────────────────┐ │ │
│  │ │ Data Reader         │ │    │ │ Data Reader                     │ │ │
│  │ │ RSI, MACD           │ │    │ │ RSI, MACD                       │ │ │
│  │ │ Momentum Logic      │ │    │ │ Mean Reversion Logic            │ │ │
│  │ │ Risk Management     │ │    │ │ Risk Management                 │ │ │
│  │ └─────────────────────┘ │    │ └─────────────────────────────────┘ │ │
│  └─────────────────────────┘    └─────────────────────────────────────┘ │
│            │                                       │                    │
│            └───────────┬───────────────────────────┘                    │
│                        │                                                │
│                        ▼                                                │
│            ┌─────────────────────────┐                                  │
│            │    EXECUTION ENGINE     │                                  │
│            └─────────────────────────┘                                  │
│                                                                         │
│  Communication Config:                                                  │
│  adapters:                                                              │
│    - type: "pipeline"                                                   │
│      containers: ["momentum_strategy", "execution"]                     │
│    - type: "pipeline"                                                   │
│      containers: ["mean_reversion_strategy", "execution"]               │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Pattern 2: Classifier-First Organization (Same Components!)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       CLASSIFIER-FIRST APPROACH                         │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    MARKET CLASSIFIER                            │   │
│  │                  (Bull/Bear Detection)                          │   │
│  └─────────────────────────┬───────────────────────────────────────┘   │
│                            │                                           │
│              ┌─────────────┼─────────────┐                             │
│              │             │             │                             │
│              ▼             ▼             ▼                             │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐           │
│  │ BULL REGIME     │ │ NEUTRAL REGIME  │ │ BEAR REGIME     │           │
│  │ PORTFOLIO       │ │ PORTFOLIO       │ │ PORTFOLIO       │           │
│  │ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │           │
│  │ │ Momentum    │ │ │ │ Both        │ │ │ │ Mean Rev    │ │           │
│  │ │ Strategy    │ │ │ │ Strategies  │ │ │ │ Strategy    │ │           │
│  │ │ (Aggressive)│ │ │ │ (Balanced)  │ │ │ │ (Defensive) │ │           │
│  │ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │           │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘           │
│                                                                         │
│  Communication Config:                                                  │
│  adapters:                                                              │
│    - type: "hierarchical"                                              │
│      parent: "market_classifier"                                       │
│      children: ["bull_portfolio", "neutral_portfolio", "bear_portfolio"]│
│    - type: "selective"                                                  │
│      source: "portfolios"                                              │
│      rules:                                                             │
│        - condition: "regime == 'BULL'"                                 │
│          target: "aggressive_execution"                                 │
│        - condition: "regime == 'BEAR'"                                 │
│          target: "defensive_execution"                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Performance Tiers in Action

### Tier Optimization Based on Event Types

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PERFORMANCE TIER MAPPING                         │
│                                                                         │
│  Event Type           Tier        Target Latency    Adapter Config     │
│  ═══════════════════════════════════════════════════════════════════    │
│                                                                         │
│  BAR, TICK, QUOTE  →  FAST     →  < 1ms latency  →  Batch 1000 events  │
│  ┌─────────────┐                                                        │
│  │ Data Stream │ ──────── FAST TIER ────────▶ Strategies                │
│  └─────────────┘         (Batched,                                      │
│                          Zero-copy)                                     │
│                                                                         │
│  SIGNAL, INDICATOR →  STANDARD  →  < 10ms       →  Async queues        │
│  ┌─────────────┐                                                        │
│  │ Strategies  │ ───── STANDARD TIER ─────────▶ Risk Management         │
│  └─────────────┘       (Async,                                          │
│                        Intelligent batching)                            │
│                                                                         │
│  ORDER, FILL      →  RELIABLE  →  < 100ms      →  Retries + persistence │
│  ┌─────────────┐                                                        │
│  │ Risk Mgmt   │ ───── RELIABLE TIER ─────────▶ Execution               │
│  └─────────────┘       (Guaranteed delivery,                            │
│                        Dead letter queue)                               │
│                                                                         │
│  Adapter Configuration:                                                  │
│  adapters:                                                              │
│    - type: "broadcast"                                                  │
│      source: "data_stream"                                              │
│      targets: ["strategy_001", "strategy_002", ...]                     │
│      tier: "fast"          # Auto-batching for speed                    │
│                                                                         │
│    - type: "pipeline"                                                   │
│      containers: ["strategies", "risk_management"]                      │
│      tier: "standard"      # Async processing                           │
│                                                                         │
│    - type: "pipeline"                                                   │
│      containers: ["risk_management", "execution"]                       │
│      tier: "reliable"      # Guaranteed delivery                        │
└─────────────────────────────────────────────────────────────────────────┘
```

## Integration with Your Logging System

### Adapters Align with Event Scope Classification

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     LOGGING INTEGRATION                                 │
│                                                                         │
│  Your Logging Event Scopes      ←→      Adapter Tiers                  │
│  ═══════════════════════════════════════════════════════════════════    │
│                                                                         │
│  internal_bus                    ←→      (Adapters don't touch)         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Container Internal Communication                                │   │
│  │ Strategy ←→ Portfolio ←→ Risk (within same container)           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  external_fast_tier              ←→      tier: "fast"                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Pipeline/Broadcast Adapters for Market Data                    │   │
│  │ Data → Indicators → Strategies (high frequency)                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  external_standard_tier          ←→      tier: "standard"               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Pipeline/Hierarchical Adapters for Business Logic              │   │
│  │ Strategies → Risk → Portfolio Management                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  external_reliable_tier          ←→      tier: "reliable"               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Selective/Pipeline Adapters for Critical Events                │   │
│  │ Risk → Execution → Order Management                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  lifecycle_management            ←→      Adapter setup/teardown         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Adapter Creation, Configuration, Health Monitoring             │   │
│  │ Automatic cleanup with container lifecycle                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Benefits:                                                              │
│  • Every adapter event logged with correlation IDs                     │
│  • Performance metrics by tier automatically tracked                   │
│  • Same log retention and archiving policies                           │
│  • Dashboard integration for adapter health monitoring                 │
└─────────────────────────────────────────────────────────────────────────┘
```

## Semantic Event Enhancement

### Problem: Generic Events vs Domain-Specific Events

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    BEFORE: GENERIC EVENT OBJECTS                        │
│                                                                         │
│  class Event:                                                           │
│      type: str          # "SIGNAL", "ORDER", etc.                     │
│      data: Dict         # Arbitrary data blob                          │
│      timestamp: datetime                                                │
│                                                                         │
│  ❌ PROBLEMS:                                                           │
│  • No type safety - runtime errors                                     │
│  • No validation - bad data propagates                                 │
│  • No correlation tracking                                              │
│  • No schema evolution                                                  │
│  • Hard to debug event flows                                           │
│  • Cannot track causation                                               │
│                                                                         │
│  Example Generic Event:                                                 │
│  event = Event(                                                         │
│      type="SIGNAL",                                                     │
│      data={"symbol": "AAPL", "action": "BUY", "strength": 0.8}       │
│  )                                                                      │
│  # No IDE support, no validation, no traceability                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### Solution: Strongly-Typed Semantic Events

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   AFTER: SEMANTIC EVENT SYSTEM                          │
│                                                                         │
│  @dataclass                                                             │
│  class TradingSignal(SemanticEventBase):                               │
│      schema_version: str = "2.0.0"                                     │
│      symbol: str = ""                                                   │
│      action: Literal["BUY", "SELL", "HOLD"] = "HOLD"                   │
│      strength: float = 0.0  # 0.0 to 1.0                              │
│      regime_context: Optional[str] = None                              │
│      risk_score: float = 0.5  # Added in v2.0.0                       │
│                                                                         │
│      # Inherited from SemanticEventBase:                               │
│      event_id: str                    # Unique identifier              │
│      correlation_id: str              # Group related events           │
│      causation_id: Optional[str]      # Parent event that caused this  │
│      source_container: str            # Which container created it     │
│                                                                         │
│  ✅ BENEFITS:                                                          │
│  • Type safety - IDE autocomplete and validation                       │
│  • Built-in validation with validate() method                          │
│  • Complete event lineage tracking                                     │
│  • Schema versioning and migration                                     │
│  • Domain-specific event types                                         │
│  • Production debugging capabilities                                    │
│                                                                         │
│  Example Semantic Event:                                                │
│  signal = TradingSignal(                                                │
│      symbol="AAPL",                                                     │
│      action="BUY",                                                      │
│      strength=0.85,                                                     │
│      regime_context="BULL",                                            │
│      causation_id=indicator_event.event_id                             │
│  )                                                                      │
│  # Full type safety, validation, traceability!                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Semantic Event Flow with Type Transformations

### Visual Event Transformation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SEMANTIC EVENT TRANSFORMATION FLOW                   │
│                                                                         │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────┐ │
│  │   Market    │    │   Indicator  │    │   Trading   │    │  Order  │ │
│  │ Data Event  │───▶│    Event     │───▶│   Signal    │───▶│  Event  │ │
│  │             │    │              │    │             │    │         │ │
│  │ symbol: AAPL│    │ name: "RSI"  │    │ symbol: AAPL│    │qty: 100 │ │
│  │ price: 150.0│    │ value: 0.7   │    │ action: BUY │    │side: BUY│ │
│  │ volume: 1000│    │ conf: 0.85   │    │ strength:0.7│    │type: MKT│ │
│  │             │    │              │    │ risk: 0.3   │    │         │ │
│  └─────────────┘    └──────────────┘    └─────────────┘    └─────────┘ │
│         │                   │                   │               │       │
│         │ TRANSFORM         │ TRANSFORM         │ TRANSFORM     │       │
│         ▼                   ▼                   ▼               ▼       │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────┐ │
│  │   Raw Bar   │    │ Momentum     │    │ Risk-Adj    │    │ Sized   │ │
│  │   Data      │    │ Indicator    │    │ Signal      │    │ Order   │ │
│  └─────────────┘    └──────────────┘    └─────────────┘    └─────────┘ │
│                                                                         │
│  Event Correlation Tracking:                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ correlation_id: "trade_flow_abc123"                             │   │
│  │                                                                 │   │
│  │ MarketDataEvent(event_id="md_001")                             │   │
│  │     ↓ caused                                                    │   │
│  │ IndicatorEvent(event_id="ind_002", causation_id="md_001")      │   │
│  │     ↓ caused                                                    │   │
│  │ TradingSignal(event_id="sig_003", causation_id="ind_002")      │   │
│  │     ↓ caused                                                    │   │
│  │ OrderEvent(event_id="ord_004", causation_id="sig_003")         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Type-Safe Transformations:                                             │
│  def indicator_to_signal(indicator: IndicatorEvent) -> TradingSignal:   │
│      return TradingSignal(                                              │
│          symbol=indicator.metadata["symbol"],                          │
│          action="BUY" if indicator.value > 0 else "SELL",              │
│          strength=abs(indicator.value),                                 │
│          causation_id=indicator.event_id,  # Preserve lineage          │
│          correlation_id=indicator.correlation_id                        │
│      )                                                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Schema Evolution in Action

### Handling Breaking Changes Without Breaking the System

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SCHEMA EVOLUTION EXAMPLE                         │
│                                                                         │
│  OLD SYSTEM (v1.0.0)          NEW SYSTEM (v2.0.0)                     │
│  ┌─────────────────────┐      ┌─────────────────────────────────────┐   │
│  │ TradingSignal v1.0  │      │ TradingSignal v2.0                  │   │
│  │ ─────────────────── │      │ ─────────────────────────────────── │   │
│  │ symbol: str         │      │ symbol: str                         │   │
│  │ action: str         │      │ action: Literal["BUY","SELL","HOLD"] │   │
│  │ strength: float     │ ───▶ │ strength: float                     │   │
│  │                     │      │ regime_context: Optional[str]      │   │
│  │                     │      │ risk_score: float = 0.5  # NEW!    │   │
│  └─────────────────────┘      └─────────────────────────────────────┘   │
│            │                                    ▲                       │
│            │                                    │                       │
│            └────── MIGRATION FUNCTION ──────────┘                       │
│                                                                         │
│  def migrate_trading_signal_v1_to_v2(v1_signal):                       │
│      return TradingSignal(                                              │
│          **{k: v for k, v in v1_signal.__dict__.items()                │
│             if k != 'schema_version'},                                  │
│          schema_version="2.0.0",                                       │
│          risk_score=0.5  # Default for new field                       │
│      )                                                                  │
│                                                                         │
│  Real-World Scenario:                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 1. Strategy A (old) sends TradingSignal v1.0                   │   │
│  │ 2. Adapter detects version mismatch                            │   │
│  │ 3. Auto-migration transforms v1.0 → v2.0                       │   │
│  │ 4. Risk Manager (new) receives TradingSignal v2.0              │   │
│  │ 5. System continues without interruption                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Benefits:                                                              │
│  • Zero-downtime upgrades                                              │
│  • Backward compatibility                                               │
│  • Gradual rollout of new features                                     │
│  • Automatic handling of version mismatches                            │
└─────────────────────────────────────────────────────────────────────────┘
```

## Execution Model Selection

### Same Semantic Events, Different Execution Models

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    EXECUTION MODEL FLEXIBILITY                          │
│                                                                         │
│  RESEARCH PHASE              BACKTESTING PHASE           LIVE TRADING   │
│  (Actor Model)               (Container Model)          (Container Model)│
│  ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐ │
│  │ 5000 Ray Actors │        │ Full Container  │        │ Max Reliability │ │
│  │ Fast Parallel   │        │ Isolation       │        │ Fault Tolerance │ │
│  │ Execution       │        │                 │        │                 │ │
│  │                 │        │                 │        │                 │ │
│  │ ┌─────────────┐ │        │ ┌─────────────┐ │        │ ┌─────────────┐ │ │
│  │ │Actor 1      │ │        │ │Container 1  │ │        │ │Container 1  │ │ │
│  │ │─────────────│ │        │ │─────────────│ │        │ │─────────────│ │ │
│  │ │TradingSignal│ │        │ │TradingSignal│ │        │ │TradingSignal│ │ │
│  │ │(semantic)   │ │        │ │(semantic)   │ │        │ │(semantic)   │ │ │
│  │ └─────────────┘ │        │ └─────────────┘ │        │ └─────────────┘ │ │
│  │                 │        │                 │        │                 │ │
│  │ ┌─────────────┐ │        │ ┌─────────────┐ │        │ ┌─────────────┐ │ │
│  │ │Actor 2      │ │        │ │Container 2  │ │        │ │Container 2  │ │ │
│  │ │─────────────│ │        │ │─────────────│ │        │ │─────────────│ │ │
│  │ │TradingSignal│ │        │ │TradingSignal│ │        │ │TradingSignal│ │ │
│  │ │(semantic)   │ │        │ │(semantic)   │ │        │ │(semantic)   │ │ │
│  │ └─────────────┘ │        │ └─────────────┘ │        │ └─────────────┘ │ │
│  │                 │        │                 │        │                 │ │
│  │    ... 5000     │        │ Full Process    │        │ Restart Policies│ │
│  └─────────────────┘        │ Boundaries      │        │ Health Checks   │ │
│                              └─────────────────┘        └─────────────────┘ │
│                                                                         │
│  SAME SEMANTIC EVENTS - DIFFERENT EXECUTION OPTIMIZATIONS               │
│                                                                         │
│  Configuration:                                                         │
│  workflow_phases:                                                       │
│    parameter_discovery:                                                 │
│      execution_model: "actors"     # 5000 parallel evaluations        │
│      adapters:                                                          │
│        - type: "semantic_broadcast"                                     │
│          semantic_events: true                                          │
│          event_types: ["TradingSignal", "IndicatorEvent"]              │
│                                                                         │
│    backtesting:                                                         │
│      execution_model: "containers" # Full isolation                    │
│      adapters:                                                          │
│        - type: "semantic_pipeline"                                      │
│          transformations:                                               │
│            - from: "IndicatorEvent"                                     │
│              to: "TradingSignal"                                        │
│                                                                         │
│    live_trading:                                                        │
│      execution_model: "containers" # Maximum reliability               │
│      adapters:                                                          │
│        - type: "semantic_pipeline"                                      │
│          tier: "reliable"  # Guaranteed delivery                        │
└─────────────────────────────────────────────────────────────────────────┘
```

## Workspace-Aware Multi-Phase Communication

### File-Based Communication Between Workflow Phases

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    WORKSPACE-AWARE COMMUNICATION                        │
│                                                                         │
│  PHASE 1: PARAMETER DISCOVERY                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 5000 Actor Containers                                           │   │
│  │ ┌──────────┐ ┌──────────┐ ┌──────────┐      ┌──────────┐       │   │
│  │ │ Actor 1  │ │ Actor 2  │ │ Actor 3  │ .... │Actor 5000│       │   │
│  │ │          │ │          │ │          │      │          │       │   │
│  │ │ Results  │ │ Results  │ │ Results  │      │ Results  │       │   │
│  │ └────┬─────┘ └────┬─────┘ └────┬─────┘      └────┬─────┘       │   │
│  │      │            │            │                 │             │   │
│  └──────┼────────────┼────────────┼─────────────────┼─────────────┘   │
│         │            │            │                 │                 │
│         ▼            ▼            ▼                 ▼                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              WORKSPACE FILE SYSTEM                             │   │
│  │                                                                 │   │
│  │ /workspace/phase1_output/                                       │   │
│  │ ├── trial_001_results.jsonl                                     │   │
│  │ ├── trial_002_results.jsonl                                     │   │
│  │ ├── trial_003_results.jsonl                                     │   │
│  │ └── ... 5000 trial files                                        │   │
│  │                                                                 │   │
│  │ Each .jsonl file contains semantic events:                      │   │
│  │ {"event_id": "res_001", "type": "ParameterResult",             │   │
│  │  "correlation_id": "param_sweep_abc",                          │   │
│  │  "parameters": {"fast_ma": 10, "slow_ma": 30},                 │   │
│  │  "performance": {"sharpe": 1.85, "drawdown": 0.12}}            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                         │
│                              ▼                                         │
│  PHASE 2: SIGNAL REPLAY                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 1000 Function Containers                                        │   │
│  │                                                                 │   │
│  │ WorkspaceAwareAdapter reads /workspace/phase1_output/*.jsonl    │   │
│  │ ┌─────────────────────────────────────────────────────────────┐ │   │
│  │ │ for trial_file in workspace.glob("phase1_output/*.jsonl"): │ │   │
│  │ │     events = adapter.deserialize_semantic_events(trial_file)│ │   │
│  │ │     for event in events:                                    │ │   │
│  │ │         if event.performance.sharpe > 1.5:                  │ │   │
│  │ │             replay_container.process_parameters(event)      │ │   │
│  │ └─────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                         │
│                              ▼                                         │
│  PHASE 3: FINAL VALIDATION                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 1 Container (Maximum Reliability)                               │   │
│  │                                                                 │   │
│  │ WorkspaceAwareAdapter reads:                                    │   │
│  │ - /workspace/phase1_output/ (parameter results)                 │   │
│  │ - /workspace/phase2_output/ (signal replay results)             │   │
│  │                                                                 │   │
│  │ Produces: /workspace/final_output/validated_strategy.yaml       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Configuration:                                                         │
│  adapters:                                                              │
│    - type: "workspace_aware"                                            │
│      workspace_integration: true                                        │
│      phase_inputs:                                                      │
│        signal_replay: "phase1_output/trial_*.jsonl"                     │
│        validation: "phase2_output/ensemble_weights.json"                │
│      semantic_events: true                                              │
│      schema_evolution: true                                             │
└─────────────────────────────────────────────────────────────────────────┘
```

## Summary: Why Adapters Solve Everything

### Before Adapters: Tightly Coupled Nightmare

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           BEFORE ADAPTERS                               │
│                                                                         │
│  ❌ Organizational pattern determines communication code                │
│  ❌ Cannot test different communication patterns                        │
│  ❌ Code changes needed to support new organizational patterns          │
│  ❌ No logging integration                                              │
│  ❌ Cannot optimize performance for different event types               │
│  ❌ Cannot easily debug communication issues                            │
│  ❌ Cannot deploy in distributed environments                           │
│  ❌ A/B testing communication patterns requires code changes            │
│  ❌ Generic events with no type safety                                  │
│  ❌ No event lineage tracking                                           │
│  ❌ Schema changes break everything                                      │
│  ❌ No correlation between workflow phases                              │
└─────────────────────────────────────────────────────────────────────────┘
```

### After Semantic Adapters: The Complete Solution

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      AFTER SEMANTIC ADAPTERS                            │
│                                                                         │
│  ✅ Any organizational pattern works with any communication pattern    │
│  ✅ A/B testing via YAML configuration changes                         │
│  ✅ Performance tiers automatically optimize for event types           │
│  ✅ Full integration with sophisticated logging system                 │
│  ✅ Easy debugging with correlation IDs and adapter metrics            │
│  ✅ Future-proof for distributed deployment                            │
│  ✅ Same code works locally and in cloud environments                  │
│  ✅ Operational excellence: health monitoring, cleanup, alerting       │
│                                                                         │
│  NEW SEMANTIC EVENT BENEFITS:                                          │
│  ✅ Type-safe events with full IDE support                             │
│  ✅ Complete event lineage tracking (correlation + causation)          │
│  ✅ Schema evolution with zero-downtime upgrades                       │
│  ✅ Domain-specific event validation                                    │
│  ✅ Multi-phase workflow coordination                                   │
│  ✅ Execution model flexibility per phase                              │
│  ✅ Workspace-aware file-based communication                           │
│                                                                         │
│  Configuration Examples:                                                │
│                                                                         │
│  # Semantic Pipeline with Type Transformations                         │
│  adapters:                                                              │
│    - type: "semantic_pipeline"                                          │
│      containers: ["indicators", "strategies", "risk"]                  │
│      transformations:                                                   │
│        - from: "IndicatorEvent"                                         │
│          to: "TradingSignal"                                            │
│          transformer: "indicator_to_signal_transform"                   │
│        - from: "TradingSignal"                                          │
│          to: "OrderEvent"                                               │
│          transformer: "signal_to_order_transform"                       │
│      semantic_events: true                                              │
│      correlation_tracking: true                                         │
│                                                                         │
│  # Schema Evolution Support                                             │
│  semantic_schemas:                                                      │
│    evolution_policy: "forward_compatible"                              │
│    event_types:                                                         │
│      TradingSignal:                                                     │
│        current_version: "2.0.0"                                        │
│        supported_versions: ["1.0.0", "1.1.0", "2.0.0"]                │
│        migrations:                                                      │
│          "1.0.0->2.0.0": "migrate_trading_signal_v1_to_v2"            │
│                                                                         │
│  # Multi-Phase Execution Models                                        │
│  workflow_phases:                                                       │
│    parameter_discovery:                                                 │
│      execution_model: "actors"     # 5000 parallel                     │
│      adapters:                                                          │
│        - type: "semantic_broadcast"                                     │
│          semantic_events: true                                          │
│    signal_replay:                                                       │
│      execution_model: "functions"  # Lightweight                       │
│      adapters:                                                          │
│        - type: "workspace_aware"                                        │
│          phase_inputs: "phase1_output/*.jsonl"                          │
│    live_trading:                                                        │
│      execution_model: "containers" # Maximum reliability               │
│      adapters:                                                          │
│        - type: "semantic_pipeline"                                      │
│          tier: "reliable"                                               │
│                                                                         │
│  # Complete Event Lineage Example                                      │
│  correlation_id: "trade_flow_abc123"                                    │
│  MarketDataEvent(id="md_001")                                          │
│      ↓ caused                                                           │
│  IndicatorEvent(id="ind_002", causation_id="md_001")                   │
│      ↓ caused                                                           │
│  TradingSignal(id="sig_003", causation_id="ind_002")                   │
│      ↓ caused                                                           │
│  OrderEvent(id="ord_004", causation_id="sig_003")                      │
│      ↓ caused                                                           │
│  FillEvent(id="fill_005", causation_id="ord_004")                      │
│                                                                         │
│  # Full traceability for compliance and debugging!                     │
└─────────────────────────────────────────────────────────────────────────┘
```

## The Complete Picture: Semantic Event-Driven Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│              SEMANTIC EVENT-DRIVEN TRADING ARCHITECTURE                 │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    ORGANIZATIONAL LAYER                         │   │
│  │  Strategy-First │ Classifier-First │ Risk-First │ Portfolio     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                 │                                       │
│                                 ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                 SEMANTIC COMMUNICATION LAYER                    │   │
│  │                                                                 │   │
│  │  • Type-safe semantic events (TradingSignal, OrderEvent, etc.) │   │
│  │  • Schema evolution with migration                             │   │
│  │  • Complete event lineage tracking                             │   │
│  │  • Pluggable adapter patterns                                  │   │
│  │  • Multi-phase workspace coordination                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                 │                                       │
│                                 ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                 EXECUTION MODEL SELECTION                       │   │
│  │                                                                 │   │
│  │  Research: 5000 Ray Actors    │  Backtest: Containers          │   │
│  │  Signal Replay: Functions     │  Live: Reliable Containers     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                 │                                       │
│                                 ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    ISOLATED EVENT BUSES                         │   │
│  │        (Maintains all isolation and performance benefits)       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

**The Bottom Line**: Semantic adapters solve the fundamental problem of **organizational flexibility vs communication requirements** by making communication patterns completely independent of how you organize your containers, while adding type safety, event lineage tracking, schema evolution, and multi-phase workflow coordination - creating a truly production-ready, institutional-grade trading system architecture.