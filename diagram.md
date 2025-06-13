# Multi-Phase Workflow Architecture for ADMF-PC

## Idealized Architecture: Infinite Compute Single-Pass

*This shows the complete end-to-end architecture if we had unlimited cores and could run everything in parallel without resource constraints.*

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MAIN PRODUCER PROCESS                                │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────────────────┐ │
│  │ Data        │  │ Shared       │  │ Feature Cache & Distribution        │ │
│  │ Streamers   │→ │ Feature      │→ │ • OHLCV, Volume, Technical Analysis │ │
│  │             │  │ Computer     │  │ • Market Microstructure             │ │
│  └─────────────┘  └──────────────┘  │ • Cross-Asset Correlations          │ │
│                                     └─────────────────────────────────────┘ │
│                           ↓                          ↓                      │
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
│ │Inference    │ │ │ │Inference    │ │ │ │Rule-based   │ │ │ │Inference    │ │
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
│PORTFOLIO STATE  │ │PORTFOLIO STATE  │ │PORTFOLIO STATE  │ │PORTFOLIO STATE  │
│  WORKER 1       │ │  WORKER 2       │ │  WORKER 3       │ │  WORKER N       │
│                 │ │                 │ │                 │ │                 │
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │
│ │Portfolio    │ │ │ │Portfolio    │ │ │ │Portfolio    │ │ │ │Portfolio    │ │
│ │State for    │ │ │ │State for    │ │ │ │State for    │ │ │ │State for    │ │
│ │VaR Risk     │ │ │ │ML Risk      │ │ │ │Correlation  │ │ │ │Custom Risk  │ │
│ │Models       │ │ │ │Models       │ │ │ │Risk Models  │ │ │ │Models       │ │
│ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │
│ │State        │ │ │ │State        │ │ │ │State        │ │ │ │State        │ │
│ │Engineering  │ │ │ │Engineering  │ │ │ │Engineering  │ │ │ │Engineering  │ │
│ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │
└─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘
        ↓                    ↓                    ↓                    ↓
        └────────────────────┼────────────────────┼────────────────────┘
                             ↓                    ↓
                    ┌─────────────────┐  ┌─────────────────┐
                    │ IPC Queue       │  │ Portfolio State │
                    │ (Portfolio      │→ │ Router &        │
                    │  States)        │  │ Distributor     │
                    └─────────────────┘  └─────────────────┘
                                                  ↓
                                         ┌─────────────────┐
                                         │ IPC Queue       │
                                         │ (Routed States) │
                                         └─────────────────┘
                                                  ↓
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
│ │Final        │ │ │ │Final        │ │ │ │Final        │ │ │ │Final        │ │
│ │Portfolio    │ │ │ │Portfolio    │ │ │ │Portfolio    │ │ │ │Portfolio    │ │
│ │Decision &   │ │ │ │Decision &   │ │ │ │Decision &   │ │ │ │Decision &   │ │
│ │Order Gen    │ │ │ │Order Gen    │ │ │ │Order Gen    │ │ │ │Order Gen    │ │
│ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │
└─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘
        ↓                    ↓                    ↓                    ↓
        └────────────────────┼────────────────────┼────────────────────┘
                             ↓                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EXECUTION PROCESS                                      │
│  ┌─────────────────┐                    ┌─────────────────┐                 │
│  │ IPC Queue       │ ←─ Receives ORDERs │ IPC Queue       │                 │
│  │ (Orders)        │                    │ (Fills)         │ ←─ Publishes    │
│  └─────────────────┘                    └─────────────────┘    FILLs        │
│           ↓                                      ↑                          │
│  ┌─────────────────┐                             │                          │
│  │ Execution       │─────────────────────────────┘                          │
│  │ Engine          │                                                        │
│  └─────────────────┘                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

*In practice, this architecture requires unlimited compute resources. The multi-phase approach below solves the resource constraint problem while maintaining identical functionality.*

---

## Real-World Multi-Phase Implementation

## Phase 1: Shared Feature Computation & Stateless Strategies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 1: FEATURES & STATELESS STRATEGIES             │
│                                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────────────────┐ │
│  │ Data        │  │ Shared       │  │ Feature Cache & Distribution        │ │
│  │ Streamers   │→ │ Feature      │→ │ • OHLCV, Volume, Technical Analysi s│ │
│  │             │  │ Computer     │  │ • Market Microstructure             │ │
│  └─────────────┘  └──────────────┘  │ • Cross-Asset Correlations          │ │
│                                      └────────────────────────────────────┘ │
│                                                 ↓                           │
│        ┌────────────────────┼────────────────────┼────────────────────┐     │
│        ↓                    ↓                    ↓                    ↓     │
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────┐ │
│ │STATELESS GROUP 1│ │STATELESS GROUP 2│ │STATELESS GROUP 3│ │STATELESS    │ │
│ │• MA Crossover   │ │• RSI Momentum   │ │• Bollinger      │ │GROUP N      │ │
│ │• Simple Momentum│ │• Volume Signals │ │• Channel Break  │ │             │ │
│ │• Price Patterns │ │• Trend Following│ │• Mean Reversion │ │             │ │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────┘ │
│                                      ↓                                      │
│                            ┌─────────────────┐                              │
│                            │ CACHE SIGNALS   │                              │
│                            │ stateless_      │                              │
│                            │ signals.parquet │                              │
│                            └─────────────────┘                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Phase 2: ML Strategy Processing

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 2: ML STRATEGY PROCESSING                      │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │ Load Cached     │  │ ML Feature      │  │ Advanced Feature Cache      │  │
│  │ Basic Features  │→ │ Engineering     │→ │ • Correlation Matrices      │  │
│  │                 │  │                 │  │ • Regime Detection          │  │
│  └─────────────────┘  └─────────────────┘  │ • Cross-Asset Features      │  │
│                                             └────────────────────────────┘  │
│                                                         ↓                   │
│        ┌────────────────────┼────────────────────┼────────────────────┐     │
│        ↓                    ↓                    ↓                    ↓     │
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────┐ │
│ │ ML STRATEGY 1   │ │ ML STRATEGY 2   │ │ ML STRATEGY 3   │ │ ML STRATEGY │ │
│ │ LSTM Predictor  │ │ XGBoost Ensemble│ │ Transformer     │ │ N           │ │
│ │ (Sequential)    │ │ (Sequential)    │ │ (Sequential)    │ │             │ │
│ │                 │ │                 │ │                 │ │             │ │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────┘ │
│                                      ↓                                      │
│                            ┌─────────────────┐                              │
│                            │ CACHE SIGNALS   │                              │
│                            │ ml_strategy_    │                              │
│                            │ signals.parquet │                              │
│                            └─────────────────┘                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Phase 3: Risk Processing & Final Portfolio Decisions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 3: RISK PROCESSING & FINAL DECISIONS               │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │ Load Cached     │  │ Load Cached     │  │ Multiple Portfolio States   │  │
│  │ Stateless       │  │ ML Strategy     │  │ Each Risk Worker Maintains  │  │
│  │ Signals         │  │ Signals         │  │ Its Own Portfolio State     │  │
│  └─────────────────┘  └─────────────────┘  │ Tailored to Risk Model      │  │
│                                             └────────────────────────────┘  │
│                                                         ↓                   │
│        ┌────────────────────┼────────────────────┼────────────────────┐     │
│        ↓                    ↓                    ↓                    ↓     │
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────┐ │
│ │PORTFOLIO STATE  │ │PORTFOLIO STATE  │ │PORTFOLIO STATE  │ │PORTFOLIO    │ │
│ │& RISK WORKER 1  │ │& RISK WORKER 2  │ │& RISK WORKER 3  │ │STATE &      │ │
│ │                 │ │                 │ │                 │ │RISK WORKER N│ │
│ │ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────┐ │ │
│ │ │Portfolio    │ │ │ │Portfolio    │ │ │ │Portfolio    │ │ │ │Portfolio│ │ │
│ │ │State for    │ │ │ │State for    │ │ │ │State for    │ │ │ │State for│ │ │
│ │ │VaR Risk     │ │ │ │ML Risk      │ │ │ │Correlation  │ │ │ │Custom   │ │ │
│ │ │Models       │ │ │ │Models       │ │ │ │Risk Models  │ │ │ │Risk     │ │ │
│ │ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │ │ │Models   │ │ │
│ │ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │ │ └─────────┘ │ │
│ │ │VaR Risk     │ │ │ │ML Portfolio │ │ │ │Correlation  │ │ │ ┌─────────┐ │ │
│ │ │Assessment   │ │ │ │Optimization │ │ │ │Risk         │ │ │ │Custom ML│ │ │
│ │ │& Orders     │ │ │ │& Orders     │ │ │ │Assessment   │ │ │ │Risk &   │ │ │
│ │ └─────────────┘ │ │ └─────────────┘ │ │ │& Orders     │ │ │ │Orders   │ │ │
│ │ ┌─────────────┐ │ │ ┌─────────────┐ │ │ └─────────────┘ │ │ └─────────┘ │ │
│ │ │Fill         │ │ │ │Fill         │ │ │ ┌─────────────┐ │ │ ┌─────────┐ │ │
│ │ │Processing   │ │ │ │Processing   │ │ │ │Fill         │ │ │ │Fill     │ │ │
│ │ │& State      │ │ │ │& State      │ │ │ │Processing   │ │ │ │Proc &   │ │ │
│ │ │Updates      │ │ │ │Updates      │ │ │ │& State      │ │ │ │State    │ │ │
│ │ └─────────────┘ │ │ └─────────────┘ │ │ │Updates      │ │ │ │Updates  │ │ │
│ └─────────────────┘ └─────────────────┘ │ └─────────────┘ │ │ └─────────┘ │ │
│          ↓                    ↓         └─────────────────┘ └─────────────┘ │
│          ↓                    ↓                    ↓                ↓       │
│    [Orders to Execution]               [Fills from Execution]               │
│    Each worker maintains                Each worker updates its             │
│    distinct portfolio state            own portfolio state                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Phase 4: Execution Processing

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 4: EXECUTION PROCESSING                            │
│                                                                             │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │                      EXECUTION PROCESS                                  │ │
│ │  ┌─────────────────┐                    ┌─────────────────┐             │ │
│ │  │ Order           │ ←─ Receives ORDERs │ Fill            │             │ │
│ │  │ Aggregation     │    from Risk       │ Distribution    │───┐         │ │
│ │  │                 │    Workers         │                 │   │         │ │
│ │  └─────────────────┘                    └─────────────────┘   │         │ │
│ │           ↓                                      ↑            │         │ │
│ │  ┌─────────────────┐                             │            │         │ │
│ │  │ Execution       │─────────────────────────────┘            │         │ │
│ │  │ Engine          │                                          │         │ │
│ │  └─────────────────┘                                          │         │ │
│ └─────────────────────────────────────────────────────────────┼────────┘  │
│                                                               │           │
│                    Fills routed back to respective Risk Workers           │
│                    to update their Portfolio State for next bar           │
│                                                               ↓           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Multi-Phase Workflow Benefits

### **Phase Isolation & Optimization**
- **Phase 1**: Fast stateless strategies complete in seconds
- **Phase 2**: ML strategies get full system resources sequentially
- **Phase 3**: Risk models process optimized signal combinations
- **Phase 4**: Portfolio decisions made with complete information

### **Intelligent Caching Strategy**
- **Features**: Computed once, reused across all phases
- **Signals**: Strategy outputs cached for risk model analysis
- **Risk Assessments**: Cached for portfolio decision making
- **Grid Search**: Rerun phases with different parameters without recomputation

### **Configurable Orchestration**
```yaml
workflow_mode: "multi_phase_optimized"

phases:
  stateless_strategies:
    grouping: "by_feature_overlap"
    parallel_cores: 8
    
  ml_strategies:
    processing: "sequential_max_resources"
    parallel_cores: 8
    
  risk_and_portfolio_processing:
    portfolio_state_management: "within_risk_workers"
    risk_workers_handle: ["portfolio_state", "risk_assessment", "order_generation", "fill_processing"]
    parallel_cores: 8
    
  execution:
    order_aggregation_and_fill_distribution: true
    parallel_cores: 1
```

### **Scalability Through Phases**
- **Research**: Run individual phases for rapid iteration
- **Grid Search**: Vary risk parameters and portfolio configurations in Phase 3
- **Validation**: Full multi-phase run with integrated risk/portfolio processing
- **Production**: Single-pass mode for real-time deployment

### **Per-Bar Processing Cycle**
- **Phase 3**: Risk Workers receive signals → update portfolio state → assess risk → generate orders → send to execution
- **Phase 4**: Execution processes orders → sends fills back to Risk Workers → Risk Workers update portfolio state
- **Next Bar**: Cycle repeats with updated portfolio state
