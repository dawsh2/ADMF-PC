# ADMF-PC Three-Tier Container Structure Diagrams

## Simple Backtest Workflow
*Mostly functions with containers only where needed*

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           SIMPLE BACKTEST PIPELINE                            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────┐  │
│  │ TIER 3: CONTAINER   │    │ TIER 3: CONTAINER   │    │ TIER 3: CONTAINER │
│  │ Market Data Pipeline│    │ Indicator Hub       │    │ Execution Engine │  │
│  │                     │    │                     │    │                  │  │
│  │ ┌─────────────────┐ │    │ ┌─────────────────┐ │    │ ┌──────────────┐ │  │
│  │ │• Multi-symbol   │ │    │ │• SMA Calculator │ │    │ │• Order Queue │ │  │
│  │ │  coordination   │ │    │ │• MACD Engine    │ │    │ │• Fill Model  │ │  │
│  │ │• Data quality   │ │    │ │• RSI Engine     │ │    │ │• Trade Log   │ │  │
│  │ │  validation     │ │    │ │• Dependency     │ │    │ │• Slippage    │ │  │
│  │ │• Feed failover  │ │    │ │  resolution     │ │    │ │  modeling    │ │  │
│  │ │• Event bus      │ │    │ │• Event bus      │ │    │ │• Event bus   │ │  │
│  │ └─────────────────┘ │    │ └─────────────────┘ │    │ └──────────────┘ │  │
│  └─────────────────────┘    └─────────────────────┘    └─────────────────┘  │
│             │                           │                           ▲        │
│             │ Market Data               │ Indicators                │        │
│             ▼                           ▼                           │        │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────┐  │
│  │ TIER 1: FUNCTION    │    │ TIER 2: STATEFUL    │    │ TIER 1: FUNCTION │  │
│  │ momentum_strategy() │    │ PositionTracker     │    │ position_sizer() │  │
│  │                     │    │                     │    │                  │  │
│  │ def momentum(...):  │    │ class Tracker:      │    │ def calculate_   │  │
│  │   if sma20 > sma50: │    │   positions = {}    │    │   size(...):     │  │
│  │     return BUY      │    │   total_value = 0   │    │   return shares  │  │
│  │   return None       │    │                     │    │                  │  │
│  │                     │    │ • Tracks positions  │    │ • Pure function  │  │
│  │ • Pure function     │    │ • Maintains state   │    │ • No side effects│  │
│  │ • No side effects   │    │ • Fresh per run     │    │ • Testable       │  │
│  └─────────────────────┘    └─────────────────────┘    └─────────────────┘  │
│             │                           │                           │        │
│             └───────── Signal ─────────▶│◄──── Position ────────────┘        │
│                                         │                                    │
│                                         ▼ Order Event                        │
└──────────────────────────────────────────────────────────────────────────────┘

Resource Usage:
┌────────────────┬──────────┬─────────────┬─────────────┐
│ Component Type │ Count    │ Memory      │ Complexity  │
├────────────────┼──────────┼─────────────┼─────────────┤
│ Containers     │ 3        │ High        │ High        │
│ Functions      │ 2        │ Minimal     │ Low         │
│ Stateful       │ 1        │ Low         │ Medium      │
└────────────────┴──────────┴─────────────┴─────────────┘
```

## Multi-Strategy Portfolio Workflow
*Balanced mix of all three tiers*

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        MULTI-STRATEGY PORTFOLIO                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         DATA & INDICATORS LAYER                         │ │
│  │                                                                         │ │
│  │  ┌─────────────────────┐              ┌─────────────────────┐          │ │
│  │  │ TIER 3: CONTAINER   │              │ TIER 3: CONTAINER   │          │ │
│  │  │ MultiAsset Data     │──── Data ───▶│ Indicator Hub       │          │ │
│  │  │ • 100+ symbols      │              │ • Complex calcs     │          │ │
│  │  │ • Multiple feeds    │              │ • Dependencies      │          │ │
│  │  │ • Real-time sync    │              │ • Caching layer     │          │ │
│  │  └─────────────────────┘              └─────────────────────┘          │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                           │                                  │
│                                           │ Indicators                       │
│                                           ▼                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                       STRATEGY LAYER (BROADCAST)                        │ │
│  │                                                                         │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │ │
│  │  │ TIER 1: FUNCTION│  │ TIER 1: FUNCTION│  │ TIER 3: CONTAINER       │ │ │
│  │  │ momentum_strat()│  │ mean_rev_strat()│  │ ML Ensemble Strategy    │ │ │
│  │  │                 │  │                 │  │                         │ │ │
│  │  │ • Dual momentum │  │ • RSI reversal  │  │ ┌─────────────────────┐ │ │ │
│  │  │ • Pure function │  │ • Pure function │  │ │• Random Forest      │ │ │ │
│  │  │ • Fast execution│  │ • Stateless     │  │ │• Neural Network     │ │ │ │
│  │  └─────────────────┘  └─────────────────┘  │ │• Feature Pipeline   │ │ │ │
│  │           │                     │          │ │• Model Retraining   │ │ │ │
│  │           │                     │          │ │• Prediction Blend   │ │ │ │
│  │           │                     │          │ └─────────────────────┘ │ │ │
│  │           └──── Signals ────────┼──────────┴─────────────┐           │ │ │
│  └─────────────────────────────────┼────────────────────────┼───────────┘ │
│                                    │                        │             │
│                                    ▼                        ▼             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                      PORTFOLIO MANAGEMENT LAYER                        │ │
│  │                                                                         │ │
│  │  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐  │ │
│  │  │ TIER 1: FUNCTION│     │ TIER 2: STATEFUL│     │ TIER 2: STATEFUL│  │ │
│  │  │ regime_class()  │     │ Portfolio       │     │ Risk Monitor    │  │ │
│  │  │                 │     │ Allocator       │     │                 │  │ │
│  │  │ • Volatility    │     │                 │     │ • Drawdown      │  │ │
│  │  │   classification│     │ • Weight calc   │     │ • Correlation   │  │ │
│  │  │ • Pure function │     │ • Rebalancing   │     │ • Exposure      │  │ │
│  │  │ • Fast regime   │     │ • State tracking│     │ • Circuit break │  │ │
│  │  │   detection     │     │                 │     │                 │  │ │
│  │  └─────────────────┘     └─────────────────┘     └─────────────────┘  │ │
│  │           │                        │                        │         │ │
│  │           └──── Regime ───────────▶│◄──── Risk Metrics ─────┘         │ │
│  └─────────────────────────────────────┼─────────────────────────────────┘ │
│                                        │                                   │
│                                        ▼ Portfolio Orders                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        EXECUTION LAYER                                  │ │
│  │                                                                         │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │  │                    TIER 3: CONTAINER                            │   │ │
│  │  │                Multi-Strategy Executor                          │   │ │
│  │  │                                                                 │   │ │
│  │  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────────────┐ │   │ │
│  │  │  │ Order Router  │ │ Risk Checker  │ │ Performance Tracker   │ │   │ │
│  │  │  │ • Priority    │ │ • Limit check │ │ • Strategy attribution│ │   │ │
│  │  │  │ • Batching    │ │ • Position    │ │ • Risk metrics        │ │   │ │
│  │  │  │ • Timing      │ │   validation  │ │ • Reporting           │ │   │ │
│  │  │  └───────────────┘ └───────────────┘ └───────────────────────┘ │   │ │
│  │  └─────────────────────────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘

Resource Usage:
┌────────────────┬──────────┬─────────────┬─────────────┐
│ Component Type │ Count    │ Memory      │ Complexity  │
├────────────────┼──────────┼─────────────┼─────────────┤
│ Containers     │ 4        │ High        │ High        │
│ Functions      │ 3        │ Minimal     │ Low         │
│ Stateful       │ 3        │ Medium      │ Medium      │
└────────────────┴──────────┴─────────────┴─────────────┘
```

## Research/Optimization Workflow  
*Function-heavy for maximum parallelization*

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       RESEARCH/OPTIMIZATION WORKFLOW                          │
│                    (5000+ Parallel Parameter Combinations)                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                      SHARED DATA LAYER                                  │ │
│  │                                                                         │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │  │                   TIER 3: CONTAINER                             │   │ │
│  │  │                Optimized Data Loader                            │   │ │
│  │  │                                                                 │   │ │
│  │  │  • Preloaded historical data (2+ years)                        │   │ │
│  │  │  • Memory-mapped files for speed                               │   │ │
│  │  │  • Pre-computed indicators cached                              │   │ │
│  │  │  • Read-only optimization (no state changes)                   │   │ │
│  │  │  • Shared across all 5000 parallel runs                       │   │ │
│  │  └─────────────────────────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                       │
│                                     │ Cached Data                           │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    PARALLEL STRATEGY TESTING                            │ │
│  │                                                                         │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     ┌─────────────┐  │ │
│  │  │TIER 1: FUNC │ │TIER 1: FUNC │ │TIER 1: FUNC │ ... │TIER 1: FUNC │  │ │
│  │  │momentum_v1()│ │momentum_v2()│ │rsi_strat()  │     │breakout()   │  │ │
│  │  │             │ │             │ │             │     │             │  │ │
│  │  │fast=10      │ │fast=12      │ │oversold=30  │     │period=20    │  │ │
│  │  │slow=30      │ │slow=26      │ │overbought=70│     │volume=1.5   │  │ │
│  │  │             │ │             │ │             │     │             │  │ │
│  │  │• Pure func  │ │• Pure func  │ │• Pure func  │     │• Pure func  │  │ │
│  │  │• No state   │ │• No state   │ │• No state   │     │• No state   │  │ │
│  │  │• Fast exec  │ │• Fast exec  │ │• Fast exec  │     │• Fast exec  │  │ │
│  │  └─────────────┘ └─────────────┘ └─────────────┘     └─────────────┘  │ │
│  │        │              │              │                     │         │ │
│  │        └──────────────┼──────────────┼─────────────────────┘         │ │
│  │                       │              │                               │ │
│  │                       ▼              ▼                               │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     ┌─────────────┐ │ │
│  │  │TIER 1: FUNC │ │TIER 1: FUNC │ │TIER 2: STATE│     │TIER 1: FUNC │ │ │
│  │  │vol_pos_siz()│ │risk_check() │ │Perf Tracker │     │result_agg() │ │ │
│  │  │             │ │             │ │             │     │             │ │ │
│  │  │• Vol-based  │ │• Limit check│ │• Win/Loss   │     │• Serialize  │ │ │
│  │  │  position   │ │• Pure func  │ │• Drawdown   │     │  results    │ │ │
│  │  │  sizing     │ │• Fast       │ │• Sharpe     │     │• To files   │ │ │
│  │  │• Pure func  │ │             │ │• Fresh inst │     │• Pure func  │ │ │
│  │  └─────────────┘ └─────────────┘ └─────────────┘     └─────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                       │
│                                     ▼ Results                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                      RESULTS AGGREGATION                                │ │
│  │                                                                         │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │  │                   TIER 3: CONTAINER                             │   │ │
│  │  │              Optimization Results Handler                       │   │ │
│  │  │                                                                 │   │ │
│  │  │  • Collect 5000+ result files                                  │   │ │
│  │  │  • Statistical analysis (best params by regime)               │   │ │
│  │  │  • Generate optimization reports                               │   │ │
│  │  │  • Prepare data for next phase (signal replay)                │   │ │
│  │  │  • Complex aggregation logic requiring event coordination     │   │ │
│  │  └─────────────────────────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘

Parallel Execution Model:
┌──────────────────────────────────────────────────────────────┐
│                    5000 PARALLEL RUNS                       │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Run 1: [Data] → [momentum_v1] → [vol_sizer] → [tracker]    │
│  Run 2: [Data] → [momentum_v2] → [vol_sizer] → [tracker]    │
│  Run 3: [Data] → [rsi_strat] → [vol_sizer] → [tracker]      │
│  ...                                                         │
│  Run 5000: [Data] → [breakout] → [vol_sizer] → [tracker]    │
│                                                              │
│  Each run:                                                   │
│  • Fresh function instances (zero state contamination)      │
│  • Shared read-only data (memory efficient)                 │
│  • Independent execution (true parallelization)             │
│  • Minimal memory footprint per run                         │
└──────────────────────────────────────────────────────────────┘

Resource Usage:
┌────────────────┬──────────┬─────────────┬─────────────┐
│ Component Type │ Count    │ Memory      │ Complexity  │
├────────────────┼──────────┼─────────────┼─────────────┤
│ Containers     │ 2        │ High        │ High        │
│ Functions      │ 20,000+  │ Minimal     │ Low         │
│ Stateful       │ 5,000    │ Low         │ Medium      │
└────────────────┴──────────┴─────────────┴─────────────┘
Total Memory: ~80% reduction vs all-container approach
```

## Live Trading Workflow
*Maximum reliability with full containers for critical components*

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            LIVE TRADING SYSTEM                                │
│                        (Production Reliability Focus)                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                      REAL-TIME DATA LAYER                               │ │
│  │                                                                         │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │  │                   TIER 3: CONTAINER                             │   │ │
│  │  │              Production Data Pipeline                           │   │ │
│  │  │                                                                 │   │ │
│  │  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │   │ │
│  │  │  │ Feed Manager    │ │ Quality Control │ │ Circuit Breaker │ │   │ │
│  │  │  │ • Primary feed  │ │ • Stale data    │ │ • Market halt   │ │   │ │
│  │  │  │ • Backup feeds  │ │ • Price spikes  │ │ • Feed failure  │ │   │ │
│  │  │  │ • Auto failover │ │ • Volume anomaly│ │ • Auto shutdown │ │   │ │
│  │  │  └─────────────────┘ └─────────────────┘ └─────────────────┘ │   │ │
│  │  │                                                                 │   │ │
│  │  │  • Event bus for feed coordination                             │   │ │
│  │  │  • Health monitoring and alerting                              │   │ │
│  │  │  • Guaranteed delivery to downstream                           │   │ │
│  │  └─────────────────────────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                       │
│                                     │ Validated Market Data                 │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                      STRATEGY EXECUTION LAYER                           │ │
│  │                                                                         │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │  │                   TIER 3: CONTAINER                             │   │ │
│  │  │                Live Strategy Coordinator                        │   │ │
│  │  │                                                                 │   │ │
│  │  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │   │ │
│  │  │  │ Regime Monitor  │ │ Signal Router   │ │ Strategy Pool   │ │   │ │
│  │  │  │ • Live regime   │ │ • Route to      │ │ • Active strats │ │   │ │
│  │  │  │   detection     │ │   active strats │ │ • Health check  │ │   │ │
│  │  │  │ • Param switch  │ │ • Event batching│ │ • Performance   │ │   │ │
│  │  │  └─────────────────┘ └─────────────────┘ └─────────────────┘ │   │ │
│  │  │                                                                 │   │ │
│  │  │  Functions hosted within container:                            │   │ │
│  │  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │   │ │
│  │  │  │ TIER 1: FUNC    │ │ TIER 1: FUNC    │ │ TIER 1: FUNC    │ │   │ │
│  │  │  │ momentum_live() │ │ mean_rev_live() │ │ regime_detect() │ │   │ │
│  │  │  │ • Optimized     │ │ • Optimized     │ │ • Fast regime   │ │   │ │
│  │  │  │   parameters    │ │   parameters    │ │   classification│ │   │ │
│  │  │  │ • Production    │ │ • Production    │ │ • Pure function │ │   │ │
│  │  │  │   proven        │ │   proven        │ │                 │ │   │ │
│  │  │  └─────────────────┘ └─────────────────┘ └─────────────────┘ │   │ │
│  │  └─────────────────────────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                       │
│                                     │ Trading Signals                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                      RISK MANAGEMENT LAYER                              │ │
│  │                                                                         │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │  │                   TIER 3: CONTAINER                             │   │ │
│  │  │                Production Risk Manager                          │   │ │
│  │  │                                                                 │   │ │
│  │  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │   │ │
│  │  │  │ Position Limits │ │ Exposure Monitor│ │ Circuit Breaker │ │   │ │
│  │  │  │ • Real-time     │ │ • Sector limits │ │ • Daily loss    │ │   │ │
│  │  │  │   validation    │ │ • Correlation   │ │ • Drawdown halt │ │   │ │
│  │  │  │ • Limit checks  │ │ • Beta exposure │ │ • Auto trading  │ │   │ │
│  │  │  │ • Override auth │ │ • VaR tracking  │ │   suspension    │ │   │ │
│  │  │  └─────────────────┘ └─────────────────┘ └─────────────────┘ │   │ │
│  │  │                                                                 │   │ │
│  │  │  Stateful components within:                                   │   │ │
│  │  │  ┌─────────────────┐ ┌─────────────────┐                     │   │ │
│  │  │  │ TIER 2: STATE   │ │ TIER 1: FUNC    │                     │   │ │
│  │  │  │ PortfolioState  │ │ position_sizer()│                     │   │ │
│  │  │  │ • Live positions│ │ • Vol-adjusted  │                     │   │ │
│  │  │  │ • P&L tracking  │ │   sizing        │                     │   │ │
│  │  │  │ • Risk metrics  │ │ • Risk-adjusted │                     │   │ │
│  │  │  └─────────────────┘ └─────────────────┘                     │   │ │
│  │  └─────────────────────────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                       │
│                                     │ Validated Orders                      │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                      EXECUTION LAYER                                    │ │
│  │                                                                         │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │  │                   TIER 3: CONTAINER                             │   │ │
│  │  │               Production Order Manager                          │   │ │
│  │  │                                                                 │   │ │
│  │  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │   │ │
│  │  │  │ Broker Gateway  │ │ Order Router    │ │ Fill Processor  │ │   │ │
│  │  │  │ • IB API        │ │ • Order queue   │ │ • Trade confirm │ │   │ │
│  │  │  │ • Reconnection  │ │ • Priority mgmt │ │ • Position sync │ │   │ │
│  │  │  │ • Heartbeat     │ │ • Rate limiting │ │ • P&L calc      │ │   │ │
│  │  │  │ • Error handle  │ │ • Batch orders  │ │ • Compliance    │ │   │ │
│  │  │  └─────────────────┘ └─────────────────┘ └─────────────────┘ │   │ │
│  │  │                                                                 │   │ │
│  │  │  ┌─────────────────┐ ┌─────────────────┐                     │   │ │
│  │  │  │ Audit Logger    │ │ Alert Manager   │                     │   │ │
│  │  │  │ • All trades    │ │ • System health │                     │   │ │
│  │  │  │ • Compliance    │ │ • Error alerts  │                     │   │ │
│  │  │  │ • Immutable log │ │ • Performance   │                     │   │ │
│  │  │  └─────────────────┘ └─────────────────┘                     │   │ │
│  │  └─────────────────────────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘

Reliability Features:
┌────────────────────────────────────────────────────────────────┐
│                    PRODUCTION RELIABILITY                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  • Container Isolation: Complete fault isolation              │
│  • Health Monitoring: All containers monitored                │
│  • Auto Recovery: Failed containers restart automatically     │
│  • Circuit Breakers: Auto halt on anomalies                  │
│  • Audit Logging: Complete trade audit trail                 │
│  • Alert System: Real-time system health alerts              │
│  • Functions: Proven strategies in lightweight execution      │
│  • State Management: Critical state in reliable containers    │
└────────────────────────────────────────────────────────────────┘

Resource Usage:
┌────────────────┬──────────┬─────────────┬─────────────┐
│ Component Type │ Count    │ Memory      │ Complexity  │
├────────────────┼──────────┼─────────────┼─────────────┤
│ Containers     │ 4        │ High        │ High        │
│ Functions      │ 4        │ Minimal     │ Low         │
│ Stateful       │ 1        │ Medium      │ Medium      │
└────────────────┴──────────┴─────────────┴─────────────┘
Focus: Maximum reliability over resource efficiency
```

## Component Distribution Summary

```
┌────────────────────────────────────────────────────────────────────────────┐
│                      COMPONENT TIER DISTRIBUTION                           │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Workflow Type        │ Containers │ Functions │ Stateful │ Total         │
│  ─────────────────────┼────────────┼───────────┼──────────┼───────────── │
│  Simple Backtest      │     3      │     2     │    1     │      6       │
│  Multi-Strategy       │     4      │     3     │    3     │     10       │
│  Research/Optimize    │     2      │  5,000+   │  5,000   │  10,000+     │
│  Live Trading         │     4      │     4     │    1     │      9       │
│                                                                            │
│  Optimization Patterns:                                                   │
│  • Simple workflows: Mostly functions, containers only for complexity    │
│  • Multi-strategy: Balanced mix based on actual component needs          │
│  • Research phase: Function-heavy for massive parallelization            │
│  • Live trading: Container-heavy for maximum reliability                 │
│                                                                            │
│  Key Insight: Infrastructure scales with actual component requirements    │
└────────────────────────────────────────────────────────────────────────────┘
```

## Container Lifecycle Patterns

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        CONTAINER LIFECYCLE PATTERNS                        │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Simple Backtest Lifecycle:                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ 1. Create 3 containers (data, indicators, execution)               │  │
│  │ 2. Initialize functions (2) and stateful (1) components            │  │
│  │ 3. Run single pipeline: data → indicators → functions → execution  │  │
│  │ 4. Collect results and dispose all components                      │  │
│  │ Duration: Minutes, Memory: Low-Medium                               │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  Research/Optimization Lifecycle:                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ 1. Create 2 shared containers (data loader, results handler)       │  │
│  │ 2. Spawn 5000 parallel lightweight pipelines                       │  │
│  │ 3. Each pipeline: functions only (maximum speed)                   │  │
│  │ 4. Stateful components: fresh instance per pipeline                │  │
│  │ 5. Aggregate results in container, dispose lightweight components  │  │
│  │ Duration: Hours, Memory: High total but efficient per component    │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  Live Trading Lifecycle:                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ 1. Create 4 production containers (persistent, monitored)          │  │
│  │ 2. Functions run within containers (hosted execution)              │  │
│  │ 3. Continuous operation with health monitoring                     │  │
│  │ 4. Auto-restart failed components                                  │  │
│  │ 5. Graceful shutdown procedures                                    │  │
│  │ Duration: Continuous, Memory: High but stable                      │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
```
