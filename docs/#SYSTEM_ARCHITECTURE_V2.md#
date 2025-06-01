# ADMF-PC: Compositional Architecture for Quantitative Trading Systems

## Table of Contents

1. [Overview](#overview)
2. [Core Architecture](#core-architecture)
3. [The Dispatcher Pattern](#the-dispatcher-pattern)
4. [Container Standardization](#container-standardization)
5. [Protocol-Based Composition](#protocol-based-composition)
6. [Configuration as System Interface](#configuration-as-system-interface)
7. [Workflow Orchestration](#workflow-orchestration)
8. [Event-Driven Execution](#event-driven-execution)
9. [Research Applications](#research-applications)

---

## Overview

ADMF-PC proposes an architecture where trading system processes are standardized and containerized to ensure reproducibility. The framework uses compositional design principles to manage complexity without requiring sophisticated individual components.

The architecture emerges from a practical observation: quantitative trading research often fails not due to poor strategy logic, but due to inconsistencies in execution environments, subtle variations in data handling, and unpredictable interactions between system components. Traditional frameworks attempt to solve this through rigid component hierarchies and complex orchestration logic, which creates brittle systems where small changes propagate unpredictably.

ADMF-PC takes a different approach. Rather than standardizing components, it standardizes the execution environment while allowing components to remain maximally flexible. This inversion creates a system where reproducibility emerges naturally from architectural constraints rather than being imposed through rigid component contracts.

The key insight is that complexity should emerge from composition rather than being embedded in individual components. A momentum strategy and a complex multi-regime classifier operate within identical container environments, receiving the same event flows and providing the same interfaces to the orchestration layer. The sophistication of the overall system—whether it's a simple moving average crossover or a sophisticated regime-aware multi-strategy ensemble—is invisible to individual components.

## Core Architecture

The system architecture uses standardized containers orchestrated through a central dispatcher that manages sequential phase processing:

### Multi-Phase Coordinator Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              COORDINATOR                                      │
│  Sequential Phase Processing - Each phase uses different container types:     │
│                                                                               │
│  Phase 1: Parameter Discovery     Phase 2: Regime Analysis                   │
│  ┌─────────────────────────┐     ┌─────────────────────────┐                 │
│  │ Multiple Full Backtest  │ ────▶ │ Analysis Container      │                 │
│  │ Containers              │     │ (No execution - reads   │                 │
│  │ (Complete execution)    │     │  results, finds optimal │                 │
│  └─────────────────────────┘     │  parameters per regime) │                 │
│                                  └─────────────────────────┘                 │
│                                                                               │
│  Phase 3: Ensemble Optimization   Phase 4: Validation                        │
│  ┌─────────────────────────┐     ┌─────────────────────────┐                 │
│  │ Signal Replay           │ ────▶ │ Final Full Backtest     │                 │
│  │ Containers              │     │ Container               │                 │
│  │ (No indicators/data -   │     │ (Complete execution     │                 │
│  │  replays saved signals) │     │  with optimizations)    │                 │
│  └─────────────────────────┘     └─────────────────────────┘                 │
│                                                                               │
│  Each phase type requires different computational patterns and components     │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Example: Standard Backtest Container Architecture

*The detailed container shown below represents **one backtest instance** used in phases like Parameter Discovery or Validation. Other phase types (Analysis, Signal Replay) use different, simpler container architectures optimized for their specific computational patterns.*

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    STANDARD BACKTEST CONTAINER                                │
│  (Used in Parameter Discovery & Validation phases)                            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌────────────────┐                                                          │
│  │ Historical Data│                                                          │
│  │    Streamer    │─────────┐                                               │
│  └────────────────┘         │                                               │
│                             │                                               │
│  ┌──────────────────────────▼───────────────────────────────────────────┐   │
│  │                    Shared Indicator Architecture                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │              Indicator Hub (Shared Computation)              │    │   │
│  │  │  • MA, RSI, ATR, etc. computed once from streamed data      │    │   │
│  │  │  • Caches results for efficiency                             │    │   │
│  │  │  • Emits indicator events to downstream consumers            │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  │                             │                                         │   │
│  │                             │ Indicator Events                       │   │
│  │                             ▼                                         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                │                                              │
│                                │ Indicator Events                            │
│                ┌───────────────┴────────────────┐                            │
│                │                                │                            │
│  ┌─────────────▼─────────────────┐  ┌──────────▼─────────────────────┐     │
│  │   HMM Classifier Container    │  │  Pattern Classifier Container   │     │
│  ├───────────────────────────────┤  ├────────────────────────────────┤     │
│  │ ┌───────────────────────────┐ │  │ ┌────────────────────────────┐ │     │
│  │ │   HMM Regime Classifier   │ │  │ │ Pattern Regime Classifier  │ │     │
│  │ │ • Consumes indicator data │ │  │ │ • Consumes indicator data  │ │     │
│  │ │ • Determines regime state │ │  │ │ • Detects market patterns  │ │     │
│  │ │ • Bull/Bear/Neutral       │ │  │ │ • Breakout/Range/Trending │ │     │
│  │ └────────────┬──────────────┘ │  │ └─────────────┬──────────────┘ │     │
│  │              │                │  │               │                │     │
│  │              ▼                │  │               ▼                │     │
│  │ ┌───────────────────────────┐ │  │ ┌────────────────────────────┐ │     │
│  │ │    Risk Container Pool    │ │  │ │    Risk Container Pool     │ │     │
│  │ │     (Subcontainers)       │ │  │ │     (Subcontainers)        │ │     │
│  │ ├───────────────────────────┤ │  │ ├────────────────────────────┤ │     │
│  │ │ ┌───────────────────────┐ │ │  │ │ ┌────────────────────────┐ │ │     │
│  │ │ │ Conservative Risk     │ │ │  │ │ │ Balanced Risk          │ │ │     │
│  │ │ │ Container             │ │ │  │ │ │ Container              │ │ │     │
│  │ │ │ • Max 2% per position │ │ │  │ │ │ • Max 3% per position │ │ │     │
│  │ │ │ • 10% total exposure  │ │ │  │ │ │ • 20% total exposure  │ │ │     │
│  │ │ │ ┌─────────────────┐   │ │ │  │ │ │ ┌──────────────────┐  │ │ │     │
│  │ │ │ │ Portfolio A     │   │ │ │  │ │ │ │ Portfolio C      │  │ │ │     │
│  │ │ │ │ $50K allocation │   │ │ │  │ │ │ │ $100K allocation │  │ │ │     │
│  │ │ │ │ ┌─────────────┐ │   │ │ │  │ │ │ │ ┌──────────────┐ │  │ │ │     │
│  │ │ │ │ │ Momentum    │ │   │ │ │  │ │ │ │ │ Pattern      │ │  │ │ │     │
│  │ │ │ │ │ Strategy    │ │   │ │ │  │ │ │ │ │ Strategy     │ │  │ │ │     │
│  │ │ │ │ │ AAPL(40%)   │ │   │ │ │  │ │ │ │ │ SPY(60%)     │ │  │ │ │     │
│  │ │ │ │ │ GOOGL(30%)  │ │   │ │ │  │ │ │ │ │ QQQ(40%)     │ │  │ │ │     │
│  │ │ │ │ │ MSFT(30%)   │ │   │ │ │  │ │ │ │ └──────────────┘ │  │ │ │     │
│  │ │ │ │ └─────────────┘ │   │ │ │  │ │ │ └──────────────────┘  │ │ │     │
│  │ │ │ └─────────────────┘   │ │ │  │ │ └────────────────────────┘ │ │     │
│  │ │ └───────────────────────┘ │ │  │ └────────────────────────────┘ │     │
│  │ └───────────────────────────┘ │  └────────────────────────────────┘     │
│  │                               │                                          │
│  │ Output: Signals & Performance │                                          │
│  └───────────────────────────────┘                                          │
│                │                                │                            │
│                └────────────────┬───────────────┘                            │
│                                 │                                            │
│  ┌──────────────────────────────▼───────────────────────────────────────┐   │
│  │                         Backtest Engine                               │   │
│  │  • Executes trades based on aggregated signals                       │   │
│  │  • Manages portfolio state and position tracking                     │   │
│  │  • Calculates performance metrics                                    │   │
│  │  • Handles multi-symbol data alignment                               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                 │                                            │
│                                 ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      Results & Performance Storage                    │   │
│  │  • Streams results to disk during execution                          │   │
│  │  • Maintains in-memory cache of top performers only                  │   │
│  │  • Provides aggregated metrics to Coordinator                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Container Type Specialization

Each phase type uses containers optimized for specific computational patterns:

- **Standard Backtest Container** (shown above): Full data processing, indicators, strategies, and execution
- **Analysis Container**: Reads saved results, performs statistical analysis, no execution components
- **Signal Replay Container**: Reads saved signals, applies weights, executes without indicator computation
- **Signal Generation Container**: Generates signals for analysis, no execution or portfolio tracking

The Coordinator manages sequential execution across these different container types, ensuring that data flows properly between phases while maintaining standardized interfaces and reproducible execution semantics.

## The Dispatcher Pattern

The Coordinator functions as a workflow dispatcher that sequences operations according to configuration specifications. This design abstracts sequencing logic into a dedicated module, allowing other components to focus on their domain-specific responsibilities.

The dispatcher embodies a crucial design principle: sophisticated behavior should emerge from simple orchestration of well-defined components, rather than from intelligent coordination logic. The Coordinator operates as a straightforward execution manager that follows specified sequences without interpretation. This design choice prevents the Coordinator from becoming an over-complex "god module" while distributing system responsibilities appropriately across modular components.

This approach eliminates a common source of non-determinism in trading systems—coordinators that make "helpful" optimizations or adaptations that make results difficult to reproduce. In ADMF-PC, identical configurations produce identical execution sequences, regardless of system load, available resources, or previous executions. The modular architecture enables this simplicity by ensuring each component handles its own domain logic while the Coordinator focuses solely on sequencing operations as specified.

### Simple Backtest Dispatch

```
Configuration: simple_backtest.yaml
┌─────────────────────────────────────────────────────────────┐
│  data:                                                      │
│    source: csv                                              │
│    symbols: [SPY]                                           │
│  strategy:                                                  │
│    type: momentum                                           │
│    parameters: {lookback: 20}                               │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Dispatcher Sequence                      │
│  1. Initialize DataContainer(csv, SPY)                      │
│  2. Initialize StrategyContainer(momentum, lookback=20)     │
│  3. Create event bindings: Data → Strategy                  │
│  4. Execute workflow until completion                       │
└─────────────────────────────────────────────────────────────┘
```

### Multi-Phase Optimization Dispatch

```
Configuration: parameter_optimization.yaml
┌─────────────────────────────────────────────────────────────┐
│  phases:                                                    │
│    - parameter_discovery:                                   │
│        grid: {lookback: [10,20,30], threshold: [0.01,0.02]} │
│    - regime_analysis:                                       │
│        classifiers: [volatility, momentum]                  │
│    - ensemble_optimization:                                 │
│        mode: signal_replay                                  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                Multi-Phase Dispatch Sequence                │
│                                                             │
│  Phase 1: Parameter Discovery                               │
│  ├─ For each parameter combination:                         │
│  │  ├─ Initialize containers with parameters                │
│  │  ├─ Execute backtest                                     │
│  │  └─ Collect performance metrics                          │
│  └─ Generate parameter rankings                             │
│                                                             │
│  Phase 2: Regime Analysis                                   │
│  ├─ Initialize classifier containers                        │
│  ├─ Process historical data for regime identification       │
│  └─ Generate regime-specific performance analysis           │
│                                                             │
│  Phase 3: Ensemble Optimization                             │
│  ├─ Load signal logs from Phase 1                          │
│  ├─ Execute signal replay optimization                      │
│  └─ Generate ensemble weights                               │
└─────────────────────────────────────────────────────────────┘
```

The dispatcher's lack of state awareness becomes advantageous here—it executes each phase exactly as specified without attempting to optimize based on intermediate results, ensuring consistent execution paths across different runs.

## Container Standardization

Every execution context operates within standardized containers that provide identical interfaces regardless of the complexity of enclosed logic. This standardization ensures that a simple momentum strategy and a complex regime-aware classifier receive identical treatment from the orchestration layer.

The container design represents a fundamental architectural decision: rather than requiring components to conform to complex interface hierarchies, the system provides a universal execution environment that abstracts away the complexities of state management, event handling, and resource allocation. This approach eliminates the common problem of component incompatibility that plagues many trading frameworks.

The containers implement a universal pattern that provides several key guarantees. First, they ensure identical execution semantics regardless of the complexity of the enclosed logic—a simple moving average strategy and a sophisticated machine learning classifier both operate within the same container framework. Second, they provide complete isolation between different execution contexts, preventing the state leakage that can make backtests non-reproducible. Third, they standardize the event flow patterns that components use to communicate, creating a predictable interaction model.

### Container Lifecycle Management

The container lifecycle follows a deterministic pattern that ensures consistent resource management and proper cleanup. Each container progresses through defined states: initialization, component registration, event bus wiring, execution, and disposal. This lifecycle is managed by the container factory system, which enforces identical creation patterns regardless of the complexity of the enclosed components.

The lifecycle management addresses a critical challenge in distributed systems: ensuring that resource allocation and cleanup happen predictably. Traditional trading frameworks often suffer from resource leaks or inconsistent initialization ordering that can affect backtest results. By enforcing a standardized lifecycle, ADMF-PC eliminates these sources of non-determinism while enabling reliable resource tracking and debugging capabilities.

### Container Universal Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                   Universal Container                       │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              Event Interface                            ││
│  │  • Receives: BAR, INDICATOR, SIGNAL events             ││
│  │  • Emits: INDICATOR, SIGNAL, ORDER events              ││
│  └─────────────────────────────────────────────────────────┘│
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │             Internal Logic                              ││
│  │  • Domain-specific processing                          ││
│  │  • State management                                    ││
│  │  • Component composition                               ││
│  └─────────────────────────────────────────────────────────┘│
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │           Resource Management                           ││
│  │  • Memory allocation                                   ││
│  │  • Event subscription                                  ││
│  │  • Lifecycle management                                ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### Nested Container Hierarchies

Complex workflows use nested container structures where each level handles specific concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Classifier Container                     │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                 Risk Container                          ││
│  │  ┌─────────────────────────────────────────────────────┐││
│  │  │             Portfolio Container                     │││
│  │  │  ┌─────────────────────────────────────────────────┐│││
│  │  │  │           Strategy Container                    ││││
│  │  │  │     (Individual strategy logic)                ││││
│  │  │  └─────────────────────────────────────────────────┘│││
│  │  │     (Portfolio allocation logic)                   │││
│  │  └─────────────────────────────────────────────────────┘││
│  │     (Risk management logic)                            ││
│  └─────────────────────────────────────────────────────────┘│
│     (Regime classification logic)                           │
└─────────────────────────────────────────────────────────────┘
```

This nesting allows complex market regime strategies to be built from simpler components while maintaining the same container interface at each level.

## Protocol-Based Composition

Components interact through event protocols rather than direct method calls or inheritance relationships. This enables arbitrary composition—any component that emits indicator events can feed any component that consumes them, regardless of their internal implementation.

This protocol-based design philosophy represents a departure from traditional object-oriented frameworks that rely on inheritance hierarchies and tight coupling between components. Instead of requiring a volatility classifier to inherit from a base classifier class and conform to specific method signatures, ADMF-PC allows any component that emits the appropriate events to participate in the system.

The protocol approach provides several advantages for quantitative research. First, it enables true composability—researchers can combine components in ways that weren't anticipated during the original design. A momentum strategy can easily consume signals from an HMM regime classifier, a pattern recognition engine, or a custom machine learning model, as long as each component emits the expected event types. Second, it simplifies testing and development, since components can be developed and tested in isolation without concern for their eventual integration context. Third, it enables incremental system evolution—new component types can be added without modifying existing components or the coordination logic.

### Dependency Injection Architecture

The protocol-based composition is enabled by a sophisticated dependency injection system that manages component relationships without requiring explicit coupling. Components declare their dependencies through protocol interfaces rather than concrete class references, allowing the injection system to wire together arbitrary combinations of compatible components.

This approach differs significantly from traditional dependency injection frameworks that rely on class hierarchies or framework-specific annotations. Instead, ADMF-PC uses protocol inspection to determine compatibility, enabling components from different libraries, frameworks, or implementation approaches to be combined seamlessly. The dependency injection system maintains a registry of available components and their capabilities, automatically resolving dependencies based on protocol compatibility rather than explicit configuration.

### Component Enhancement Through Capabilities

The architecture supports dynamic component enhancement through a capability system that can augment any component with cross-cutting concerns without modifying the original implementation. Components can be enhanced with capabilities such as logging, monitoring, validation, performance profiling, or memory optimization by wrapping them in capability providers that implement the same protocols.

This capability enhancement system addresses a common challenge in quantitative trading frameworks: how to add operational concerns like logging or monitoring to components without polluting their core logic. Traditional approaches often require components to inherit from framework-specific base classes or implement framework-specific interfaces, creating tight coupling. The ADMF-PC capability system preserves component independence while enabling comprehensive operational enhancement.

### Protocol Flow Example

```
Market Data    Indicator Hub    Strategy    Risk & Portfolio    Execution Engine
    │              │               │              │                    │
    │──BAR event──▶│               │              │                    │
    │              │──INDICATOR──▶ │              │                    │
    │              │   event       │──SIGNAL──────▶│                    │
    │              │               │     event     │──ORDER event──────▶│
    │              │               │              │                    │
    │              │               │              │◄──FILL event───────┤
    │              │               │              │                    │
    │              │               │        Update Portfolio:          │
    │              │               │        - Track positions          │
    │              │               │        - Update exposure          │
    │              │               │        - Risk metrics             │
    │              │               │        - Portfolio state          │
```

The protocol design means that adding a new indicator type requires no changes to existing strategies, and adding a new strategy type requires no changes to existing risk management or execution components. The bidirectional flow between Risk & Portfolio and Execution ensures proper portfolio state management through the complete order lifecycle.

## Configuration as System Interface

The configuration layer serves as the primary interface for defining workflows. Rather than writing code to specify how components should interact, users declare the desired composition and let the dispatcher handle the implementation details.

The configuration-driven approach serves a deeper purpose than user convenience—it acts as an architectural safeguard that ensures all system entry points route through the standardized coordination layer. By making configuration the primary interface, ADMF-PC prevents the ad-hoc execution paths that often lead to non-reproducible results in research environments.

When a researcher specifies a trading strategy through configuration, they're not just setting parameters—they're defining a complete execution graph that the Coordinator will follow deterministically. This configuration becomes a complete specification of the research experiment, capturing not just what algorithms to run but how they should interact, what data they should process, and how results should be aggregated. The configuration layer also serves as a natural abstraction boundary, allowing researchers to think in terms of trading concepts—strategies, indicators, risk models—rather than implementation details like thread management, memory allocation, or event routing.

### Configuration-Driven Component Discovery

```yaml
strategy:
  type: momentum
  parameters:
    lookback_period: 20
    momentum_threshold: 0.0002
    rsi_period: 14
```

From this configuration, the system automatically:
- Infers required indicators (SMA_20, RSI)
- Creates appropriate container hierarchies
- Establishes event flow relationships
- Configures component parameters

This approach ensures that all system entry points route through the standardized dispatcher, preventing ad-hoc execution paths that could compromise reproducibility.

## Workflow Orchestration

Complex research workflows emerge from composition of standardized operations. The dispatcher treats a multi-phase parameter optimization with the same operational semantics as a simple backtest—both are sequences of container operations specified in configuration.

The workflow orchestration capabilities represent the culmination of the architectural design principles. By standardizing execution environments and simplifying coordination logic, the system enables complex research workflows to be expressed as compositions of simpler operations. A typical quantitative research workflow might involve strategy development, parameter optimization, regime analysis, ensemble construction, and out-of-sample validation—each phase building on the results of previous phases.

The beauty of this approach is that each phase uses the same underlying container and dispatching infrastructure, ensuring consistent execution semantics across the entire research process. A parameter optimization that tests thousands of strategy variants uses the same container creation and event flow patterns as a simple single-strategy backtest. This consistency enables reliable composition—researchers can confidently combine phases knowing that the execution semantics will remain predictable.

### Multi-Phase Optimization Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        COORDINATOR (The Factory Manager)             │
│  "I create the workspace and tell each station where to save/load"  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼                               ▼
            ┌──────────────┐              ┌──────────────┐
            │  OPTIMIZER   │              │  BACKTESTER  │
            │ "I expand    │              │ "I execute   │
            │  parameters  │              │  strategies" │
            │  AND analyze │              │              │
            │  results"    │              │              │
            └──────────────┘              └──────────────┘
```

### Phase 1: Parameter Discovery Example

```
COORDINATOR says: "Here's your prepared workspace and paths"
    │
    ├─→ Workspace already created with structure:
    │   ./results/workflow_123/
    │   ├── signals/          ← Path passed to backtester
    │   ├── performance/      ← Path passed to backtester
    │   ├── analysis/         ← Path for optimizer outputs
    │   └── metadata/         ← Configuration tracking
    │
    └─→ Tells OPTIMIZER: "Expand this parameter space"
        │
        │   Parameter Space:
        │   ┌─────────────────────────┐
        │   │ lookback: [10, 20, 30] │
        │   │ threshold: [0.01, 0.02] │
        │   │ regime_cls: [hmm, pat]  │
        │   └─────────────────────────┘
        │
        └─→ OPTIMIZER returns 18 combinations:
            │
            ├── Combination 1:  {lookback: 10, threshold: 0.01, regime: hmm}
            ├── Combination 2:  {lookback: 10, threshold: 0.01, regime: pattern}
            ├── Combination 3:  {lookback: 10, threshold: 0.02, regime: hmm}
            └── ... (15 more combinations)

COORDINATOR then orchestrates 18 backtests with explicit paths:
    │
    └─→ For each combination:
        │
        ├─→ Creates backtest config:
        │   {
        │     "parameters": {combination},
        │     "capture_signals": true,
        │     "output_paths": {
        │       "signals": "./results/workflow_123/signals/trial_0.jsonl",
        │       "performance": "./results/workflow_123/performance/trial_0.json"
        │     }
        │   }
        │
        ├─→ BACKTESTER executes strategy
        ├─→ BACKTESTER saves signals to specified path
        └─→ BACKTESTER saves performance to specified path

Output Files After Phase 1:
    signals/
    ├── trial_0.jsonl    (all signals from combination 1)
    ├── trial_1.jsonl    (all signals from combination 2)
    └── ... (16 more files)
    
    performance/
    ├── trial_0.json     (metrics from combination 1)
    ├── trial_1.json     (metrics from combination 2)
    └── ... (16 more files)
```

### Complete Multi-Phase Data Flow

```
COORDINATOR CREATES WORKSPACE:
└── ./results/workflow_123/
    ├── signals/
    ├── performance/
    ├── analysis/
    ├── metadata/
    └── checkpoints/

PHASE 1: Parameter Discovery
├── COORDINATOR → OPTIMIZER: "Expand parameters"
├── OPTIMIZER → COORDINATOR: [param_set_1, param_set_2, ...]
├── COORDINATOR → BACKTESTER: "Execute with paths.signals/trial_N.jsonl"
└── OUTPUT: signals/trial_*.jsonl, performance/trial_*.json

PHASE 2: Regime Analysis  
├── COORDINATOR → OPTIMIZER: "Analyze at paths.performance/"
├── OPTIMIZER reads: performance/trial_*.json
├── OPTIMIZER analyzes: Find best params per regime
└── OPTIMIZER writes: analysis/regime_optimal_params.json

PHASE 3: Ensemble Optimization
├── COORDINATOR → OPTIMIZER: "Expand weight space"
├── OPTIMIZER → COORDINATOR: [weight_set_1, weight_set_2, ...]
├── COORDINATOR → BACKTESTER: "Replay signals at paths.signals/"
├── BACKTESTER reads: signals/*, analysis/regime_optimal_params.json
├── BACKTESTER writes: performance/ensemble_trial_*.json
├── COORDINATOR → OPTIMIZER: "Analyze ensemble results"
└── OPTIMIZER writes: analysis/ensemble_weights.json

PHASE 4: Validation
├── COORDINATOR → BACKTESTER: "Full test with all optimizations"
├── BACKTESTER reads: analysis/regime_optimal_params.json, analysis/ensemble_weights.json
└── BACKTESTER writes: performance/validation_results.json
```

### Workspace Management and File-Based Communication

The Coordinator implements a sophisticated workspace management system that enables complex multi-phase workflows through standardized file-based communication patterns. Each workflow execution creates a structured workspace with defined directories for different types of intermediate results: signals, performance metrics, analysis outputs, metadata, and checkpoints.

This file-based communication approach provides several architectural advantages. First, it enables natural checkpointing and resumability—any phase can be restarted from its last successful completion without affecting previous phases. Second, it facilitates debugging and analysis by making all intermediate results inspectable and modifiable. Third, it enables parallel execution of different workflow branches, since file-based communication naturally supports concurrent readers and writers.

The workspace management system follows a strict naming convention that encodes workflow identity, phase information, and result types in file paths. This enables automated discovery of workflow artifacts and systematic cleanup of temporary files. The structured approach also supports workflow composition, where the outputs of one workflow can become inputs to another, enabling hierarchical research processes.

### Key Workflow Benefits

1. **File-Based Communication**: Phase 1 writes → Phase 2 reads → Phase 2 writes → Phase 3 reads
2. **Checkpointing**: Resume from any phase using saved state
3. **Debugging**: Inspect intermediate results at each phase
4. **Parallelization**: Phases can read/write concurrently
5. **Manual Intervention**: Edit files between phases if needed
6. **Workflow Composition**: Outputs from one workflow become inputs to another

## Event-Driven Execution

The system operates as an event-driven architecture where strategy logic remains identical between backtesting and live execution. The same container that processes historical BAR events during backtesting processes real-time BAR events during live trading.

### Event Flow: Signals → Orders → Fills

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        Event Flow Within Backtest Container                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Strategies           Risk & Portfolio Containers      Backtest Engine       │
│      │                           │                          │                 │
│      │    SIGNAL Event           │                          │                 │
│      │  (Buy AAPL, strength=0.8) │                          │                 │
│      ├──────────────────────────►│                          │                 │
│      │                           │                          │                 │
│      │                    Risk Assessment:                  │                 │
│      │                    - Check position limits           │                 │
│      │                    - Check exposure limits           │                 │
│      │                    - Apply position sizing           │                 │
│      │                    - May VETO signal                 │                 │
│      │                           │                          │                 │
│      │                           │     ORDER Event          │                 │
│      │                           │  (Buy AAPL, 100 shares) │                 │
│      │                           ├─────────────────────────►│                 │
│      │                           │                          │                 │
│      │                           │                   Execute Order:           │
│      │                           │                   - Check market data      │
│      │                           │                   - Apply slippage         │
│      │                           │                   - Update positions       │
│      │                           │                          │                 │
│      │                           │      FILL Event          │                 │
│      │                           │◄─────────────────────────┤                 │
│      │                           │  (Filled @ $150.25)      │                 │
│      │                           │                          │                 │
│      │                 Update Risk & Portfolio:             │                 │
│      │                    - Track positions                 │                 │
│      │                    - Update exposure                 │                 │
│      │                    - Risk metrics                    │                 │
│      │                    - Portfolio state                │                 │
│      │                           │                          │                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Key Event Flow Points:
1. **Strategies** generate SIGNAL events based on market data and indicators
2. **Risk & Portfolio Containers** convert SIGNAL events to ORDER events (or veto them)
3. **Backtest Engine** executes ORDER events and generates FILL events
4. **Risk & Portfolio Containers** update portfolio state based on FILL events

### Event Flow Standardization

```
Historical Backtest:              Live Trading:
    │                                 │
CSV File ────▶ BAR events         Market Feed ────▶ BAR events
    │                                 │
    ▼                                 ▼
Strategy Container               Strategy Container
    │                                 │
    ▼                                 ▼
SIGNAL events                    SIGNAL events
    │                                 │
    ▼                                 ▼
Simulated Execution             Live Execution
```

This consistency eliminates implementation discrepancies between research and production phases, since the same event-processing logic runs in both environments.

### Production Consistency Guarantees

The architectural design provides strong guarantees about consistency between research and production environments. The same container patterns, event flows, and component compositions that execute during backtesting operate identically during live trading. This consistency is achieved through several mechanisms: standardized container interfaces that abstract away execution environment differences, protocol-based component communication that remains invariant across deployment contexts, and deterministic state management that produces identical results given identical inputs.

The production consistency extends beyond simple algorithm execution to encompass the entire trading pipeline. Risk management logic, portfolio tracking, order generation, and position sizing all operate through the same event-driven patterns in both environments. This eliminates the common problem in quantitative trading where strategies perform differently in live trading due to subtle implementation differences between research and production systems.

### State Management and Isolation Theory

The container-based architecture implements a sophisticated approach to state management that addresses fundamental challenges in distributed systems. Each container maintains complete isolation of state, preventing the subtle interactions between components that can make systems non-deterministic. This isolation is achieved through several layers: separate memory spaces for each container, independent event bus instances that prevent cross-container communication, and explicit resource management that ensures proper cleanup.

The state isolation design draws from principles in functional programming and actor model concurrency, where components communicate through message passing rather than shared mutable state. This approach eliminates entire classes of bugs related to unexpected state interactions while enabling confident parallel execution of multiple containers. The isolation guarantees extend to temporal aspects as well—containers maintain no memory of previous executions, ensuring that backtest results depend only on input data and configuration rather than execution history.

## Research Applications

The architecture enables rapid iteration on research questions by abstracting away infrastructure concerns. Researchers can focus on strategy logic, parameter sensitivity, regime analysis, and ensemble construction without managing the complexities of data flow coordination.

The practical implications of this architectural approach for quantitative research are significant. Traditional trading frameworks often require researchers to spend substantial time on infrastructure concerns—managing data flows between components, ensuring consistent execution across different test scenarios, debugging subtle interaction effects between loosely related modules. ADMF-PC eliminates many of these concerns by providing a standardized execution environment where researchers can focus on the substantive questions: which combinations of strategies perform well, how sensitive are results to parameter choices, what market regimes favor different approaches.

The framework particularly excels in scenarios that require systematic exploration of large parameter spaces or complex workflow orchestration. Because the container and coordination infrastructure handles the mechanics of execution consistently, researchers can implement sophisticated multi-phase optimization workflows that would be prohibitively complex to manage manually. The configuration-driven approach allows these workflows to be modified, extended, or repeated by editing YAML files rather than rewriting code, enabling rapid experimentation and reliable reproduction of results.

### Research Workflow Example

A typical research workflow demonstrates how the architectural principles work in practice:

1. **Strategy Development**: Implement strategy logic in isolated container, leveraging the protocol-based design to focus purely on signal generation logic without concern for execution details

2. **Parameter Optimization**: Use grid search across parameter space, with the Coordinator automatically managing thousands of container instances while ensuring identical execution semantics for each parameter combination

3. **Regime Analysis**: Analyze strategy performance across market conditions using the same container infrastructure, enabling fair comparison across different market environments  

4. **Ensemble Construction**: Combine multiple strategies through signal replay, leveraging the standardized event flow to test different combination approaches without re-running computationally expensive strategy logic

5. **Validation**: Walk-forward analysis with out-of-sample testing, using the same container patterns to ensure that validation results reflect actual strategy performance rather than implementation artifacts

Each phase builds naturally on the previous phases while maintaining the same execution guarantees. The modular design means that insights from one phase can inform modifications to previous phases without requiring complete workflow reconstruction.

### Testing Architecture and Validation

The protocol-based design enables a uniform approach to testing that transcends individual component implementations. Since components communicate through standardized protocols rather than concrete interfaces, any component implementing a given protocol can be tested using the same test suite. This enables systematic validation of component behavior and facilitates regression testing when components are modified or replaced.

The container isolation properties also enable sophisticated testing approaches. Individual containers can be tested in complete isolation, eliminating concerns about test interactions or shared state corruption. Mock components can be substituted for real implementations without changing container structure, enabling controlled testing of specific interaction patterns. The deterministic execution guarantees mean that test results are completely reproducible, facilitating automated testing and continuous integration processes.

### Scalability Characteristics and Performance Analysis

The architectural design exhibits favorable scalability characteristics across multiple dimensions. Horizontal scaling is achieved through container replication—multiple identical containers can execute in parallel without interaction, enabling linear scaling of computational workloads. The shared-nothing architecture means that adding computational resources translates directly to increased throughput without coordination overhead.

Vertical scaling within individual containers is achieved through the shared computation model, where expensive operations like indicator calculation are performed once and shared among multiple consumers. This eliminates redundant computation while maintaining component independence. The signal replay architecture provides an additional scaling dimension, enabling rapid exploration of ensemble and risk parameter combinations without repeating expensive strategy computations.

### Extensibility Patterns and System Evolution

The architectural design supports multiple patterns for system extension without requiring modifications to existing components. New component types can be introduced by implementing existing protocols, enabling seamless integration with existing system components. New protocols can be defined to support novel interaction patterns, with existing components remaining unaffected. The capability enhancement system allows operational concerns to be added to any component without source code modification.

The container pattern system supports extension through composition—new container types can be defined by combining existing container components in novel ways. The workflow orchestration system enables extension through new phase types that can be integrated into existing workflow templates. This extensibility is achieved without sacrificing the deterministic execution guarantees that ensure reproducible results.

### Implications for Quantitative Research

The architectural decisions in ADMF-PC have several important implications for how quantitative research can be conducted and validated. The standardized execution environments and deterministic coordination ensure that published results can be replicated exactly, addressing a persistent problem in quantitative finance research. The modular design means that adding new components or capabilities doesn't require understanding or modifying existing code, lowering the barrier to implementing sophisticated strategies and workflows.

Perhaps most importantly, the configuration-driven approach allows rapid experimentation with different strategy compositions, parameter sets, and market regimes. Ideas can be tested quickly without the implementation overhead that typically slows quantitative research. The complete research workflows are captured in human-readable configurations that serve as documentation of both the research methodology and the implementation details, making it easier to build on previous work and maintain institutional knowledge.

The architecture enables a new paradigm for quantitative research where complex investigations can be decomposed into sequences of standardized operations. Rather than building monolithic analysis scripts that are difficult to debug and impossible to reuse, researchers can compose sophisticated workflows from proven components. This compositional approach enables systematic exploration of parameter spaces and methodological variations while maintaining confidence in the reliability of individual components.

---

*ADMF-PC demonstrates that sophisticated quantitative trading systems can be built through architectural elegance rather than component complexity. By standardizing execution environments while maintaining component flexibility, implementing straightforward coordination with deterministic behavior, and providing configuration-driven workflow abstraction, the framework enables research that is both powerful and reproducible. The key insight is that complexity should emerge from composition rather than being embedded in individual components—a principle that proves remarkably effective for quantitative trading research.*