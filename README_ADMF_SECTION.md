# ADMF-PC: Advanced Data Mining Framework - Pattern Centric

## Project-Specific Style Guide Implementation

This document shows how ADMF-PC implements the general principles from CLAUDE.md and STYLE.md.

## Architecture Overview

ADMF-PC is an event-driven backtesting and trading framework that emphasizes:
- Protocol + Composition architecture
- Container-based isolation
- Event-driven communication
- Pattern-centric design

## Canonical Implementations

### Core Modules and Their Canonical Files

```
src/
├── core/
│   ├── containers/
│   │   ├── container.py           # THE container implementation
│   │   ├── protocols.py          # Container protocols
│   │   └── factory.py            # Container factory
│   ├── events/
│   │   ├── event_bus.py          # THE event bus
│   │   └── protocols.py          # Event protocols
│   └── coordinator/
│       ├── coordinator.py        # THE coordinator
│       └── topology.py           # Topology management
├── strategy/
│   ├── strategies/
│   │   ├── momentum.py           # Momentum strategy
│   │   ├── mean_reversion.py    # Mean reversion strategy
│   │   └── trend_following.py   # Trend following strategy
│   ├── classifiers/
│   │   └── classifiers.py        # Regime classifiers
│   └── optimization/
│       └── optimizers.py         # Strategy optimizers
├── risk/
│   ├── validators.py             # Risk validators
│   ├── portfolio_state.py        # Portfolio tracking
│   └── position_sizing.py        # Position sizing
├── execution/
│   ├── engine.py                 # Execution engine
│   └── brokers/
│       └── simulated.py          # Market simulation
└── data/
    ├── handlers.py               # Data handlers
    └── loaders.py                # Data loaders
```

## ADMF-PC Specific Patterns

### 1. Container Pattern Implementation

In ADMF-PC, containers provide isolated execution environments:

```python
# ✅ ADMF-PC Way: Single container.py with configuration
class Container:
    def __init__(self, config: ContainerConfig):
        # Features enabled via config
        if config.enable_subcontainers:
            self.subcontainer_manager = SubcontainerManager()
        if config.scoped_event_bus:
            self.event_bus = ScopedEventBus()

# ❌ NOT the ADMF-PC Way: Multiple container files
# enhanced_container.py, portfolio_container.py, etc.
```

### 2. Event-Driven Architecture

ADMF-PC uses semantic events for all communication:

```python
# Event types (in src/core/types/events.py)
@dataclass
class SignalEvent:
    timestamp: pd.Timestamp
    symbol: str
    signal: Signal
    metadata: Dict

@dataclass  
class OrderRequestEvent:
    signal: Signal
    risk_params: Dict
    portfolio_id: str
```

### 3. Strategy Implementation Pattern

All strategies follow the stateless service pattern:

```python
# ✅ ADMF-PC Strategy Pattern
@strategy(
    feature_config={
        'sma': {'params': ['sma_period'], 'default': 20},
        'rsi': {'params': ['rsi_period'], 'default': 14}
    }
)
def momentum_strategy(features: Dict, bar: Dict, params: Dict) -> Dict:
    """Pure function - parameters passed at runtime"""
    return {
        'signal': Signal.BUY if features['rsi'] < 30 else Signal.FLAT,
        'confidence': 0.8
    }

# ❌ NOT the ADMF-PC Way
class MomentumStrategy:
    def __init__(self, fast_period, slow_period):
        self.fast_period = fast_period  # Storing parameters
```

### 4. Decorator-Based Discovery

ADMF-PC uses decorators for component registration:

```python
# Strategies
@strategy(feature_config={...})
def my_strategy(...): ...

# Classifiers  
@classifier(regime_types=[...])
def my_classifier(...): ...

# Features/Indicators
@feature(params=['period'])
def sma(data, period): ...
```

## Configuration Examples

### Backtest Configuration

```yaml
# config/simple_backtest.yaml
topology:
  type: simple_backtest
  
containers:
  - type: data
    config:
      source: CSV
      file: data/SPY.csv
      
  - type: strategy
    config:
      type: momentum
      parameters:
        sma_period: 20
        rsi_period: 14
      features:
        - volume_filter
        
  - type: risk
    config:
      max_position_size: 0.1
      max_portfolio_heat: 0.06
      
  - type: execution
    config:
      commission: 0.001
      slippage_model: linear
```

### Parameter Search Configuration

```yaml
# Multiple parameter combinations, single strategy function
strategies:
  - type: momentum
    parameters:
      sma_period: [10, 20, 30]  # 3 values
      rsi_period: [14, 21]      # 2 values
    # Creates 6 portfolios, but only 1 momentum function!
```

## ADMF-PC Specific Guidelines

### Workflow Patterns

Workflows are defined ONCE in the WorkflowManager:

```python
# ✅ Correct: Single pattern definition
self._workflow_patterns = {
    'simple_backtest': {
        'container_pattern': 'simple_backtest',
        'communication_config': [...]
    }
}

# ❌ Wrong: Pattern definitions scattered across files
```

### Communication Architecture

- **Containers** create containers (via ContainerFactory)
- **Communication** configures adapters (via CommunicationFactory)
- **Workflows** orchestrate patterns (via WorkflowManager)
- Never mix these responsibilities

### Risk Management

Risk validators are stateless with runtime parameters:

```python
# ✅ ADMF-PC Risk Pattern
def validate_position_size(order: Order, params: Dict) -> ValidationResult:
    max_size = params.get('max_position_size', 0.1)
    # Validation logic
    
# ❌ NOT ADMF-PC Pattern
class PositionSizeValidator:
    def __init__(self, max_size):
        self.max_size = max_size  # Storing parameters
```

## Migration Guide for ADMF-PC

When refactoring existing ADMF-PC code:

1. **Identify enhanced files**:
   ```bash
   find src/ -name "*enhanced*" -o -name "*improved*"
   ```

2. **Merge into canonical implementations**:
   - `enhanced_container.py` → merge into `container.py`
   - `portfolio_container.py` → configuration of `container.py`
   - `symbol_timeframe_container.py` → configuration of `container.py`

3. **Update imports**:
   ```python
   # Before
   from src.core.containers.enhanced_container import EnhancedContainer
   
   # After  
   from src.core.containers.container import Container
   ```

## ADMF-PC File Placement Rules

### Source Code (`src/`)
- Only canonical implementations
- No experimental code
- No enhanced/improved variants

### Configuration (`config/`)
- YAML workflow definitions
- Example configurations
- Test configurations

### Temporary Work (`tmp/`)
```
tmp/
├── experiments/      # Experimental code
├── analysis/        # Performance analysis
├── debug/          # Debug scripts
└── reports/        # Status reports
```

### Documentation (`docs/`)
- Architecture documentation
- API documentation
- Migration guides

## Required Reading for ADMF-PC Contributors

Before contributing to ADMF-PC:

1. **CLAUDE.md** - General LLM interaction guidelines
2. **STYLE.md** - General style principles
3. **This section** - ADMF-PC specific implementations
4. **docs/architecture/STANDARD_PATTERN_ARCHITECTURE.md** - Pattern details
5. **src/core/README.md** - Core module documentation

## ADMF-PC Anti-Patterns to Avoid

1. **Creating specialized containers** - Use configuration instead
2. **Storing parameters in strategy classes** - Use stateless functions
3. **Deep inheritance hierarchies** - Use protocols and composition
4. **Backward compatibility layers** - Make clean breaks
5. **Multiple event bus implementations** - One canonical event bus
6. **Factory proliferation** - Minimal factories for orchestration only

---

**Remember**: ADMF-PC values architectural clarity over backward compatibility. When in doubt, consult the canonical implementations and ask for clarification.