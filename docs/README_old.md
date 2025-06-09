# ADMF-PC: Adaptive Decision Making Framework - Protocol Components

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**A quantitative trading platform built on protocol-based architecture and composable containers.**

## 🎯 **What is ADMF-PC?**

ADMF-PC transforms trading system development from complex programming into simple configuration using protocol-based architecture and composable containers. The system enables:

- **Configuration-driven development** through YAML files
- **Protocol + Composition architecture** instead of inheritance hierarchies
- **Isolated container execution** for parallel backtesting
- **Event-driven communication** between components
- **Composable backtest patterns** for different optimization phases

### **Unified Topology Configuration**

Define workflows through universal topology specification:

```yaml
workflow:
  execution_mode: "backtest"  # Simple mode: backtest, signal_generation, signal_replay

symbols:
  - symbol: SPY
    data_source: csv
    features: [sma_20, rsi_14]
  - symbol: QQQ
    data_source: csv
    features: [sma_20, rsi_14]

portfolios:
  - name: momentum_portfolio
    strategies: [momentum_10_20, momentum_5_15]
    symbols: [SPY, QQQ]
  - name: reversion_portfolio
    strategies: [mean_reversion]
    symbols: [SPY]

strategies:
  - name: momentum_10_20
    type: momentum
    fast_period: 10
    slow_period: 20
  - name: momentum_5_15
    type: momentum
    fast_period: 5
    slow_period: 15
  - name: mean_reversion
    type: mean_reversion
    lookback: 20
```

## 🏗️ **Unified Architecture**

ADMF-PC uses a **unified architecture** that achieves **60% container reduction** through state-based separation:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        WORKFLOW MANAGER                                     │
│  • Universal topology builder (one method for all workflows)               │
│  • Three simple execution modes: backtest, signal_generation, signal_replay│
│  • No pattern detection - direct topology specification                    │
└─────────────────────────┬───────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ SYMBOL          │ │ PORTFOLIO       │ │ SHARED COMPONENT│
│ CONTAINERS      │ │ CONTAINERS      │ │ POOLS           │
│ (Stateful)      │ │ (Stateful)      │ │ (Stateless)     │
│                 │ │                 │ │                 │
│ • Market data   │ │ • Position      │ │ • Strategy      │
│ • Feature       │ │   tracking      │ │   functions     │
│   computation   │ │ • P&L           │ │ • Classifiers   │
│ • Per-symbol    │ │   accumulation  │ │ • Risk          │
│   isolation     │ │ • Portfolio     │ │   validators    │
│                 │ │   state         │ │ • Shared across │
│                 │ │                 │ │   all contexts  │
└─────────────────┘ └─────────────────┘ └─────────────────┘
          │               │               │
          └───────────────┼───────────────┘
                          ▼
                ┌─────────────────┐
                │ EXECUTION ENGINE│
                │ (Universal)     │
                │                 │
                │ • Order         │
                │   management    │
                │ • Market        │
                │   simulation    │
                │ • Universal     │
                │   topology      │
                │   execution     │
                └─────────────────┘
```

### **Protocol + Composition Benefits**

Unlike inheritance-based systems, ADMF-PC uses protocols for maximum flexibility:

```python
# Mix any components regardless of inheritance
class TradingStrategy:
    def __init__(self):
        self.components = [
            SimpleMovingAverage(20),              # Your indicator
            ta.RSI(period=14),                    # TA-Lib indicator  
            sklearn.RandomForestClassifier(),     # ML model
            custom_momentum_calc,                 # Simple function
            ExternalLibrarySignal("momentum"),    # Third-party library
        ]
    
    def generate_signals(self, data):
        # Use them all together seamlessly
        signals = []
        for component in self.components:
            if hasattr(component, 'calculate'):
                signals.append(component.calculate(data))
            elif hasattr(component, 'predict'):  # ML model
                signals.append(component.predict(data))
            elif callable(component):  # Function
                signals.append(component(data))
        return self.combine_signals(signals)
```

**Result**: Complete freedom to mix your code, external libraries, ML models, and simple functions.

## 📁 **Project Structure**

```
ADMF-PC/
├── 📁 config/                    # Example configurations
│   ├── simple_backtest.yaml     # Basic backtest example
│   ├── optimization_workflow.yaml # Multi-phase optimization
│   └── regime_aware_optimization.yaml # Regime detection
│
├── 📁 data/                      # Sample data files
│   └── SPY_1m.csv               # Example market data
│
├── 📁 docs/                      # Comprehensive documentation
│   ├── MULTIPHASE_OPTIMIZATION.MD # Multi-phase workflows
│   ├── BACKTEST_README.MD       # Container architecture
│   ├── YAML_CONFIG.MD           # Configuration guide
│   └── PC/BENEFITS.MD           # Protocol + Composition benefits
│
├── 📁 src/                       # Core system modules
│   ├── 📁 core/                 # Foundation architecture
│   │   ├── 📁 coordinator/      # Workflow orchestration
│   │   ├── 📁 containers/       # Isolation and lifecycle
│   │   ├── 📁 components/       # Protocol-based components
│   │   ├── 📁 events/           # Event system
│   │   └── 📁 infrastructure/   # Cross-cutting concerns
│   │
│   ├── 📁 data/                 # Data management
│   │   ├── protocols.py         # Data provider interfaces
│   │   ├── handlers.py          # CSV, database, API handlers
│   │   └── streamers.py         # Real-time data streaming
│   │
│   ├── 📁 strategy/             # Strategy development
│   │   ├── 📁 strategies/       # Pre-built strategies
│   │   ├── 📁 classifiers/      # Regime detection
│   │   ├── 📁 optimization/     # Parameter optimization
│   │   └── 📁 components/       # Strategy building blocks
│   │
│   ├── 📁 risk/                 # Risk and portfolio management
│   │   ├── position_sizing.py   # Position sizing algorithms
│   │   ├── risk_limits.py       # Risk constraint enforcement
│   │   ├── portfolio_state.py   # Portfolio tracking
│   │   └── signal_processing.py # Signal to order conversion
│   │
│   └── 📁 execution/            # Order execution
│       ├── backtest_engine.py   # Historical simulation
│       ├── market_simulation.py # Realistic execution models
│       ├── order_manager.py     # Order lifecycle
│       └── protocols.py         # Execution interfaces
│
├── 📁 tests/                     # Comprehensive test suite
├── main.py                       # Command-line entry point
└── README.md                     # This file
```

## 🚀 **Quick Start**

### **Installation**

```bash
git clone https://github.com/your-org/ADMF-PC.git
cd ADMF-PC
pip install -r requirements.txt
```

### **Run Your First Backtest**

```bash
# 1. Copy example configuration
cp config/simple_backtest.yaml my_test.yaml

# 2. Edit symbols and dates (optional)
# 3. Run backtest
python main.py --config my_test.yaml

# 4. View results
ls results/my_test/
```

### **Simple Configuration Example**

```yaml
workflow:
  execution_mode: "backtest"  # backtest, signal_generation, signal_replay

symbols:
  - symbol: AAPL
    data_source: csv
    features: [sma_20, rsi_14]
  - symbol: MSFT
    data_source: csv
    features: [sma_20, rsi_14]

portfolios:
  - name: momentum_portfolio
    strategies: [momentum_10_30]
    symbols: [AAPL, MSFT]

strategies:
  - name: momentum_10_30
    type: momentum
    fast_period: 10
    slow_period: 30

risk:
  initial_capital: 100000
  position_size_pct: 2.0
```

## 🔧 **Universal Topology**

ADMF-PC uses a unified topology that works across all execution modes:

### **1. Backtest Mode**

Universal topology executing full backtests:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         UNIVERSAL TOPOLOGY - BACKTEST MODE                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌────────────────┐     ┌────────────────┐     ┌────────────────┐            │
│  │ Symbol         │     │ Symbol         │     │ Symbol         │            │
│  │ Container SPY  │     │ Container QQQ  │     │ Container IWM  │            │
│  ├────────────────┤     ├────────────────┤     ├────────────────┤            │
│  │• Market data   │     │• Market data   │     │• Market data   │            │
│  │• Feature hub   │     │• Feature hub   │     │• Feature hub   │            │
│  │• Isolation     │     │• Isolation     │     │• Isolation     │            │
│  └────────┬───────┘     └────────┬───────┘     └────────┬───────┘            │
│           │                      │                      │                    │
│           └──────────┬───────────┼──────────────────────┘                    │
│                      │           │                                           │
│                      ▼           ▼                                           │
│           ┌─────────────────────────────────────────┐                        │
│           │        Shared Component Pools           │                        │
│           │  • Strategy functions (stateless)       │                        │
│           │  • Classifiers (stateless)              │                        │
│           │  • Risk validators (stateless)          │                        │
│           └─────────────────┬───────────────────────┘                        │
│                             │                                                │
│                             ▼                                                │
│           ┌─────────────────────────────────────────┐                        │
│           │      Portfolio Containers               │                        │
│           │  Portfolio A: momentum_portfolio        │                        │
│           │  • Position tracking                    │                        │
│           │  • P&L accumulation                     │                        │
│           │  • Subscribes to strategies + symbols   │                        │
│           └─────────────────┬───────────────────────┘                        │
│                             │                                                │
│                             ▼                                                │
│                    ┌─────────────────┐                                       │
│                    │ Execution Engine│                                       │
│                    │ (Universal)     │                                       │
│                    └─────────────────┘                                       │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Usage**: Complete strategy evaluation with natural multi-asset scaling

### **2. Signal Replay Mode**

Same universal topology, but replaying pre-captured signals:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       UNIVERSAL TOPOLOGY - SIGNAL REPLAY MODE                │
│  (10-100x faster - no indicator computation)                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌────────────────┐                                                          │
│  │  Signal Store  │                                                          │
│  │   (From prev   │─────────┐                                               │
│  │    backtest)   │         │                                               │
│  └────────────────┘         │                                               │
│                             ▼                                               │
│           ┌─────────────────────────────────────────┐                        │
│           │        Shared Component Pools           │                        │
│           │  • Strategy functions (NO computation)  │                        │
│           │  • Classifiers (replay mode)           │                        │
│           │  • Risk validators (stateless)          │                        │
│           │  • Read signals, skip calculation       │                        │
│           └─────────────────┬───────────────────────┘                        │
│                             │                                                │
│                             ▼                                                │
│           ┌─────────────────────────────────────────┐                        │
│           │      Portfolio Containers               │                        │
│           │  Portfolio A: momentum_portfolio        │                        │
│           │  • Position tracking                    │                        │
│           │  • P&L accumulation                     │                        │
│           │  • Ensemble weight optimization         │                        │
│           └─────────────────┬───────────────────────┘                        │
│                             │                                                │
│                             ▼                                                │
│                    ┌─────────────────┐                                       │
│                    │ Execution Engine│                                       │
│                    │ (Universal)     │                                       │
│                    └─────────────────┘                                       │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Usage**: Ensemble weight optimization, risk parameter tuning

### **3. Signal Generation Mode**

Same universal topology, but stopping after signal creation:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     UNIVERSAL TOPOLOGY - SIGNAL GENERATION MODE              │
│  (Stop after signal creation - research & analysis)                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌────────────────┐     ┌────────────────┐     ┌────────────────┐            │
│  │ Symbol         │     │ Symbol         │     │ Symbol         │            │
│  │ Container SPY  │     │ Container QQQ  │     │ Container IWM  │            │
│  ├────────────────┤     ├────────────────┤     ├────────────────┤            │
│  │• Market data   │     │• Market data   │     │• Market data   │            │
│  │• Feature hub   │     │• Feature hub   │     │• Feature hub   │            │
│  │• Isolation     │     │• Isolation     │     │• Isolation     │            │
│  └────────┬───────┘     └────────┬───────┘     └────────┬───────┘            │
│           │                      │                      │                    │
│           └──────────┬───────────┼──────────────────────┘                    │
│                      │           │                                           │
│                      ▼           ▼                                           │
│           ┌─────────────────────────────────────────┐                        │
│           │        Shared Component Pools           │                        │
│           │  • Strategy functions (generate)        │                        │
│           │  • Classifiers (regime detection)      │                        │
│           │  • Signal analysis & quality metrics   │                        │
│           │  • STOP: No risk/execution components   │                        │
│           └─────────────────┬───────────────────────┘                        │
│                             │                                                │
│                             ▼                                                │
│           ┌─────────────────────────────────────────┐                        │
│           │       Signal Storage & Analysis         │                        │
│           │  • Capture signals + metadata           │                        │
│           │  • Calculate quality metrics            │                        │
│           │  • Store for later replay               │                        │
│           │  • NO position tracking or execution    │                        │
│           └─────────────────────────────────────────┘                        │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Usage**: Signal quality research, MAE/MFE analysis, classifier comparison

## ⚡ **Event-Driven Architecture**

Components communicate through isolated event buses, preventing cross-contamination:

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

### **Event Flow Process**

1. **Strategies** generate SIGNAL events based on market data and indicators
2. **Risk & Portfolio Containers** convert SIGNAL events to ORDER events (or veto them)
3. **Backtest Engine** executes ORDER events and generates FILL events
4. **Risk & Portfolio Containers** update portfolio state based on FILL events

### **Event Flow Benefits**

- **Unidirectional Flow**: Clear data lineage, no circular dependencies
- **Container Isolation**: Events scoped to individual containers
- **Loose Coupling**: Components only know about event interfaces
- **Easy Testing**: Mock event sources and sinks independently

## 🔄 **Unified Execution Modes**

ADMF-PC uses a universal topology across three simple execution modes:

### **Mode 1: Backtest**
- Full pipeline execution with live data flow
- Complete strategy evaluation for deployment
- Natural multi-asset scaling through symbol containers

### **Mode 2: Signal Generation**  
- Same topology, stop after signal creation
- Research signal quality and classifier effectiveness
- Store signals for ensemble optimization

### **Mode 3: Signal Replay**
- Load pre-generated signals, skip computation
- 10-100x faster ensemble weight optimization
- Test portfolio allocations and risk parameters

### **Benefits of Universal Topology**

```
Traditional: Different containers for different patterns (complex)
ADMF-PC:    Same containers, different execution flows (simple)
```

All workflows use identical topology - only the execution flow changes.

## 🧪 **Testing**

### **Run Test Suite**

```bash
# Run all tests
python -m pytest tests/

# Run specific module tests
python -m pytest tests/unit/core/
python -m pytest tests/integration/

# Run with coverage
python -m pytest tests/ --cov=src/ --cov-report=html
```

### **Protocol-Based Testing**

Components implement protocols, making testing simple:

```python
# Test any component that implements SignalGenerator protocol
def test_signal_generator(signal_generator: SignalGenerator):
    signal = signal_generator.generate_signal(mock_data)
    assert signal['strength'] > 0
    assert signal['direction'] in ['BUY', 'SELL']

# Works with any implementation
test_signal_generator(MomentumStrategy())
test_signal_generator(sklearn.RandomForestClassifier())  
test_signal_generator(custom_function)
```

## 📋 **Configuration Examples**

### **Simple Momentum Strategy**

```yaml
workflow:
  type: "backtest"
  name: "Tech Stock Momentum"

data:
  symbols: ["AAPL", "GOOGL", "MSFT", "AMZN"]
  start_date: "2022-01-01"
  end_date: "2023-12-31"

strategies:
  - name: "momentum"
    type: "momentum"
    fast_period: 10
    slow_period: 30

risk:
  initial_capital: 100000
  position_size_pct: 2.0
```

### **Multi-Strategy Portfolio**

```yaml
strategies:
  - name: "tech_momentum"
    type: "momentum"
    symbols: ["AAPL", "GOOGL", "MSFT"]
    
  - name: "etf_mean_reversion"
    type: "mean_reversion"
    symbols: ["SPY", "QQQ"]
    
  - name: "commodity_breakout"
    type: "breakout"
    symbols: ["GLD"]

strategy_allocation:
  tech_momentum: 0.5
  etf_mean_reversion: 0.3
  commodity_breakout: 0.2
```

### **Parameter Optimization**

```yaml
workflow:
  type: "optimization"

optimization:
  algorithm: "genetic"
  objective: "sharpe_ratio"
  
  parameter_space:
    fast_period: [5, 8, 10, 12, 15]
    slow_period: [20, 26, 30, 35, 40]
    signal_threshold: [0.01, 0.015, 0.02]
    
  population_size: 100
  generations: 50
```

### **Live Trading**

```yaml
workflow:
  type: "live_trading"

broker:
  name: "alpaca"
  paper_trading: true
  api_key: "${ALPACA_API_KEY}"
  secret_key: "${ALPACA_SECRET_KEY}"

risk:
  real_time_limits:
    max_daily_loss: 500
    max_position_size: 1000
    
monitoring:
  email_alerts: true
  slack_webhook: "${SLACK_WEBHOOK}"
```

## 📚 **Architecture Benefits**

### **1. Protocol + Composition vs Inheritance**

**Research Speed**: Test any idea in minutes, not hours
- Mix your strategies with external libraries, ML models, and simple functions
- No need to inherit from complex base classes
- Add capabilities incrementally without breaking existing code

**Production Flexibility**: Deploy any combination of strategies
- Runtime algorithm switching based on market conditions
- Configuration-driven behavior without code changes
- Easy integration with third-party tools and data sources

### **2. Container Isolation**

**Reproducible Results**: Same configuration → identical execution
- No state leakage between parallel backtests
- Complete isolation of event buses and resources
- Standardized creation patterns eliminate setup variations

**Massive Parallelization**: Run thousands of backtests simultaneously
- Each container runs independently
- Linear scaling with compute resources
- Easy cloud deployment and resource management

### **3. Event-Driven Design**

**Clean Architecture**: Unidirectional data flow
- No circular dependencies or hidden coupling
- Clear component boundaries and responsibilities
- Easy to understand, debug, and maintain

**Flexible Communication**: Components only know about events
- Add new components without changing existing ones
- Mock any component for testing
- Monitor system behavior through event streams

## 📚 **Documentation**

### **Essential Reading**
- **[YAML Configuration Guide](docs/YAML_CONFIG.MD)**: Complete configuration reference
- **[Multi-Phase Optimization](docs/MULTIPHASE_OPTIMIZATION.MD)**: Advanced optimization workflows
- **[Container Architecture](docs/BACKTEST_README.MD)**: Deep dive into isolation system
- **[Protocol Benefits](docs/PC/BENEFITS.MD)**: Protocol + Composition advantages

### **API Documentation**
- **[Core Module](src/core/README.md)**: Foundation architecture
- **[Data Module](src/data/README.md)**: Data management and protocols
- **[Strategy Module](src/strategy/README.md)**: Strategy development framework
- **[Risk Module](src/risk/README.md)**: Risk and portfolio management
- **[Execution Module](src/execution/README.md)**: Order execution and simulation

## 🤝 **Contributing**

We welcome contributions! Please see:

1. **Code Style**: We use Black for formatting
2. **Testing**: All new features need tests
3. **Documentation**: Update docs for any changes
4. **Architecture**: Follow protocol-based patterns

### **Development Setup**

```bash
# Clone and setup development environment
git clone https://github.com/your-org/ADMF-PC.git
cd ADMF-PC

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
python -m pytest tests/
```

## 🔒 **Security & Compliance**

### **Security Features**
- **Container Isolation**: Complete separation between executions
- **Input Validation**: Comprehensive validation at all boundaries
- **Audit Logging**: Complete audit trails for all operations
- **Environment Variables**: Secure credential management

### **Compliance**
- **Regulatory Logging**: 7-year audit retention
- **Trade Validation**: Pre and post-trade checks
- **Risk Controls**: Circuit breakers and position limits
- **Reproducibility**: Exact result reproduction for audits

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Ready to explore protocol-based quantitative trading?**

```bash
git clone https://github.com/your-org/ADMF-PC.git
cd ADMF-PC
python main.py --config config/simple_backtest.yaml
```

**Experience configuration-driven trading system development.**
