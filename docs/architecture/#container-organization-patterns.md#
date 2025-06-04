# Container Organization Patterns

## Overview

ADMF-PC's composable container architecture supports multiple organizational patterns, each optimized for different use cases. This document analyzes the primary patterns, explains when to use each, and provides implementation examples.

## Pattern Comparison Matrix

| Pattern | Best For | Mental Model | Development | Performance Attribution | Combinatorial Testing |
|---------|----------|--------------|-------------|------------------------|----------------------|
| **Strategy-First** | Simple backtests, strategy development | Intuitive - matches trader thinking | Independent strategy teams | Clear per-strategy metrics | Limited |
| **Classifier-First** | Research, regime optimization | Abstract - academic research | Coupled development | Mixed attribution | Excellent |
| **Risk-First** | Risk management focus | Risk-centric thinking | Risk team ownership | Clear risk attribution | Good |
| **Portfolio-First** | Asset allocation research | Portfolio construction | Portfolio team focus | Clear allocation metrics | Good |

---

# Strategy-First Pattern

## When to Use
- **Simple strategy backtests**: Testing individual trading strategies
- **Strategy development**: Building and iterating on strategy logic
- **Performance comparison**: Clear A/B testing between strategies
- **Intuitive workflows**: Teams familiar with traditional trading approaches
- **Independent development**: Multiple strategy teams working in parallel

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    STRATEGY-FIRST BACKTEST CONTAINER                          │
│  (Organizes around trading strategies as primary units)                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                          SHARED DATA LAYER                             │  │
│  │  ┌────────────────┐    ┌─────────────────────────────────────────────┐ │  │
│  │  │ Market Data    │───▶│          Indicator Engine                   │ │  │
│  │  │ Streamer       │    │  • All technical indicators                 │ │  │
│  │  └────────────────┘    │  • Shared computation                       │ │  │
│  │                        └─────────────────────────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                         │                                     │
│                                         │ Market Data + Indicators            │
│                                         ▼                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                       STRATEGY CONTAINERS                               │  │
│  │  (Each strategy owns its complete logic stack)                         │  │
│  │                                                                         │  │
│  │  ┌─────────────────────────┐    ┌─────────────────────────────────────┐ │  │
│  │  │  Momentum Strategy      │    │  Pattern Strategy Container         │ │  │
│  │  │  Container              │    │                                     │ │  │
│  │  │                         │    │ ┌─────────────────────────────────┐ │ │  │
│  │  │ ┌─────────────────────┐ │    │ │    Pattern Recognition          │ │ │  │
│  │  │ │  Regime Detection   │ │    │ │    • Chart patterns             │ │ │  │
│  │  │ │  • HMM Bull/Bear    │ │    │ │    • Support/Resistance         │ │ │  │
│  │  │ │  • Strategy-specific│ │    │ │    • Breakout detection         │ │ │  │
│  │  │ └─────────────────────┘ │    │ └─────────────────────────────────┘ │ │  │
│  │  │                         │    │                                     │ │  │
│  │  │ ┌─────────────────────┐ │    │ ┌─────────────────────────────────┐ │ │  │
│  │  │ │  Signal Generator   │ │    │ │    Signal Generator             │ │ │  │
│  │  │ │  • Entry/Exit logic │ │    │ │    • Pattern-based entries     │ │ │  │
│  │  │ │  • Momentum signals │ │    │ │    • Breakout confirmations     │ │ │  │
│  │  │ └─────────────────────┘ │    │ └─────────────────────────────────┘ │ │  │
│  │  │                         │    │                                     │ │  │
│  │  │ ┌─────────────────────┐ │    │ ┌─────────────────────────────────┐ │ │  │
│  │  │ │   Risk Management   │ │    │ │    Risk Management              │ │ │  │
│  │  │ │  • Strategy-specific│ │    │ │    • Pattern-specific rules     │ │ │  │
│  │  │ │  • Position sizing  │ │    │ │    • Breakout position sizing   │ │ │  │
│  │  │ └─────────────────────┘ │    │ └─────────────────────────────────┘ │ │  │
│  │  │                         │    │                                     │ │  │
│  │  │ ┌─────────────────────┐ │    │ ┌─────────────────────────────────┐ │ │  │
│  │  │ │  Portfolio Config   │ │    │ │    Portfolio Config             │ │ │  │
│  │  │ │  • Symbols: AAPL,   │ │    │ │    • Symbols: SPY, QQQ          │ │ │  │
│  │  │ │    GOOGL, MSFT      │ │    │ │    • Allocation: Equal weight   │ │ │  │
│  │  │ │  • Allocation: 60%  │ │    │ │    • Allocation: 40%            │ │ │  │
│  │  │ └─────────────────────┘ │    │ └─────────────────────────────────┘ │ │  │
│  │  └─────────────────────────┘    └─────────────────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                         │                                     │
│                                         │ Strategy Signals                    │
│                                         ▼                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                      EXECUTION AGGREGATOR                              │  │
│  │  • Collects signals from all strategies                                │  │
│  │  • Resolves signal conflicts                                           │  │
│  │  • Aggregates portfolio allocations                                    │  │
│  │  • Executes trades and tracks performance                              │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

## YAML Configuration Example

```yaml
# Strategy-First Pattern Configuration
workflow:
  name: "multi_strategy_backtest"
  pattern: "strategy_first"
  
data:
  source: "csv"
  file_path: "data/SPY_1m.csv"
  symbols: ["SPY", "AAPL", "GOOGL", "MSFT", "QQQ"]

strategies:
  - name: "momentum_strategy"
    type: "momentum"
    allocation: 0.6
    symbols: ["AAPL", "GOOGL", "MSFT"]
    parameters:
      lookback_period: 20
      rsi_period: 14
    risk:
      max_position_pct: 5.0
      stop_loss_pct: 2.0
    regime_detection:
      type: "hmm"
      parameters:
        n_states: 3
        
  - name: "pattern_strategy" 
    type: "pattern"
    allocation: 0.4
    symbols: ["SPY", "QQQ"]
    parameters:
      pattern_types: ["breakout", "support_resistance"]
    risk:
      max_position_pct: 8.0
      stop_loss_pct: 3.0
    regime_detection:
      type: "pattern_based"
      
execution:
  mode: "backtest"
  initial_capital: 100000
  commission: 1.0
```

## Advantages

**1. Intuitive Mental Model**
- Aligns with how traders think: "I have a momentum strategy and a pattern strategy"
- Clear ownership: each strategy container owns its complete logic
- Easy to understand signal attribution and performance

**2. Independent Development**
- Strategy teams can work independently
- Different parameter sets and risk rules per strategy
- Clear A/B testing: Strategy A vs Strategy B
- Simple to add/remove strategies

**3. Clear Performance Attribution**
```python
# Easy to track performance by strategy
results = {
    "momentum_strategy": {
        "sharpe_ratio": 1.2,
        "max_drawdown": 0.08,
        "total_trades": 45
    },
    "pattern_strategy": {
        "sharpe_ratio": 0.9, 
        "max_drawdown": 0.12,
        "total_trades": 23
    }
}
```

**4. Simplified Debugging**
- When something goes wrong, you know which strategy caused it
- Clear signal lineage from strategy to execution
- Easier to isolate and fix strategy-specific issues

## Use Cases

### Simple Strategy Development
```yaml
# Single strategy development and testing
strategies:
  - name: "rsi_mean_reversion"
    type: "mean_reversion"
    allocation: 1.0
    parameters:
      rsi_period: 14
      oversold_threshold: 30
      overbought_threshold: 70
```

### Strategy Comparison Studies
```yaml
# Compare multiple momentum approaches
strategies:
  - name: "sma_crossover"
    type: "momentum"
    parameters: {fast_period: 10, slow_period: 30}
  - name: "rsi_momentum"  
    type: "momentum"
    parameters: {rsi_period: 14, momentum_threshold: 50}
  - name: "macd_momentum"
    type: "momentum" 
    parameters: {fast_period: 12, slow_period: 26, signal_period: 9}
```

## Limitations

**1. Potential Duplication**
- Each strategy may implement similar regime detection logic
- Risk management code may be duplicated across strategies
- Indicator calculations might be repeated

**2. Complex Signal Aggregation**
- Harder to implement sophisticated ensemble methods
- Signal conflicts between strategies can be difficult to resolve
- Limited ability to dynamically weight strategies based on market conditions

---

# Classifier-First Pattern (Current ADMF-PC)

## When to Use
- **Systematic research**: Testing all combinations of classifiers, risk profiles, and strategies
- **Regime-based optimization**: Different strategy parameters for different market regimes
- **Combinatorial parameter studies**: 3×3×5×20 parameter space exploration
- **Academic research**: Publications requiring systematic methodology
- **Advanced ensemble methods**: Dynamic strategy weighting based on regime confidence

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                   CLASSIFIER-FIRST BACKTEST CONTAINER                         │
│  (Organizes around market regime classification as primary driver)            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                          SHARED DATA LAYER                             │  │
│  │  [Same as Strategy-First - shared indicator computation]                │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                         │                                     │
│                                         ▼                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                    CLASSIFIER CONTAINERS                                │  │
│  │  (Primary organizational unit - classifiers own downstream logic)      │  │
│  │                                                                         │  │
│  │  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐│  │
│  │  │   HMM Classifier Container      │  │ Pattern Classifier Container    ││  │
│  │  │                                 │  │                                 ││  │
│  │  │ ┌─────────────────────────────┐ │  │ ┌─────────────────────────────┐ ││  │
│  │  │ │    HMM Regime Engine        │ │  │ │  Pattern Recognition Engine │ ││  │
│  │  │ │    • Bull/Bear/Neutral      │ │  │ │  • Chart patterns           │ ││  │
│  │  │ │    • Confidence scoring     │ │  │ │  • Pattern strength scoring │ ││  │
│  │  │ └─────────────────────────────┘ │  │ └─────────────────────────────┘ ││  │
│  │  │               │                 │  │               │                 ││  │
│  │  │               ▼                 │  │               ▼                 ││  │
│  │  │ ┌─────────────────────────────┐ │  │ ┌─────────────────────────────┐ ││  │
│  │  │ │      Risk Container Pool    │ │  │ │     Risk Container Pool     │ ││  │
│  │  │ │                             │ │  │ │                             │ ││  │
│  │  │ │  ┌─────────────────────────┐│ │  │ │ ┌─────────────────────────┐ │ ││  │
│  │  │ │  │ Conservative Risk       ││ │  │ │ │ Balanced Risk           │ │ ││  │
│  │  │ │  │ • Max 2% per position   ││ │  │ │ │ • Max 3% per position   │ │ ││  │
│  │  │ │  │ • 10% total exposure    ││ │  │ │ │ • 20% total exposure    │ │ ││  │
│  │  │ │  │   ┌─────────────────┐   ││ │  │ │ │   ┌─────────────────┐   │ │ ││  │
│  │  │ │  │   │ Portfolio A     │   ││ │  │ │ │   │ Portfolio C     │   │ │ ││  │
│  │  │ │  │   │ $50K allocation │   ││ │  │ │ │   │ $100K allocation │   │ │ ││  │
│  │  │ │  │   │ ┌─────────────┐ │   ││ │  │ │ │   │ ┌─────────────┐ │   │ │ ││  │
│  │  │ │  │   │ │ Momentum    │ │   ││ │  │ │ │   │ │ Pattern     │ │   │ │ ││  │
│  │  │ │  │   │ │ Strategy    │ │   ││ │  │ │ │   │ │ Strategy    │ │   │ │ ││  │
│  │  │ │  │   │ │ Params Set A│ │   ││ │  │ │ │   │ │ Params Set B│ │   │ │ ││  │
│  │  │ │  │   │ └─────────────┘ │   ││ │  │ │ │   │ └─────────────┘ │   │ │ ││  │
│  │  │ │  │   └─────────────────┘   ││ │  │ │ │   └─────────────────┘   │ │ ││  │
│  │  │ │  └─────────────────────────┘│ │  │ │ └─────────────────────────┘ │ ││  │
│  │  │ └─────────────────────────────┘ │  │ └─────────────────────────────┘ ││  │
│  │  └─────────────────────────────────┘  └─────────────────────────────────┘│  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

## YAML Configuration Example

```yaml
# Classifier-First Pattern Configuration
workflow:
  name: "regime_aware_backtest"
  pattern: "classifier_first"
  
data:
  source: "csv"
  file_path: "data/SPY_1m.csv"
  symbols: ["SPY", "AAPL", "GOOGL"]

classifiers:
  - name: "hmm_classifier"
    type: "hmm"
    parameters:
      n_states: 3
      lookback_period: 20
    risk_profiles:
      - name: "conservative"
        max_position_pct: 2.0
        max_exposure_pct: 10.0
        stop_loss_pct: 1.5
        portfolios:
          - name: "portfolio_a"
            allocation: 50000
            strategies:
              - name: "momentum"
                type: "momentum"
                parameters:
                  # Parameters optimized for HMM Bull regime
                  lookback_period: 20
                  rsi_oversold: 35
                  rsi_overbought: 65
              - name: "mean_reversion"
                type: "mean_reversion" 
                parameters:
                  # Parameters optimized for HMM Neutral regime
                  lookback_period: 15
                  entry_threshold: 0.8
                  
  - name: "pattern_classifier"
    type: "pattern"
    parameters:
      pattern_types: ["breakout", "support_resistance"]
    risk_profiles:
      - name: "balanced"
        max_position_pct: 3.0
        max_exposure_pct: 20.0
        portfolios:
          - name: "portfolio_b"
            allocation: 100000
            strategies:
              - name: "pattern_strategy"
                type: "pattern"
                parameters:
                  # Parameters optimized for pattern detection
                  pattern_strength_threshold: 0.7
                  confirmation_periods: 3
```

## Combinatorial Testing Benefits

**1. Systematic Parameter Space Exploration**
```python
# Test all combinations efficiently
classifiers = ["HMM", "Pattern", "Hybrid"]           # 3 options
risk_profiles = ["Conservative", "Balanced", "Aggressive"]  # 3 options  
portfolios = ["Portfolio_A", "Portfolio_B", "Portfolio_C", "Portfolio_D", "Portfolio_E"]  # 5 options
strategy_params = [param_set_1, param_set_2, ..., param_set_20]  # 20 variations

total_experiments = 3 × 3 × 5 × 20 = 900 experiments
```

**2. Efficient Container Reuse**
```python
# Same risk container used across multiple contexts
conservative_risk = RiskContainer(max_position=2%, total_exposure=10%)

# Reused in multiple classifier contexts without duplication
hmm_classifier.add_risk_container(conservative_risk)
pattern_classifier.add_risk_container(conservative_risk)
hybrid_classifier.add_risk_container(conservative_risk)
```

**3. Clean Performance Attribution by Hierarchy**
```python
# Analyze performance at each level
results = {
    "HMM": {
        "Conservative": {
            "Portfolio_A": {"sharpe": 1.2, "max_dd": 0.08},
            "Portfolio_B": {"sharpe": 1.1, "max_dd": 0.10}
        },
        "Balanced": {
            "Portfolio_A": {"sharpe": 1.4, "max_dd": 0.12},
            "Portfolio_B": {"sharpe": 1.3, "max_dd": 0.15}
        }
    },
    "Pattern": {
        "Conservative": {
            "Portfolio_A": {"sharpe": 0.9, "max_dd": 0.07},
            "Portfolio_B": {"sharpe": 1.0, "max_dd": 0.09}
        }
    }
}

# Easy aggregation by any dimension
best_classifier = max(results.keys(), key=lambda c: avg_sharpe(results[c]))
best_risk_profile = max(risk_profiles, key=lambda r: avg_sharpe_for_risk(r))
```

## Research Questions This Pattern Enables

### Question 1: "Which classifier performs best?"
```python
# Aggregate across all risk profiles and portfolios
classifier_performance = {}
for classifier in ["HMM", "Pattern", "Hybrid"]:
    all_results = flatten_results(results[classifier])
    classifier_performance[classifier] = {
        "avg_sharpe": np.mean([r.sharpe for r in all_results]),
        "avg_max_dd": np.mean([r.max_drawdown for r in all_results]),
        "consistency": np.std([r.sharpe for r in all_results])
    }
```

### Question 2: "How does risk tolerance affect regime classification?"
```python
# Compare same portfolio across different risk profiles
for classifier in classifiers:
    for portfolio in portfolios:
        conservative_result = results[classifier]["Conservative"][portfolio]
        aggressive_result = results[classifier]["Aggressive"][portfolio] 
        
        print(f"{classifier} + {portfolio}:")
        print(f"  Conservative: Sharpe={conservative_result.sharpe:.2f}")
        print(f"  Aggressive: Sharpe={aggressive_result.sharpe:.2f}")
        print(f"  Risk Premium: {aggressive_result.sharpe - conservative_result.sharpe:.2f}")
```

### Question 3: "What's the optimal parameter set for each regime?"
```python
# Optimize strategy parameters within each context
for classifier in classifiers:
    for risk_profile in risk_profiles:
        for portfolio in portfolios:
            context_results = results[classifier][risk_profile][portfolio]
            best_params = optimize_strategy_parameters(context_results)
            
            print(f"Best params for {classifier}+{risk_profile}+{portfolio}:")
            print(f"  {best_params}")
```

## Advantages

**1. Regime-Aware Optimization**
- Different strategy parameters optimized for each market regime
- Prevents strategy conflicts in different market conditions
- More sophisticated regime-based risk management

**2. Computational Efficiency**
- Shared regime classification across multiple strategies
- Avoids duplicate regime detection logic
- Better suited for ensemble methods and meta-strategies

**3. Advanced Research Capabilities**
- Systematic exploration of parameter combinations
- Clean isolation of variables for statistical analysis
- Support for sophisticated ensemble and meta-learning approaches

**4. Container Reuse and Modularity**
- Risk containers reused across classifiers
- Portfolio containers reused across risk profiles
- Strategy containers reused across portfolios
- Efficient memory usage and development time

## Limitations

**1. Complex Mental Model**
- Less intuitive than strategy-first for simple use cases
- Requires understanding of regime-based trading concepts
- Harder to explain to traditional traders

**2. Coupled Development**
- Changes to classifiers can affect downstream components
- More complex dependency management
- Harder to isolate and test individual strategies

**3. Mixed Performance Attribution**
- Harder to attribute performance to individual strategies
- Signal lineage goes through multiple layers
- More complex debugging when things go wrong

---

# Implementation Guide

## Creating Custom Patterns

The ADMF-PC composable container system allows you to implement any organizational pattern as a coordinator workflow:

### 1. Define the Pattern Structure

```python
# src/core/containers/composition_engine.py

strategy_first_pattern = ContainerPattern(
    name="strategy_first",
    description="Strategy-first backtest pattern",
    structure={
        "root": {
            "role": "backtest",
            "children": {
                "data": {"role": "data"},
                "strategies": {
                    "role": "strategy_aggregator",
                    "children": {
                        "momentum_strategy": {
                            "role": "strategy",
                            "children": {
                                "regime_detection": {"role": "classifier"},
                                "risk_management": {"role": "risk"},
                                "portfolio": {"role": "portfolio"}
                            }
                        },
                        "pattern_strategy": {
                            "role": "strategy", 
                            "children": {
                                "pattern_detection": {"role": "classifier"},
                                "risk_management": {"role": "risk"},
                                "portfolio": {"role": "portfolio"}
                            }
                        }
                    }
                },
                "execution": {"role": "execution"}
            }
        }
    }
)
```

### 2. Create Pattern-Specific Coordinator

```python
# src/core/coordinator/strategy_first_coordinator.py

class StrategyFirstCoordinator(BaseCoordinator):
    """Coordinator optimized for strategy-first pattern"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.pattern = "strategy_first"
    
    def setup_containers(self) -> None:
        """Set up containers using strategy-first pattern"""
        
        # Create shared data layer
        data_config = self.config.get('data', {})
        self.data_container = self.create_container('data', data_config)
        
        # Create strategy containers
        strategies_config = self.config.get('strategies', [])
        self.strategy_containers = []
        
        for strategy_config in strategies_config:
            strategy_container = self.create_strategy_container(strategy_config)
            self.strategy_containers.append(strategy_container)
        
        # Create execution aggregator
        execution_config = self.config.get('execution', {})
        self.execution_container = self.create_container('execution', execution_config)
    
    def create_strategy_container(self, config: Dict[str, Any]) -> StrategyContainer:
        """Create a complete strategy container with all sub-components"""
        
        strategy_container = StrategyContainer(config)
        
        # Add regime detection specific to this strategy
        regime_config = config.get('regime_detection', {})
        classifier = self.create_container('classifier', regime_config)
        strategy_container.add_child(classifier)
        
        # Add risk management specific to this strategy  
        risk_config = config.get('risk', {})
        risk_manager = self.create_container('risk', risk_config)
        strategy_container.add_child(risk_manager)
        
        # Add portfolio allocation specific to this strategy
        portfolio_config = config.get('portfolio', {})
        portfolio = self.create_container('portfolio', portfolio_config)
        strategy_container.add_child(portfolio)
        
        return strategy_container
```

### 3. Register the Pattern

```yaml
# config/strategy_first_example.yaml

workflow:
  coordinator_type: "strategy_first"
  pattern: "strategy_first"

data:
  source: "csv"
  file_path: "data/SPY_1m.csv"

strategies:
  - name: "momentum_strategy"
    type: "momentum"
    allocation: 0.6
    regime_detection:
      type: "hmm"
      parameters: {n_states: 3}
    risk:
      max_position_pct: 5.0
      stop_loss_pct: 2.0
    portfolio:
      symbols: ["AAPL", "GOOGL", "MSFT"]
      
  - name: "pattern_strategy"
    type: "pattern"
    allocation: 0.4
    regime_detection:
      type: "pattern_based"
    risk:
      max_position_pct: 8.0
      stop_loss_pct: 3.0
    portfolio:
      symbols: ["SPY", "QQQ"]

execution:
  mode: "backtest"
  initial_capital: 100000
```

## Pattern Selection Guide

### Use Strategy-First When:
- **Simple backtests**: Testing 1-3 strategies
- **Strategy development**: Iterating on strategy logic
- **Team structure**: Independent strategy teams
- **Mental model**: Traders familiar with traditional approaches
- **Debugging priority**: Need clear signal attribution

### Use Classifier-First When:
- **Research focus**: Academic or systematic research
- **Combinatorial testing**: Testing many parameter combinations
- **Regime optimization**: Different parameters per market regime  
- **Ensemble methods**: Advanced signal combination techniques
- **Computational efficiency**: Sharing expensive calculations

### Use Risk-First When:
- **Risk management focus**: Risk team owns the process
- **Regulatory requirements**: Need clear risk attribution
- **Multiple strategies**: With shared risk constraints
- **Capital allocation**: Dynamic risk-based allocation

### Use Portfolio-First When:
- **Asset allocation research**: Testing allocation strategies
- **Multi-asset strategies**: Cross-asset momentum, etc.
- **Rebalancing studies**: Different rebalancing frequencies
- **Benchmark tracking**: Tracking multiple benchmarks

## Performance Characteristics

| Pattern | Memory Usage | CPU Usage | Development Time | Debug Complexity | Research Capability |
|---------|--------------|-----------|------------------|------------------|-------------------|
| Strategy-First | Medium | Medium | Low | Low | Low |
| Classifier-First | High | High | High | High | Very High |
| Risk-First | Medium | Medium | Medium | Medium | Medium |
| Portfolio-First | Medium | Low | Medium | Medium | Medium |

## Conclusion

ADMF-PC's composable container architecture provides the flexibility to implement multiple organizational patterns, each optimized for different use cases:

- **Strategy-First**: Best for simple strategy development and testing
- **Classifier-First**: Best for systematic research and regime-based optimization  
- **Risk-First**: Best for risk-focused workflows
- **Portfolio-First**: Best for asset allocation research

The choice of pattern should be driven by:
1. **Team structure and expertise**
2. **Research objectives** 
3. **Computational requirements**
4. **Development timeline**
5. **Debugging and maintenance needs**

By documenting these patterns and providing implementation examples, teams can choose the organizational approach that best fits their specific requirements while leveraging the full power of the ADMF-PC architecture.