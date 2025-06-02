# Concrete Scenarios Where Event Communication Adapters Pay Off

## Scenario 1: Multi-Phase Research Workflows

### **Problem**: Different phases need different communication patterns

Your research workflow has distinct phases that need different event routing:

```yaml
# Phase 1: Parameter Discovery - Need full pipeline
phase_1_communication:
  adapters:
    - type: "pipeline"
      containers: ["data", "indicators", "strategies", "risk", "execution"]

# Phase 2: Regime Analysis - Need analysis-only routing  
phase_2_communication:
  adapters:
    - type: "pipeline"
      containers: ["results_reader", "regime_analyzer", "report_generator"]
    # Skip execution entirely

# Phase 3: Signal Replay - Need replay-specific routing
phase_3_communication:
  adapters:
    - type: "pipeline"
      containers: ["signal_reader", "ensemble_optimizer", "risk", "execution"]
    # Skip data/indicators/strategies
```

### **BacktestContainer Router Problem**:
```python
class BacktestContainer:
    def _route_events(self, event):
        # This becomes a mess with different phases
        if self.phase == "parameter_discovery":
            self._route_full_pipeline(event)
        elif self.phase == "regime_analysis":
            self._route_analysis_only(event)
        elif self.phase == "signal_replay":
            self._route_replay_pipeline(event)
        # 15 more phases... this gets unwieldy
```

### **Adapter Solution**:
```python
# Clean, declarative configuration per phase
workflow_configs = {
    "parameter_discovery": load_config("phase1_communication.yaml"),
    "regime_analysis": load_config("phase2_communication.yaml"), 
    "signal_replay": load_config("phase3_communication.yaml")
}

# Simple to switch between phases
current_communication = EventCommunicationFactory().create_communication_layer(
    workflow_configs[current_phase], containers
)
```

**Payoff**: Clean separation of complex multi-phase routing vs. growing if-else routing logic.

---

## Scenario 2: Dynamic Strategy Allocation Based on Performance

### **Problem**: Route signals based on real-time performance

You want to dynamically route signals to different risk containers based on strategy performance:

```python
# High-performing strategies get aggressive risk allocation
# Poor-performing strategies get conservative risk allocation
# Mediocre strategies get balanced risk allocation

# This changes every 100 bars based on rolling performance
```

### **BacktestContainer Router Problem**:
```python
class BacktestContainer:
    def _route_strategy_signals(self, signal_event):
        strategy_name = signal_event.source
        recent_performance = self.performance_tracker.get_recent_sharpe(strategy_name)
        
        # This routing logic gets complex and hard to test
        if recent_performance > 1.5:
            target = self.aggressive_risk_container
        elif recent_performance < 0.5:
            target = self.conservative_risk_container
        else:
            target = self.balanced_risk_container
            
        # What if you want different rules for different market regimes?
        # What if you want to test different performance thresholds?
        # This becomes unmaintainable
```

### **Adapter Solution**:
```yaml
# Easy to configure and test different allocation rules
dynamic_allocation:
  adapters:
    - type: "selective"
      source: "strategies"
      rules:
        - condition: "performance.sharpe > 1.5"
          target: "aggressive_risk"
        - condition: "performance.sharpe < 0.5" 
          target: "conservative_risk"
        - condition: "default"
          target: "balanced_risk"

# Easy to A/B test different thresholds
alternative_allocation:
  adapters:
    - type: "selective"
      source: "strategies"
      rules:
        - condition: "performance.sharpe > 2.0"  # Different threshold
          target: "aggressive_risk"
        # ... rest of rules
```

**Payoff**: Easy to test different allocation rules without code changes, complex routing logic becomes configurable.

---

## Scenario 3: Distributed Computing Deployment

### **Problem**: Scale to thousands of containers across multiple machines

You need to run 10,000 parameter combinations, but your laptop can only handle 100 at a time:

```python
# Local machine: 100 containers
# AWS Instance 1: 1000 containers  
# AWS Instance 2: 1000 containers
# AWS Instance 3: 1000 containers
# etc.
```

### **BacktestContainer Router Problem**:
```python
class BacktestContainer:
    def _route_orders(self, order_event):
        # This only works if ExecutionContainer is local
        execution_container = self.execution_container
        execution_container.receive_event(order_event)
        
        # What if ExecutionContainer is on a different machine?
        # Need to rewrite all routing logic for network communication
```

### **Adapter Solution**:
```yaml
# Local configuration
local_communication:
  adapters:
    - type: "pipeline"
      transport: "local"  # Same machine
      containers: ["strategies", "risk", "execution"]

# Distributed configuration  
distributed_communication:
  adapters:
    - type: "pipeline"
      transport: "grpc"   # Different machines
      containers: 
        - "strategies@local"
        - "risk@aws-instance-1" 
        - "execution@aws-instance-2"
```

```python
# Same adapter, different transport
class PipelineCommunicationAdapter:
    def transform_and_forward(self, event, target):
        if self.transport == "local":
            target.receive_event(event)
        elif self.transport == "grpc":
            self.grpc_client.send(target.network_address, event)
```

**Payoff**: Same configuration works locally and distributed, no code changes needed for scaling.

---

## Scenario 4: A/B Testing Communication Patterns

### **Problem**: Test if different communication patterns affect performance

You want to test whether broadcast vs. selective routing affects strategy performance:

```yaml
# Test A: Broadcast all indicator updates to all strategies
test_a_communication:
  adapters:
    - type: "broadcast"
      source: "indicators"
      targets: ["momentum_strategy", "mean_reversion_strategy", "breakout_strategy"]

# Test B: Selective routing based on indicator type
test_b_communication:
  adapters:
    - type: "selective"
      source: "indicators"
      rules:
        - condition: "indicator.type == 'momentum'"
          target: "momentum_strategy"
        - condition: "indicator.type == 'mean_reversion'"
          target: "mean_reversion_strategy"
        - condition: "indicator.type == 'volatility'"
          target: "breakout_strategy"
```

### **BacktestContainer Router Problem**:
```python
class BacktestContainer:
    def __init__(self, config):
        self.communication_mode = config.get('communication_mode', 'broadcast')
    
    def _route_indicators(self, indicator_event):
        if self.communication_mode == 'broadcast':
            # Broadcast logic here
            for strategy in self.strategies:
                strategy.receive_event(indicator_event)
        elif self.communication_mode == 'selective':
            # Selective logic here
            if indicator_event.type == 'momentum':
                self.momentum_strategy.receive_event(indicator_event)
            # ... more if-else logic
            
        # Adding new communication patterns requires code changes
        # Hard to isolate the impact of communication vs strategy logic
```

### **Adapter Solution**:
```python
# Clean A/B test - same code, different configs
results_a = run_backtest("test_a_communication.yaml")
results_b = run_backtest("test_b_communication.yaml")

# Easy to isolate the impact of communication pattern
performance_difference = results_a.sharpe_ratio - results_b.sharpe_ratio
```

**Payoff**: Easy to A/B test communication patterns, clean isolation of variables.

---

## Scenario 5: Complex Ensemble Strategy Routing

### **Problem**: Route signals based on regime, strategy confidence, and market conditions

You have a sophisticated ensemble that routes signals based on multiple criteria:

```yaml
ensemble_routing:
  adapters:
    - type: "selective"
      source: "strategies"
      rules:
        # High confidence momentum signals in bull market
        - condition: "regime == 'BULL' and strategy.type == 'momentum' and signal.confidence > 0.8"
          target: "aggressive_portfolio"
          
        # Any signal in bear market
        - condition: "regime == 'BEAR'"
          target: "defensive_portfolio"
          
        # High confidence mean reversion in neutral market
        - condition: "regime == 'NEUTRAL' and strategy.type == 'mean_reversion' and signal.confidence > 0.7"
          target: "balanced_portfolio"
          
        # Low confidence signals get paper trading
        - condition: "signal.confidence < 0.5"
          target: "paper_trading_portfolio"
          
        # Default case
        - condition: "default"
          target: "conservative_portfolio"
```

### **BacktestContainer Router Problem**:
```python
class BacktestContainer:
    def _route_strategy_signals(self, signal_event):
        # This becomes a nightmare of nested if-else statements
        regime = self.current_regime
        strategy_type = signal_event.strategy_type
        confidence = signal_event.confidence
        
        if regime == 'BULL':
            if strategy_type == 'momentum':
                if confidence > 0.8:
                    target = self.aggressive_portfolio
                else:
                    target = self.balanced_portfolio
            elif strategy_type == 'mean_reversion':
                # ... more nested logic
        elif regime == 'BEAR':
            # ... different logic tree
        # This becomes unmaintainable and hard to test
```

### **Adapter Solution**:
```python
# Rules are declarative and easy to test
selective_adapter = SelectiveCommunicationAdapter()
selective_adapter.load_rules_from_config("ensemble_routing.yaml")

# Easy to unit test individual rules
test_rule = "regime == 'BULL' and strategy.type == 'momentum' and signal.confidence > 0.8"
assert selective_adapter.evaluate_rule(test_rule, test_signal) == True

# Easy to modify rules without touching code
```

**Payoff**: Complex routing logic becomes configurable and testable, easier to modify and debug.

---

## Scenario 6: Live Trading with Different Data Sources

### **Problem**: Live trading needs different event routing than backtesting

In backtesting, you replay historical data. In live trading, you need real-time data from multiple sources:

```yaml
# Backtesting communication
backtest_communication:
  adapters:
    - type: "pipeline"
      containers: ["csv_data_reader", "indicators", "strategies", "simulated_execution"]

# Live trading communication  
live_communication:
  adapters:
    - type: "broadcast"
      source: "market_data_feed"
      targets: ["indicators", "risk_monitor", "position_tracker"]
    - type: "pipeline"
      containers: ["indicators", "strategies", "live_risk_manager", "broker_execution"]
    - type: "selective"
      source: "strategies"
      rules:
        - condition: "signal.urgency == 'high'"
          target: "fast_execution_queue"
        - condition: "signal.urgency == 'normal'"
          target: "normal_execution_queue"
```

### **BacktestContainer Router Problem**:
```python
class BacktestContainer:
    def __init__(self, config):
        self.mode = config.get('mode', 'backtest')  # 'backtest' or 'live'
    
    def _route_events(self, event):
        if self.mode == 'backtest':
            # Backtest routing logic
            self._route_backtest_pipeline(event)
        elif self.mode == 'live':
            # Completely different routing logic
            self._route_live_pipeline(event)
            
        # Maintaining two different routing systems in one class
        # Live vs backtest differences leak into core logic
```

### **Adapter Solution**:
```python
# Same container code, different communication configs
if config.mode == 'backtest':
    communication_config = load_config("backtest_communication.yaml")
else:
    communication_config = load_config("live_communication.yaml")

# Containers don't need to know about mode differences
communication_layer = EventCommunicationFactory().create_communication_layer(
    communication_config, containers
)
```

**Payoff**: Clean separation between container logic and deployment environment, same code works in backtest and live.

---

## When BacktestContainer Router is Still Better

The adapter approach is **overkill** for:

1. **Simple linear pipelines**: Data → Strategy → Risk → Execution
2. **Single organizational pattern**: Only using strategy-first
3. **No performance routing**: All signals go through same path
4. **Local-only deployment**: Never need distributed containers
5. **Stable communication patterns**: Routing logic doesn't change

## Summary: Adapter Payoff Threshold

**Use adapters when you need:**
- ✅ **Multiple communication patterns** for different phases/modes
- ✅ **Dynamic routing** based on performance/regime/confidence
- ✅ **Distributed deployment** across multiple machines
- ✅ **A/B testing** of communication patterns
- ✅ **Complex routing rules** that change frequently
- ✅ **Environment-specific routing** (backtest vs live vs research)

**Stick with BacktestContainer router when you have:**
- ✅ **Simple, stable routing** that doesn't change
- ✅ **Single deployment environment** (local only)
- ✅ **Linear pipeline flows** without complex branching
- ✅ **Development speed priority** over flexibility

The key insight is that adapters pay off when **routing logic becomes complex enough to be its own concern** rather than simple glue code in BacktestContainer.
