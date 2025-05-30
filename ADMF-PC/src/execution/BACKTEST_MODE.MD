# Backtest Mode Implementation Guide

This document describes how the backtest mode should create the nested container hierarchy described in BACKTEST.MD.

## Current State vs Target Architecture

### Current Implementation (What We Have)
```
Coordinator
    │
    └── ContainerLifecycleManager.create_and_start_container("backtest")
            │
            └── Generic Backtest Container
                    ├── Portfolio
                    ├── RiskManager
                    ├── Strategy
                    └── ExecutionSimulator
```

### Target Architecture (From BACKTEST.MD)
```
Coordinator
    │
    └── BacktestContainer (Top-level container for entire backtest)
            │
            ├── DataStreamer
            ├── IndicatorHub (Shared computation)
            │
            ├── Classifier Container (e.g., HMM)
            │   ├── Classifier Component
            │   │
            │   └── Risk & Portfolio Containers (Multiple)
            │       ├── Conservative Risk & Portfolio
            │       │   ├── Risk Manager
            │       │   ├── Portfolio
            │       │   └── Strategies
            │       │       ├── Momentum Strategy
            │       │       └── Mean Reversion Strategy
            │       │
            │       └── Aggressive Risk & Portfolio
            │           ├── Risk Manager
            │           ├── Portfolio
            │           └── Strategies
            │               └── Breakout Strategy
            │
            └── BacktestEngine (Executes trades)
```

## Implementation Requirements

### 1. BacktestContainerFactory
Should create the complete hierarchy following BACKTEST.MD:

```python
class BacktestContainerFactory:
    """Creates standardized backtest containers per BACKTEST.MD."""
    
    @staticmethod
    def create_instance(config: BacktestConfig) -> BacktestContainer:
        """
        Creates a backtest container with the full hierarchy:
        1. Data layer (DataStreamer)
        2. Shared computation (IndicatorHub)
        3. Classifier layer (with Risk & Portfolio sub-containers)
        4. Execution layer (BacktestEngine)
        """
        # ALWAYS creates components in this exact order
        container = BacktestContainer(config.container_id)
        
        # 1. Data layer
        container.add_component(DataStreamer(config.data_config))
        container.add_component(IndicatorHub(config.indicator_config))
        
        # 2. Classifier layer (with nested containers)
        for classifier_config in config.classifiers:
            classifier_container = container.create_subcontainer(
                f"{config.container_id}_{classifier_config.type}"
            )
            
            # Add classifier
            classifier_container.add_component(
                create_classifier(classifier_config)
            )
            
            # Add Risk & Portfolio sub-containers
            for risk_profile in classifier_config.risk_profiles:
                risk_container = classifier_container.create_subcontainer(
                    f"{classifier_container.id}_{risk_profile.name}"
                )
                
                # Add risk and portfolio components
                risk_container.add_component(RiskManager(risk_profile.params))
                risk_container.add_component(Portfolio(risk_profile.capital))
                
                # Add strategies
                for strategy_config in risk_profile.strategies:
                    risk_container.add_component(
                        create_strategy(strategy_config)
                    )
        
        # 3. Execution layer
        container.add_component(BacktestEngine(config.execution_config))
        
        # 4. Wire event buses
        container.wire_event_flows()
        
        return container
```

### 2. Event Flow Implementation

The event flow should follow the pattern from BACKTEST.MD:

```python
def wire_event_flows(self):
    """Wire up event flows per BACKTEST.MD."""
    
    # 1. Market Data Flow
    # DataStreamer -> IndicatorHub
    self.event_bus.connect(
        source=self.data_streamer,
        event_type=EventType.BAR,
        target=self.indicator_hub.process_market_data
    )
    
    # 2. Indicator Events
    # IndicatorHub -> All Classifiers
    for classifier in self.classifiers:
        self.event_bus.connect(
            source=self.indicator_hub,
            event_type=EventType.INDICATOR,
            target=classifier.process_indicator_event
        )
    
    # 3. Signal Flow
    # Strategies -> Risk & Portfolio -> BacktestEngine
    for risk_container in self.get_all_risk_containers():
        # Strategies emit signals
        for strategy in risk_container.strategies:
            risk_container.event_bus.connect(
                source=strategy,
                event_type=EventType.SIGNAL,
                target=risk_container.risk_manager.process_signal
            )
        
        # Risk & Portfolio emits orders
        self.event_bus.connect(
            source=risk_container.risk_manager,
            event_type=EventType.ORDER,
            target=self.backtest_engine.process_order
        )
    
    # 4. Fill Events
    # BacktestEngine -> Risk & Portfolio containers
    for risk_container in self.get_all_risk_containers():
        self.event_bus.connect(
            source=self.backtest_engine,
            event_type=EventType.FILL,
            target=risk_container.portfolio.process_fill
        )
```

### 3. Coordinator Integration

The Coordinator should use the specialized factory:

```python
async def _create_backtest_workflow_container(
    self,
    workflow_id: str,
    config: WorkflowConfig,
    context: ExecutionContext
) -> str:
    """Create a backtest container following BACKTEST.MD architecture."""
    
    # Convert workflow config to backtest config
    backtest_config = BacktestConfig(
        container_id=f"backtest_{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        data_config=config.data_config,
        indicator_config=self._extract_indicators(config),
        classifiers=self._extract_classifiers(config),
        execution_config=config.backtest_config.get('execution', {})
    )
    
    # Use the specialized factory
    container = BacktestContainerFactory.create_instance(backtest_config)
    
    # Register with lifecycle manager
    self.container_manager.register_container(container)
    
    return container.container_id
```

## Benefits of Proper Implementation

### 1. **Complete Isolation**
- Each backtest runs in its own container
- No state leakage between backtests
- Clean lifecycle management

### 2. **Shared Computation**
- IndicatorHub computes indicators once
- All strategies receive same indicator values
- Significant performance improvement

### 3. **Flexible Risk Management**
- Multiple risk profiles in same backtest
- Each with own capital allocation
- Different position sizing rules

### 4. **Regime-Aware Execution**
- Classifiers provide market context
- Strategies adapt to regime
- Risk limits adjust dynamically

### 5. **Standardized Creation**
- Same setup process every time
- Reproducible results
- Easy to debug

## Migration Path

### Phase 1: Create Factory
1. Implement `BacktestContainerFactory`
2. Test with simple single-classifier setup
3. Verify event flow works correctly

### Phase 2: Update Coordinator
1. Modify `_create_workflow_container` for backtest mode
2. Extract classifier and strategy configs from workflow
3. Use factory to create proper hierarchy

### Phase 3: Add Missing Components
1. Implement `IndicatorHub` integration
2. Add classifier containers
3. Support multiple risk profiles

### Phase 4: Validation
1. Compare against BACKTEST.MD architecture
2. Verify all event flows work
3. Test parallel execution

## Example Configuration

```yaml
workflow_type: backtest

data:
  start_date: "2023-01-01"
  end_date: "2023-12-31"
  symbols: ["AAPL", "GOOGL", "MSFT"]

indicators:
  - name: "SMA_20"
    type: "SMA"
    parameters: {period: 20}
  - name: "RSI_14"
    type: "RSI"
    parameters: {period: 14}

classifiers:
  - type: "hmm"
    parameters:
      n_states: 3
    risk_profiles:
      - name: "conservative"
        capital_allocation: 300000
        risk_parameters:
          max_position_size: 0.02
          max_total_exposure: 0.1
        strategies:
          - name: "momentum"
            class: "MomentumStrategy"
            symbols: ["AAPL", "GOOGL"]
            parameters:
              fast_period: 10
              slow_period: 30
              
      - name: "aggressive"
        capital_allocation: 500000
        risk_parameters:
          max_position_size: 0.05
          max_total_exposure: 0.3
        strategies:
          - name: "breakout"
            class: "BreakoutStrategy"
            symbols: ["AAPL", "GOOGL", "MSFT"]
            parameters:
              breakout_period: 20

execution:
  slippage_model: "fixed"
  slippage_value: 0.01
  commission_model: "percentage"
  commission_value: 0.001
```

## Testing the Implementation

### 1. Container Creation Test
```python
def test_backtest_container_creation():
    """Test that container hierarchy matches BACKTEST.MD."""
    config = load_test_config()
    container = BacktestContainerFactory.create_instance(config)
    
    # Verify structure
    assert container.has_component("DataStreamer")
    assert container.has_component("IndicatorHub")
    assert container.has_component("BacktestEngine")
    
    # Verify classifier containers
    classifier_containers = container.get_subcontainers_by_type("classifier")
    assert len(classifier_containers) == 1  # HMM
    
    # Verify risk containers
    hmm_container = classifier_containers[0]
    risk_containers = hmm_container.get_subcontainers_by_type("risk_portfolio")
    assert len(risk_containers) == 2  # Conservative and Aggressive
    
    # Verify strategies
    conservative = risk_containers[0]
    assert len(conservative.get_components_by_type(Strategy)) == 1  # Momentum
    
    aggressive = risk_containers[1]
    assert len(aggressive.get_components_by_type(Strategy)) == 1  # Breakout
```

### 2. Event Flow Test
```python
def test_event_flow():
    """Test that events flow correctly through the hierarchy."""
    container = create_test_container()
    
    # Track events
    events_received = []
    
    def track_event(event):
        events_received.append((event.event_type, event.source_id))
    
    # Subscribe to all event types
    container.event_bus.subscribe(EventType.BAR, track_event)
    container.event_bus.subscribe(EventType.INDICATOR, track_event)
    container.event_bus.subscribe(EventType.SIGNAL, track_event)
    container.event_bus.subscribe(EventType.ORDER, track_event)
    container.event_bus.subscribe(EventType.FILL, track_event)
    
    # Simulate market data
    container.process_market_data(test_market_data)
    
    # Verify event sequence
    assert events_received[0][0] == EventType.BAR
    assert events_received[1][0] == EventType.INDICATOR
    assert events_received[2][0] == EventType.SIGNAL
    assert events_received[3][0] == EventType.ORDER
    assert events_received[4][0] == EventType.FILL
```

## Conclusion

The current implementation needs to be enhanced to create the full nested container hierarchy from BACKTEST.MD. This will provide:

1. Better organization and separation of concerns
2. Shared computation through IndicatorHub
3. Support for multiple classifiers and risk profiles
4. Standardized creation process
5. Clean event flow architecture

The implementation should follow the factory pattern to ensure every backtest is created identically, supporting the massive parallelization goals of the ADMF-PC system.