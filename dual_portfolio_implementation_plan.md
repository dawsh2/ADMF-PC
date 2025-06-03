# Dual Portfolio Implementation Plan

## Quick Implementation Approach

### Step 1: Modify Pipeline Adapter for Namespaces

```python
# In pipeline_adapter.py
class PipelineAdapter:
    def __init__(self, name: str, config: Dict[str, Any], event_bus):
        self.pipelines = {}  # namespace -> pipeline
        
    def add_pipeline(self, namespace: str, containers: List[str]):
        """Add a named pipeline."""
        self.pipelines[namespace] = {
            'containers': containers,
            'sequence': self._determine_sequence(containers)
        }
    
    def route_event(self, event: Event):
        """Route event to appropriate pipeline based on namespace."""
        namespace = getattr(event, 'namespace', 'default')
        if namespace in self.pipelines:
            self._process_pipeline(event, self.pipelines[namespace])
```

### Step 2: Extend Event Class

```python
# In events/types.py
@dataclass
class Event:
    event_type: EventType
    timestamp: datetime
    source: str
    payload: Dict[str, Any]
    namespace: Optional[str] = None  # Add namespace field
```

### Step 3: Create TradingSystemContainer

```python
# New container that encapsulates a complete trading system
class TradingSystemContainer(BaseComposableContainer):
    """Container that holds a complete trading pipeline."""
    
    def __init__(self, config: Dict[str, Any], namespace: str):
        super().__init__(
            role=ContainerRole.SYSTEM,
            name=f"TradingSystem_{namespace}",
            config=config
        )
        self.namespace = namespace
        
    async def _initialize_self(self):
        # Create all sub-containers
        self.indicator_container = IndicatorContainer(
            config=self.config.get('indicators', {}),
            container_id=f"indicator_{self.namespace}"
        )
        
        self.strategy_container = StrategyContainer(
            config=self.config.get('strategy', {}),
            container_id=f"strategy_{self.namespace}"
        )
        
        self.risk_container = RiskContainer(
            config=self.config.get('risk', {}),
            container_id=f"risk_{self.namespace}"
        )
        
        self.execution_container = ExecutionContainer(
            config=self.config.get('execution', {}),
            container_id=f"execution_{self.namespace}"
        )
        
        self.portfolio_container = PortfolioContainer(
            config=self.config.get('portfolio', {}),
            container_id=f"portfolio_{self.namespace}"
        )
        
        # Add all as children
        for container in [self.indicator_container, self.strategy_container,
                         self.risk_container, self.execution_container,
                         self.portfolio_container]:
            self.add_child_container(container)
```

### Step 4: Modify ComposableWorkflowManager

```python
# In composable_workflow_manager_pipeline.py
async def _execute_multi_portfolio_backtest(self, config):
    """Execute backtest with multiple independent portfolios."""
    
    # Create shared data container
    data_container = DataContainer(config['data'])
    
    # Create trading systems
    trading_systems = []
    for system_config in config['trading_systems']:
        namespace = system_config['name']
        system = TradingSystemContainer(system_config, namespace)
        await system.initialize()
        trading_systems.append(system)
    
    # Set up routing
    self._setup_multi_portfolio_routing(data_container, trading_systems)
    
    # Start all systems
    await data_container.start()
    for system in trading_systems:
        await system.start()
```

### Step 5: Configuration Example

```yaml
# config/dual_portfolio_test.yaml
workflow:
  type: "multi_portfolio_backtest"

data:
  source: "csv"
  file_path: "data/SPY_1m.csv"
  symbols: ["SPY"]
  max_bars: 100

trading_systems:
  - name: "momentum"
    portfolio:
      initial_capital: 50000
    
    strategy:
      type: "momentum"
      parameters:
        lookback_period: 20
        momentum_threshold: 0.0002
    
    risk:
      position_sizers:
        - type: "fixed"
          size: 1000

  - name: "mean_reversion"
    portfolio:
      initial_capital: 50000
    
    strategy:
      type: "mean_reversion"
      parameters:
        lookback_period: 15
        entry_threshold: 1.0
```

## Minimal Changes Approach

If we want to test this with minimal changes to existing code:

### Option 1: Run Two Separate Backtests
```bash
# Run momentum strategy
python main.py --config config/momentum_only.yaml --output results/momentum/

# Run mean reversion strategy  
python main.py --config config/mean_reversion_only.yaml --output results/mean_reversion/

# Combine reports
python scripts/combine_reports.py results/momentum/ results/mean_reversion/
```

### Option 2: Modify Existing Multi-Strategy
1. Change RiskContainer to track portfolios by strategy name
2. Change PortfolioContainer to maintain separate states
3. Change reporting to generate per-strategy reports

```python
# In PortfolioContainer
class PortfolioContainer:
    def __init__(self, config):
        # Multiple portfolio states
        self.portfolio_states = {}
        for strategy in config.get('strategies', []):
            name = strategy['name']
            capital = config['initial_capital'] * strategy.get('allocation', 1.0)
            self.portfolio_states[name] = PortfolioState(capital)
    
    async def _handle_fill_event(self, event):
        # Route to correct portfolio based on signal source
        strategy_name = event.payload.get('strategy_name')
        if strategy_name in self.portfolio_states:
            portfolio = self.portfolio_states[strategy_name]
            # Update specific portfolio
```

## Recommendation

**Start with Option 1 (Run Separate Backtests)** because:
1. No code changes required
2. Complete isolation guaranteed
3. Easy to compare results
4. Can automate with shell script

**Then move to TradingSystemContainer approach** for:
1. True parallel execution
2. Shared data streaming
3. Unified reporting interface
4. Better resource efficiency

Would you like me to:
1. Create the shell script for running separate backtests?
2. Implement the TradingSystemContainer approach?
3. Modify the existing containers for multi-portfolio support?