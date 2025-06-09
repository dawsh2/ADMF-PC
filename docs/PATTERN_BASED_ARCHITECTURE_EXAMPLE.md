# Pattern-Based Architecture Example

## The Clean Separation Approach

Instead of mixing container creation with communication setup, we have clean separation:

### 1. Container Factory (core/containers/factory.py)
**Responsibility**: Only creates container hierarchies

```python
# Defines container structure only
simple_backtest = ContainerPattern(
    name="simple_backtest",
    structure={
        "root": {
            "role": "backtest",
            "children": {
                "data": {"role": "data"},
                "features": {"role": "feature"},
                "strategy": {"role": "strategy"},
                "risk": {"role": "risk"},
                "execution": {"role": "execution"},
                "portfolio": {"role": "portfolio"}
            }
        }
    }
)
```

### 2. Communication Factory (core/communication/factory.py)
**Responsibility**: Only creates communication adapters

```python
# Creates adapters based on configuration
adapter = factory.create_adapter('backtest_pipeline', {
    'type': 'pipeline',
    'event_flow': [
        {'from': 'DataContainer', 'to': ['FeatureContainer'], 'events': ['BAR']},
        {'from': 'FeatureContainer', 'to': ['StrategyContainer'], 'events': ['FEATURE']},
        # ... etc
    ]
})
```

### 3. Workflow Manager (coordinator/workflows/workflow_manager.py)
**Responsibility**: Orchestrates both according to named patterns

```python
class WorkflowManager:
    def execute(self, config):
        # 1. Determine pattern
        pattern_name = self._determine_pattern(config)  # e.g., "simple_backtest"
        
        # 2. Create containers using container factory
        containers = self.factory.compose_pattern(
            pattern_name=pattern_name,
            config_overrides=pattern_config
        )
        
        # 3. Setup communication using communication factory
        comm_config = self._get_communication_config(pattern_name, containers)
        adapters = self.adapter_factory.create_adapters_from_config(
            comm_config, container_map
        )
        
        # 4. Start everything
        await containers.start()
        self.adapter_factory.start_all()
```

## Example: "simple_backtest" Pattern

When you request a `simple_backtest` workflow:

### Step 1: Container Creation
WorkflowManager → Container Factory → Creates hierarchy:
```
BacktestContainer (root)
├── DataContainer
├── FeatureContainer 
├── StrategyContainer
├── RiskContainer
├── ExecutionContainer
└── PortfolioContainer
```

### Step 2: Communication Setup
WorkflowManager → Communication Factory → Creates pipeline:
```
Data → Features → Strategy → Risk → Execution → Portfolio
  ↓        ↓        ↓       ↓        ↓         ↓
[BAR]  [FEATURE] [SIGNAL] [ORDER]  [FILL]  [PORTFOLIO]
```

### Step 3: Event Flow
The pipeline adapter routes events according to the pattern:
- `BAR` events: Data → Features & Strategy
- `FEATURE` events: Features → Strategy  
- `SIGNAL` events: Strategy → Risk
- `ORDER` events: Risk → Execution
- `FILL` events: Execution → Portfolio
- `PORTFOLIO` events: Portfolio → Risk (feedback)

## Benefits

1. **Single Pattern Definition**: Each workflow pattern (like "simple_backtest") is defined once in WorkflowManager
2. **Separation of Concerns**: 
   - Container factory only knows about containers
   - Communication factory only knows about adapters
   - Workflow manager coordinates both
3. **Reusable Components**: Same containers can be used in different communication patterns
4. **Easy Testing**: Can test container creation and communication setup independently
5. **Clear Architecture**: No mixing of responsibilities

## Pattern Registry

WorkflowManager maintains pattern definitions:

```python
self._workflow_patterns = {
    'simple_backtest': {
        'description': 'Simple backtest workflow with linear pipeline',
        'container_pattern': 'simple_backtest',  # → Container Factory
        'communication_pattern': 'pipeline'      # → Communication Factory
    },
    'full_backtest': {
        'description': 'Full backtest with classifier and complex routing',
        'container_pattern': 'full_backtest',
        'communication_pattern': 'hierarchical'
    }
}
```

This way, adding a new workflow pattern just requires:
1. Adding container structure to Container Factory (if new)
2. Adding communication config to WorkflowManager
3. No duplication, clear responsibilities