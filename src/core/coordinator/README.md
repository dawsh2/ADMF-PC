# Coordinator Module

The Coordinator is the central orchestration component for the ADMF-PC system. It serves as the single entry point for all high-level operations, ensuring reproducibility and consistency across all workflows.

## Overview

The Coordinator:
- Reads and validates workflow configurations (YAML or programmatic)
- Sets up shared infrastructure (indicators, data feeds)
- Creates isolated containers for each workflow/trial
- Delegates execution to specialized workflow managers
- Aggregates results in a standardized format
- Ensures proper cleanup of resources

## Key Features

### 1. Single Entry Point
All workflows (optimization, backtest, live trading, etc.) go through the Coordinator:

```python
coordinator = Coordinator(shared_services={...})
result = await coordinator.execute_workflow(config)
```

### 2. Configuration-Driven
Complex workflows are defined in YAML, not code:

```yaml
workflow:
  type: "optimization"
  optimization_config:
    algorithm: "genetic"
    objective: "maximize_sharpe"
```

### 3. Container Isolation
Each workflow runs in its own container with:
- Isolated event bus (no cross-contamination)
- Independent component instances
- Separate state management
- Clean resource boundaries

### 4. Workflow Types

- **Optimization**: Multi-stage parameter optimization with regime awareness
- **Backtest**: Historical simulation with realistic execution
- **Live Trading**: Real-time trading with risk controls
- **Analysis**: Market analysis and performance evaluation
- **Validation**: Strategy validation and robustness testing

### 5. Reproducibility

The single execution path ensures:
- Consistent initialization sequence
- Same default values applied
- Controlled random seeds
- Complete configuration capture
- Automatic audit trail

## Architecture

```
Coordinator
    ├── Infrastructure Setup
    │   ├── Shared Indicators
    │   ├── Data Feeds
    │   └── Computation Resources
    │
    ├── Container Management
    │   ├── Workflow Containers
    │   ├── Component Isolation
    │   └── Resource Cleanup
    │
    └── Workflow Managers
        ├── OptimizationManager
        ├── BacktestManager
        ├── LiveTradingManager
        ├── AnalysisManager
        └── ValidationManager
```

## Usage Examples

### Simple Backtest

```python
from admf.core.coordinator import Coordinator, WorkflowConfig, WorkflowType

# Create coordinator
coordinator = Coordinator(
    shared_services={
        'market_data': market_data_provider
    }
)

# Configure backtest
config = WorkflowConfig(
    workflow_type=WorkflowType.BACKTEST,
    data_config={...},
    backtest_config={
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'strategy': {...}
    }
)

# Execute
result = await coordinator.execute_workflow(config)
```

### Complex Optimization

```yaml
# optimization_workflow.yaml
workflow:
  type: "optimization"
  
  optimization_config:
    # Multi-stage optimization
    stages:
      - name: "regime_detector_optimization"
        component: "RegimeDetector"
        algorithm: "grid"
        
      - name: "strategy_optimization_by_regime"
        component: "TrendStrategy"
        algorithm: "bayesian"
        regime_specific: true
```

```python
# Load and execute
coordinator = Coordinator(config_path="optimization_workflow.yaml")
result = await coordinator.execute_workflow()
```

## Workflow Lifecycle

1. **Configuration Validation**: Ensures all required fields are present
2. **Container Creation**: Isolated environment for the workflow
3. **Infrastructure Setup**: Shared resources initialization
4. **Phase Execution**: Sequential execution of workflow phases
5. **Result Aggregation**: Collecting and formatting results
6. **Resource Cleanup**: Proper teardown of containers and resources

## Integration Points

### With Container System
- Each workflow gets its own `UniversalScopedContainer`
- Components are created with proper isolation
- Dependencies are resolved within containers

### With Event System
- Container-isolated event buses
- Workflow lifecycle events
- Phase transition notifications

### With Component System
- Protocol-based component creation
- Capability composition
- Automatic optimization support

## Best Practices

1. **Always use configuration files** for complex workflows
2. **Specify random seeds** for reproducibility
3. **Set appropriate timeouts** for long-running workflows
4. **Monitor active workflows** using status methods
5. **Handle errors gracefully** with proper cleanup

## Advanced Features

### Parallel Execution
```python
# Run multiple workflows concurrently
tasks = [
    coordinator.execute_workflow(config1),
    coordinator.execute_workflow(config2),
    coordinator.execute_workflow(config3)
]
results = await asyncio.gather(*tasks)
```

### Workflow Monitoring
```python
# Check active workflows
active = await coordinator.list_active_workflows()

# Get detailed status
status = await coordinator.get_workflow_status(workflow_id)

# Cancel if needed
await coordinator.cancel_workflow(workflow_id)
```

### Custom Workflow Managers
Extend `BaseWorkflowManager` to add new workflow types:

```python
class CustomWorkflowManager(BaseWorkflowManager):
    def get_execution_phases(self) -> List[WorkflowPhase]:
        return [...]
    
    async def execute_phase(self, phase, config, context):
        # Custom phase logic
        pass
```

## Troubleshooting

### Common Issues

1. **Configuration Validation Errors**
   - Check required fields in workflow config
   - Ensure data sources are properly configured

2. **Container Creation Failures**
   - Verify shared services are available
   - Check component class registrations

3. **Resource Cleanup Issues**
   - Always call `coordinator.shutdown()`
   - Use context managers for automatic cleanup

### Debugging Tips

- Enable detailed logging: `log_level: "DEBUG"`
- Monitor container statistics
- Track event flow through containers
- Use validation workflows to test configurations

## Future Enhancements

- GPU resource management
- Distributed execution support
- Real-time progress visualization
- Advanced caching mechanisms
- Workflow templates and presets