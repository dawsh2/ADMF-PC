# Coordinator Workflows

This directory contains workflow orchestration logic for ADMF-PC. All container creation, workflow management, and execution modes live here as part of the coordinator's responsibilities.

## Directory Structure

```
workflows/
├── backtest_workflow.py      # Backtest workflow manager (WorkflowManager protocol)
├── container_factories.py    # Factory functions for creating containers
├── containers_pipeline.py    # Pipeline-based container orchestration  
└── modes/                    # Execution mode configurations
    ├── backtest.py          # Standard backtest mode
    ├── signal_generation.py # Signal generation without execution
    └── signal_replay.py     # Fast signal replay for optimization
```

## Key Components

### BacktestWorkflowManager
- Implements the `WorkflowManager` protocol
- Creates container hierarchy following BACKTEST.MD architecture
- Manages container lifecycle (initialize, start, stop)
- Coordinates data streaming and event flow

### Container Factories
Factory functions for creating protocol-based containers:
- `create_backtest_container()` - Root coordination container
- `create_data_container()` - Data loading and streaming
- `create_indicator_container()` - Indicator computation
- `create_strategy_container()` - Strategy signal generation
- `create_risk_container()` - Risk management and position sizing
- `create_portfolio_container()` - Portfolio state tracking
- `create_execution_container()` - Order execution
- `create_classifier_container()` - Regime detection
- `create_optimization_container()` - Parameter optimization
- `create_signal_capture_container()` - Signal recording
- `create_signal_replay_container()` - Signal replay

### Execution Modes

#### Backtest Mode
Standard backtesting with full execution pipeline.

#### Signal Generation Mode  
Generates signals without execution for:
- Signal analysis and MAE/MFE optimization
- Classifier performance comparison
- Strategy development

#### Signal Replay Mode
Fast signal replay (10-100x faster) for:
- Ensemble weight optimization
- Risk parameter tuning
- Rapid iteration on pre-computed signals

## Architecture Principles

1. **Workflow Composition**: Workflows can compose other workflows
2. **Container Isolation**: Each container has its own event bus
3. **Adapter-Based Communication**: Cross-container communication via adapters
4. **Configuration-Driven**: All behavior configured through YAML/dict configs

## Usage Example

```python
# Create a backtest workflow
from core.coordinator.workflows import BacktestWorkflowManager

manager = BacktestWorkflowManager(
    container_id="backtest_001",
    shared_services={"logger": logger}
)

# Execute with configuration
result = await manager.execute(
    config=workflow_config,
    context=execution_context
)
```

## Container Hierarchy

The typical backtest container hierarchy:

```
BacktestContainer (root)
├── DataContainer
├── IndicatorContainer  
└── ClassifierContainers
    ├── RiskContainer
    │   └── StrategyContainers
    └── PortfolioContainer
└── ExecutionContainer
```

## Integration with Coordinator

These workflows are invoked by the main Coordinator through:
1. YAML configuration interpretation
2. WorkflowManager protocol implementation
3. Phase management for complex multi-phase workflows

See the parent coordinator README for more details on the overall architecture.