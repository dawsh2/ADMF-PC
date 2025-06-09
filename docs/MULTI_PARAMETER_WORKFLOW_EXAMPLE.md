# Multi-Parameter Workflow Example

## Problem Solved
How to generate **separate portfolios per parameter combination** with **separate strategy containers** for each, without hardcoding.

## Solution Architecture

The enhanced WorkflowManager automatically detects when multi-parameter execution is needed and creates the appropriate container structure dynamically.

## Example Configuration

### Single Strategy with Parameter Grid
```yaml
# config/multi_strategy_optimization.yaml
type: optimization
data:
  source: csv
  file_path: data/SPY.csv
  
strategies:
  - type: momentum
    parameters:
      lookback_period: [10, 20, 30]        # 3 values = 3 combinations
      signal_threshold: [0.01, 0.02, 0.05] # 3 values = 3 combinations
      # Total: 3 × 3 = 9 parameter combinations

optimization:
  parameter_grids:
    initial_capital: [10000, 50000, 100000]  # 3 values
    # Combined with strategies: 9 × 3 = 27 total combinations

backtest:
  commission: 0.001
  slippage: 0.0005
```

### Multiple Strategies with Grids
```yaml
strategies:
  - type: momentum
    parameters:
      lookback_period: [10, 20]  # 2 values
      signal_threshold: [0.01, 0.02]  # 2 values = 4 combinations
      
  - type: mean_reversion
    parameters:
      rsi_period: [14, 21]  # 2 values
      rsi_oversold: [20, 30]  # 2 values = 4 combinations
      
# Total: 4 + 4 = 8 strategy combinations
```

## How It Works

### 1. Automatic Detection
```python
def _requires_multi_parameter(self, config: WorkflowConfig) -> bool:
    # Detects parameter grids like:
    # - lookback_period: [10, 20, 30]
    # - signal_threshold: {'min': 0.01, 'max': 0.05, 'step': 0.01}
    # - optimization.parameter_grids: {...}
    return True  # if any found
```

### 2. Pattern Selection
```python
def _determine_pattern(self, config: WorkflowConfig) -> str:
    if self._requires_multi_parameter(config):
        if config.workflow_type == WorkflowType.OPTIMIZATION:
            return 'optimization_grid'      # → Multi-parameter optimization
        else:
            return 'multi_parameter_backtest'  # → Multi-parameter backtest
```

### 3. Dynamic Container Generation
```python
# For each parameter combination:
for i, param_combo in enumerate(param_combinations):
    # Create separate portfolio
    portfolio_container = self.factory.create_container(
        role=ContainerRole.PORTFOLIO,
        config={
            'parameter_combination': param_combo,
            'combination_index': i,
            'initial_capital': param_combo.get('initial_capital', 100000)
        },
        container_id=f"portfolio_combo_{i}"
    )
    
    # Create separate strategy for this combination
    strategy_container = self.factory.create_container(
        role=ContainerRole.STRATEGY,
        config={
            'type': param_combo['strategy']['type'],
            'parameters': param_combo['strategy']['parameters'],
            'target_portfolio': portfolio_container.metadata.container_id
        },
        container_id=f"strategy_combo_{i}"
    )
    
    # Create separate risk + execution for this combination
    risk_container = ...
    execution_container = ...
```

### 4. Communication Architecture
```
Hub Container (Data Coordinator)
├── Broadcasts data to all portfolios
│
├── Portfolio_Combo_0 (momentum: lookback=10, threshold=0.01, capital=10000)
│   ├── Strategy_Combo_0
│   ├── Risk_Combo_0
│   └── Execution_Combo_0
│
├── Portfolio_Combo_1 (momentum: lookback=10, threshold=0.01, capital=50000)
│   ├── Strategy_Combo_1
│   ├── Risk_Combo_1
│   └── Execution_Combo_1
│
└── Portfolio_Combo_N (...)
    ├── Strategy_Combo_N
    ├── Risk_Combo_N
    └── Execution_Combo_N
```

### 5. Results Collection
```python
results = {
    'summary': {
        'total_combinations': 27,
        'best_combination': {
            'combination_id': 'combo_15',
            'parameters': {
                'strategy': {'type': 'momentum', 'lookback_period': 20, 'signal_threshold': 0.02},
                'initial_capital': 100000
            },
            'pnl': 15234.56
        }
    },
    'portfolios': {
        'combo_0': {
            'parameters': {'strategy': {...}, 'initial_capital': 10000},
            'portfolio_pnl': 1234.56,
            'final_value': 11234.56,
            'trades': 45
        },
        'combo_1': {
            'parameters': {'strategy': {...}, 'initial_capital': 50000},
            'portfolio_pnl': 2345.67,
            'final_value': 52345.67,
            'trades': 38
        }
        # ... for all combinations
    }
}
```

## Usage Examples

### Basic Multi-Parameter Backtest
```python
from coordinator.workflows.workflow_manager import WorkflowManager

async def run_multi_parameter_backtest():
    config = WorkflowConfig(
        workflow_type=WorkflowType.BACKTEST,
        data_config={'source': 'csv', 'file_path': 'data/SPY.csv'},
        strategies=[{
            'type': 'momentum',
            'parameters': {
                'lookback_period': [10, 20, 30],  # Auto-detected as parameter grid
                'signal_threshold': [0.01, 0.02, 0.05]
            }
        }],
        backtest_config={'initial_capital': 100000}
    )
    
    manager = WorkflowManager()
    result = await manager.execute(config, ExecutionContext())
    
    # Results contain all 9 parameter combinations
    print(f"Tested {result.metadata['parameter_combinations']} combinations")
    print(f"Best PnL: {result.final_results['summary']['best_combination']['pnl']}")
```

### Multi-Strategy Optimization
```python
async def run_multi_strategy_optimization():
    config = WorkflowConfig(
        workflow_type=WorkflowType.OPTIMIZATION,
        strategies=[
            {
                'type': 'momentum',
                'parameters': {
                    'lookback_period': {'min': 10, 'max': 30, 'step': 5},  # Range notation
                    'signal_threshold': [0.01, 0.02, 0.03]
                }
            },
            {
                'type': 'mean_reversion', 
                'parameters': {
                    'rsi_period': [14, 21, 28],
                    'rsi_oversold': [20, 25, 30]
                }
            }
        ],
        optimization_config={
            'parameter_grids': {
                'initial_capital': [10000, 50000, 100000]
            }
        }
    )
    
    manager = WorkflowManager()
    result = await manager.execute(config, ExecutionContext())
    
    # Each strategy × parameter grid combination gets separate portfolio
    return result
```

### Convenience Method
```python
# For backward compatibility
pipeline_manager = ComposableWorkflowManagerPipeline()

result = await pipeline_manager.execute_multi_parameter_optimization(
    strategy_configs=[
        {'type': 'momentum', 'parameters': {'lookback_period': [10, 20, 30]}},
        {'type': 'mean_reversion', 'parameters': {'rsi_period': [14, 21]}}
    ],
    optimization_grids={'initial_capital': [10000, 50000]},
    base_config={'data': {'source': 'csv', 'file_path': 'data/SPY.csv'}}
)
```

## Benefits

### 1. **No Hardcoding**
- Automatically detects parameter grids in configuration
- Dynamically generates required number of containers
- Scales to any number of parameter combinations

### 2. **Complete Isolation**
- Each parameter combination gets separate portfolio
- No cross-contamination between combinations
- Parallel execution of all combinations

### 3. **Standard Architecture**
- Uses existing container factory for creation
- Uses existing communication factory for wiring
- Follows pattern-based architecture standard

### 4. **Rich Results**
- Detailed results for each combination
- Summary with best/worst performers
- Full parameter tracking for reproducibility

### 5. **Flexible Configuration**
- Supports lists: `[10, 20, 30]`
- Supports ranges: `{'min': 10, 'max': 30, 'step': 5}`
- Supports multiple strategies with different grids
- Supports optimization-level parameter grids

This approach allows you to specify parameter grids in configuration and automatically get separate portfolios and strategies for each combination, making optimization and multi-parameter testing seamless.