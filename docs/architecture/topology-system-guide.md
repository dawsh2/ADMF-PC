# ADMF-PC Topology System Guide

## Overview

The ADMF-PC system follows a clean architectural hierarchy with clear single responsibilities:

```
main.py (entry point)
    ↓
Coordinator (workflow management, result streaming, data flow)
    ↓
Sequencer (phase execution, delegates topology building)
    ↓
TopologyBuilder (ONLY builds topologies - never executes!)
    ↓
Topology Modules (pipeline definitions)
    ↓
Containers (execution units)
```

Users specify a workflow (e.g., `adaptive_ensemble`), and the system orchestrates the execution through clearly separated layers.

## ⚠️ Important: Architecture Migration in Progress

**Note**: The system is currently being migrated to this clean architecture. The current implementation still has TopologyBuilder doing execution, which violates single responsibility. See `MIGRATION_PLAN.md` for details.

## Architecture Hierarchy

The correct hierarchy is: **Coordinator → Workflows → Sequences → Topologies**

### Complete Flow

```
User Config (workflow: adaptive_ensemble)
    ↓
Coordinator (entry point, manages overall execution)
    ↓
Workflow Definition (defines multi-phase process)
    ↓
Sequencer (orchestrates phase execution)
    ↓
TopologyBuilder (creates topologies per phase)
    ↓
Topology Modules (define pipeline structure)
    ↓
Containers + Adapters (execute the pipeline)
```

### Layer Breakdown

#### 1. Coordinator
- **Role**: Entry point and overall orchestration
- **Responsibilities**:
  - Receives user configuration
  - Determines workflow to execute
  - Manages global resources
  - Handles cleanup and error recovery

#### 2. Workflows (User-Facing)
- **What users specify**: `workflow: adaptive_ensemble`
- **Defines**: Multi-phase processes with different topologies per phase
- **Location**: `src/core/coordinator/workflows/`
- **Examples**: 
  - `adaptive_ensemble` - Grid search → Regime analysis → Ensemble optimization → Validation
  - `walk_forward_optimization` - Rolling window parameter optimization
  - `regime_aware_trading` - Regime detection → Strategy selection → Execution
  - `parameter_optimization` - Parameter sweep → Performance analysis → Selection

#### 3. Sequences
- **Role**: Phase orchestration within a workflow
- **Responsibilities**:
  - Execute phases in order
  - Pass results between phases
  - Handle phase dependencies
  - Manage phase-specific configuration
- **Location**: `src/core/coordinator/sequences/`

#### 4. Topologies (Pipeline Building Blocks)
- **What sequences use**: Different pipeline structures for each phase
- **Types**:
  - `backtest` - Full pipeline: data → features → strategies → portfolios → risk → execution
  - `signal_generation` - Generate and save: data → features → strategies → disk
  - `signal_replay` - Load and execute: disk → portfolios → risk → execution
  - `analysis` - Custom analysis pipelines
  - `optimization` - Parameter optimization pipelines
- **Location**: `src/core/coordinator/topologies/`

#### 5. Containers (Execution Units)
- **What topologies create**: Processing units with components
- **Examples**:
  - Data containers with DataStreamer components
  - Feature containers with FeatureCalculator components
  - Portfolio containers with PortfolioState, SignalProcessor components
  - Analysis containers with custom analysis components

### Key Components

#### 1. TopologyBuilder (`src/core/coordinator/topology.py`)

**After migration**, TopologyBuilder will be a pure factory that:
- ONLY builds topologies based on mode
- Returns topology structure (containers + adapters)
- Has NO execution logic

```python
class TopologyBuilder:
    def build_topology(self, mode: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Build and return topology structure - NO execution!"""
        # Import topology modules
        from .topologies import (
            create_backtest_topology,
            create_signal_generation_topology,
            create_signal_replay_topology
        )
        
        # Map mode to topology function
        topology_functions = {
            'backtest': create_backtest_topology,
            'signal_generation': create_signal_generation_topology,
            'signal_replay': create_signal_replay_topology
        }
        
        # Create and return topology
        topology_function = topology_functions[mode]
        return topology_function(config)
```

#### 2. Topology Modules (`src/core/coordinator/topologies/*.py`)

Each topology module defines:
- What containers to create (based on config)
- What components to add to each container
- How to wire them together with adapters

Example structure:
```
topologies/
├── __init__.py              # Exports topology functions
├── backtest.py              # Full pipeline: data → features → strategies → portfolios → risk → execution
├── signal_generation.py     # Signal generation: data → features → strategies (save signals)
├── signal_replay.py         # Signal replay: saved signals → portfolios → risk → execution
└── helpers/
    ├── component_builder.py # Creates components from config
    └── adapter_wiring.py    # Wires containers together
```

#### 3. Container + Components Pattern

Generic containers with pluggable components provide flexibility:

```python
# Generic container with role
container = Container(ContainerConfig(
    role=ContainerRole.DATA,
    name=container_id,
    container_id=container_id
))

# Add components based on role
container.add_component('data_streamer', DataStreamer(...))
container.add_component('bar_aggregator', BarAggregator(...))
```

## How It Works

### 1. Configuration Drives Everything

The system reads configuration to determine:
- Which symbols to process
- What strategies to run
- Risk parameters
- Feature calculations needed
- Parameter combinations to test

```yaml
mode: backtest
symbols: [SPY, QQQ]
strategies:
  - type: momentum
    fast_period: 10
    slow_period: 20
  - type: mean_reversion
    rsi_period: 14
risk_profiles:
  - type: conservative
    max_position_size: 0.1
```

### 2. Dynamic Container Creation

Topology modules create containers dynamically based on config:

```python
def create_backtest_topology(config, tracing_enabled):
    topology = {
        'containers': {},
        'adapters': [],
        'parameter_combinations': []
    }
    
    # Extract symbol configurations from config
    symbol_configs = _extract_symbol_timeframe_configs(config)
    
    # Create containers for each symbol/timeframe
    for st_config in symbol_configs:
        symbol = st_config['symbol']
        timeframe = st_config.get('timeframe', '1d')
        
        # Create data container
        data_container = Container(ContainerConfig(
            role=ContainerRole.DATA,
            name=f"{symbol}_{timeframe}_data"
        ))
        data_container.add_component('data_streamer', 
            DataStreamer(
                symbol=symbol,
                timeframe=timeframe,
                data_source=st_config.get('data_config', {})
            ))
        
        topology['containers'][data_container.name] = data_container
```

### 3. Event-Driven Communication

Containers communicate through events:

```
Data Container → BAR event → Feature Container
Feature Container → FEATURES event → Feature Dispatcher
Feature Dispatcher → Filtered FEATURES → Strategy Services
Strategy Services → SIGNAL events → Portfolio Containers
Portfolio Containers → ORDER events → Risk Service
Risk Service → Validated ORDER → Execution Container
Execution Container → FILL events → Portfolio Containers
```

### 4. Adapter-Based Wiring

Adapters handle cross-container communication:

```python
def wire_backtest_topology(containers, config):
    adapters = []
    
    # Create Feature Dispatcher for routing
    feature_dispatcher = FeatureDispatcher(root_event_bus)
    
    # Wire containers based on topology needs
    for feature_container in feature_containers:
        feature_container.event_bus.subscribe(
            EventType.FEATURES, 
            feature_dispatcher.handle_features
        )
    
    # Create broadcast adapters for fills
    fill_broadcast = adapter_factory.create_adapter(
        name='fill_broadcast',
        config={
            'type': 'broadcast',
            'source': 'execution',
            'targets': portfolio_container_names,
            'allowed_types': [EventType.FILL]
        }
    )
    adapters.append(fill_broadcast)
    
    return adapters
```

## Creating New Topologies

### Step 1: Define the Pipeline

Determine what your topology needs:
- What data sources?
- What processing steps?
- What outputs?

Example: ML Training Topology
- Pipeline: data → features → training → model storage
- No portfolios or execution needed

### Step 2: Create the Topology Module

Create a new file in `src/core/coordinator/topologies/`:

```python
# ml_training.py
from typing import Dict, Any, List
import logging

from ...containers.container import Container, ContainerConfig, ContainerRole
from ...containers.components import DataStreamer, FeatureCalculator
from ...events import EventBus

logger = logging.getLogger(__name__)

def create_ml_training_topology(config: Dict[str, Any], tracing_enabled: bool = True) -> Dict[str, Any]:
    """
    Create topology for ML model training.
    
    Pipeline: data → features → training → model storage
    """
    topology = {
        'containers': {},
        'adapters': [],
        'models': {},
        'root_event_bus': EventBus("root_event_bus")
    }
    
    # Read configuration to determine what to create
    ml_config = config.get('ml_training', {})
    symbols = ml_config.get('symbols', ['SPY'])
    
    # Create containers dynamically based on config
    for symbol in symbols:
        # Data container
        data_container = Container(ContainerConfig(
            role=ContainerRole.DATA,
            name=f"{symbol}_training_data"
        ))
        
        # Configure based on config, not hardcoded
        data_config = ml_config.get('data', {})
        data_container.add_component('data_streamer', DataStreamer(
            symbol=symbol,
            start_date=data_config.get('start_date'),
            end_date=data_config.get('end_date'),
            data_source=data_config.get('source', 'csv')
        ))
        
        topology['containers'][data_container.name] = data_container
    
    # Create feature engineering container
    feature_config = ml_config.get('features', {})
    if feature_config.get('enabled', True):
        feature_container = Container(ContainerConfig(
            role=ContainerRole.FEATURE,
            name="ml_feature_engineering"
        ))
        
        # Add components based on config
        indicators = feature_config.get('indicators', [])
        feature_container.add_component('feature_engineer',
            MLFeatureEngineer(
                indicators=indicators,
                lookback=feature_config.get('lookback', 100),
                feature_transformations=feature_config.get('transformations', [])
            ))
        
        topology['containers']['ml_features'] = feature_container
    
    # Create training container based on config
    training_config = ml_config.get('training', {})
    model_type = training_config.get('model_type', 'random_forest')
    
    training_container = Container(ContainerConfig(
        role=ContainerRole.ANALYSIS,  # New role for ML
        name="ml_training"
    ))
    
    # Dynamically create model based on config
    if model_type == 'random_forest':
        from ...ml.models import RandomForestModel
        model = RandomForestModel(**training_config.get('model_params', {}))
    elif model_type == 'neural_network':
        from ...ml.models import NeuralNetworkModel
        model = NeuralNetworkModel(**training_config.get('model_params', {}))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    training_container.add_component('model_trainer', ModelTrainer(
        model=model,
        validation_split=training_config.get('validation_split', 0.2),
        metrics=training_config.get('metrics', ['accuracy', 'sharpe'])
    ))
    
    topology['containers']['ml_training'] = training_container
    
    # Wire containers
    from .helpers.adapter_wiring import wire_ml_training_topology
    topology['adapters'] = wire_ml_training_topology(
        topology['containers'], 
        {'root_event_bus': topology['root_event_bus']}
    )
    
    return topology
```

### Step 3: Best Practices

#### 1. **Always Read from Config**

```python
# ❌ BAD - Hardcoded values
data_container.add_component('data_streamer', DataStreamer(
    symbol="SPY",  # Hardcoded!
    timeframe="1d",  # Hardcoded!
    lookback=100  # Hardcoded!
))

# ✅ GOOD - Config-driven
symbol = config.get('symbol', 'SPY')  # Default only as fallback
timeframe = config.get('timeframe', '1d')
lookback = config.get('lookback', 100)

data_container.add_component('data_streamer', DataStreamer(
    symbol=symbol,
    timeframe=timeframe,
    lookback=lookback
))
```

#### 2. **Create Containers Dynamically**

```python
# ❌ BAD - Fixed number of containers
container1 = Container(...)
container2 = Container(...)

# ✅ GOOD - Dynamic based on config
for symbol in config.get('symbols', []):
    container = Container(ContainerConfig(
        name=f"{symbol}_data"
    ))
    topology['containers'][container.name] = container
```

#### 3. **Use Helper Functions**

```python
# Extract common patterns into helpers
def _extract_symbol_configs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract and normalize symbol configurations."""
    # Handle different config formats
    if 'symbols' in config:
        # Simple list format
        return [{'symbol': s} for s in config['symbols']]
    elif 'symbol_configs' in config:
        # Detailed format
        return config['symbol_configs']
    else:
        # Fallback
        return [{'symbol': 'SPY'}]
```

#### 4. **Make Components Optional**

```python
# Only create components if enabled in config
if config.get('risk_management', {}).get('enabled', True):
    risk_container = Container(...)
    topology['containers']['risk'] = risk_container

if config.get('ml_scoring', {}).get('enabled', False):
    ml_container = Container(...)
    topology['containers']['ml_scoring'] = ml_container
```

#### 5. **Support Multiple Modes**

```python
def create_ml_topology(config: Dict[str, Any], tracing_enabled: bool = True) -> Dict[str, Any]:
    ml_mode = config.get('ml_mode', 'training')
    
    if ml_mode == 'training':
        return _create_training_topology(config, tracing_enabled)
    elif ml_mode == 'inference':
        return _create_inference_topology(config, tracing_enabled)
    elif ml_mode == 'backtesting':
        return _create_ml_backtest_topology(config, tracing_enabled)
    else:
        raise ValueError(f"Unknown ML mode: {ml_mode}")
```

### Step 4: Register the Topology

Add to `topologies/__init__.py`:

```python
from .ml_training import create_ml_training_topology

__all__ = [
    'create_backtest_topology',
    'create_signal_generation_topology', 
    'create_signal_replay_topology',
    'create_ml_training_topology'  # New topology
]
```

Update TopologyBuilder to recognize new mode:

```python
# In topology.py
class WorkflowMode(str, Enum):
    BACKTEST = "backtest"
    SIGNAL_GENERATION = "signal_generation"
    SIGNAL_REPLAY = "signal_replay"
    ML_TRAINING = "ml_training"  # New mode

# In _create_topology method
topology_functions = {
    WorkflowMode.BACKTEST: create_backtest_topology,
    WorkflowMode.SIGNAL_GENERATION: create_signal_generation_topology,
    WorkflowMode.SIGNAL_REPLAY: create_signal_replay_topology,
    WorkflowMode.ML_TRAINING: create_ml_training_topology  # New mapping
}
```

### Step 5: Create Adapter Wiring

Add wiring logic to `helpers/adapter_wiring.py`:

```python
def wire_ml_training_topology(containers: Dict[str, Any], config: Dict[str, Any]) -> List[Any]:
    """Wire containers for ML training topology."""
    adapters = []
    root_event_bus = config.get('root_event_bus')
    
    # Wire data → features
    data_containers = {k: v for k, v in containers.items() if '_data' in k}
    feature_container = containers.get('ml_features')
    
    if feature_container:
        for data_container in data_containers.values():
            # Subscribe feature container to data events
            data_container.event_bus.subscribe(
                EventType.BAR,
                feature_container.get_component('feature_engineer').on_bar
            )
    
    # Wire features → training
    training_container = containers.get('ml_training')
    if feature_container and training_container:
        feature_container.event_bus.subscribe(
            EventType.FEATURES,
            training_container.get_component('model_trainer').on_features
        )
    
    return adapters
```

## Testing Your Topology

### 1. Unit Test the Topology Creation

```python
def test_ml_training_topology_creation():
    config = {
        'ml_training': {
            'symbols': ['SPY', 'QQQ'],
            'model_type': 'random_forest',
            'features': {
                'indicators': ['rsi', 'macd'],
                'lookback': 50
            }
        }
    }
    
    topology = create_ml_training_topology(config)
    
    # Verify containers created
    assert 'SPY_training_data' in topology['containers']
    assert 'QQQ_training_data' in topology['containers']
    assert 'ml_features' in topology['containers']
    assert 'ml_training' in topology['containers']
```

### 2. Integration Test the Flow

```python
def test_ml_training_flow():
    config = load_test_config('ml_training_test.yaml')
    
    builder = TopologyBuilder()
    result = builder.execute(
        WorkflowConfig(workflow_type=WorkflowType.ML_TRAINING, parameters=config),
        ExecutionContext(workflow_id='test_ml_001')
    )
    
    assert result.success
    assert 'model_path' in result.final_results
```

## Common Patterns

### 1. Multi-Symbol Processing

```python
# Process multiple symbols in parallel
for symbol in config.get('symbols', []):
    create_symbol_container(symbol, config)
```

### 2. Optional Components

```python
# Only add components if configured
if config.get('use_technical_indicators', True):
    container.add_component('ta_calculator', TechnicalAnalysis(...))

if config.get('use_ml_features', False):
    container.add_component('ml_features', MLFeatureExtractor(...))
```

### 3. Dynamic Pipelines

```python
# Build pipeline based on config
pipeline_steps = config.get('pipeline', ['data', 'features', 'signals'])

for step in pipeline_steps:
    if step == 'data':
        create_data_containers(config)
    elif step == 'features':
        create_feature_containers(config)
    elif step == 'ml':
        create_ml_containers(config)
```

### 4. Parameter Expansion

```python
# Expand parameter combinations
def _expand_parameters(config):
    params = []
    for model in config.get('models', []):
        for feature_set in config.get('feature_sets', []):
            params.append({
                'model': model,
                'features': feature_set
            })
    return params
```

## Debugging Tips

1. **Enable Tracing**: Set `tracing.enabled: true` in config
2. **Check Event Flow**: Look at trace files in `./results/traces/`
3. **Verify Wiring**: Log subscription counts
4. **Test Incrementally**: Start with minimal config and add complexity

## Workflow Integration

### How Workflows Use Topologies

Workflows orchestrate multiple topologies to achieve complex objectives. The WorkflowManager defines the sequence and configuration for each phase.

#### Example: Adaptive Ensemble Workflow

```python
# In WorkflowManager
class WorkflowManager:
    def __init__(self):
        self._workflow_patterns = {
            'adaptive_ensemble': {
                'phases': [
                    {
                        'name': 'grid_search',
                        'topology': 'signal_generation',
                        'config_override': {
                            'mode': 'walk_forward',
                            'save_signals': True
                        }
                    },
                    {
                        'name': 'regime_analysis',
                        'topology': 'analysis',
                        'config_override': {
                            'analysis_type': 'regime_performance',
                            'input': 'phase_1_results'
                        }
                    },
                    {
                        'name': 'ensemble_optimization',
                        'topology': 'signal_replay',
                        'config_override': {
                            'mode': 'walk_forward',
                            'optimize_weights': True
                        }
                    },
                    {
                        'name': 'final_validation',
                        'topology': 'backtest',
                        'config_override': {
                            'use_ensemble': True,
                            'regime_adaptive': True
                        }
                    }
                ]
            }
        }
```

### Sequencer Orchestration

The Sequencer handles the execution of each phase:

```python
class Sequencer:
    def __init__(self):
        self.topology_builder = TopologyBuilder()  # Composition, not inheritance!
        
    async def execute_phases(
        self, 
        pattern: Dict[str, Any], 
        config: Dict[str, Any], 
        context: Dict[str, Any],
        result_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Execute phases - Sequencer handles execution, not TopologyBuilder!"""
        results = {}
        
        for phase in pattern['phases']:
            # Merge base config with phase overrides
            phase_config = self._merge_configs(config, phase.get('config_override', {}))
            
            # Add context from previous phases
            if phase.get('input'):
                phase_config['input_data'] = results[phase['input']]
            
            # TopologyBuilder ONLY builds topology structure
            topology = self.topology_builder.build_topology(
                mode=phase['topology'],
                config=phase_config
            )
            
            # Sequencer executes the topology
            phase_result = await self._execute_topology(
                topology,
                phase_config,
                {**context, 'phase': phase['name']}
            )
            
            # Stream results to Coordinator
            if result_callback:
                await result_callback(
                    workflow_id=context['workflow_id'],
                    phase=phase['name'],
                    result=phase_result
                )
            
            results[f"phase_{phase['name']}"] = phase_result
            
        return results
```

### Complete Adaptive Ensemble Flow

```yaml
# User specifies workflow
workflow: adaptive_ensemble

# Base configuration used by all phases
data:
  symbols: [SPY, QQQ]
  start_date: '2020-01-01'
  end_date: '2023-12-31'
  
strategies:
  - type: momentum
    parameters:
      fast_period: [5, 10, 20]
      slow_period: [20, 40, 60]
  - type: mean_reversion
    parameters:
      rsi_period: [7, 14, 21]
      
classifiers:
  - type: hmm_regime
    parameters:
      n_states: [2, 3, 4]
  - type: volatility_regime
    parameters:
      lookback: [20, 40]
```

#### Phase 1: Grid Search (Signal Generation)
```
┌─────────────────────────────────────────────────────────┐
│ Uses: signal_generation topology                        │
│                                                         │
│ • Runs all parameter combinations                       │
│ • Walk-forward validation with multiple windows        │
│ • Saves signals to disk for later analysis             │
│ • Groups results by regime within each window          │
│                                                         │
│ Output: signals/grid_search/*.parquet                  │
└─────────────────────────────────────────────────────────┘
```

#### Phase 2: Regime Analysis
```
┌─────────────────────────────────────────────────────────┐
│ Uses: analysis topology (custom)                        │
│                                                         │
│ • Loads Phase 1 results                                │
│ • Analyzes performance by regime                        │
│ • Identifies best parameters per regime                 │
│ • Stores regime-optimal configurations                  │
│                                                         │
│ Output: regime_configs.json                            │
└─────────────────────────────────────────────────────────┘
```

#### Phase 3: Ensemble Weight Optimization
```
┌─────────────────────────────────────────────────────────┐
│ Uses: signal_replay topology                            │
│                                                         │
│ • Loads signals from regime-optimal strategies         │
│ • Walk-forward optimization of ensemble weights        │
│ • Finds optimal weights per regime                     │
│ • 3x+ faster than regenerating signals                 │
│                                                         │
│ Output: ensemble_weights.json                          │
└─────────────────────────────────────────────────────────┘
```

#### Phase 4: Final Validation
```
┌─────────────────────────────────────────────────────────┐
│ Uses: backtest topology                                 │
│                                                         │
│ • Full backtest on out-of-sample data                  │
│ • Uses regime-adaptive ensemble                         │
│ • Dynamically switches parameters based on regime      │
│ • Complete performance metrics                          │
│                                                         │
│ Output: final_results.json, performance_report.html     │
└─────────────────────────────────────────────────────────┘
```

### Creating Custom Workflows

#### Step 1: Define Workflow Pattern

```python
# In WorkflowManager
self._workflow_patterns['my_custom_workflow'] = {
    'phases': [
        {
            'name': 'data_preparation',
            'topology': 'data_prep',  # Custom topology
            'config_override': {
                'clean_data': True,
                'fill_missing': 'forward'
            }
        },
        {
            'name': 'feature_engineering',
            'topology': 'feature_eng',  # Custom topology
            'config_override': {
                'create_synthetic': True
            }
        },
        {
            'name': 'backtesting',
            'topology': 'backtest',  # Existing topology
            'config_override': {
                'use_prepared_data': True
            }
        }
    ]
}
```

#### Step 2: Implement Required Topologies

Create any custom topologies needed by your workflow following the patterns described earlier.

#### Step 3: Handle Phase Dependencies

```python
def execute_workflow(self, workflow_name: str, config: Dict):
    workflow_def = self._workflow_patterns[workflow_name]
    context = {'phase_results': {}}
    
    for phase in workflow_def['phases']:
        # Check dependencies
        if phase.get('depends_on'):
            for dep in phase['depends_on']:
                if dep not in context['phase_results']:
                    raise ValueError(f"Phase {phase['name']} depends on {dep}")
        
        # Execute phase with context
        result = self._execute_phase(phase, config, context)
        context['phase_results'][phase['name']] = result
    
    return context['phase_results']
```

## Example: Complete Adaptive Ensemble Flow

Let's trace how a user request flows through all layers:

### 1. User Configuration
```yaml
# config/adaptive_ensemble_config.yaml
workflow: adaptive_ensemble  # User specifies workflow, not topology!

data:
  symbols: [SPY, QQQ]
  start_date: '2020-01-01'
  end_date: '2023-12-31'

strategies:
  - type: momentum
    parameter_grid:
      fast_period: [5, 10, 20]
      slow_period: [20, 40, 60]

walk_forward:
  train_periods: 252
  test_periods: 63
  step_size: 21
```

### 2. Coordinator Receives Request
```python
# main.py
coordinator = Coordinator()
result = coordinator.run(config_file='adaptive_ensemble_config.yaml')
```

### 3. Coordinator Finds Workflow
```python
# In Coordinator
workflow_name = config.get('workflow')  # 'adaptive_ensemble'
workflow = self.workflow_manager.get_workflow(workflow_name)
result = workflow.execute(config)
```

### 4. Workflow Defines Phases
```python
# In adaptive_ensemble_workflow.py
class AdaptiveEnsembleWorkflow:
    def __init__(self):
        self.phases = [
            {
                'name': 'grid_search',
                'sequence': 'walk_forward_sequence',
                'topology': 'signal_generation',
                'config': {
                    'save_signals': True,
                    'group_by_regime': True
                }
            },
            {
                'name': 'regime_analysis',
                'sequence': 'single_pass_sequence',
                'topology': 'analysis',
                'config': {
                    'analysis_type': 'regime_performance',
                    'input': '{phase.grid_search.output}'
                }
            },
            # ... more phases
        ]
```

### 5. Sequencer Orchestrates Each Phase

```python
# For Phase 1: Grid Search
sequencer = WalkForwardSequence()
sequencer.execute(
    topology='signal_generation',
    config=merged_config,
    windows=[(train_start, train_end, test_start, test_end), ...]
)

# For each window, sequencer:
for window in windows:
    # 1. Asks TopologyBuilder to BUILD topology (no execution!)
    topology = topology_builder.build_topology(
        mode='signal_generation',
        config={**config, 'window': window}
    )
    
    # 2. SEQUENCER executes the topology (not TopologyBuilder!)
    result = await sequencer._execute_topology(topology, config, context)
```

### 6. Topology Creates Pipeline

```python
# In signal_generation topology
def create_signal_generation_topology(config):
    # Create containers based on config
    for symbol in config['symbols']:
        data_container = Container(...)
        data_container.add_component('streamer', DataStreamer(symbol=symbol))
        
        feature_container = Container(...)
        feature_container.add_component('calc', FeatureCalculator(...))
    
    # Wire them together
    wire_signal_generation_topology(containers)
    
    return topology
```

### 7. Results Flow Back Up

```
Containers execute → Topology completes → Sequence aggregates → 
Workflow combines → Coordinator returns → User gets results
```

## Creating New Components

### 1. Stateless Strategy Component

```python
# src/strategy/strategies/bollinger_mean_reversion.py
from typing import Dict, Any, Optional

@strategy(
    feature_config={
        'bollinger_upper': {'params': ['bb_period', 'bb_std'], 'default': [20, 2]},
        'bollinger_lower': {'params': ['bb_period', 'bb_std'], 'default': [20, 2]},
        'close': {}
    }
)
def bollinger_mean_reversion_strategy(
    features: Dict[str, Any], 
    bar: Dict[str, Any], 
    params: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Mean reversion strategy using Bollinger Bands."""
    close = features['close']
    upper = features['bollinger_upper']
    lower = features['bollinger_lower']
    
    threshold = params.get('entry_threshold', 0.02)
    
    # Generate signals
    if close < lower * (1 - threshold):
        return {
            'symbol': bar['symbol'],
            'direction': 'long',
            'strength': min(1.0, (lower - close) / lower),
            'reason': 'oversold_bollinger'
        }
    elif close > upper * (1 + threshold):
        return {
            'symbol': bar['symbol'],
            'direction': 'short',
            'strength': min(1.0, (close - upper) / upper),
            'reason': 'overbought_bollinger'
        }
    
    return None  # No signal
```

### 2. Stateless Classifier Component

```python
# src/strategy/classifiers/volume_regime_classifier.py
from typing import Dict, Any

@classifier(
    regime_types=['low_volume', 'normal_volume', 'high_volume'],
    features=['volume_sma', 'volume']
)
def volume_regime_classifier(
    features: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """Classify market regime based on volume patterns."""
    volume = features['volume']
    volume_sma = features['volume_sma']
    
    # Thresholds from params
    low_threshold = params.get('low_threshold', 0.7)
    high_threshold = params.get('high_threshold', 1.3)
    
    ratio = volume / volume_sma
    
    if ratio < low_threshold:
        regime = 'low_volume'
        confidence = 1.0 - (ratio / low_threshold)
    elif ratio > high_threshold:
        regime = 'high_volume'
        confidence = min(1.0, (ratio - high_threshold) / high_threshold)
    else:
        regime = 'normal_volume'
        confidence = 1.0 - abs(ratio - 1.0)
    
    return {
        'regime': regime,
        'confidence': confidence,
        'metadata': {'volume_ratio': ratio}
    }
```

## Creating New Sequences

### 1. Monte Carlo Sequence

```python
# src/core/coordinator/sequences/monte_carlo.py
from typing import Dict, Any, List
import numpy as np

class MonteCarloSequence:
    """Run multiple iterations with randomized parameters."""
    
    def __init__(self, topology_builder: TopologyBuilderProtocol):
        self.topology_builder = topology_builder
        
    async def execute(
        self,
        topology: str,
        config: Dict[str, Any],
        context: Dict[str, Any],
        iterations: int = 100
    ) -> Dict[str, Any]:
        """Execute Monte Carlo simulation."""
        results = []
        base_params = config.get('parameters', {})
        
        for i in range(iterations):
            # Randomize parameters
            iteration_config = config.copy()
            iteration_config['parameters'] = self._randomize_parameters(
                base_params,
                config.get('randomization', {})
            )
            
            # Build and execute topology
            topology_struct = self.topology_builder.build_topology(
                topology,
                iteration_config
            )
            
            # Execute (via appropriate executor)
            result = await self._execute_iteration(
                topology_struct,
                iteration_config,
                {**context, 'iteration': i}
            )
            
            results.append({
                'iteration': i,
                'parameters': iteration_config['parameters'],
                'result': result
            })
        
        return {
            'iterations': iterations,
            'results': results,
            'statistics': self._calculate_statistics(results)
        }
    
    def _randomize_parameters(
        self,
        base_params: Dict[str, Any],
        randomization: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply randomization to parameters."""
        randomized = base_params.copy()
        
        for param, config in randomization.items():
            if param in base_params:
                method = config.get('method', 'uniform')
                
                if method == 'uniform':
                    low = base_params[param] * (1 - config['range'])
                    high = base_params[param] * (1 + config['range'])
                    randomized[param] = np.random.uniform(low, high)
                elif method == 'normal':
                    mean = base_params[param]
                    std = mean * config['std_ratio']
                    randomized[param] = np.random.normal(mean, std)
        
        return randomized
```

## Creating New Workflows

### 1. Regime-Adaptive Trading Workflow

```python
# src/core/coordinator/workflows/regime_adaptive_trading.py
from typing import Dict, Any, List

@workflow(
    name='regime_adaptive_trading',
    description='Detect regime then adapt strategy parameters',
    tags=['regime', 'adaptive', 'production']
)
def regime_adaptive_trading_workflow():
    """
    Multi-phase workflow for regime-adaptive trading:
    1. Historical regime detection
    2. Parameter optimization per regime
    3. Live regime detection
    4. Adaptive execution
    """
    return {
        'phases': [
            {
                'name': 'historical_regime_detection',
                'topology': 'analysis',
                'sequence': 'single_pass',
                'config_override': {
                    'analysis_type': 'regime_detection',
                    'classifiers': ['hmm_regime', 'volatility_regime'],
                    'lookback_periods': 500
                }
            },
            {
                'name': 'regime_parameter_optimization',
                'topology': 'signal_generation',
                'sequence': 'walk_forward',
                'config_override': {
                    'mode': 'parameter_sweep',
                    'group_by': 'regime',
                    'save_signals': True
                },
                'depends_on': ['historical_regime_detection']
            },
            {
                'name': 'ensemble_weight_optimization',
                'topology': 'signal_replay',
                'sequence': 'single_pass',
                'config_override': {
                    'optimize': 'ensemble_weights',
                    'per_regime': True
                },
                'depends_on': ['regime_parameter_optimization']
            },
            {
                'name': 'live_trading_simulation',
                'topology': 'backtest',
                'sequence': 'single_pass',
                'config_override': {
                    'mode': 'paper_trading',
                    'regime_adaptive': True,
                    'parameter_lookup': '{phase.ensemble_weight_optimization.output}'
                },
                'depends_on': ['ensemble_weight_optimization']
            }
        ],
        'data_flow': {
            'regime_labels': 'historical_regime_detection → regime_parameter_optimization',
            'optimal_parameters': 'regime_parameter_optimization → ensemble_weight_optimization',
            'ensemble_weights': 'ensemble_weight_optimization → live_trading_simulation'
        }
    }
```

### 2. Using the Workflow

```yaml
# config/regime_adaptive_trading.yaml
workflow: regime_adaptive_trading

data:
  symbols: [SPY, QQQ, IWM]
  start_date: '2020-01-01'
  end_date: '2023-12-31'

strategies:
  - type: momentum
    parameter_ranges:
      fast_period: [5, 10, 20]
      slow_period: [20, 40, 60]
  - type: bollinger_mean_reversion
    parameter_ranges:
      bb_period: [10, 20, 30]
      entry_threshold: [0.01, 0.02, 0.03]

classifiers:
  - type: hmm_regime
    n_states: 3
  - type: volume_regime
    thresholds:
      low: 0.7
      high: 1.3

risk:
  base_position_size: 0.05
  regime_adjustments:
    high_volatility: 0.5  # Reduce size by 50%
    low_volume: 0.8       # Reduce size by 20%
```

## Component Registration and Discovery

### How Components Self-Register

```python
# The @strategy decorator automatically registers the function
@strategy(...)
def my_strategy(...):
    pass

# Behind the scenes:
def strategy(feature_config=None, **metadata):
    def decorator(func):
        # Register with discovery system
        registry = get_component_registry()
        registry.register(
            component_type='strategy',
            name=func.__name__,
            factory=func,
            metadata={
                'feature_config': feature_config,
                **metadata
            }
        )
        return func
    return decorator
```

### Discovery at Runtime

```python
# TopologyBuilder can discover all registered components
from ..containers.discovery import get_component_registry

registry = get_component_registry()

# Get all strategies
strategies = registry.get_components_by_type('strategy')

# Get specific strategy
momentum = registry.get_component('momentum_strategy')
```

## Data Flow Between Phases

### 1. Result Streaming

During execution, results stream through the Coordinator:

```
Container → Event → Sequencer → ResultCallback → Coordinator → ResultStreamer → Disk
```

### 2. Inter-Phase Data Transfer

The DataFlowManager handles passing data between phases:

```python
# Phase 1 stores output
await data_manager.store_phase_output(
    workflow_id='wf_001',
    phase_name='regime_detection',
    output={'regimes': regime_labels}
)

# Phase 2 retrieves it
regime_data = await data_manager.get_phase_output(
    workflow_id='wf_001',
    phase_name='regime_detection'
)
```

### 3. Phase Dependencies

Workflows can specify dependencies:

```python
{
    'name': 'optimization',
    'depends_on': ['regime_detection'],
    'input': '{phase.regime_detection.output}'  # Template syntax
}
```

## Summary

The ADMF-PC system provides a five-layer architecture:

1. **Coordinator** - Entry point, result streaming, and data flow management
2. **Workflows** - User-facing multi-phase processes
3. **Sequences** - Phase orchestration logic (walk-forward, monte carlo, etc.)
4. **Topologies** - Reusable pipeline building blocks  
5. **Containers** - Isolated execution units with components

Key benefits:
- **Abstraction**: Users think in workflows, not implementation details
- **Flexibility**: Each layer can evolve independently
- **Modularity**: Workflows reuse sequences, sequences reuse topologies
- **Composability**: Complex workflows from simple building blocks
- **Configuration-Driven**: No hardcoded values, everything from YAML
- **Extensibility**: Easy to add new components via decorators
- **Scalability**: Ready for distributed execution
- **Testability**: Each layer can be tested independently

Remember: 
- **Users specify workflows, not topologies or modes**
- **Components self-register via decorators (@strategy, @classifier, @feature)**
- **Workflows define the "what", sequences handle the "when", topologies handle the "how"**
- **Data flows between phases via DataFlowManager**
- **Results stream in real-time via ResultStreamer**
- **Let configuration drive everything at every layer!**
