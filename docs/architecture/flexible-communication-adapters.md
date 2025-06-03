# Enhanced Flexible Event Communication Architecture

## The Core Problem

We need to maintain the benefits of isolated event buses (resource isolation, parallelization, reproducibility) while supporting different container organization patterns (Strategy-First, Classifier-First, Risk-First, Portfolio-First) without forcing tight coupling between organizational structure and event communication.

**New Context**: The ADMF-PC system supports multiple organizational approaches, sophisticated workspace management with file-based communication, and needs to handle both research workflows (parameter sweeps, signal replay) and production trading scenarios.

## Solution: Pluggable Event Communication Adapters with Semantic Events

Instead of hardcoding event flows based on organizational patterns, use **Event Communication Adapters** that can be configured independently of container organization, enhanced with semantic event types and multiple execution models.

### Enhanced Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONTAINER ORGANIZATION                        │
│  (Strategy-First, Classifier-First, Risk-First, Portfolio-First) │
│                             │                                   │
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │            SEMANTIC EVENT COMMUNICATION LAYER               │ │
│  │                                                             │ │
│  │  ┌─────────────────┐    ┌─────────────────────────────────┐ │ │
│  │  │ Communication   │    │ Event Flow Patterns             │ │ │
│  │  │ Adapter Factory │────│ • Pipeline                      │ │ │
│  │  │                 │    │ • Hierarchical                  │ │ │
│  │  └─────────────────┘    │ • Broadcast                     │ │ │
│  │                         │ • Selective                     │ │ │
│  │                         │ • Workspace-Aware               │ │ │
│  │                         └─────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                             │                                   │
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │           EXECUTION MODEL SELECTION                         │ │
│  │  Containers | Functions | Parallel Containers             │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                             │                                   │
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              ISOLATED EVENT BUSES                           │ │
│  │  (Per container, maintaining all isolation benefits)        │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Semantic Event System

### Strongly Typed Events with Schema Evolution

```python
from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any, List, Callable, Iterator
from datetime import datetime
from abc import ABC, abstractmethod
import uuid
import math
import copy
import json
import pickle
from pathlib import Path

@dataclass
class SemanticEventBase:
    """Base class for all semantic events"""
    # Core identification
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: str = "1.0.0"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Correlation and tracing
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    causation_id: Optional[str] = None  # Parent event that caused this
    
    # Source context
    source_container: str = ""
    source_component: str = ""
    
    # Business context
    strategy_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    regime_context: Optional[str] = None
    
    def validate(self) -> bool:
        """Override in subclasses for validation"""
        return True

@dataclass
class MarketDataEvent(SemanticEventBase):
    """Market data events - high frequency, low latency tier"""
    schema_version: str = "1.2.0"
    
    symbol: str = ""
    price: float = 0.0
    volume: int = 0
    bid: Optional[float] = None
    ask: Optional[float] = None
    data_type: Literal["BAR", "TICK", "QUOTE"] = "BAR"
    
    def validate(self) -> bool:
        return self.price > 0 and self.volume >= 0

@dataclass  
class IndicatorEvent(SemanticEventBase):
    """Technical indicator events - standard tier"""
    schema_version: str = "1.1.0"
    
    indicator_name: str = ""
    value: float = 0.0
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        return self.indicator_name and not math.isnan(self.value)

@dataclass
class TradingSignal(SemanticEventBase):
    """Trading signal events - standard tier"""
    schema_version: str = "2.0.0"
    
    symbol: str = ""
    action: Literal["BUY", "SELL", "HOLD"] = "HOLD"
    strength: float = 0.0  # 0.0 to 1.0
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Enhanced in v2.0.0
    regime_context: Optional[str] = None
    risk_score: float = 0.5  # Added in v2.0.0
    
    def validate(self) -> bool:
        return (0.0 <= self.strength <= 1.0 and 
                self.symbol and 
                0.0 <= self.risk_score <= 1.0)

@dataclass
class OrderEvent(SemanticEventBase):
    """Order events - reliable tier"""
    schema_version: str = "1.0.0"
    
    order_type: Literal["MARKET", "LIMIT", "STOP"] = "MARKET"
    symbol: str = ""
    quantity: int = 0
    price: Optional[float] = None
    side: Literal["BUY", "SELL"] = "BUY"
    
    # Risk management fields
    max_position_pct: float = 0.02
    stop_loss_pct: Optional[float] = None
    
    def validate(self) -> bool:
        return (self.quantity > 0 and 
                self.symbol and 
                0.0 < self.max_position_pct <= 1.0)

@dataclass
class RegimeChangeEvent(SemanticEventBase):
    """Market regime change events - critical tier"""
    schema_version: str = "1.0.0"
    
    previous_regime: str = ""
    new_regime: str = ""
    confidence: float = 0.0
    classifier_type: str = ""
    
    def validate(self) -> bool:
        return (self.new_regime and 
                0.0 <= self.confidence <= 1.0 and
                self.classifier_type)
```

### Schema Evolution and Migration

```python
class EventSchemaRegistry:
    """Manages event schemas and migrations"""
    
    def __init__(self):
        self.schemas = {}
        self.migrations = {}
        
    def register_schema(self, event_type: type, version: str):
        """Register event schema version"""
        if event_type not in self.schemas:
            self.schemas[event_type] = {}
        self.schemas[event_type][version] = event_type
        
    def register_migration(self, event_type: type, from_version: str, 
                          to_version: str, migration_fn: callable):
        """Register migration between schema versions"""
        key = (event_type, from_version, to_version)
        self.migrations[key] = migration_fn
        
    def migrate_event(self, event: SemanticEventBase, target_version: str):
        """Migrate event to target schema version"""
        current_version = event.schema_version
        if current_version == target_version:
            return event
            
        # Find migration path
        migration_key = (type(event), current_version, target_version)
        if migration_key in self.migrations:
            return self.migrations[migration_key](event)
        
        raise ValueError(f"No migration path from {current_version} to {target_version}")

# Example migration
def migrate_trading_signal_v1_to_v2(v1_signal: TradingSignal) -> TradingSignal:
    """Migration from v1.0.0 to v2.0.0"""
    return TradingSignal(
        **{k: v for k, v in v1_signal.__dict__.items() if k != 'schema_version'},
        schema_version="2.0.0",
        risk_score=0.5  # Default value for new field
    )

# Register migration
schema_registry = EventSchemaRegistry()
schema_registry.register_migration(
    TradingSignal, "1.0.0", "2.0.0", migrate_trading_signal_v1_to_v2
)
```

## Container Execution Configuration

### Container Configuration Options

```python
from dataclasses import dataclass, field
from typing import Literal, Dict, Any

class ContainerExecutionConfig:
    """Configuration for container execution modes"""
    
    @dataclass
    class ContainerConfig:
        isolation_level: Literal["full", "process", "thread"] = "full"
        resource_limits: Dict[str, Any] = field(default_factory=dict)
        restart_policy: str = "on_failure"
        parallelism: int = 1  # Number of parallel containers
        
    @dataclass
    class FunctionConfig:
        runtime: Literal["local", "lambda", "k8s_jobs"] = "local"
        timeout_seconds: int = 300
        memory_limit_mb: int = 512
        
execution_config = {
    "parameter_discovery": {
        "model": "containers",  # Parallel containers for parameter sweep
        "config": ContainerExecutionConfig.ContainerConfig(
            isolation_level="full",
            parallelism=100  # Run 100 parameter combinations in parallel
        )
    },
    "backtesting": {
        "model": "containers",  # Full isolation
        "config": ContainerExecutionConfig.ContainerConfig(isolation_level="full")
    },
    "live_trading": {
        "model": "containers",  # Maximum reliability  
        "config": ContainerExecutionConfig.ContainerConfig(
            isolation_level="full",
            restart_policy="always"
        )
    },
    "signal_replay": {
        "model": "functions",  # Lightweight and fast
        "config": ContainerExecutionConfig.FunctionConfig(runtime="local")
    }
}
```

### Communication Adapter Implementations

```python
class CommunicationAdapter(ABC):
    """Base communication adapter"""
    @abstractmethod
    def setup_flow(self, containers: List['Container']) -> None:
        pass

class PipelineCommunicationAdapter(CommunicationAdapter):
    """Pipeline adapter for sequential data flow"""
    
    def __init__(self, containers: List[str]):
        self.containers = containers
        
    def setup_flow(self, containers: List['Container']) -> None:
        """Set up pipeline flow between containers"""
        for i, container in enumerate(containers[:-1]):
            next_container = containers[i + 1]
            
            # Set up event forwarding
            container.on_output_event(
                lambda event: self.forward_event(event, next_container)
            )
    
    def forward_event(self, event: SemanticEventBase, target: 'Container'):
        """Forward event to target container"""
        # Events are forwarded through the container's event bus
        target.receive_event(event)

class IsolatedPipelineAdapter(PipelineCommunicationAdapter):
    """Pipeline adapter with serialization boundary for complete isolation"""
    
    def forward_event(self, event: SemanticEventBase, target: 'Container'):
        """Forward event with full serialization for isolation"""
        # Serialize event to ensure complete isolation
        serialized = pickle.dumps(event)
        deserialized = pickle.loads(serialized)
        
        # Forward to isolated container
        target.receive_event(deserialized)

class ParallelPipelineAdapter(PipelineCommunicationAdapter):
    """Pipeline adapter that supports parallel container execution"""
    
    def __init__(self, containers: List[str], parallelism: int = 1):
        super().__init__(containers)
        self.parallelism = parallelism
        self.container_pool = {}
        
    def setup_flow(self, containers: List['Container']) -> None:
        """Set up pipeline with parallel execution support"""
        # Create pools of containers for parallel stages
        for container_name in self.containers:
            if self.should_parallelize(container_name):
                # Create multiple instances for parallel execution
                self.container_pool[container_name] = [
                    self.create_container_instance(container_name) 
                    for _ in range(self.parallelism)
                ]
            else:
                # Single instance for sequential stages
                self.container_pool[container_name] = [containers[container_name]]
        
        # Set up load-balanced forwarding
        self._setup_load_balanced_forwarding()
    
    def should_parallelize(self, container_name: str) -> bool:
        """Determine if a container stage should be parallelized"""
        # Parallelize compute-intensive stages like strategy evaluation
        return container_name in ['strategy_container', 'optimization_container']
```

## Workspace-Aware Communication Adapters

### File-Based Communication Integration

```python
class WorkspaceAwareAdapter(CommunicationAdapter):
    """Adapter that integrates with workspace management for multi-phase workflows"""
    
    def __init__(self, workspace_manager):
        self.workspace_manager = workspace_manager
        self.phase_outputs = {}
        
    def setup_phase_communication(self, phase_name: str, 
                                 input_containers: List['Container'],
                                 output_containers: List['Container']):
        """Set up communication that writes to workspace"""
        
        # Set up input from previous phase files
        for container in input_containers:
            input_path = self.workspace_manager.get_phase_input_path(phase_name)
            container.set_input_source(FileBasedInputSource(input_path))
            
        # Set up output to workspace files  
        for container in output_containers:
            output_path = self.workspace_manager.get_phase_output_path(phase_name)
            container.add_output_sink(FileBasedOutputSink(output_path))

class FileBasedInputSource:
    """Input source that reads from workspace files"""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        
    def get_events(self) -> Iterator[SemanticEventBase]:
        """Stream events from file"""
        if self.file_path.suffix == '.jsonl':
            with open(self.file_path) as f:
                for line in f:
                    event_data = json.loads(line)
                    # Reconstruct semantic event from JSON
                    event = self.deserialize_event(event_data)
                    yield event
    
    def deserialize_event(self, event_data: Dict[str, Any]) -> SemanticEventBase:
        """Deserialize event from JSON data"""
        # Implementation would map event types to classes
        pass

class FileBasedOutputSink:
    """Output sink that writes to workspace files"""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.file_handle = None
        
    def write_event(self, event: SemanticEventBase):
        """Write semantic event to file"""
        if not self.file_handle:
            self.file_handle = open(self.file_path, 'w')
            
        # Serialize semantic event to JSON
        event_json = self.serialize_event(event)
        self.file_handle.write(json.dumps(event_json) + '\n')
        self.file_handle.flush()
    
    def serialize_event(self, event: SemanticEventBase) -> Dict[str, Any]:
        """Serialize event to JSON-compatible dict"""
        # Implementation would handle datetime conversion etc
        pass
```

## Enhanced Communication Patterns

### 1. Pipeline Pattern with Semantic Awareness

```python
class SemanticPipelineAdapter(PipelineCommunicationAdapter):
    """Pipeline adapter with semantic event transformation"""
    
    def __init__(self):
        super().__init__()
        self.event_transformers = {}
        
    def register_transformation(self, input_type: type, output_type: type, 
                              transformer: callable):
        """Register semantic event transformation"""
        self.event_transformers[(input_type, output_type)] = transformer
        
    def transform_and_forward(self, event: SemanticEventBase, target: 'Container'):
        """Transform semantic event for target container"""
        # Determine target event type based on container capabilities
        target_event_type = self.get_target_event_type(target)
        
        # Find appropriate transformer
        transformer_key = (type(event), target_event_type)
        if transformer_key in self.event_transformers:
            transformer = self.event_transformers[transformer_key]
            transformed_event = transformer(event)
        else:
            # Default: pass through if compatible
            transformed_event = event
            
        # Preserve correlation and causation
        if hasattr(transformed_event, 'causation_id'):
            transformed_event.causation_id = event.event_id
            transformed_event.correlation_id = event.correlation_id
            
        target.receive_event(transformed_event)
    
    def get_target_event_type(self, target: 'Container') -> type:
        """Get expected event type for target container"""
        # Implementation would check container capabilities
        pass

# Example transformations
def indicator_to_signal_transform(indicator_event: IndicatorEvent) -> TradingSignal:
    """Transform indicator event to trading signal"""
    return TradingSignal(
        correlation_id=indicator_event.correlation_id,
        causation_id=indicator_event.event_id,
        symbol=indicator_event.metadata.get('symbol', ''),
        action="BUY" if indicator_event.value > 0 else "SELL",
        strength=abs(indicator_event.value),
        strategy_id=indicator_event.strategy_id,
        regime_context=indicator_event.regime_context
    )

def signal_to_order_transform(signal_event: TradingSignal) -> OrderEvent:
    """Transform trading signal to order"""
    return OrderEvent(
        correlation_id=signal_event.correlation_id,
        causation_id=signal_event.event_id,
        symbol=signal_event.symbol,
        side="BUY" if signal_event.action == "BUY" else "SELL",
        order_type="MARKET",
        quantity=100,  # Position sizing logic would go here
        strategy_id=signal_event.strategy_id
    )
```

### 2. Hierarchical Pattern with Context Propagation

```python
class HierarchicalCommunicationAdapter(CommunicationAdapter):
    """Base hierarchical adapter"""
    pass

class SemanticHierarchicalAdapter(HierarchicalCommunicationAdapter):
    """Hierarchical adapter with semantic context propagation"""
    
    def broadcast_to_children(self, event: SemanticEventBase, children: List['Container']):
        """Broadcast context event to children with semantic enrichment"""
        for child in children:
            # Clone event and enrich with child-specific context
            child_event = copy.deepcopy(event)
            
            # Add child-specific context
            if hasattr(child_event, 'regime_context') and isinstance(event, RegimeChangeEvent):
                child_event.regime_context = event.new_regime
                
            # Update source tracking
            child_event.causation_id = event.event_id
            child_event.source_container = child.name
            
            child.receive_context_event(child_event)
            
    def aggregate_to_parent(self, event: SemanticEventBase, parent: 'Container'):
        """Aggregate child results with semantic correlation"""
        # Create aggregation event
        aggregated_event = copy.deepcopy(event)
        aggregated_event.causation_id = event.event_id
        
        # Add aggregation metadata
        if not hasattr(aggregated_event, 'metadata'):
            aggregated_event.metadata = {}
        aggregated_event.metadata['aggregated_from'] = event.source_container
        
        parent.receive_child_result(aggregated_event)
```

### 3. Broadcast Pattern with Semantic Filtering

```python
class BroadcastCommunicationAdapter(CommunicationAdapter):
    """Base broadcast adapter"""
    pass

class SemanticBroadcastAdapter(BroadcastCommunicationAdapter):
    """Broadcast adapter with semantic event filtering"""
    
    def __init__(self):
        super().__init__()
        self.filters = {}
        
    def add_semantic_filter(self, target: 'Container', filter_fn: callable):
        """Add semantic filter for specific target"""
        self.filters[target.name] = filter_fn
        
    def broadcast_to_targets(self, event: SemanticEventBase, targets: List['Container']):
        """Broadcast with semantic filtering"""
        for target in targets:
            # Apply semantic filter if exists
            if target.name in self.filters:
                filter_fn = self.filters[target.name]
                if not filter_fn(event):
                    continue  # Skip this target
                    
            # Clone and enrich event for target
            target_event = copy.deepcopy(event)
            target_event.causation_id = event.event_id
            
            target.receive_event(target_event)

# Example semantic filters
def momentum_strategy_filter(event: SemanticEventBase) -> bool:
    """Filter for momentum strategy - only accept trend indicators"""
    if isinstance(event, IndicatorEvent):
        return event.indicator_name in ['MACD', 'RSI', 'SMA']
    return True

def high_confidence_filter(event: SemanticEventBase) -> bool:
    """Filter for high-confidence signals only"""
    if isinstance(event, TradingSignal):
        return event.strength > 0.7
    return True
```

### 4. Selective Pattern (For Complex Routing)
**Best for**: Conditional event routing based on content

```python
class SelectiveCommunicationAdapter(CommunicationAdapter):
    """Route events based on content and rules"""
    
    def __init__(self):
        self.routing_rules = {}
    
    def add_routing_rule(self, condition: Callable, target: 'Container') -> None:
        """Add conditional routing rule"""
        self.routing_rules[condition] = target
    
    def setup_selective_routing(self, source: 'Container') -> None:
        """Set up content-based routing"""
        
        source.on_event(
            lambda event: self.route_event(event)
        )
    
    def route_event(self, event: SemanticEventBase) -> None:
        """Route event based on rules"""
        for condition, target in self.routing_rules.items():
            if condition(event):
                target.receive_event(event)
                break

# Usage
selective_adapter = SelectiveCommunicationAdapter()
selective_adapter.add_routing_rule(
    lambda event: hasattr(event, 'regime') and event.regime == "BULL", 
    aggressive_risk_container
)
selective_adapter.add_routing_rule(
    lambda event: hasattr(event, 'regime') and event.regime == "BEAR", 
    conservative_risk_container
)
selective_adapter.setup_selective_routing(classifier_container)
```

## Configuration Integration

### Enhanced YAML Configuration with Semantic Events

```yaml
# Event communication with semantic awareness
event_communication:
  semantic_events: true
  schema_registry: "global"
  
  # Execution configuration per phase
  execution_config:
    parameter_discovery:
      model: "containers"
      config:
        parallelism: 100  # Run 100 parallel containers
        isolation_level: "full"
    signal_replay:
      model: "functions" 
      config:
        runtime: "local"
        timeout_seconds: 60
    backtesting:
      model: "containers"
      config:
        isolation_level: "full"
        
  adapters:
    - type: "semantic_pipeline"
      name: "main_data_flow"
      containers: ["data", "indicators", "strategies", "risk", "execution"]
      transformations:
        - from: "IndicatorEvent"
          to: "TradingSignal"
          transformer: "indicator_to_signal_transform"
        - from: "TradingSignal" 
          to: "OrderEvent"
          transformer: "signal_to_order_transform"
      tier: "standard"
      
    - type: "semantic_hierarchical"
      name: "regime_distribution"
      parent: "hmm_classifier"
      children: ["conservative_risk", "aggressive_risk"]
      context_propagation:
        - event_type: "RegimeChangeEvent"
          enrich_children: true
      tier: "reliable"
      
    - type: "semantic_broadcast"
      name: "filtered_distribution"
      source: "indicator_hub"
      targets: 
        - container: "momentum_strategy"
          filters: ["momentum_strategy_filter"]
        - container: "mean_reversion_strategy"
          filters: ["mean_reversion_filter"]
      tier: "fast"
      
    - type: "workspace_aware"
      name: "phase_coordination"
      workspace_integration: true
      phase_inputs:
        signal_replay: "signals/trial_*.jsonl"
        validation: "analysis/ensemble_weights.json"

# Schema evolution configuration
semantic_schemas:
  evolution_policy: "forward_compatible"
  migration_timeout: 30
  
  event_types:
    TradingSignal:
      current_version: "2.0.0"
      supported_versions: ["1.0.0", "1.1.0", "2.0.0"]
      migrations:
        "1.0.0->2.0.0": "migrate_trading_signal_v1_to_v2"
        
    OrderEvent:
      current_version: "1.0.0"
      supported_versions: ["1.0.0"]

# Container organization configured separately
organization: "classifier_first"
classifiers:
  - name: "hmm_classifier"
    # ... classifier config
```

## Adapter Factory Pattern

```python
class EventCommunicationFactory:
    """Factory to create appropriate communication adapters"""
    
    def __init__(self):
        self.adapters = {
            'pipeline': PipelineCommunicationAdapter,
            'semantic_pipeline': SemanticPipelineAdapter,
            'hierarchical': HierarchicalCommunicationAdapter,
            'semantic_hierarchical': SemanticHierarchicalAdapter,
            'broadcast': BroadcastCommunicationAdapter,
            'semantic_broadcast': SemanticBroadcastAdapter,
            'selective': SelectiveCommunicationAdapter,
            'workspace_aware': WorkspaceAwareAdapter
        }
    
    def create_communication_layer(self, config: Dict, containers: Dict[str, 'Container']) -> 'CommunicationLayer':
        """Create communication layer from config"""
        
        communication_layer = CommunicationLayer()
        
        for adapter_config in config.get('adapters', []):
            adapter_type = adapter_config['type']
            adapter_class = self.adapters[adapter_type]
            adapter = adapter_class()
            
            # Configure adapter based on type
            self.configure_adapter(adapter, adapter_config, containers)
            communication_layer.add_adapter(adapter)
        
        return communication_layer
    
    def configure_adapter(self, adapter, config: Dict, containers: Dict[str, 'Container']) -> None:
        """Configure specific adapter based on its type"""
        
        if isinstance(adapter, SemanticPipelineAdapter):
            container_names = config['containers']
            container_list = [containers[name] for name in container_names]
            
            # Register transformations
            for transform in config.get('transformations', []):
                transformer_fn = globals()[transform['transformer']]
                adapter.register_transformation(
                    globals()[transform['from']],
                    globals()[transform['to']],
                    transformer_fn
                )
            
            adapter.setup_flow(container_list)
            
        elif isinstance(adapter, PipelineCommunicationAdapter):
            container_names = config['containers']
            container_list = [containers[name] for name in container_names]
            adapter.setup_flow(container_list)
            
        elif isinstance(adapter, (HierarchicalCommunicationAdapter, SemanticHierarchicalAdapter)):
            parent = containers[config['parent']]
            children = [containers[name] for name in config['children']]
            adapter.setup_hierarchy(parent, children)
            
        elif isinstance(adapter, (BroadcastCommunicationAdapter, SemanticBroadcastAdapter)):
            source = containers[config['source']]
            targets = []
            
            # Handle semantic filtering
            if isinstance(adapter, SemanticBroadcastAdapter) and 'targets' in config:
                for target_config in config['targets']:
                    if isinstance(target_config, dict):
                        container = containers[target_config['container']]
                        targets.append(container)
                        
                        # Add filters
                        for filter_name in target_config.get('filters', []):
                            filter_fn = globals()[filter_name]
                            adapter.add_semantic_filter(container, filter_fn)
                    else:
                        # Simple string target
                        targets.append(containers[target_config])
            else:
                targets = [containers[name] for name in config.get('targets', [])]
                
            adapter.setup_broadcast(source, targets)
            
        elif isinstance(adapter, SelectiveCommunicationAdapter):
            source = containers[config['source']]
            for rule in config['rules']:
                condition = self.parse_condition(rule['condition'])
                target = containers[rule['target']]
                adapter.add_routing_rule(condition, target)
            adapter.setup_selective_routing(source)
```

## Container-Based Execution Strategy

### When Full Container Isolation is Best

Given the sophisticated workspace management and multi-organizational support:

1. **Research Reproducibility**: Container isolation ensures identical runs across parameter sweeps
2. **Multi-Phase Workflows**: File-based communication between phases requires process boundaries
3. **Organizational Flexibility**: Different container hierarchies for different organizational patterns
4. **Production Deployment**: Live trading requires maximum isolation and fault tolerance
5. **Compliance and Auditing**: Container-level logging and resource tracking

### When Lightweight Functions Work Well

1. **Signal Replay**: Fast iteration over pre-computed signals
2. **Simple Transformations**: Stateless data transformations
3. **Quick Analysis**: Short-lived computational tasks
4. **Utility Operations**: File I/O, data formatting, etc.

### Parallel Container Strategy

```yaml
# Use different execution strategies for different phases
workflow_phases:
  parameter_discovery:
    execution_model: "containers"  # Parallel container instances
    parallelism: 100               # 100 concurrent containers
    
  regime_analysis:
    execution_model: "containers"  # Isolation for complex analysis
    parallelism: 10                # Moderate parallelism
    
  signal_replay:
    execution_model: "functions"   # Lightweight execution
    parallelism: 50                # Function-level concurrency
    
  validation:
    execution_model: "containers"  # Reliable final validation
    parallelism: 1                 # Single instance for consistency
    
  live_trading:
    execution_model: "containers"  # Maximum reliability
    parallelism: 1                 # Single instance, fault-tolerant
```

## Benefits of Enhanced Approach

### 1. Semantic Clarity
- **Type safety**: Catch errors at configuration time
- **Schema evolution**: Handle versioning and migration automatically
- **Rich metadata**: Full correlation and causation tracking
- **Domain modeling**: Events model trading domain concepts

### 2. Execution Flexibility  
- **Model selection**: Choose optimal execution model per phase
- **Performance optimization**: Parallel containers for compute-intensive work, isolated containers for production
- **Resource efficiency**: Functions for lightweight tasks
- **Scaling options**: Horizontal scale with appropriate isolation

### 3. Workspace Integration
- **Multi-phase coordination**: File-based communication between phases
- **Resumable workflows**: Checkpoint and resume long-running optimizations
- **Debugging support**: Inspect intermediate results between phases
- **Compliance**: Complete audit trail of all decisions

### 4. Organizational Agnostic
- **Pattern independence**: Same adapters work with any organizational approach
- **Configuration flexibility**: Change patterns without code changes
- **Migration support**: Convert between organizational styles
- **Team compatibility**: Support different team preferences

## Default Communication Patterns by Organization

### Strategy-First Default
```yaml
event_communication:
  adapters:
    - type: "semantic_pipeline"
      containers: ["data", "strategy_a", "execution"]
    - type: "semantic_pipeline"  
      containers: ["data", "strategy_b", "execution"]
    - type: "semantic_broadcast"
      source: "execution"
      targets: ["performance_tracker"]
```

### Classifier-First Default
```yaml
event_communication:
  adapters:
    - type: "semantic_pipeline"
      containers: ["data", "indicators", "classifier"]
    - type: "semantic_hierarchical"
      parent: "classifier"
      children: ["risk_conservative", "risk_aggressive"]
    - type: "semantic_pipeline"
      containers: ["risk_conservative", "execution"]
    - type: "semantic_pipeline"
      containers: ["risk_aggressive", "execution"]
```

### Risk-First Default
```yaml
event_communication:
  adapters:
    - type: "semantic_broadcast"
      source: "data"
      targets: ["strategy_a", "strategy_b"]
    - type: "semantic_pipeline"
      containers: ["strategy_a", "risk_manager", "execution"]
    - type: "semantic_pipeline"
      containers: ["strategy_b", "risk_manager", "execution"]
```

## Implementation Strategy

### Phase 1: Basic Adapters
- Implement Pipeline and Broadcast adapters
- Create configuration system
- Migrate existing isolated buses to use adapters

### Phase 2: Advanced Patterns
- Add Hierarchical and Selective adapters
- Implement adapter factory and auto-configuration
- Add monitoring and debugging tools

### Phase 3: Semantic Events
- Define semantic event types
- Implement schema evolution
- Add type-safe transformations

### Phase 4: Optimization
- Add performance monitoring for communication patterns
- Implement adapter performance optimizations
- Add adaptive communication pattern selection

## Implementation Considerations

### Adapter Lifecycle Management

```python
import logging
logger = logging.getLogger(__name__)

class CommunicationLayer:
    """Manages adapter lifecycle and coordination"""
    
    def __init__(self):
        self.adapters: Dict[str, CommunicationAdapter] = {}
        self.adapter_health: Dict[str, bool] = {}
        
    async def startup(self):
        """Initialize all adapters in correct order"""
        # Start workspace-aware adapters first (they setup file I/O)
        workspace_adapters = [a for a in self.adapters.values() 
                            if isinstance(a, WorkspaceAwareAdapter)]
        for adapter in workspace_adapters:
            await adapter.initialize()
            
        # Then start other adapters
        for name, adapter in self.adapters.items():
            if adapter not in workspace_adapters:
                await adapter.initialize()
                self.adapter_health[name] = True
                
    async def shutdown(self):
        """Graceful shutdown with cleanup"""
        for adapter in self.adapters.values():
            try:
                await adapter.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up adapter: {e}")
                
    async def health_check(self) -> Dict[str, Any]:
        """Monitor adapter health"""
        health_status = {}
        for name, adapter in self.adapters.items():
            health_status[name] = {
                "healthy": self.adapter_health.get(name, False),
                "metrics": await adapter.get_metrics()
            }
        return health_status
    
    def add_adapter(self, adapter: CommunicationAdapter):
        """Add adapter to layer"""
        name = getattr(adapter, 'name', adapter.__class__.__name__)
        self.adapters[name] = adapter
```

### Error Handling and Recovery

```python
class AdapterErrorHandling:
    """Comprehensive error handling for adapters"""
    
    def __init__(self):
        self.error_counts = {}
        self.circuit_breakers = {}
        
    def on_transformation_error(self, event: SemanticEventBase, error: Exception):
        """Handle transformation failures"""
        logger.error(f"Transformation error for event {event.event_id}: {error}")
        
        # Try to forward original event
        if event.validate():
            self.forward_as_fallback(event)
        else:
            self.send_to_dead_letter(event, error)
            
    def on_delivery_failure(self, event: SemanticEventBase, target: str, error: Exception):
        """Handle delivery failures with retry logic"""
        key = f"{event.event_id}:{target}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        
        if self.error_counts[key] < 3:  # Retry up to 3 times
            self.schedule_retry(event, target, delay=2 ** self.error_counts[key])
        else:
            self.trigger_circuit_breaker(target)
            self.send_to_dead_letter(event, error)
            
    def on_validation_failure(self, event: SemanticEventBase):
        """Handle invalid events"""
        logger.error(f"Validation failed for event {event.event_id}")
        self.send_to_dead_letter(event, ValueError("Validation failed"))
    
    def forward_as_fallback(self, event: SemanticEventBase):
        """Implementation for fallback forwarding"""
        pass
    
    def send_to_dead_letter(self, event: SemanticEventBase, error: Exception):
        """Implementation for dead letter queue"""
        pass
    
    def schedule_retry(self, event: SemanticEventBase, target: str, delay: int):
        """Implementation for retry scheduling"""
        pass
    
    def trigger_circuit_breaker(self, target: str):
        """Implementation for circuit breaker"""
        pass
```

### Performance Optimization

```python
class PerformanceTiers:
    """Performance tiers for different event types"""
    
    FAST = {
        "batch_size": 1000,
        "timeout_ms": 10,
        "delivery_guarantee": "at_most_once",
        "serialization": "msgpack"
    }
    
    STANDARD = {
        "batch_size": 100,
        "timeout_ms": 100,
        "delivery_guarantee": "at_least_once",
        "serialization": "json"
    }
    
    RELIABLE = {
        "batch_size": 1,
        "timeout_ms": 1000,
        "delivery_guarantee": "exactly_once",
        "serialization": "protobuf",
        "persistence": True
    }
```

### Testing Strategy

```python
class MockSemanticAdapter(CommunicationAdapter):
    """Test adapter that records all events"""
    
    def __init__(self):
        self.events_sent = []
        self.events_received = []
        self.transformations_applied = []
        
    async def send_event(self, event: SemanticEventBase, target: str):
        """Record sent events"""
        self.events_sent.append((event, target))
        
    def verify_event_flow(self, expected_flow: List[Tuple[type, type]]):
        """Verify events flowed as expected"""
        actual_flow = [(type(e[0]), e[1]) for e in self.events_sent]
        assert actual_flow == expected_flow
    
    def setup_flow(self, containers: List['Container']) -> None:
        """Mock setup"""
        pass

class AdapterTestHarness:
    """Test harness for adapter configurations"""
    
    def test_semantic_transformation(self):
        """Test event transformation logic"""
        adapter = SemanticPipelineAdapter()
        adapter.register_transformation(
            IndicatorEvent, TradingSignal, indicator_to_signal_transform
        )
        
        # Create test event
        indicator = IndicatorEvent(
            indicator_name="RSI",
            value=0.7,
            metadata={"symbol": "AAPL"}
        )
        
        # Transform
        signal = adapter.transform_event(indicator, TradingSignal)
        
        # Verify
        assert isinstance(signal, TradingSignal)
        assert signal.symbol == "AAPL"
        assert signal.action == "BUY"
        assert signal.causation_id == indicator.event_id
```

## Implementation Checklist

### Phase 1: Foundation (Week 1-2)
- [ ] Define all semantic event types
- [ ] Implement SemanticEventBase with validation
- [ ] Create schema registry with basic migrations
- [ ] Build EventCommunicationFactory

### Phase 2: Core Adapters (Week 3-4)
- [ ] Implement SemanticPipelineAdapter with transformations
- [ ] Implement SemanticHierarchicalAdapter with context
- [ ] Add error handling and retry logic
- [ ] Create adapter lifecycle management

### Phase 3: Integration (Week 5-6)
- [ ] Integrate with existing containers
- [ ] Update Coordinator to use adapters
- [ ] Add monitoring and metrics
- [ ] Implement health checks

### Phase 4: Testing & Deployment (Week 7-8)
- [ ] Unit tests for each adapter type
- [ ] Integration tests with containers
- [ ] Performance benchmarks
- [ ] Migration from existing system

## Production Considerations

### Deployment
- Adapter configuration hot-reload without restarts
- Rolling updates with backward compatibility
- Feature flags for gradual rollout

### Monitoring
- Event flow visualization dashboard
- Latency tracking per adapter and transformation
- Error rate alerts with automatic remediation
- Dead letter queue monitoring

### Scaling
- Horizontal scaling of adapter instances
- Backpressure handling with flow control
- Resource limits and quotas per adapter
- Auto-scaling based on event throughput

## Conclusion

The enhanced adapter system provides:

1. **Semantic Events**: Type-safe, versioned, correlation-aware event system
2. **Execution Flexibility**: Choose optimal execution model per use case
3. **Workspace Integration**: Seamless multi-phase workflow coordination
4. **Pattern Independence**: Support all organizational approaches
5. **Production Readiness**: Reliability, tracing, and compliance features

The key insight is that **different phases of trading system workflows have different requirements** - parameter sweeps need massive parallelization (parallel containers), live trading needs maximum reliability (isolated containers), and signal replay needs fast turnaround (lightweight functions). The adapter system enables this flexibility while maintaining semantic consistency and organizational pattern independence.