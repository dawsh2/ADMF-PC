# Event Communication Adapter Implementation Guide

## Table of Contents
1. [Problem Context & Motivation](#problem-context--motivation)
2. [Solution Architecture](#solution-architecture)
3. [Phase 1: Foundation Implementation](#phase-1-foundation-implementation)
4. [Phase 2: Coordinator Integration](#phase-2-coordinator-integration)
5. [Phase 3: Additional Adapters](#phase-3-additional-adapters)
6. [Phase 4: Production Deployment](#phase-4-production-deployment)
7. [Testing Strategy](#testing-strategy)
8. [Migration Guide](#migration-guide)

## Problem Context & Motivation

### The Circular Dependency Problem

During multi-strategy backtest runs, the system experiences circular dependencies in event routing:

```bash
$ python main.py --config config/multi_strategy_test.yaml --bars 50

# Results in:
WARNING - Event routing cycle detected!
WARNING - Duplicate trade generated
ERROR - Negative cash balance: -$100,304
ERROR - Signal rejected: Failed risk checks
INFO - Bars processed: 0  # Despite --bars 50
```

### Root Cause Analysis

The current implementation conflates two orthogonal concerns:

1. **Container Hierarchy** (Configuration & Organization)
   ```
   ClassifierContainer
   └── RiskContainer
       └── PortfolioContainer
           └── StrategyContainer
   ```

2. **Event Flow** (Trading Pipeline)
   ```
   Data → Indicators → Strategies → Risk → Execution
   ```

When containers use external event routing for all communication, circular dependencies emerge because the hierarchy and flow patterns conflict.

### Why Adapters Are Essential

Beyond fixing the immediate issue, the system needs flexibility for:

1. **Multi-Phase Research Workflows**
   - Phase 1: Parameter discovery needs full pipeline
   - Phase 2: Regime analysis needs analysis-only routing
   - Phase 3: Signal replay needs to skip data/indicators

2. **Dynamic Performance Routing**
   - Route high-performing strategies to aggressive risk
   - Route poor performers to conservative risk
   - Change routing every 100 bars based on rolling Sharpe

3. **Distributed Deployment**
   - Run 10,000 parameter combinations
   - Distribute across AWS instances
   - Maintain same configuration for local and distributed

4. **A/B Testing Communication Patterns**
   - Test broadcast vs selective routing impact
   - Isolate communication effects from strategy logic

## Solution Architecture

### Event Communication Adapters

Separate container organization from event communication using pluggable adapters:

```python
# Instead of hard-coded routing in BacktestContainer:
if self.mode == 'backtest':
    self._route_backtest_pipeline(event)
elif self.mode == 'live':
    self._route_live_pipeline(event)
# ... 500 lines of routing logic

# Use configurable adapters:
communication_config = load_config("phase1_pipeline.yaml")
communication_layer = EventCommunicationFactory().create_communication_layer(
    communication_config, containers
)
```

### Adapter Types

1. **Pipeline Adapter** - Linear flow transformation
2. **Hierarchical Adapter** - Parent-child with context
3. **Broadcast Adapter** - One-to-many distribution
4. **Selective Adapter** - Rule-based routing

## Phase 1: Foundation Implementation

### Step 1: Create Base Adapter Interface

Create the directory structure:
```bash
mkdir -p src/core/communication
touch src/core/communication/__init__.py
touch src/core/communication/base_adapter.py
```

Implement `src/core/communication/base_adapter.py`:

```python
# src/core/communication/base_adapter.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import uuid
import time
from datetime import datetime

class CommunicationAdapter(ABC):
    """Base interface for all communication adapters"""
    
    def __init__(self, coordinator_id: str, log_manager: 'LogManager'):
        self.coordinator_id = coordinator_id
        self.log_manager = log_manager
        self.adapter_id = f"{self.__class__.__name__}_{uuid.uuid4().hex[:8]}"
        
        # Create adapter-specific logger
        from src.core.logging.container_logger import ContainerLogger
        self.logger = ContainerLogger(
            coordinator_id,
            f"adapter_{self.__class__.__name__.lower()}",
            base_log_dir=str(log_manager.base_log_dir)
        )
        
        # Adapter metrics
        self.metrics = {
            'events_processed': 0,
            'errors': 0,
            'total_latency_ms': 0,
            'start_time': datetime.utcnow()
        }
        
        self.logger.log_info(
            "Communication adapter initialized",
            adapter_type=self.__class__.__name__,
            adapter_id=self.adapter_id,
            lifecycle_operation="adapter_initialization"
        )
    
    @abstractmethod
    def setup(self, config: Dict[str, Any], containers: Dict[str, Any]) -> None:
        """Setup adapter with given configuration"""
        pass
    
    def track_event(self, event_id: str, operation: str, **context):
        """Track event processing for metrics"""
        self.metrics['events_processed'] += 1
        
        self.logger.log_debug(
            f"Adapter event: {operation}",
            event_id=event_id,
            adapter_id=self.adapter_id,
            **context
        )
    
    def track_error(self, error: Exception, **context):
        """Track adapter errors"""
        self.metrics['errors'] += 1
        
        self.logger.log_error(
            "Adapter error occurred",
            error=str(error),
            adapter_id=self.adapter_id,
            **context
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get adapter performance metrics"""
        uptime_seconds = (datetime.utcnow() - self.metrics['start_time']).total_seconds()
        
        return {
            'adapter_type': self.__class__.__name__,
            'adapter_id': self.adapter_id,
            'events_processed': self.metrics['events_processed'],
            'errors': self.metrics['errors'],
            'error_rate': self.metrics['errors'] / max(self.metrics['events_processed'], 1),
            'average_latency_ms': self.metrics['total_latency_ms'] / max(self.metrics['events_processed'], 1),
            'uptime_seconds': uptime_seconds,
            'events_per_second': self.metrics['events_processed'] / max(uptime_seconds, 1)
        }
    
    def cleanup(self):
        """Cleanup adapter resources"""
        self.logger.log_info(
            "Cleaning up communication adapter",
            adapter_id=self.adapter_id,
            final_metrics=self.get_metrics(),
            lifecycle_operation="adapter_cleanup"
        )
        self.logger.close()
```

### Step 2: Implement Pipeline Adapter

Create `src/core/communication/pipeline_adapter.py`:

```python
# src/core/communication/pipeline_adapter.py
from .base_adapter import CommunicationAdapter
from typing import List, Callable, Any, Dict
import asyncio
import time
import uuid

class PipelineCommunicationAdapter(CommunicationAdapter):
    """Linear pipeline communication adapter"""
    
    def __init__(self, coordinator_id: str, log_manager: 'LogManager'):
        super().__init__(coordinator_id, log_manager)
        self.pipeline_stages: List[Any] = []
        self.event_transformer = EventTransformer()
        
    def setup(self, config: Dict[str, Any], containers: Dict[str, Any]) -> None:
        """Setup linear pipeline flow"""
        container_names = config.get('containers', [])
        self.pipeline_stages = [containers[name] for name in container_names]
        
        self.logger.log_info(
            "Setting up pipeline communication",
            container_count=len(self.pipeline_stages),
            container_names=container_names,
            event_flow="external_standard_tier",
            lifecycle_operation="pipeline_setup"
        )
        
        # Wire up the pipeline
        for i, container in enumerate(self.pipeline_stages[:-1]):
            next_container = self.pipeline_stages[i + 1]
            self._wire_pipeline_stage(container, next_container, i)
    
    def _wire_pipeline_stage(self, source: Any, target: Any, stage_number: int):
        """Wire one stage of the pipeline"""
        
        def pipeline_handler(event):
            return self._handle_pipeline_event(event, source, target, stage_number)
        
        # Connect source output to our handler
        source.on_output_event(pipeline_handler)
        
        self.logger.log_debug(
            "Pipeline stage wired",
            stage_number=stage_number,
            source_container=getattr(source, 'container_id', 'unknown'),
            target_container=getattr(target, 'container_id', 'unknown'),
            event_flow="external_standard_tier"
        )
    
    def _handle_pipeline_event(self, event, source, target, stage_number: int):
        """Handle event flowing through pipeline"""
        start_time = time.time()
        
        try:
            # Set correlation ID if not present
            correlation_id = getattr(event, 'correlation_id', None) or f"pipeline_{uuid.uuid4().hex[:8]}"
            if hasattr(event, 'set_correlation_id'):
                event.set_correlation_id(correlation_id)
            
            # Track the event
            self.track_event(
                event_id=getattr(event, 'id', 'unknown'),
                operation="pipeline_route",
                stage_number=stage_number,
                source_container=getattr(source, 'container_id', 'unknown'),
                target_container=getattr(target, 'container_id', 'unknown'),
                correlation_id=correlation_id,
                event_flow="external_standard_tier"
            )
            
            # Transform event if needed
            transformed_event = self.event_transformer.transform(event, source, target)
            
            # Forward to target
            target.receive_event(transformed_event)
            
            # Track latency
            latency_ms = (time.time() - start_time) * 1000
            self.metrics['total_latency_ms'] += latency_ms
            
            self.logger.log_debug(
                "Pipeline event processed",
                event_id=getattr(event, 'id', 'unknown'),
                latency_ms=round(latency_ms, 2),
                correlation_id=correlation_id,
                event_flow="external_standard_tier"
            )
            
        except Exception as e:
            self.track_error(
                e,
                event_id=getattr(event, 'id', 'unknown'),
                stage_number=stage_number,
                source_container=getattr(source, 'container_id', 'unknown'),
                target_container=getattr(target, 'container_id', 'unknown'),
                event_flow="external_standard_tier"
            )
            raise


class EventTransformer:
    """Handles event transformation between pipeline stages"""
    
    def __init__(self):
        self.transformation_rules = {
            # Add transformation rules as needed
            ('BAR', 'INDICATOR'): self._transform_bar_to_indicator,
            ('SIGNAL', 'ORDER'): self._transform_signal_to_order,
            ('ORDER', 'FILL'): self._transform_order_to_fill,
        }
    
    def transform(self, event, source, target):
        """Transform event based on source and target types"""
        
        # Get event type
        event_type = getattr(event, 'type', None)
        if not event_type:
            return event  # No transformation needed
        
        # Get target's expected input type
        target_input_type = getattr(target, 'expected_input_type', None)
        if not target_input_type:
            return event  # Target accepts anything
        
        # Check if transformation is needed
        transformation_key = (event_type, target_input_type)
        if transformation_key in self.transformation_rules:
            return self.transformation_rules[transformation_key](event)
        
        return event  # No transformation rule found
    
    def _transform_bar_to_indicator(self, event):
        """Transform BAR event to INDICATOR event"""
        # Implementation depends on your event structure
        return event
    
    def _transform_signal_to_order(self, event):
        """Transform SIGNAL event to ORDER event"""
        # Implementation depends on your event structure
        return event
    
    def _transform_order_to_fill(self, event):
        """Transform ORDER event to FILL event"""
        # Implementation depends on your event structure
        return event
```

### Step 3: Create Communication Factory

Create `src/core/communication/factory.py`:

```python
# src/core/communication/factory.py
from typing import Dict, Any, Type, List
from .base_adapter import CommunicationAdapter
from .pipeline_adapter import PipelineCommunicationAdapter
import numpy as np

class EventCommunicationFactory:
    """Factory for creating communication adapters"""
    
    def __init__(self, coordinator_id: str, log_manager: 'LogManager'):
        self.coordinator_id = coordinator_id
        self.log_manager = log_manager
        
        # Registry of available adapter types
        self.adapter_registry: Dict[str, Type[CommunicationAdapter]] = {
            'pipeline': PipelineCommunicationAdapter,
            # Add more as you implement them
        }
        
        # Create factory logger
        from src.core.logging.container_logger import ContainerLogger
        self.logger = ContainerLogger(
            coordinator_id,
            "communication_factory",
            base_log_dir=str(log_manager.base_log_dir)
        )
        
        self.active_adapters: List[CommunicationAdapter] = []
    
    def create_communication_layer(self, config: Dict[str, Any], containers: Dict[str, Any]) -> 'CommunicationLayer':
        """Create communication layer from configuration"""
        
        self.logger.log_info(
            "Creating communication layer",
            adapter_configs=len(config.get('adapters', [])),
            available_containers=len(containers),
            lifecycle_operation="communication_layer_creation"
        )
        
        communication_layer = CommunicationLayer(self.coordinator_id, self.log_manager)
        
        for adapter_config in config.get('adapters', []):
            adapter = self._create_adapter(adapter_config, containers)
            adapter_name = adapter_config.get('name', f"{adapter_config['type']}_adapter")
            communication_layer.add_adapter(adapter, adapter_name)
            self.active_adapters.append(adapter)
        
        return communication_layer
    
    def _create_adapter(self, adapter_config: Dict[str, Any], containers: Dict[str, Any]) -> CommunicationAdapter:
        """Create single adapter from config"""
        
        adapter_type = adapter_config['type']
        
        if adapter_type not in self.adapter_registry:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
        
        # Create adapter
        adapter_class = self.adapter_registry[adapter_type]
        adapter = adapter_class(self.coordinator_id, self.log_manager)
        
        # Setup adapter
        adapter.setup(adapter_config, containers)
        
        self.logger.log_info(
            "Adapter created",
            adapter_type=adapter_type,
            adapter_id=adapter.adapter_id,
            lifecycle_operation="adapter_creation"
        )
        
        return adapter
    
    def cleanup_all_adapters(self):
        """Cleanup all created adapters"""
        for adapter in self.active_adapters:
            try:
                adapter.cleanup()
            except Exception as e:
                self.logger.log_error(
                    "Error cleaning up adapter",
                    adapter_id=adapter.adapter_id,
                    error=str(e),
                    lifecycle_operation="adapter_cleanup_error"
                )
        
        self.active_adapters.clear()


class CommunicationLayer:
    """Manages all communication adapters"""
    
    def __init__(self, coordinator_id: str, log_manager: 'LogManager'):
        self.coordinator_id = coordinator_id
        self.log_manager = log_manager
        
        from src.core.logging.container_logger import ContainerLogger
        self.logger = ContainerLogger(
            coordinator_id,
            "communication_layer",
            base_log_dir=str(log_manager.base_log_dir)
        )
        
        self.adapters: Dict[str, CommunicationAdapter] = {}
    
    def add_adapter(self, adapter: CommunicationAdapter, name: str):
        """Add adapter to communication layer"""
        self.adapters[name] = adapter
        
        self.logger.log_info(
            "Adapter added to communication layer",
            adapter_name=name,
            adapter_type=type(adapter).__name__,
            total_adapters=len(self.adapters),
            lifecycle_operation="adapter_registration"
        )
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get metrics for all adapters"""
        adapter_metrics = {}
        
        for name, adapter in self.adapters.items():
            adapter_metrics[name] = adapter.get_metrics()
        
        return {
            'total_adapters': len(self.adapters),
            'adapters': adapter_metrics,
            'overall_health': self._calculate_overall_health(adapter_metrics)
        }
    
    def _calculate_overall_health(self, adapter_metrics: Dict[str, Any]) -> str:
        """Calculate overall communication system health"""
        if not adapter_metrics:
            return 'unknown'
        
        error_rates = [metrics['error_rate'] for metrics in adapter_metrics.values()]
        max_error_rate = max(error_rates) if error_rates else 0
        
        if max_error_rate < 0.01:
            return 'healthy'
        elif max_error_rate < 0.05:
            return 'warning'
        else:
            return 'critical'
    
    async def cleanup(self):
        """Cleanup all adapters"""
        self.logger.log_info(
            "Starting communication layer cleanup",
            adapter_count=len(self.adapters),
            lifecycle_operation="communication_layer_cleanup"
        )
        
        for name, adapter in self.adapters.items():
            try:
                adapter.cleanup()
            except Exception as e:
                self.logger.log_error(
                    "Error cleaning up adapter",
                    adapter_name=name,
                    error=str(e),
                    lifecycle_operation="adapter_cleanup_error"
                )
        
        self.logger.close()
```

### Step 4: Create Module Init File

Create `src/core/communication/__init__.py`:

```python
# src/core/communication/__init__.py
"""Event Communication Adapters for ADMF-PC

This module provides pluggable communication adapters that separate
container organization from event communication patterns.
"""

from .base_adapter import CommunicationAdapter
from .pipeline_adapter import PipelineCommunicationAdapter
from .factory import EventCommunicationFactory, CommunicationLayer

__all__ = [
    'CommunicationAdapter',
    'PipelineCommunicationAdapter', 
    'EventCommunicationFactory',
    'CommunicationLayer'
]
```

## Phase 2: Coordinator Integration

### Step 5: Integrate with WorkflowCoordinator

Modify `src/core/coordinator/coordinator.py`:

```python
# src/core/coordinator/coordinator.py (additions to existing class)
from typing import Dict, Any, Optional
from datetime import datetime
from ..communication import EventCommunicationFactory, CommunicationLayer

class WorkflowCoordinator:
    def __init__(self, config: Dict[str, Any]):
        # Your existing initialization...
        self.coordinator_id = config.get('coordinator_id', 'default_coordinator')
        self.config = config
        self.containers = {}
        self.start_time = datetime.utcnow()
        
        # Initialize logging (existing)
        self.log_manager = LogManager(
            config=config.get('logging', {}),
            base_log_dir=config.get('log_dir', 'logs')
        )
        self.logger = self.log_manager.system_logger
        
        # Add communication components
        self.communication_factory = EventCommunicationFactory(
            self.coordinator_id, 
            self.log_manager
        )
        self.communication_layer: Optional[CommunicationLayer] = None
        
        self.logger.log_info(
            "WorkflowCoordinator initialized with communication support",
            coordinator_id=self.coordinator_id,
            lifecycle_operation="coordinator_initialization"
        )
        
    async def setup_communication(self, communication_config: Dict[str, Any]):
        """Setup event communication system"""
        
        self.logger.log_info(
            "Initializing event communication system",
            communication_pattern=communication_config.get('pattern', 'default'),
            adapter_count=len(communication_config.get('adapters', [])),
            lifecycle_operation="communication_initialization"
        )
        
        try:
            # Create communication layer with all registered containers
            self.communication_layer = self.communication_factory.create_communication_layer(
                communication_config,
                self.containers
            )
            
            self.logger.log_info(
                "Event communication system ready",
                active_adapters=len(self.communication_layer.adapters),
                lifecycle_operation="communication_ready"
            )
            
        except Exception as e:
            self.logger.log_error(
                "Failed to initialize communication system",
                error=str(e),
                lifecycle_operation="communication_initialization_error"
            )
            raise
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Enhanced system status with communication metrics"""
        
        # Get base status
        status = {
            "coordinator": {
                "id": self.coordinator_id,
                "active_containers": len(self.containers),
                "uptime_hours": (datetime.utcnow() - self.start_time).total_seconds() / 3600
            },
            "logging": self.log_manager.get_log_summary() if hasattr(self.log_manager, 'get_log_summary') else {},
            "containers": {
                container_id: {
                    "type": type(container).__name__,
                    "status": getattr(container, 'status', 'unknown'),
                }
                for container_id, container in self.containers.items()
            }
        }
        
        # Add communication status
        if self.communication_layer:
            try:
                communication_metrics = self.communication_layer.get_system_metrics()
                status['communication'] = communication_metrics
                
            except Exception as e:
                self.logger.log_error(
                    "Error getting communication status",
                    error=str(e),
                    lifecycle_operation="status_collection_error"
                )
                status['communication'] = {"status": "error", "error": str(e)}
        else:
            status['communication'] = {"status": "not_initialized"}
        
        return status
    
    async def shutdown(self):
        """Enhanced shutdown with communication cleanup"""
        
        self.logger.log_info(
            "Starting coordinator shutdown with communication cleanup",
            lifecycle_operation="coordinator_shutdown"
        )
        
        # Cleanup communication layer first
        if self.communication_layer:
            await self.communication_layer.cleanup()
        
        # Cleanup communication factory
        self.communication_factory.cleanup_all_adapters()
        
        # Continue with existing shutdown process
        for container_id, container in self.containers.items():
            try:
                if hasattr(container, 'cleanup'):
                    await container.cleanup()
                self.logger.log_info(
                    f"Container {container_id} cleaned up",
                    lifecycle_operation="container_cleanup"
                )
            except Exception as e:
                self.logger.log_error(
                    f"Error cleaning up container {container_id}",
                    error=str(e),
                    lifecycle_operation="container_cleanup_error"
                )
        
        # Final logging cleanup
        if hasattr(self.log_manager, 'cleanup_and_archive_logs'):
            await self.log_manager.cleanup_and_archive_logs()
        
        self.logger.log_info(
            "Coordinator shutdown complete",
            lifecycle_operation="coordinator_shutdown_complete"
        )
        
        if hasattr(self.logger, 'close'):
            self.logger.close()
```

### Step 6: Update Container Base Classes

Update containers to support adapter-based communication:

```python
# src/core/containers/enhanced_container.py (additions)
class EnhancedContainer:
    """Enhanced container with adapter support"""
    
    def __init__(self, config: Dict[str, Any], container_id: str):
        # Existing initialization...
        self.container_id = container_id
        self.config = config
        
        # Add output event handlers for adapters
        self._output_event_handlers = []
        
    def on_output_event(self, handler: Callable):
        """Register handler for output events (used by adapters)"""
        self._output_event_handlers.append(handler)
        
    def emit_output_event(self, event):
        """Emit event to all registered handlers"""
        for handler in self._output_event_handlers:
            try:
                handler(event)
            except Exception as e:
                self.logger.log_error(
                    "Error in output event handler",
                    error=str(e),
                    event_type=getattr(event, 'type', 'unknown')
                )
    
    def receive_event(self, event):
        """Receive event from adapter"""
        # Process event based on container logic
        if hasattr(self, 'process_event'):
            self.process_event(event)
        else:
            self.logger.log_warning(
                "Container has no process_event method",
                container_id=self.container_id,
                event_type=getattr(event, 'type', 'unknown')
            )
```

### Step 7: Create Configuration Examples

Create `config/communication_examples.yaml`:

```yaml
# config/communication_examples.yaml

# Simple pipeline for basic workflow
simple_pipeline:
  adapters:
    - type: "pipeline"
      name: "main_flow"
      containers: ["data_container", "indicator_container", "strategy_container", "risk_container", "execution_container"]

# Strategy-first organizational pattern
strategy_first_communication:
  adapters:
    - type: "pipeline"
      name: "data_to_strategies"
      containers: ["data_container", "strategy_container_001"]
    - type: "pipeline"
      name: "data_to_strategies_2"
      containers: ["data_container", "strategy_container_002"]
    - type: "pipeline"
      name: "strategy_1_to_execution"
      containers: ["strategy_container_001", "execution_container"]
    - type: "pipeline"
      name: "strategy_2_to_execution"
      containers: ["strategy_container_002", "execution_container"]

# Multi-phase research workflow
multi_phase_research:
  # Phase 1: Parameter Discovery
  phase_1_communication:
    adapters:
      - type: "pipeline"
        name: "discovery_pipeline"
        containers: ["data_container", "indicator_container", "strategy_container", "risk_container", "execution_container"]
  
  # Phase 2: Regime Analysis
  phase_2_communication:
    adapters:
      - type: "pipeline"
        name: "analysis_pipeline"
        containers: ["results_reader", "regime_analyzer", "report_generator"]
  
  # Phase 3: Signal Replay
  phase_3_communication:
    adapters:
      - type: "pipeline"
        name: "replay_pipeline"
        containers: ["signal_reader", "ensemble_optimizer", "risk_container", "execution_container"]

# Fix for current multi-strategy backtest
multi_strategy_fixed:
  adapters:
    - type: "pipeline"
      name: "main_pipeline"
      containers: ["data_container", "indicator_container", "classifier_container", "risk_container", "portfolio_container", "strategy_container", "execution_container"]
```

## Phase 3: Additional Adapters

### Step 8: Implement Broadcast Adapter

Create `src/core/communication/broadcast_adapter.py`:

```python
# src/core/communication/broadcast_adapter.py
from .base_adapter import CommunicationAdapter
from typing import List, Any, Dict
import time
import uuid

class BroadcastCommunicationAdapter(CommunicationAdapter):
    """One-to-many broadcast adapter"""
    
    def __init__(self, coordinator_id: str, log_manager: 'LogManager'):
        super().__init__(coordinator_id, log_manager)
        self.source_container = None
        self.target_containers: List[Any] = []
    
    def setup(self, config: Dict[str, Any], containers: Dict[str, Any]) -> None:
        """Setup broadcast communication"""
        source_name = config.get('source')
        target_names = config.get('targets', [])
        
        self.source_container = containers[source_name]
        self.target_containers = [containers[name] for name in target_names]
        
        self.logger.log_info(
            "Setting up broadcast communication",
            source_container=source_name,
            target_containers=target_names,
            target_count=len(self.target_containers),
            event_flow="external_standard_tier",
            lifecycle_operation="broadcast_setup"
        )
        
        # Wire up broadcast
        if hasattr(self.source_container, 'on_broadcast_event'):
            self.source_container.on_broadcast_event(self._handle_broadcast_event)
        else:
            # Fallback to generic output event
            self.source_container.on_output_event(self._handle_broadcast_event)
    
    def _handle_broadcast_event(self, event):
        """Handle broadcast event to multiple targets"""
        start_time = time.time()
        
        correlation_id = getattr(event, 'correlation_id', None) or f"broadcast_{uuid.uuid4().hex[:8]}"
        if hasattr(event, 'set_correlation_id'):
            event.set_correlation_id(correlation_id)
        
        self.track_event(
            event_id=getattr(event, 'id', 'unknown'),
            operation="broadcast",
            source_container=getattr(self.source_container, 'container_id', 'unknown'),
            target_count=len(self.target_containers),
            correlation_id=correlation_id,
            event_flow="external_standard_tier"
        )
        
        success_count = 0
        for target in self.target_containers:
            try:
                # Clone event for each target to maintain isolation
                cloned_event = self._clone_event(event)
                target.receive_event(cloned_event)
                success_count += 1
                
                self.logger.log_debug(
                    "Event delivered to target",
                    target_container=getattr(target, 'container_id', 'unknown'),
                    event_id=getattr(event, 'id', 'unknown'),
                    correlation_id=correlation_id
                )
                
            except Exception as e:
                self.track_error(
                    e,
                    target_container=getattr(target, 'container_id', 'unknown'),
                    event_id=getattr(event, 'id', 'unknown'),
                    correlation_id=correlation_id
                )
        
        latency_ms = (time.time() - start_time) * 1000
        self.metrics['total_latency_ms'] += latency_ms
        
        self.logger.log_info(
            "Broadcast completed",
            event_id=getattr(event, 'id', 'unknown'),
            successful_deliveries=success_count,
            total_targets=len(self.target_containers),
            latency_ms=round(latency_ms, 2),
            correlation_id=correlation_id,
            event_flow="external_standard_tier"
        )
    
    def _clone_event(self, event):
        """Clone event for broadcast isolation"""
        # Deep copy implementation depends on your event structure
        # For now, return the same event (you'll want proper cloning in production)
        import copy
        try:
            return copy.deepcopy(event)
        except:
            # Fallback if deepcopy fails
            return event
```

Update the factory to include broadcast adapter:

```python
# In src/core/communication/factory.py, update the registry:
from .broadcast_adapter import BroadcastCommunicationAdapter

class EventCommunicationFactory:
    def __init__(self, coordinator_id: str, log_manager: 'LogManager'):
        # ... existing code ...
        
        # Registry of available adapter types
        self.adapter_registry: Dict[str, Type[CommunicationAdapter]] = {
            'pipeline': PipelineCommunicationAdapter,
            'broadcast': BroadcastCommunicationAdapter,  # Add this line
            # Add more as you implement them
        }
```

### Step 9: Implement Hierarchical Adapter

Create `src/core/communication/hierarchical_adapter.py`:

```python
# src/core/communication/hierarchical_adapter.py
from .base_adapter import CommunicationAdapter
from typing import List, Any, Dict
import time
import uuid

class HierarchicalCommunicationAdapter(CommunicationAdapter):
    """Hierarchical adapter for parent-child communication with context"""
    
    def __init__(self, coordinator_id: str, log_manager: 'LogManager'):
        super().__init__(coordinator_id, log_manager)
        self.parent_container = None
        self.child_containers: List[Any] = []
        self.hierarchy_map: Dict[str, List[str]] = {}
    
    def setup(self, config: Dict[str, Any], containers: Dict[str, Any]) -> None:
        """Setup hierarchical communication"""
        parent_name = config.get('parent')
        child_names = config.get('children', [])
        
        self.parent_container = containers[parent_name]
        self.child_containers = [containers[name] for name in child_names]
        
        # Track hierarchy
        self.hierarchy_map[parent_name] = child_names
        
        self.logger.log_info(
            "Setting up hierarchical communication",
            parent_container=parent_name,
            child_containers=child_names,
            child_count=len(self.child_containers),
            event_flow="external_standard_tier",
            lifecycle_operation="hierarchical_setup"
        )
        
        # Wire up bidirectional communication
        self._setup_parent_to_children()
        self._setup_children_to_parent()
    
    def _setup_parent_to_children(self):
        """Setup parent broadcasting context to children"""
        def broadcast_handler(event):
            return self._broadcast_to_children(event)
        
        if hasattr(self.parent_container, 'on_context_event'):
            self.parent_container.on_context_event(broadcast_handler)
        else:
            self.parent_container.on_output_event(broadcast_handler)
    
    def _setup_children_to_parent(self):
        """Setup children aggregating results to parent"""
        for child in self.child_containers:
            def aggregate_handler(event, child_id=getattr(child, 'container_id', 'unknown')):
                return self._aggregate_to_parent(event, child_id)
            
            if hasattr(child, 'on_result_event'):
                child.on_result_event(aggregate_handler)
            else:
                child.on_output_event(aggregate_handler)
    
    def _broadcast_to_children(self, event):
        """Broadcast context event from parent to all children"""
        start_time = time.time()
        
        correlation_id = getattr(event, 'correlation_id', None) or f"hierarchy_{uuid.uuid4().hex[:8]}"
        if hasattr(event, 'set_correlation_id'):
            event.set_correlation_id(correlation_id)
        
        self.logger.log_info(
            "Broadcasting context to children",
            event_type=getattr(event, 'type', 'unknown'),
            parent_container=getattr(self.parent_container, 'container_id', 'unknown'),
            child_count=len(self.child_containers),
            correlation_id=correlation_id,
            event_flow="external_standard_tier"
        )
        
        success_count = 0
        for child in self.child_containers:
            try:
                # Add hierarchical context
                contextualized_event = self._add_hierarchical_context(event, child)
                
                if hasattr(child, 'receive_context_event'):
                    child.receive_context_event(contextualized_event)
                else:
                    child.receive_event(contextualized_event)
                    
                success_count += 1
                
            except Exception as e:
                self.track_error(
                    e,
                    child_container=getattr(child, 'container_id', 'unknown'),
                    event_type=getattr(event, 'type', 'unknown'),
                    correlation_id=correlation_id
                )
        
        latency_ms = (time.time() - start_time) * 1000
        self.metrics['total_latency_ms'] += latency_ms
        
        self.logger.log_info(
            "Context broadcast completed",
            successful_deliveries=success_count,
            total_children=len(self.child_containers),
            latency_ms=round(latency_ms, 2),
            correlation_id=correlation_id
        )
    
    def _aggregate_to_parent(self, event, child_id: str):
        """Aggregate results from child to parent"""
        start_time = time.time()
        
        correlation_id = getattr(event, 'correlation_id', None)
        
        self.logger.log_debug(
            "Aggregating child result to parent",
            child_container=child_id,
            parent_container=getattr(self.parent_container, 'container_id', 'unknown'),
            event_type=getattr(event, 'type', 'unknown'),
            correlation_id=correlation_id,
            event_flow="external_standard_tier"
        )
        
        try:
            # Add child context
            event_with_source = self._add_child_source(event, child_id)
            
            if hasattr(self.parent_container, 'receive_child_result'):
                self.parent_container.receive_child_result(event_with_source)
            else:
                self.parent_container.receive_event(event_with_source)
            
            latency_ms = (time.time() - start_time) * 1000
            self.metrics['total_latency_ms'] += latency_ms
            
            self.logger.log_debug(
                "Child result aggregated",
                child_container=child_id,
                latency_ms=round(latency_ms, 2),
                correlation_id=correlation_id
            )
            
        except Exception as e:
            self.track_error(
                e,
                child_container=child_id,
                parent_container=getattr(self.parent_container, 'container_id', 'unknown'),
                event_type=getattr(event, 'type', 'unknown'),
                correlation_id=correlation_id
            )
            raise
    
    def _add_hierarchical_context(self, event, child):
        """Add hierarchical context to event"""
        # Clone event and add context
        import copy
        contextualized = copy.copy(event)
        
        if hasattr(contextualized, 'metadata'):
            contextualized.metadata['parent_container'] = getattr(self.parent_container, 'container_id', 'unknown')
            contextualized.metadata['hierarchy_level'] = getattr(child, 'hierarchy_level', 1)
        
        return contextualized
    
    def _add_child_source(self, event, child_id: str):
        """Add child source information to event"""
        import copy
        sourced_event = copy.copy(event)
        
        if hasattr(sourced_event, 'metadata'):
            sourced_event.metadata['source_child'] = child_id
            sourced_event.metadata['aggregation_timestamp'] = time.time()
        
        return sourced_event
```

### Step 10: Implement Selective Adapter

Create `src/core/communication/selective_adapter.py`:

```python
# src/core/communication/selective_adapter.py
from .base_adapter import CommunicationAdapter
from typing import List, Callable, Any, Dict, Tuple
import time
import uuid

class SelectiveCommunicationAdapter(CommunicationAdapter):
    """Rule-based selective routing adapter"""
    
    def __init__(self, coordinator_id: str, log_manager: 'LogManager'):
        super().__init__(coordinator_id, log_manager)
        self.source_container = None
        self.routing_rules: List[Tuple[Callable, Any, str]] = []
        self.rule_metrics: Dict[str, Dict[str, int]] = {}
    
    def setup(self, config: Dict[str, Any], containers: Dict[str, Any]) -> None:
        """Setup selective routing"""
        source_name = config.get('source')
        self.source_container = containers[source_name]
        
        # Setup routing rules
        for rule_config in config.get('rules', []):
            self._add_rule_from_config(rule_config, containers)
        
        self.logger.log_info(
            "Setting up selective routing",
            source_container=source_name,
            rule_count=len(self.routing_rules),
            event_flow="external_reliable_tier",
            lifecycle_operation="selective_setup"
        )
        
        # Wire up selective routing
        self.source_container.on_output_event(self._handle_selective_routing)
    
    def _add_rule_from_config(self, rule_config: Dict, containers: Dict[str, Any]):
        """Add routing rule from configuration"""
        condition_str = rule_config['condition']
        target_name = rule_config['target']
        rule_name = rule_config.get('name', f"rule_{len(self.routing_rules)}")
        
        # Create condition function
        condition = self._create_condition_function(condition_str)
        
        # Get target container
        target = containers[target_name]
        
        # Add rule
        self.routing_rules.append((condition, target, rule_name))
        
        # Initialize metrics
        self.rule_metrics[rule_name] = {
            'evaluations': 0,
            'matches': 0,
            'routes': 0,
            'errors': 0
        }
        
        self.logger.log_debug(
            "Added routing rule",
            rule_name=rule_name,
            condition=condition_str,
            target_container=target_name
        )
    
    def _create_condition_function(self, condition_str: str) -> Callable:
        """Create condition function from string expression"""
        # Simple implementation - in production, use safe evaluation
        def condition_func(event):
            try:
                # Get event attributes for evaluation
                event_dict = {
                    'type': getattr(event, 'type', None),
                    'source': getattr(event, 'source', None),
                    'confidence': getattr(event.payload, 'confidence', 0) if hasattr(event, 'payload') else 0,
                    'urgency': getattr(event.payload, 'urgency', 'normal') if hasattr(event, 'payload') else 'normal',
                    'regime': getattr(event.payload, 'regime', None) if hasattr(event, 'payload') else None,
                }
                
                # Special handling for performance metrics
                if 'performance' in condition_str:
                    # This would integrate with your performance tracking
                    event_dict['performance'] = {
                        'sharpe': getattr(event.payload, 'sharpe', 0) if hasattr(event, 'payload') else 0
                    }
                
                # Evaluate condition (simplified - use ast.literal_eval in production)
                # For now, handle simple comparisons
                if 'default' in condition_str:
                    return True
                    
                # Parse simple conditions like "performance.sharpe > 1.5"
                if '>' in condition_str:
                    left, right = condition_str.split('>')
                    left_val = eval(left.strip(), {"__builtins__": {}}, event_dict)
                    right_val = float(right.strip())
                    return left_val > right_val
                elif '<' in condition_str:
                    left, right = condition_str.split('<')
                    left_val = eval(left.strip(), {"__builtins__": {}}, event_dict)
                    right_val = float(right.strip())
                    return left_val < right_val
                elif '==' in condition_str:
                    left, right = condition_str.split('==')
                    left_val = eval(left.strip(), {"__builtins__": {}}, event_dict)
                    right_val = eval(right.strip(), {"__builtins__": {}}, event_dict)
                    return left_val == right_val
                else:
                    # Direct boolean evaluation
                    return eval(condition_str, {"__builtins__": {}}, event_dict)
                    
            except Exception as e:
                self.logger.log_warning(
                    "Error evaluating condition",
                    condition=condition_str,
                    error=str(e)
                )
                return False
        
        return condition_func
    
    def _handle_selective_routing(self, event):
        """Route event based on rules"""
        start_time = time.time()
        
        correlation_id = getattr(event, 'correlation_id', None) or f"selective_{uuid.uuid4().hex[:8]}"
        if hasattr(event, 'set_correlation_id'):
            event.set_correlation_id(correlation_id)
        
        self.logger.log_debug(
            "Starting selective routing evaluation",
            event_id=getattr(event, 'id', 'unknown'),
            event_type=getattr(event, 'type', 'unknown'),
            source_container=getattr(self.source_container, 'container_id', 'unknown'),
            rule_count=len(self.routing_rules),
            correlation_id=correlation_id,
            event_flow="external_reliable_tier"
        )
        
        routed = False
        evaluation_results = []
        
        for condition, target, rule_name in self.routing_rules:
            try:
                # Update metrics
                self.rule_metrics[rule_name]['evaluations'] += 1
                
                # Evaluate condition
                rule_start = time.time()
                matches = condition(event)
                evaluation_time_ms = (time.time() - rule_start) * 1000
                
                evaluation_results.append({
                    'rule_name': rule_name,
                    'matches': matches,
                    'evaluation_time_ms': round(evaluation_time_ms, 2),
                    'target_container': getattr(target, 'container_id', 'unknown')
                })
                
                if matches:
                    self.rule_metrics[rule_name]['matches'] += 1
                    
                    # Route the event
                    target.receive_event(event)
                    routed = True
                    
                    self.rule_metrics[rule_name]['routes'] += 1
                    
                    self.logger.log_info(
                        "Event routed by selective adapter",
                        rule_name=rule_name,
                        event_id=getattr(event, 'id', 'unknown'),
                        source_container=getattr(self.source_container, 'container_id', 'unknown'),
                        target_container=getattr(target, 'container_id', 'unknown'),
                        correlation_id=correlation_id,
                        event_flow="external_reliable_tier"
                    )
                    
                    break  # First matching rule wins
                    
            except Exception as e:
                self.rule_metrics[rule_name]['errors'] += 1
                self.track_error(
                    e,
                    rule_name=rule_name,
                    event_id=getattr(event, 'id', 'unknown'),
                    correlation_id=correlation_id
                )
        
        # Track routing completion
        if not routed:
            self.logger.log_warning(
                "No routing rule matched event",
                event_id=getattr(event, 'id', 'unknown'),
                event_type=getattr(event, 'type', 'unknown'),
                rules_evaluated=len(self.routing_rules),
                correlation_id=correlation_id
            )
        
        latency_ms = (time.time() - start_time) * 1000
        self.metrics['total_latency_ms'] += latency_ms
        
        self.logger.log_debug(
            "Selective routing completed",
            event_id=getattr(event, 'id', 'unknown'),
            routed=routed,
            rules_evaluated=len(evaluation_results),
            total_latency_ms=round(latency_ms, 2),
            evaluation_results=evaluation_results,
            correlation_id=correlation_id
        )
    
    def get_rule_performance_report(self) -> Dict[str, Any]:
        """Get detailed rule performance metrics"""
        rule_performance = {}
        
        for rule_name, metrics in self.rule_metrics.items():
            total_evaluations = max(metrics['evaluations'], 1)
            
            rule_performance[rule_name] = {
                'total_evaluations': metrics['evaluations'],
                'total_matches': metrics['matches'],
                'total_routes': metrics['routes'],
                'total_errors': metrics['errors'],
                'match_rate': metrics['matches'] / total_evaluations,
                'error_rate': metrics['errors'] / total_evaluations,
                'route_success_rate': metrics['routes'] / max(metrics['matches'], 1)
            }
        
        return {
            'adapter_type': 'selective',
            'total_rules': len(self.routing_rules),
            'rule_performance': rule_performance
        }
```

### Step 11: Update Factory with All Adapters

Update `src/core/communication/factory.py`:

```python
# src/core/communication/factory.py (update imports and registry)
from .base_adapter import CommunicationAdapter
from .pipeline_adapter import PipelineCommunicationAdapter
from .broadcast_adapter import BroadcastCommunicationAdapter
from .hierarchical_adapter import HierarchicalCommunicationAdapter
from .selective_adapter import SelectiveCommunicationAdapter

class EventCommunicationFactory:
    def __init__(self, coordinator_id: str, log_manager: 'LogManager'):
        # ... existing code ...
        
        # Registry of available adapter types
        self.adapter_registry: Dict[str, Type[CommunicationAdapter]] = {
            'pipeline': PipelineCommunicationAdapter,
            'broadcast': BroadcastCommunicationAdapter,
            'hierarchical': HierarchicalCommunicationAdapter,
            'selective': SelectiveCommunicationAdapter,
        }
```

Update `src/core/communication/__init__.py`:

```python
# src/core/communication/__init__.py
"""Event Communication Adapters for ADMF-PC

This module provides pluggable communication adapters that separate
container organization from event communication patterns.
"""

from .base_adapter import CommunicationAdapter
from .pipeline_adapter import PipelineCommunicationAdapter
from .broadcast_adapter import BroadcastCommunicationAdapter
from .hierarchical_adapter import HierarchicalCommunicationAdapter
from .selective_adapter import SelectiveCommunicationAdapter
from .factory import EventCommunicationFactory, CommunicationLayer

__all__ = [
    'CommunicationAdapter',
    'PipelineCommunicationAdapter',
    'BroadcastCommunicationAdapter', 
    'HierarchicalCommunicationAdapter',
    'SelectiveCommunicationAdapter',
    'EventCommunicationFactory',
    'CommunicationLayer'
]
```

## Phase 4: Production Deployment

### Step 12: Production Configuration

Create `config/production_communication.yaml`:

```yaml
# config/production_communication.yaml
coordinator:
  coordinator_id: "prod_trading_coordinator"
  log_dir: "/var/logs/admf-pc"

logging:
  retention_policy:
    max_age_days: 30
    archive_after_days: 7
    max_total_size_gb: 10.0
  performance:
    async_writing: true
    batch_size: 1000

communication:
  pattern: "classifier_first"
  adapters:
    # Data distribution using broadcast
    - type: "broadcast"
      name: "data_distribution"
      source: "data_container"
      targets: ["indicator_container", "market_monitor", "data_recorder"]
    
    # Main processing pipeline
    - type: "pipeline"
      name: "main_flow"
      containers: ["indicator_container", "classifier_container"]
    
    # Hierarchical classifier to strategies
    - type: "hierarchical"
      name: "classifier_hierarchy"
      parent: "classifier_container"
      children: ["strategy_container_001", "strategy_container_002", "strategy_container_003"]
    
    # Selective routing based on performance
    - type: "selective"
      name: "performance_routing"
      source: "strategy_ensemble"
      rules:
        - condition: "performance.sharpe > 1.5"
          target: "aggressive_risk_container"
          name: "high_performance_route"
        
        - condition: "performance.sharpe < 0.5"
          target: "conservative_risk_container"
          name: "low_performance_route"
        
        - condition: "default"
          target: "balanced_risk_container"
          name: "default_route"
    
    # Final execution pipeline
    - type: "pipeline"
      name: "execution_flow"
      containers: ["risk_container", "compliance_container", "execution_container"]

containers:
  data_container:
    type: "data"
    logging: {level: "INFO"}
    
  classifier_container:
    type: "classifier"
    logging: {level: "DEBUG"}
    
  # Multiple strategy containers
  strategy_container_001:
    type: "strategy"
    strategy_type: "momentum"
    logging: {level: "INFO"}
    
  strategy_container_002:
    type: "strategy"
    strategy_type: "mean_reversion"
    logging: {level: "INFO"}
    
  strategy_container_003:
    type: "strategy"
    strategy_type: "breakout"
    logging: {level: "INFO"}
    
  # Risk containers
  aggressive_risk_container:
    type: "risk"
    max_position_pct: 5.0
    max_exposure_pct: 30.0
    
  conservative_risk_container:
    type: "risk"
    max_position_pct: 1.0
    max_exposure_pct: 5.0
    
  balanced_risk_container:
    type: "risk"
    max_position_pct: 2.5
    max_exposure_pct: 15.0
```

### Step 13: Monitoring Dashboard Integration

Create `src/monitoring/communication_dashboard.py`:

```python
# src/monitoring/communication_dashboard.py
from typing import Dict, Any
import numpy as np

class CommunicationDashboard:
    """Dashboard integration for communication system"""
    
    def __init__(self, coordinator: 'WorkflowCoordinator'):
        self.coordinator = coordinator
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time communication metrics for dashboard"""
        
        if not self.coordinator.communication_layer:
            return {"status": "not_initialized"}
        
        metrics = self.coordinator.communication_layer.get_system_metrics()
        
        # Calculate aggregate statistics
        total_events = sum(
            adapter['events_processed'] 
            for adapter in metrics['adapters'].values()
        )
        
        total_errors = sum(
            adapter['errors'] 
            for adapter in metrics['adapters'].values()
        )
        
        latencies = [
            adapter['average_latency_ms'] 
            for adapter in metrics['adapters'].values()
            if adapter['events_processed'] > 0
        ]
        
        return {
            "communication_health": {
                "status": metrics['overall_health'],
                "total_adapters": metrics['total_adapters'],
                "active_adapters": len([
                    name for name, adapter_metrics in metrics['adapters'].items()
                    if adapter_metrics['events_processed'] > 0
                ])
            },
            "performance": {
                "total_events_processed": total_events,
                "total_errors": total_errors,
                "overall_error_rate": total_errors / max(total_events, 1),
                "average_latency_ms": np.mean(latencies) if latencies else 0,
                "p95_latency_ms": np.percentile(latencies, 95) if latencies else 0,
                "p99_latency_ms": np.percentile(latencies, 99) if latencies else 0
            },
            "adapter_breakdown": {
                name: {
                    "type": adapter['adapter_type'],
                    "events_per_second": adapter['events_per_second'],
                    "error_rate": adapter['error_rate'],
                    "average_latency_ms": adapter['average_latency_ms'],
                    "health": self._calculate_adapter_health(adapter)
                }
                for name, adapter in metrics['adapters'].items()
            }
        }
    
    def _calculate_adapter_health(self, adapter_metrics: Dict[str, Any]) -> str:
        """Calculate individual adapter health status"""
        error_rate = adapter_metrics.get('error_rate', 0)
        
        if error_rate < 0.01:
            return "healthy"
        elif error_rate < 0.05:
            return "warning"
        else:
            return "critical"
    
    def get_adapter_specific_metrics(self, adapter_name: str) -> Dict[str, Any]:
        """Get detailed metrics for specific adapter"""
        
        if not self.coordinator.communication_layer:
            return {"error": "Communication layer not initialized"}
        
        adapters = self.coordinator.communication_layer.adapters
        
        if adapter_name not in adapters:
            return {"error": f"Adapter '{adapter_name}' not found"}
        
        adapter = adapters[adapter_name]
        
        # Get base metrics
        metrics = adapter.get_metrics()
        
        # Add adapter-specific metrics
        if hasattr(adapter, 'get_rule_performance_report'):
            # Selective adapter
            metrics['rule_performance'] = adapter.get_rule_performance_report()
        
        return metrics
    
    def get_event_flow_visualization(self) -> Dict[str, Any]:
        """Get data for event flow visualization"""
        
        if not self.coordinator.communication_layer:
            return {"error": "Communication layer not initialized"}
        
        # Build graph representation of event flow
        nodes = []
        edges = []
        
        for container_id in self.coordinator.containers:
            nodes.append({
                "id": container_id,
                "type": "container",
                "label": container_id
            })
        
        for adapter_name, adapter in self.coordinator.communication_layer.adapters.items():
            nodes.append({
                "id": adapter_name,
                "type": "adapter",
                "label": adapter_name,
                "adapter_type": type(adapter).__name__
            })
            
            # Add edges based on adapter type
            if isinstance(adapter, PipelineCommunicationAdapter):
                # Pipeline creates sequential edges
                for i in range(len(adapter.pipeline_stages) - 1):
                    source = getattr(adapter.pipeline_stages[i], 'container_id', f'stage_{i}')
                    target = getattr(adapter.pipeline_stages[i+1], 'container_id', f'stage_{i+1}')
                    edges.append({
                        "source": source,
                        "target": target,
                        "adapter": adapter_name,
                        "type": "pipeline"
                    })
            
            elif isinstance(adapter, BroadcastCommunicationAdapter):
                # Broadcast creates one-to-many edges
                source = getattr(adapter.source_container, 'container_id', 'source')
                for target in adapter.target_containers:
                    target_id = getattr(target, 'container_id', 'target')
                    edges.append({
                        "source": source,
                        "target": target_id,
                        "adapter": adapter_name,
                        "type": "broadcast"
                    })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "statistics": {
                "total_containers": len(self.coordinator.containers),
                "total_adapters": len(self.coordinator.communication_layer.adapters),
                "total_connections": len(edges)
            }
        }
```

## Testing Strategy

### Step 14: Integration Tests

Create `tests/test_communication_integration.py`:

```python
# tests/test_communication_integration.py
import pytest
import asyncio
from src.core.coordinator.coordinator import WorkflowCoordinator
from src.core.logging.log_manager import LogManager
from src.core.events.types import Event, EventType

@pytest.mark.asyncio
async def test_pipeline_adapter_integration():
    """Test pipeline adapter with full logging integration"""
    
    # Setup coordinator with communication
    config = {
        'coordinator_id': 'test_coordinator',
        'log_dir': 'test_logs',
        'logging': {
            'retention_policy': {'max_age_days': 1},
            'performance': {'async_writing': False}  # Sync for testing
        }
    }
    
    coordinator = WorkflowCoordinator(config)
    
    # Create test containers
    await coordinator.create_container('data_container', {'type': 'data'})
    await coordinator.create_container('strategy_container', {'type': 'strategy'})
    await coordinator.create_container('execution_container', {'type': 'execution'})
    
    # Setup communication
    communication_config = {
        'adapters': [
            {
                'type': 'pipeline',
                'name': 'test_pipeline',
                'containers': ['data_container', 'strategy_container', 'execution_container']
            }
        ]
    }
    
    await coordinator.setup_communication(communication_config)
    
    # Verify communication layer is setup
    assert coordinator.communication_layer is not None
    assert len(coordinator.communication_layer.adapters) == 1
    
    # Get system status
    status = await coordinator.get_system_status()
    assert 'communication' in status
    assert status['communication']['total_adapters'] == 1
    assert status['communication']['overall_health'] in ['healthy', 'warning', 'critical']
    
    # Test event flow
    data_container = coordinator.containers['data_container']
    test_event = Event(
        event_type=EventType.BAR,
        payload={'symbol': 'AAPL', 'price': 150.0},
        timestamp=None
    )
    
    # Emit event and verify it flows through pipeline
    data_container.emit_output_event(test_event)
    
    # Check metrics after event
    await asyncio.sleep(0.1)  # Allow async processing
    
    metrics = coordinator.communication_layer.get_system_metrics()
    pipeline_metrics = metrics['adapters']['test_pipeline']
    assert pipeline_metrics['events_processed'] > 0
    assert pipeline_metrics['error_rate'] == 0
    
    # Cleanup
    await coordinator.shutdown()

@pytest.mark.asyncio 
async def test_broadcast_adapter():
    """Test broadcast adapter functionality"""
    
    config = {
        'coordinator_id': 'test_broadcast',
        'log_dir': 'test_logs'
    }
    
    coordinator = WorkflowCoordinator(config)
    
    # Create containers
    await coordinator.create_container('indicator_container', {'type': 'indicator'})
    await coordinator.create_container('strategy_001', {'type': 'strategy'})
    await coordinator.create_container('strategy_002', {'type': 'strategy'})
    await coordinator.create_container('strategy_003', {'type': 'strategy'})
    
    # Setup broadcast communication
    communication_config = {
        'adapters': [
            {
                'type': 'broadcast',
                'name': 'indicator_broadcast',
                'source': 'indicator_container',
                'targets': ['strategy_001', 'strategy_002', 'strategy_003']
            }
        ]
    }
    
    await coordinator.setup_communication(communication_config)
    
    # Test broadcast
    indicator_container = coordinator.containers['indicator_container']
    test_event = Event(
        event_type=EventType.INDICATOR,
        payload={'indicator': 'RSI', 'value': 65.5},
        timestamp=None
    )
    
    # Track events received by strategies
    received_events = {
        'strategy_001': [],
        'strategy_002': [],
        'strategy_003': []
    }
    
    for strategy_id in received_events:
        strategy = coordinator.containers[strategy_id]
        strategy.received_events = received_events[strategy_id]
        
        def make_handler(events_list):
            def handler(event):
                events_list.append(event)
            return handler
        
        strategy.receive_event = make_handler(received_events[strategy_id])
    
    # Broadcast event
    indicator_container.emit_output_event(test_event)
    
    await asyncio.sleep(0.1)  # Allow processing
    
    # Verify all strategies received the event
    for strategy_id, events in received_events.items():
        assert len(events) == 1
        assert events[0].type == EventType.INDICATOR
        assert events[0].payload['value'] == 65.5
    
    # Check broadcast metrics
    metrics = coordinator.communication_layer.get_system_metrics()
    broadcast_metrics = metrics['adapters']['indicator_broadcast']
    assert broadcast_metrics['events_processed'] == 1
    assert broadcast_metrics['error_rate'] == 0
    
    await coordinator.shutdown()

@pytest.mark.asyncio
async def test_selective_adapter():
    """Test selective routing based on conditions"""
    
    config = {
        'coordinator_id': 'test_selective',
        'log_dir': 'test_logs'
    }
    
    coordinator = WorkflowCoordinator(config)
    
    # Create containers
    await coordinator.create_container('strategy_ensemble', {'type': 'strategy'})
    await coordinator.create_container('aggressive_risk', {'type': 'risk'})
    await coordinator.create_container('conservative_risk', {'type': 'risk'})
    await coordinator.create_container('balanced_risk', {'type': 'risk'})
    
    # Setup selective routing
    communication_config = {
        'adapters': [
            {
                'type': 'selective',
                'name': 'performance_routing',
                'source': 'strategy_ensemble',
                'rules': [
                    {
                        'condition': 'performance.sharpe > 1.5',
                        'target': 'aggressive_risk',
                        'name': 'high_perf'
                    },
                    {
                        'condition': 'performance.sharpe < 0.5',
                        'target': 'conservative_risk',
                        'name': 'low_perf'
                    },
                    {
                        'condition': 'default',
                        'target': 'balanced_risk',
                        'name': 'default'
                    }
                ]
            }
        ]
    }
    
    await coordinator.setup_communication(communication_config)
    
    # Track which container receives events
    received_by = {'aggressive': 0, 'conservative': 0, 'balanced': 0}
    
    def make_counter(counter_key):
        def handler(event):
            received_by[counter_key] += 1
        return handler
    
    coordinator.containers['aggressive_risk'].receive_event = make_counter('aggressive')
    coordinator.containers['conservative_risk'].receive_event = make_counter('conservative')
    coordinator.containers['balanced_risk'].receive_event = make_counter('balanced')
    
    ensemble = coordinator.containers['strategy_ensemble']
    
    # Test high performance signal
    high_perf_event = Event(
        event_type=EventType.SIGNAL,
        payload={'sharpe': 2.0},
        timestamp=None
    )
    ensemble.emit_output_event(high_perf_event)
    
    # Test low performance signal
    low_perf_event = Event(
        event_type=EventType.SIGNAL,
        payload={'sharpe': 0.3},
        timestamp=None
    )
    ensemble.emit_output_event(low_perf_event)
    
    # Test average performance signal
    avg_perf_event = Event(
        event_type=EventType.SIGNAL,
        payload={'sharpe': 1.0},
        timestamp=None
    )
    ensemble.emit_output_event(avg_perf_event)
    
    await asyncio.sleep(0.1)  # Allow processing
    
    # Verify routing worked correctly
    assert received_by['aggressive'] == 1  # High sharpe
    assert received_by['conservative'] == 1  # Low sharpe
    assert received_by['balanced'] == 1  # Default
    
    # Check rule performance
    adapter = coordinator.communication_layer.adapters['performance_routing']
    rule_report = adapter.get_rule_performance_report()
    
    assert rule_report['total_rules'] == 3
    assert rule_report['rule_performance']['high_perf']['total_matches'] == 1
    assert rule_report['rule_performance']['low_perf']['total_matches'] == 1
    assert rule_report['rule_performance']['default']['total_matches'] == 1
    
    await coordinator.shutdown()

@pytest.mark.asyncio
async def test_hierarchical_adapter():
    """Test hierarchical parent-child communication"""
    
    config = {
        'coordinator_id': 'test_hierarchical',
        'log_dir': 'test_logs'
    }
    
    coordinator = WorkflowCoordinator(config)
    
    # Create containers
    await coordinator.create_container('classifier', {'type': 'classifier'})
    await coordinator.create_container('strategy_001', {'type': 'strategy'})
    await coordinator.create_container('strategy_002', {'type': 'strategy'})
    await coordinator.create_container('strategy_003', {'type': 'strategy'})
    
    # Setup hierarchical communication
    communication_config = {
        'adapters': [
            {
                'type': 'hierarchical',
                'name': 'classifier_hierarchy',
                'parent': 'classifier',
                'children': ['strategy_001', 'strategy_002', 'strategy_003']
            }
        ]
    }
    
    await coordinator.setup_communication(communication_config)
    
    # Test parent to children broadcast
    classifier = coordinator.containers['classifier']
    
    # Track context events received by children
    child_contexts = {
        'strategy_001': [],
        'strategy_002': [],
        'strategy_003': []
    }
    
    for strategy_id in child_contexts:
        strategy = coordinator.containers[strategy_id]
        contexts = child_contexts[strategy_id]
        
        def make_context_handler(ctx_list):
            def handler(event):
                ctx_list.append(event)
            return handler
        
        if hasattr(strategy, 'receive_context_event'):
            strategy.receive_context_event = make_context_handler(contexts)
        else:
            strategy.receive_event = make_context_handler(contexts)
    
    # Broadcast regime context
    context_event = Event(
        event_type=EventType.REGIME,
        payload={'regime': 'BULL', 'confidence': 0.85},
        timestamp=None
    )
    
    if hasattr(classifier, 'emit_context_event'):
        classifier.emit_context_event(context_event)
    else:
        classifier.emit_output_event(context_event)
    
    await asyncio.sleep(0.1)
    
    # Verify all children received context
    for strategy_id, contexts in child_contexts.items():
        assert len(contexts) == 1
        assert contexts[0].payload['regime'] == 'BULL'
    
    # Test child to parent aggregation
    parent_results = []
    
    def parent_handler(event):
        parent_results.append(event)
    
    if hasattr(classifier, 'receive_child_result'):
        classifier.receive_child_result = parent_handler
    else:
        classifier.receive_event = parent_handler
    
    # Children send results
    for i, strategy_id in enumerate(['strategy_001', 'strategy_002', 'strategy_003']):
        strategy = coordinator.containers[strategy_id]
        result_event = Event(
            event_type=EventType.SIGNAL,
            payload={'signal': f'BUY_{i}', 'confidence': 0.7 + i * 0.1},
            timestamp=None
        )
        
        if hasattr(strategy, 'emit_result_event'):
            strategy.emit_result_event(result_event)
        else:
            strategy.emit_output_event(result_event)
    
    await asyncio.sleep(0.1)
    
    # Verify parent received all results
    assert len(parent_results) == 3
    
    await coordinator.shutdown()


# Performance tests
@pytest.mark.asyncio
async def test_adapter_performance():
    """Test adapter performance meets targets"""
    
    config = {
        'coordinator_id': 'test_performance',
        'log_dir': 'test_logs'
    }
    
    coordinator = WorkflowCoordinator(config)
    
    # Create containers for pipeline test
    containers = ['data', 'indicators', 'strategies', 'risk', 'execution']
    for c in containers:
        await coordinator.create_container(c, {'type': c})
    
    # Setup pipeline
    communication_config = {
        'adapters': [
            {
                'type': 'pipeline',
                'name': 'perf_pipeline',
                'containers': containers
            }
        ]
    }
    
    await coordinator.setup_communication(communication_config)
    
    # Send many events through pipeline
    data_container = coordinator.containers['data']
    num_events = 1000
    
    import time
    start_time = time.time()
    
    for i in range(num_events):
        event = Event(
            event_type=EventType.BAR,
            payload={'index': i, 'price': 100 + i * 0.1},
            timestamp=None
        )
        data_container.emit_output_event(event)
    
    # Wait for processing
    await asyncio.sleep(1.0)
    
    elapsed = time.time() - start_time
    
    # Check performance
    metrics = coordinator.communication_layer.get_system_metrics()
    pipeline_metrics = metrics['adapters']['perf_pipeline']
    
    assert pipeline_metrics['events_processed'] >= num_events * (len(containers) - 1)  # Events per stage
    
    # Verify latency target (< 10ms for pipeline)
    assert pipeline_metrics['average_latency_ms'] < 10.0
    
    # Verify throughput
    events_per_second = pipeline_metrics['events_per_second']
    assert events_per_second > 100  # Should handle > 100 events/second
    
    print(f"Performance test results:")
    print(f"  Events processed: {pipeline_metrics['events_processed']}")
    print(f"  Average latency: {pipeline_metrics['average_latency_ms']:.2f}ms")
    print(f"  Events per second: {events_per_second:.2f}")
    
    await coordinator.shutdown()
```

## Migration Guide

### Step 15: Migration from Current System

1. **Identify Current Event Routes**
   ```python
   # Map current container communications
   # Example: RiskContainer → ExecutionContainer (ORDER events)
   ```

2. **Create Adapter Configuration**
   ```yaml
   # Replace external event routing with adapters
   adapters:
     - type: "pipeline"
       containers: ["risk_container", "execution_container"]
   ```

3. **Update Container Implementations**
   - Remove external event router configuration
   - Add adapter support methods (on_output_event, receive_event)

4. **Test with Simple Pipeline First**
   - Start with 3-container pipeline
   - Verify no circular dependencies
   - Check event flow with logging

5. **Gradually Add Complexity**
   - Add broadcast for data distribution
   - Add selective routing for performance
   - Add hierarchical for classifier patterns

### Fix for Current Multi-Strategy Issue

The immediate fix for your multi-strategy backtest:

```yaml
# config/multi_strategy_test_fixed.yaml
communication:
  adapters:
    # Simple pipeline to fix circular dependencies
    - type: "pipeline"
      name: "main_flow"
      containers: [
        "data_container",
        "indicator_container", 
        "classifier_container",
        "risk_container",
        "portfolio_container",
        "strategy_container",
        "execution_container"
      ]
```

Then in your main.py:

```python
# Load communication config
with open('config/multi_strategy_test_fixed.yaml') as f:
    comm_config = yaml.safe_load(f)

# Setup communication before running backtest
await coordinator.setup_communication(comm_config['communication'])

# Now run backtest - no more circular dependencies!
```

## Summary

This comprehensive implementation guide provides:

1. **All Code from Your Colleague** - Every adapter implementation included
2. **Complete Integration** - Full coordinator and logging integration
3. **Production Configuration** - Real-world configuration examples
4. **Testing Suite** - Comprehensive tests for all adapters
5. **Migration Path** - Clear steps to fix current issues

Start with Phase 1 (Pipeline adapter) to immediately solve your circular dependency problem, then gradually add more sophisticated adapters as needed for your research workflows.