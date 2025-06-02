# Container-Aware Logging and Debugging System v3

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Design Motivation](#design-motivation)
3. [Solution Overview](#solution-overview)
4. [Protocol + Composition Approach](#protocol--composition-approach)
5. [Architecture Design](#architecture-design)
6. [Implementation Components](#implementation-components)
7. [Coordinator Integration and Lifecycle Management](#coordinator-integration-and-lifecycle-management)
8. [Production Operations and Performance](#production-operations-and-performance)
9. [Usage Examples](#usage-examples)
10. [Benefits and Advantages](#benefits-and-advantages)

---

## Problem Statement

### The Container Isolation Challenge

Modern quantitative trading systems using the hybrid tiered communication architecture face a critical debugging challenge: **how do you maintain observability across 50+ isolated containers without creating logging chaos?**

```
Traditional Logging Nightmare:
┌─────────────────────────────────────────────────────────────────┐
│                    Single Log Stream                            │
│  2024-01-15T10:30:01 Container_001: Processing signal          │
│  2024-01-15T10:30:01 Container_047: Risk check failed          │
│  2024-01-15T10:30:01 Container_012: Generating momentum signal  │
│  2024-01-15T10:30:01 Container_033: Order submitted             │
│  2024-01-15T10:30:01 Container_001: Processing signal          │
│  2024-01-15T10:30:01 Container_028: Portfolio update            │
│  2024-01-15T10:30:01 Container_015: Mean reversion signal      │
│  ... 1000s more entries per second ...                         │
│                                                                 │
│  Result: IMPOSSIBLE TO DEBUG! 🔥                               │
└─────────────────────────────────────────────────────────────────┘
```

### Debugging Complexity Matrix

The hybrid architecture creates multiple debugging scenarios, each requiring different approaches:

| Scenario | Challenge | Current Limitation |
|----------|-----------|-------------------|
| **Single Container Debug** | Understanding internal event flow | No container isolation in logs |
| **Cross-Container Flow** | Tracing signals through multiple containers | No correlation tracking |
| **Event Bus Isolation** | Separating internal vs external communication | Mixed event scopes in logs |
| **Performance Analysis** | Identifying bottlenecks across tiers | No tier-specific metrics |
| **Error Investigation** | Finding root cause across boundaries | No event flow reconstruction |
| **Log Management** | Handling log growth and cleanup | No centralized lifecycle management |
| **Production Operations** | Monitoring disk usage and performance | No automated maintenance |

### The Inheritance Problem

Traditional logging frameworks force inheritance hierarchies that violate the protocol + composition philosophy:

```python
# WRONG: Traditional inheritance approach
class LoggableStrategy(BaseLoggableComponent):  # Framework coupling!
    def __init__(self):
        super().__init__()  # Required framework initialization
        # Can only work with other framework components
        
# PROBLEM: Can't add logging to external libraries!
sklearn_model = RandomForestClassifier()  # How do you log this?
external_function = talib.RSI  # Or this?
```

### The Lifecycle Management Gap

Existing logging solutions fail to address operational concerns:

- ❌ **Centralized log management**
- ❌ **Log rotation and cleanup**  
- ❌ **Coordinator-level orchestration**
- ❌ **Log retention policies**
- ❌ **Cleanup on container shutdown**
- ❌ **Disk space monitoring**
- ❌ **Performance optimization under load**

---

## Design Motivation

### Container Boundary Respect

The logging system must **respect the same architectural boundaries** that make the hybrid communication system successful:

```
Container Architecture Alignment:
┌─────────────────────────────────────────────────────────┐
│             Market Regime Container                      │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Internal Communication: Direct Event Bus       │   │
│  │  Logging Scope: "internal_bus"                  │   │
│  │  Log File: logs/containers/regime/classifier.log│   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
└─────────────────────────┼───────────────────────────────┘
                          │
         External Communication: Tiered Event Router
         Logging Scope: "external_standard_tier"
         Log File: logs/flows/cross_container_flows.log
                          │
┌─────────────────────────▼───────────────────────────────┐
│            Strategy Container Pool                      │
│  ┌─────────────────────────────────────────────────┐   │
│  │  50+ Individual Strategy Containers             │   │
│  │  Each with isolated internal communication      │   │
│  │  Each with separate log file                    │   │
│  │  Log Files: logs/containers/strategy_001/*.log  │   │
│  │             logs/containers/strategy_002/*.log  │   │
│  │             ... (50+ separate log streams)      │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Protocol + Composition Benefits

The logging system should exemplify the same principles that make the overall architecture successful:

**Zero Inheritance**: No component should be forced to inherit from logging base classes
**Universal Compatibility**: Any component (custom, external library, simple function) should be able to gain logging capability
**Composable Enhancement**: Components should be able to opt into logging, tracing, debugging capabilities independently
**Runtime Flexibility**: Logging behavior should be configurable and changeable at runtime

### Performance and Scalability

With 50+ containers processing thousands of events per second, the logging system must:

- **Minimize Performance Impact**: Logging shouldn't slow down critical trading paths
- **Scale Linearly**: Adding more containers shouldn't degrade logging performance
- **Isolate Failures**: A logging failure in one container shouldn't affect others
- **Enable Selective Observability**: Only log what's needed for each component

### Operational Excellence Requirements

Production trading systems require enterprise-grade log management:

- **Automated Lifecycle Management**: Create, rotate, archive, and delete logs automatically
- **Resource Monitoring**: Track disk usage and prevent storage exhaustion
- **Performance Optimization**: Async writing, batching, and compression for high-throughput environments
- **Reliability**: Graceful degradation and recovery from logging failures

---

## Solution Overview

### Hybrid Logging Architecture with Lifecycle Management

The solution implements a **hybrid logging architecture** that mirrors the hybrid communication architecture while providing complete operational lifecycle management:

```
┌─────────────────────────────────────────────────────────────────┐
│                    HYBRID LOGGING SYSTEM v3                     │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         Container-Isolated Logging (Internal)          │   │
│  │                                                         │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │   │
│  │  │Container_001│    │Container_002│    │Container_003│ │   │
│  │  │strategy.log │    │strategy.log │    │risk.log     │ │   │
│  │  │risk.log     │    │data.log     │    │execution.log│ │   │
│  │  └─────────────┘    └─────────────┘    └─────────────┘ │   │
│  │                                                         │   │
│  │  Benefits:                                              │   │
│  │  • Complete isolation per container                    │   │
│  │  • Fast, local file I/O                               │   │
│  │  • Easy single-container debugging                     │   │
│  │  • No cross-container interference                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         Cross-Container Flow Tracking (External)       │   │
│  │                                                         │   │
│  │  Event Flow Tracing:                                   │   │
│  │  • Container_001 → Container_015 (standard_tier)       │   │
│  │  • Container_015 → Container_032 (reliable_tier)       │   │
│  │  • Container_032 → Execution (reliable_tier)           │   │
│  │                                                         │   │
│  │  Master Correlation Log:                               │   │
│  │  • Cross-reference events across containers            │   │
│  │  • Trace signal flows end-to-end                       │   │
│  │  • Identify performance bottlenecks                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         Coordinator-Managed Lifecycle (Operations)     │   │
│  │                                                         │   │
│  │  Lifecycle Management:                                  │   │
│  │  • Automated log creation/cleanup                      │   │
│  │  • Log rotation and archiving                          │   │
│  │  • Retention policy enforcement                        │   │
│  │  • Disk usage monitoring                               │   │
│  │  • Performance optimization                            │   │
│  │                                                         │   │
│  │  Operational Benefits:                                  │   │
│  │  • Zero manual maintenance                             │   │
│  │  • Predictable resource usage                          │   │
│  │  • Automated error recovery                            │   │
│  │  • Production-ready reliability                        │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Four-Layer Approach

#### Layer 1: Component-Level Logging
- **Protocol-based**: Any component can implement `Loggable` protocol
- **Zero inheritance**: No base classes required
- **Composable**: Logging capability added through composition

#### Layer 2: Container-Isolated Streams
- **Boundary respect**: Each container gets separate log files
- **Performance isolation**: Container logging failures don't cascade
- **Debug locality**: Easy to focus on single container issues

#### Layer 3: Cross-Container Correlation
- **Flow tracking**: Trace events across container boundaries
- **Correlation IDs**: Track signals from source to execution
- **System-wide observability**: Understand multi-container workflows

#### Layer 4: Coordinator-Managed Lifecycle
- **Automated management**: Complete log lifecycle automation
- **Resource optimization**: Disk usage monitoring and cleanup
- **Production readiness**: Enterprise-grade operational features

---

## Protocol + Composition Approach

### Core Protocols (Zero Inheritance!)

The system is built entirely on protocols - no inheritance anywhere:

```python
from typing import Protocol, runtime_checkable
from datetime import datetime
from pathlib import Path
import asyncio

@runtime_checkable
class Loggable(Protocol):
    """Anything that can log messages"""
    def log(self, level: str, message: str, **context) -> None: ...

@runtime_checkable
class EventTrackable(Protocol):
    """Anything that can track event flows"""
    def trace_event(self, event_id: str, source: str, target: str, **context) -> None: ...

@runtime_checkable
class ContainerAware(Protocol):
    """Anything that knows about container context"""
    @property
    def container_id(self) -> str: ...
    @property
    def component_name(self) -> str: ...

@runtime_checkable
class CorrelationAware(Protocol):
    """Anything that can track correlation across boundaries"""
    def set_correlation_id(self, correlation_id: str) -> None: ...
    def get_correlation_id(self) -> Optional[str]: ...

@runtime_checkable
class Debuggable(Protocol):
    """Anything that can be debugged"""
    def capture_state(self) -> Dict[str, Any]: ...
    def enable_tracing(self, enabled: bool) -> None: ...

@runtime_checkable
class LifecycleManaged(Protocol):
    """Anything that can be managed through its lifecycle"""
    def initialize(self) -> None: ...
    def cleanup(self) -> None: ...
    def get_status(self) -> Dict[str, Any]: ...
```

### Composition Over Inheritance

Instead of forcing components to inherit from base classes, capabilities are **composed** from smaller components:

```python
# COMPOSITION: Build loggers from smaller components
class ContainerLogger:
    def __init__(self, container_id: str, component_name: str):
        # Compose with various specialized components
        self.container_context = ContainerContext(container_id, component_name)
        self.correlation_tracker = CorrelationTracker()
        self.scope_detector = EventScopeDetector()
        self.container_writer = LogWriter(container_log_path)
        self.master_writer = LogWriter(master_log_path)
    
    # Implement Loggable protocol through composition
    def log(self, level: str, message: str, **context) -> None:
        # Use composed components to build functionality
        correlation_id = self.correlation_tracker.get_correlation_id()
        event_scope = self.scope_detector.detect_scope(context)
        # ... rest of logging logic
```

### Universal Component Enhancement

Any component can be enhanced with logging capabilities regardless of its origin:

```python
# Works with YOUR components
class MyStrategy:
    def generate_signal(self, data):
        return {"action": "BUY", "strength": 0.8}

# Works with EXTERNAL LIBRARIES
sklearn_model = RandomForestClassifier()

# Works with SIMPLE FUNCTIONS
def momentum_signal(data):
    return data['close'].pct_change() > 0.02

# ALL can get logging capability through pure composition!
my_strategy = add_logging_to_any_component(MyStrategy(), "container_001", "my_strategy")
sklearn_model = add_logging_to_any_component(sklearn_model, "container_002", "ml_model")
momentum_signal = add_logging_to_any_component(momentum_signal, "container_003", "momentum")

# Now they can all log!
my_strategy.log_info("Generating signal", signal_strength=0.8)
sklearn_model.log_debug("Model prediction", prediction=prediction_value)
momentum_signal.log_warning("Low volume detected", volume=current_volume)
```

---

## Architecture Design

### Container-Aware Log Organization

The logging system creates a directory structure that mirrors the container architecture with full lifecycle management:

```
logs/
├── containers/                    # Container-isolated logs
│   ├── market_regime_container/
│   │   ├── hmm_classifier.log     # Component-specific logs
│   │   ├── volatility_detector.log
│   │   └── regime_coordinator.log
│   ├── strategy_container_001/
│   │   ├── momentum_strategy.log
│   │   ├── risk_manager.log
│   │   └── portfolio_tracker.log
│   ├── strategy_container_002/
│   │   ├── mean_reversion.log
│   │   ├── risk_manager.log
│   │   └── portfolio_tracker.log
│   ├── ... (50+ strategy containers)
│   ├── execution_container/
│   │   ├── order_manager.log
│   │   ├── fill_processor.log
│   │   └── position_tracker.log
├── flows/                         # Cross-container event flows
│   ├── main_coordinator_event_flows.log
│   ├── optimization_coordinator_flows.log
│   └── validation_coordinator_flows.log
├── correlations/                  # Correlation tracking
│   ├── trade_correlations.log
│   ├── signal_correlations.log
│   └── error_correlations.log
├── system/                        # System-level logs
│   ├── coordinator.log
│   ├── log_manager.log
│   └── performance_metrics.log
├── archived/                      # Archived logs (compressed)
│   ├── containers/
│   │   └── 2024-01-01/           # Date-organized archives
│   │       └── strategy_001.log.gz
│   └── flows/
│       └── 2024-01-01/
│           └── event_flows.log.gz
└── master.log                     # High-level system overview
```

### Event Scope Classification

The system automatically classifies events by their communication scope:

```python
class EventScope(Enum):
    INTERNAL_BUS = "internal_bus"               # Within container
    EXTERNAL_FAST = "external_fast_tier"       # Cross-container fast
    EXTERNAL_STANDARD = "external_standard_tier" # Cross-container standard
    EXTERNAL_RELIABLE = "external_reliable_tier" # Cross-container reliable
    COMPONENT_INTERNAL = "component_internal"   # Within component
    LIFECYCLE_MANAGEMENT = "lifecycle_management" # Log management operations
```

This enables filtering and analysis by communication pattern:

```bash
# Debug internal container issues
grep "internal_bus" logs/containers/strategy_001/*.log

# Analyze cross-container performance
grep "external_fast_tier" logs/flows/*.log

# Track reliable order execution
grep "external_reliable_tier" logs/correlations/*.log

# Monitor lifecycle management
grep "lifecycle_management" logs/system/*.log
```

### Correlation Tracking Across Boundaries

The system tracks correlation IDs across container boundaries to enable end-to-end signal tracing:

```
Signal Flow with Correlation Tracking:
┌─────────────────┐    correlation_id: "trade_xyz_123"
│ Data Container  │────────────────────────────────────┐
└─────────────────┘                                    │
         │                                             │
         ▼ (internal_bus)                               │
┌─────────────────┐                                    │
│ Indicator Calc  │                                    │
└─────────────────┘                                    │
         │                                             │
         ▼ (external_standard_tier)                     │
┌─────────────────┐    correlation_id: "trade_xyz_123" │
│Strategy_001     │◄───────────────────────────────────┘
│momentum.log     │
└─────────────────┘
         │
         ▼ (external_standard_tier)
┌─────────────────┐    correlation_id: "trade_xyz_123"
│ Risk Container  │◄───────────────────────────────────┐
│ risk_check.log  │                                    │
└─────────────────┘                                    │
         │                                             │
         ▼ (external_reliable_tier)                     │
┌─────────────────┐    correlation_id: "trade_xyz_123" │
│Execution        │◄───────────────────────────────────┘
│order_mgmt.log   │
└─────────────────┘
```

---

## Implementation Components

### Core Composable Components

The system is built from small, focused components that can be composed together:

#### 1. LogWriter - Enhanced File I/O Component
```python
class LogWriter:
    """Writes logs to files with lifecycle management - composable component"""
    
    def __init__(self, log_file: Path, max_size_mb: int = 100, enable_compression: bool = False):
        self.log_file = log_file
        self.max_size_mb = max_size_mb
        self.enable_compression = enable_compression
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._handle = None
        self._init_handle()
        self._bytes_written = 0
    
    def _init_handle(self):
        try:
            self._handle = open(self.log_file, 'a', encoding='utf-8')
        except Exception as e:
            print(f"Failed to open log file {self.log_file}: {e}")
    
    def write(self, entry: Dict[str, Any]) -> None:
        if self._handle:
            try:
                json_str = json.dumps(entry, default=str)
                self._handle.write(json_str + '\n')
                self._handle.flush()
                self._bytes_written += len(json_str) + 1
                
                # Check if rotation is needed
                if self._bytes_written > (self.max_size_mb * 1024 * 1024):
                    self._rotate_log()
                    
            except Exception as e:
                print(f"Failed to write log entry: {e}")
    
    def _rotate_log(self):
        """Rotate log file when size limit reached"""
        if self._handle:
            self._handle.close()
            
            # Create rotated filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rotated_path = self.log_file.with_suffix(f".{timestamp}.log")
            
            # Move current log to rotated name
            self.log_file.rename(rotated_path)
            
            # Optionally compress
            if self.enable_compression:
                self._compress_log(rotated_path)
            
            # Reinitialize handle with original filename
            self._init_handle()
            self._bytes_written = 0
    
    def _compress_log(self, log_path: Path):
        """Compress rotated log file"""
        import gzip
        compressed_path = log_path.with_suffix('.log.gz')
        
        with open(log_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                f_out.writelines(f_in)
        
        log_path.unlink()  # Delete uncompressed version
    
    def close(self):
        if self._handle:
            self._handle.close()
```

#### 2. ContainerContext - Container Awareness Component
```python
class ContainerContext:
    """Container context - composable component"""
    
    def __init__(self, container_id: str, component_name: str):
        self.container_id = container_id
        self.component_name = component_name
        self.creation_time = datetime.utcnow()
        self.metrics = {
            'log_count': 0,
            'error_count': 0,
            'last_activity': None
        }
    
    def update_metrics(self, level: str):
        """Update context metrics"""
        self.metrics['log_count'] += 1
        self.metrics['last_activity'] = datetime.utcnow()
        
        if level in ['ERROR', 'CRITICAL']:
            self.metrics['error_count'] += 1
```

#### 3. CorrelationTracker - Cross-Boundary Tracking Component
```python
class CorrelationTracker:
    """Correlation tracking - composable component"""
    
    def __init__(self):
        self.context = threading.local()
        self.correlation_history: Dict[str, List[str]] = {}
        self._lock = threading.Lock()
    
    def set_correlation_id(self, correlation_id: str) -> None:
        self.context.correlation_id = correlation_id
    
    def get_correlation_id(self) -> Optional[str]:
        return getattr(self.context, 'correlation_id', None)
    
    def track_event(self, event_id: str):
        """Track event in correlation chain"""
        correlation_id = self.get_correlation_id()
        if correlation_id:
            with self._lock:
                if correlation_id not in self.correlation_history:
                    self.correlation_history[correlation_id] = []
                self.correlation_history[correlation_id].append(event_id)
    
    def get_correlation_chain(self, correlation_id: str) -> List[str]:
        """Get full event chain for correlation"""
        with self._lock:
            return self.correlation_history.get(correlation_id, [])
```

#### 4. EventScopeDetector - Communication Pattern Detection Component
```python
class EventScopeDetector:
    """Event scope detection - composable component"""
    
    def detect_scope(self, context: Dict[str, Any]) -> str:
        if 'internal_scope' in context:
            return EventScope.INTERNAL_BUS.value
        elif 'publish_tier' in context:
            tier = context['publish_tier']
            return f"external_{tier}_tier"
        elif 'event_flow' in context:
            return context['event_flow']
        elif 'lifecycle_operation' in context:
            return EventScope.LIFECYCLE_MANAGEMENT.value
        else:
            return EventScope.COMPONENT_INTERNAL.value
```

---

## Coordinator Integration and Lifecycle Management

### LogManager - Centralized Lifecycle Management

```python
# src/core/logging/log_manager.py
class LogManager:
    """Centralized log lifecycle management for coordinator"""
    
    def __init__(self, coordinator_id: str, base_log_dir: str = "logs", config: Dict[str, Any] = None):
        self.coordinator_id = coordinator_id
        self.base_log_dir = Path(base_log_dir)
        self.active_containers: Set[str] = set()
        self.log_writers: Dict[str, LogWriter] = {}
        self.container_registries: Dict[str, 'ContainerLogRegistry'] = {}
        
        # Initialize retention policy from config
        retention_config = config.get('retention_policy', {}) if config else {}
        self.retention_policy = LogRetentionPolicy(**retention_config)
        
        # Performance settings
        performance_config = config.get('performance', {}) if config else {}
        self.async_writing = performance_config.get('async_writing', True)
        self.batch_size = performance_config.get('batch_size', 1000)
        self.flush_interval = performance_config.get('flush_interval_seconds', 5)
        
        # Create base log structure
        self._initialize_log_structure()
        
        # Start background maintenance if async enabled
        if self.async_writing:
            self._start_background_tasks()
    
    def _initialize_log_structure(self):
        """Create standardized log directory structure"""
        directories = [
            self.base_log_dir / "containers",
            self.base_log_dir / "flows", 
            self.base_log_dir / "system",
            self.base_log_dir / "correlations",
            self.base_log_dir / "archived"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Create coordinator system log
        self.system_logger = ContainerLogger(
            self.coordinator_id, 
            "coordinator",
            base_log_dir=str(self.base_log_dir)
        )
        
        self.system_logger.log_info(
            "Log management system initialized",
            base_log_dir=str(self.base_log_dir),
            async_writing=self.async_writing,
            lifecycle_operation="initialization"
        )
        
    def register_container(self, container_id: str) -> 'ContainerLogRegistry':
        """Register new container and setup its logging"""
        self.active_containers.add(container_id)
        
        # Create container log directory
        container_log_dir = self.base_log_dir / "containers" / container_id
        container_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create registry for this container's loggers
        registry = ContainerLogRegistry(container_id, container_log_dir, self)
        self.container_registries[container_id] = registry
        
        self.system_logger.log_info(
            "Registered container for logging",
            container_id=container_id,
            log_dir=str(container_log_dir),
            lifecycle_operation="container_registration"
        )
        
        return registry
        
    def unregister_container(self, container_id: str):
        """Cleanup container logging on shutdown"""
        if container_id in self.active_containers:
            self.active_containers.remove(container_id)
            
            # Close container registry
            if container_id in self.container_registries:
                registry = self.container_registries[container_id]
                registry.close_all_loggers()
                del self.container_registries[container_id]
            
            # Close all log writers for this container
            container_writers = [
                key for key in self.log_writers.keys() 
                if key.startswith(f"{container_id}.")
            ]
            
            for writer_key in container_writers:
                writer = self.log_writers.pop(writer_key)
                writer.close()
                
            self.system_logger.log_info(
                "Unregistered container logging",
                container_id=container_id,
                lifecycle_operation="container_cleanup"
            )
            
    async def cleanup_and_archive_logs(self):
        """Periodic cleanup and archiving"""
        self.system_logger.log_info(
            "Starting log cleanup and archiving",
            lifecycle_operation="maintenance_start"
        )
        
        try:
            stats = await self.retention_policy.apply_retention_rules(self.base_log_dir)
            
            self.system_logger.log_info(
                "Log cleanup completed",
                **stats,
                lifecycle_operation="maintenance_complete"
            )
            
        except Exception as e:
            self.system_logger.log_error(
                "Error during log cleanup",
                error=str(e),
                lifecycle_operation="maintenance_error"
            )
            
    def get_log_summary(self) -> Dict[str, Any]:
        """Get logging system status"""
        container_log_dirs = list((self.base_log_dir / "containers").iterdir())
        
        return {
            "coordinator_id": self.coordinator_id,
            "active_containers": len(self.active_containers),
            "total_log_directories": len(container_log_dirs),
            "base_log_dir": str(self.base_log_dir),
            "disk_usage_mb": self._calculate_disk_usage(),
            "oldest_log": self._get_oldest_log_date(),
            "log_structure": self._get_log_structure(),
            "performance_settings": {
                "async_writing": self.async_writing,
                "batch_size": self.batch_size,
                "flush_interval": self.flush_interval
            },
            "retention_policy": {
                "max_age_days": self.retention_policy.max_age_days,
                "archive_after_days": self.retention_policy.archive_after_days,
                "max_size_gb": self.retention_policy.max_size_gb
            }
        }
        
    def _calculate_disk_usage(self) -> float:
        """Calculate total disk usage of logs"""
        total_size = 0
        for file_path in self.base_log_dir.rglob("*.log*"):
            try:
                total_size += file_path.stat().st_size
            except:
                pass
        return total_size / (1024 * 1024)  # Convert to MB
        
    def _get_oldest_log_date(self) -> Optional[str]:
        """Get oldest log file date"""
        oldest_time = None
        for file_path in self.base_log_dir.rglob("*.log"):
            try:
                mtime = file_path.stat().st_mtime
                if oldest_time is None or mtime < oldest_time:
                    oldest_time = mtime
            except:
                pass
        
        if oldest_time:
            return datetime.fromtimestamp(oldest_time).isoformat()
        return None
        
    def _get_log_structure(self) -> Dict[str, Any]:
        """Get current log directory structure"""
        structure = {}
        
        for container_dir in (self.base_log_dir / "containers").iterdir():
            if container_dir.is_dir():
                log_files = [f.name for f in container_dir.glob("*.log")]
                structure[container_dir.name] = {
                    "log_files": log_files,
                    "file_count": len(log_files),
                    "total_size_mb": sum(
                        f.stat().st_size for f in container_dir.glob("*.log*") 
                        if f.is_file()
                    ) / (1024 * 1024)
                }
                
        return structure
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        if self.async_writing:
            # Start periodic maintenance
            asyncio.create_task(self._periodic_maintenance())
    
    async def _periodic_maintenance(self):
        """Background task for periodic maintenance"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self.cleanup_and_archive_logs()
            except Exception as e:
                self.system_logger.log_error(
                    "Error in periodic maintenance", 
                    error=str(e),
                    lifecycle_operation="maintenance_error"
                )
                await asyncio.sleep(600)  # Retry in 10 minutes


class ContainerLogRegistry:
    """Manages loggers for a specific container with lifecycle support"""
    
    def __init__(self, container_id: str, log_dir: Path, log_manager: LogManager):
        self.container_id = container_id
        self.log_dir = log_dir
        self.log_manager = log_manager
        self.component_loggers: Dict[str, ContainerLogger] = {}
        self.creation_time = datetime.utcnow()
        
    def create_component_logger(self, component_name: str) -> ContainerLogger:
        """Create logger for a component within this container"""
        if component_name in self.component_loggers:
            return self.component_loggers[component_name]
            
        logger = ContainerLogger(
            self.container_id,
            component_name,
            base_log_dir=str(self.log_dir.parent.parent)  # Back to base logs dir
        )
        
        self.component_loggers[component_name] = logger
        
        # Register with log manager
        writer_key = f"{self.container_id}.{component_name}"
        self.log_manager.log_writers[writer_key] = logger.container_writer
        
        return logger
        
    def close_all_loggers(self):
        """Close all loggers for this container"""
        for logger in self.component_loggers.values():
            logger.close()
        self.component_loggers.clear()
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            "container_id": self.container_id,
            "component_count": len(self.component_loggers),
            "components": list(self.component_loggers.keys()),
            "creation_time": self.creation_time.isoformat(),
            "log_directory": str(self.log_dir)
        }


class LogRetentionPolicy:
    """Manages log retention and cleanup policies"""
    
    def __init__(self, 
                 max_age_days: int = 30,
                 max_size_gb: float = 10.0,
                 archive_after_days: int = 7,
                 compression_enabled: bool = True):
        self.max_age_days = max_age_days
        self.max_size_gb = max_size_gb  
        self.archive_after_days = archive_after_days
        self.compression_enabled = compression_enabled
        
    async def apply_retention_rules(self, base_log_dir: Path) -> Dict[str, Any]:
        """Apply all retention policies and return statistics"""
        stats = {
            "archived_files": 0,
            "deleted_files": 0,
            "compressed_files": 0,
            "space_freed_mb": 0,
            "errors": []
        }
        
        # Apply policies
        archived_stats = await self._archive_old_logs(base_log_dir)
        deleted_stats = await self._delete_expired_logs(base_log_dir)
        compressed_stats = await self._compress_large_logs(base_log_dir)
        
        # Combine statistics
        for key in stats:
            if key in archived_stats:
                stats[key] += archived_stats[key] if isinstance(archived_stats[key], (int, float)) else 0
            if key in deleted_stats:
                stats[key] += deleted_stats[key] if isinstance(deleted_stats[key], (int, float)) else 0
            if key in compressed_stats:
                stats[key] += compressed_stats[key] if isinstance(compressed_stats[key], (int, float)) else 0
        
        return stats
        
    async def _archive_old_logs(self, base_log_dir: Path) -> Dict[str, Any]:
        """Archive logs older than archive threshold"""
        cutoff_time = datetime.now().timestamp() - (self.archive_after_days * 24 * 3600)
        archive_dir = base_log_dir / "archived" 
        stats = {"archived_files": 0, "space_freed_mb": 0, "errors": []}
        
        for log_file in base_log_dir.rglob("*.log"):
            try:
                if log_file.stat().st_mtime < cutoff_time:
                    # Don't archive system logs or already archived logs
                    if "system" in log_file.parts or "archived" in log_file.parts:
                        continue
                    
                    # Move to archive
                    relative_path = log_file.relative_to(base_log_dir)
                    date_dir = datetime.fromtimestamp(log_file.stat().st_mtime).strftime("%Y-%m-%d")
                    archive_path = archive_dir / date_dir / relative_path
                    archive_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    original_size = log_file.stat().st_size
                    
                    # Compress and move
                    if self.compression_enabled:
                        import gzip
                        with open(log_file, 'rb') as f_in:
                            with gzip.open(f"{archive_path}.gz", 'wb') as f_out:
                                f_out.writelines(f_in)
                    else:
                        # Just move without compression
                        log_file.rename(archive_path)
                    
                    if self.compression_enabled:
                        log_file.unlink()  # Delete original after compression
                    
                    stats["archived_files"] += 1
                    stats["space_freed_mb"] += original_size / (1024 * 1024)
                    
            except Exception as e:
                stats["errors"].append(f"Error archiving {log_file}: {e}")
                
        return stats
                
    async def _delete_expired_logs(self, base_log_dir: Path) -> Dict[str, Any]:
        """Delete logs older than max age"""
        cutoff_time = datetime.now().timestamp() - (self.max_age_days * 24 * 3600)
        stats = {"deleted_files": 0, "space_freed_mb": 0, "errors": []}
        
        # Delete old archived logs
        archive_dir = base_log_dir / "archived"
        if archive_dir.exists():
            for archived_file in archive_dir.rglob("*"):
                try:
                    if archived_file.is_file() and archived_file.stat().st_mtime < cutoff_time:
                        original_size = archived_file.stat().st_size
                        archived_file.unlink()
                        stats["deleted_files"] += 1
                        stats["space_freed_mb"] += original_size / (1024 * 1024)
                        
                except Exception as e:
                    stats["errors"].append(f"Error deleting archived log {archived_file}: {e}")
        
        return stats
    
    async def _compress_large_logs(self, base_log_dir: Path) -> Dict[str, Any]:
        """Compress large log files to save space"""
        stats = {"compressed_files": 0, "space_freed_mb": 0, "errors": []}
        
        if not self.compression_enabled:
            return stats
        
        # Compress logs larger than 50MB
        size_threshold = 50 * 1024 * 1024  # 50MB
        
        for log_file in base_log_dir.rglob("*.log"):
            try:
                if log_file.stat().st_size > size_threshold:
                    # Don't compress if already compressed recently
                    compressed_version = log_file.with_suffix('.log.gz')
                    if compressed_version.exists():
                        continue
                    
                    original_size = log_file.stat().st_size
                    
                    # Compress the file
                    import gzip
                    with open(log_file, 'rb') as f_in:
                        with gzip.open(compressed_version, 'wb') as f_out:
                            f_out.writelines(f_in)
                    
                    # Remove original
                    log_file.unlink()
                    
                    compressed_size = compressed_version.stat().st_size
                    space_saved = (original_size - compressed_size) / (1024 * 1024)
                    
                    stats["compressed_files"] += 1
                    stats["space_freed_mb"] += space_saved
                    
            except Exception as e:
                stats["errors"].append(f"Error compressing {log_file}: {e}")
        
        return stats
```

### Coordinator Integration

```python
# src/core/coordinator/coordinator.py
class WorkflowCoordinator:
    def __init__(self, config: Dict[str, Any]):
        self.coordinator_id = config.get('coordinator_id', f"coord_{uuid.uuid4().hex[:8]}")
        
        # Initialize log manager FIRST
        self.log_manager = LogManager(
            self.coordinator_id,
            base_log_dir=config.get('log_dir', 'logs'),
            config=config.get('logging', {})
        )
        
        # Create coordinator logger
        self.logger = self.log_manager.system_logger
        
        # Container management
        self.containers: Dict[str, Any] = {}
        self.container_log_registries: Dict[str, ContainerLogRegistry] = {}
        
        # Start background maintenance
        if config.get('logging', {}).get('performance', {}).get('async_writing', True):
            asyncio.create_task(self.periodic_log_maintenance())
        
    async def create_container(self, container_id: str, container_config: Dict) -> Any:
        """Create container with automatic log setup"""
        
        # Register logging for this container
        log_registry = self.log_manager.register_container(container_id)
        self.container_log_registries[container_id] = log_registry
        
        # Create the actual container
        container = await self._instantiate_container(container_id, container_config)
        
        # Add logging capability to container
        add_logging_to_any_component(
            container, 
            container_id, 
            "container_manager"
        )
        
        # Store container
        self.containers[container_id] = container
        
        self.logger.log_info(
            "Container created with logging",
            container_id=container_id,
            container_type=container_config.get('type'),
            lifecycle_operation="container_creation"
        )
        
        return container
        
    async def shutdown_container(self, container_id: str):
        """Shutdown container and cleanup logs"""
        if container_id in self.containers:
            container = self.containers[container_id]
            
            # Shutdown container
            if hasattr(container, 'shutdown'):
                await container.shutdown()
                
            # Cleanup logging
            if container_id in self.container_log_registries:
                registry = self.container_log_registries[container_id]
                registry.close_all_loggers()
                del self.container_log_registries[container_id]
                
            self.log_manager.unregister_container(container_id)
            del self.containers[container_id]
            
            self.logger.log_info(
                "Container shutdown and logs cleaned",
                container_id=container_id,
                lifecycle_operation="container_shutdown"
            )
            
    async def periodic_log_maintenance(self):
        """Background task for log maintenance"""
        while True:
            try:
                await self.log_manager.cleanup_and_archive_logs()
                await asyncio.sleep(3600)  # Run every hour
            except Exception as e:
                self.logger.log_error(
                    "Error in log maintenance", 
                    error=str(e),
                    lifecycle_operation="maintenance_error"
                )
                await asyncio.sleep(600)  # Retry in 10 minutes
                
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including logging"""
        return {
            "coordinator": {
                "id": self.coordinator_id,
                "active_containers": len(self.containers),
                "uptime_hours": (datetime.utcnow() - self.start_time).total_seconds() / 3600
            },
            "logging": self.log_manager.get_log_summary(),
            "containers": {
                container_id: {
                    "type": type(container).__name__,
                    "status": getattr(container, 'status', 'unknown'),
                    "log_registry": self.container_log_registries[container_id].get_registry_stats()
                }
                for container_id, container in self.containers.items()
            }
        }
    
    async def shutdown(self):
        """Graceful shutdown with log cleanup"""
        self.logger.log_info(
            "Starting coordinator shutdown",
            lifecycle_operation="coordinator_shutdown"
        )
        
        # Shutdown all containers
        for container_id in list(self.containers.keys()):
            await self.shutdown_container(container_id)
        
        # Final log cleanup
        await self.log_manager.cleanup_and_archive_logs()
        
        # Close system logger
        self.logger.close()
```

---

## Production Operations and Performance

### Configuration Management

```yaml
# config/coordinator_config.yaml
coordinator:
  coordinator_id: "main_trading_coordinator"
  log_dir: "/var/logs/admf-pc"
  
logging:
  retention_policy:
    max_age_days: 30
    archive_after_days: 7
    max_total_size_gb: 10.0
    compression_enabled: true
    
  log_levels:
    system: "INFO"
    containers: "DEBUG"
    event_flows: "INFO"
    lifecycle_management: "INFO"
    
  performance:
    async_writing: true
    batch_size: 1000
    flush_interval_seconds: 5
    max_file_size_mb: 100
    enable_compression: true
    
  monitoring:
    disk_usage_alert_threshold_gb: 8.0
    error_rate_alert_threshold: 10  # errors per minute
    performance_metrics_enabled: true

containers:
  data_container:
    type: "data"
    logging:
      level: "INFO"
      components: ["data_handler", "stream_processor"]
      
  strategy_ensemble:
    type: "strategy_ensemble"
    logging:
      level: "DEBUG"
      components: ["strategy_manager", "signal_aggregator"]
      
  risk_container:
    type: "risk"
    logging:
      level: "INFO"
      components: ["risk_manager", "portfolio_tracker"]
      
  execution_container:
    type: "execution"
    logging:
      level: "INFO"
      components: ["order_manager", "fill_processor"]
```

### Performance Optimization Features

#### 1. Asynchronous Batch Writing
```python
class AsyncBatchLogWriter:
    """High-performance async batch log writer"""
    
    def __init__(self, log_file: Path, batch_size: int = 1000, flush_interval: int = 5):
        self.log_file = log_file
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.batch_buffer: List[Dict[str, Any]] = []
        self.last_flush = datetime.utcnow()
        self._lock = asyncio.Lock()
        
        # Start background flush task
        asyncio.create_task(self._periodic_flush())
    
    async def write_async(self, entry: Dict[str, Any]):
        """Add entry to batch buffer"""
        async with self._lock:
            self.batch_buffer.append(entry)
            
            # Flush if batch is full
            if len(self.batch_buffer) >= self.batch_size:
                await self._flush_batch()
    
    async def _periodic_flush(self):
        """Periodically flush buffered entries"""
        while True:
            await asyncio.sleep(self.flush_interval)
            
            async with self._lock:
                if self.batch_buffer and \
                   (datetime.utcnow() - self.last_flush).seconds >= self.flush_interval:
                    await self._flush_batch()
    
    async def _flush_batch(self):
        """Flush all buffered entries to disk"""
        if not self.batch_buffer:
            return
            
        try:
            # Use aiofiles for async file I/O
            import aiofiles
            
            async with aiofiles.open(self.log_file, 'a') as f:
                for entry in self.batch_buffer:
                    await f.write(json.dumps(entry, default=str) + '\n')
            
            self.batch_buffer.clear()
            self.last_flush = datetime.utcnow()
            
        except Exception as e:
            print(f"Error flushing batch to {self.log_file}: {e}")
```

#### 2. Memory-Optimized Event Scope Detection
```python
class OptimizedEventScopeDetector:
    """Memory-optimized event scope detection"""
    
    def __init__(self):
        # Pre-compile scope detection patterns
        self.scope_patterns = {
            'internal_scope': EventScope.INTERNAL_BUS.value,
            'lifecycle_operation': EventScope.LIFECYCLE_MANAGEMENT.value,
        }
        
        self.tier_mapping = {
            'fast': EventScope.EXTERNAL_FAST.value,
            'standard': EventScope.EXTERNAL_STANDARD.value,
            'reliable': EventScope.EXTERNAL_RELIABLE.value,
        }
    
    def detect_scope(self, context: Dict[str, Any]) -> str:
        """Optimized scope detection with minimal allocations"""
        # Fast path for common patterns
        for key, scope in self.scope_patterns.items():
            if key in context:
                return scope
        
        # Handle tier-based scopes
        if 'publish_tier' in context:
            tier = context['publish_tier']
            return self.tier_mapping.get(tier, EventScope.EXTERNAL_STANDARD.value)
        
        if 'event_flow' in context:
            return context['event_flow']
        
        return EventScope.COMPONENT_INTERNAL.value
```

#### 3. Resource Monitoring and Alerting
```python
class LoggingResourceMonitor:
    """Monitor logging system resource usage"""
    
    def __init__(self, log_manager: LogManager, alert_thresholds: Dict[str, Any]):
        self.log_manager = log_manager
        self.alert_thresholds = alert_thresholds
        self.metrics_history: Dict[str, List[float]] = defaultdict(list)
        
    async def monitor_resources(self):
        """Continuous resource monitoring"""
        while True:
            try:
                # Check disk usage
                disk_usage_gb = self.log_manager._calculate_disk_usage() / 1024
                self.metrics_history['disk_usage_gb'].append(disk_usage_gb)
                
                if disk_usage_gb > self.alert_thresholds.get('disk_usage_alert_threshold_gb', 8.0):
                    await self._alert_disk_usage(disk_usage_gb)
                
                # Check error rates
                error_rate = self._calculate_error_rate()
                self.metrics_history['error_rate'].append(error_rate)
                
                if error_rate > self.alert_thresholds.get('error_rate_alert_threshold', 10):
                    await self._alert_error_rate(error_rate)
                
                # Check log processing performance
                processing_latency = self._calculate_processing_latency()
                self.metrics_history['processing_latency_ms'].append(processing_latency)
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                print(f"Error in resource monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _alert_disk_usage(self, usage_gb: float):
        """Alert on high disk usage"""
        self.log_manager.system_logger.log_warning(
            "High disk usage detected",
            current_usage_gb=usage_gb,
            threshold_gb=self.alert_thresholds['disk_usage_alert_threshold_gb'],
            lifecycle_operation="resource_alert"
        )
        
        # Trigger emergency cleanup
        await self.log_manager.cleanup_and_archive_logs()
    
    async def _alert_error_rate(self, error_rate: float):
        """Alert on high error rate"""
        self.log_manager.system_logger.log_warning(
            "High error rate detected",
            current_error_rate=error_rate,
            threshold=self.alert_thresholds['error_rate_alert_threshold'],
            lifecycle_operation="resource_alert"
        )
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate per minute"""
        # Implementation would analyze recent logs for error count
        # This is a simplified version
        return 0.0
    
    def _calculate_processing_latency(self) -> float:
        """Calculate log processing latency"""
        # Implementation would measure time from log creation to disk write
        # This is a simplified version
        return 1.0
```

### Operational Dashboard Data

```python
class LoggingDashboard:
    """Provide data for operational dashboards"""
    
    def __init__(self, log_manager: LogManager):
        self.log_manager = log_manager
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time logging system metrics"""
        summary = self.log_manager.get_log_summary()
        
        return {
            "system_health": {
                "status": "healthy" if summary["disk_usage_mb"] < 8192 else "warning",
                "active_containers": summary["active_containers"],
                "total_disk_usage_mb": summary["disk_usage_mb"],
                "disk_usage_percentage": (summary["disk_usage_mb"] / (10 * 1024)) * 100,  # % of 10GB limit
            },
            "container_breakdown": summary["log_structure"],
            "performance": summary["performance_settings"],
            "retention": summary["retention_policy"],
            "recent_activity": self._get_recent_activity(),
            "error_summary": self._get_error_summary()
        }
    
    def _get_recent_activity(self) -> List[Dict[str, Any]]:
        """Get recent logging activity"""
        # Implementation would parse recent system logs
        return [
            {
                "timestamp": "2024-01-15T10:30:00Z",
                "event": "container_created",
                "container_id": "strategy_001",
                "status": "success"
            },
            {
                "timestamp": "2024-01-15T10:25:00Z", 
                "event": "log_rotation",
                "file": "execution_container/order_manager.log",
                "status": "success"
            }
        ]
    
    def _get_error_summary(self) -> Dict[str, Any]:
        """Get error summary for monitoring"""
        return {
            "last_24h_errors": 5,
            "critical_errors": 0,
            "most_common_error": "Failed to write log entry",
            "error_rate_trend": "stable"
        }
```

---

## Usage Examples

### Basic Container Logging with Lifecycle Management

```python
# Create logging components for a container with full lifecycle support
def setup_production_container_logging(container_id: str, coordinator: WorkflowCoordinator):
    """Setup production-ready logging for a container"""
    
    # Container automatically gets logging via coordinator
    log_registry = coordinator.container_log_registries[container_id]
    
    # Create component-specific loggers
    data_logger = log_registry.create_component_logger("data_handler")
    strategy_logger = log_registry.create_component_logger("strategy")
    risk_logger = log_registry.create_component_logger("risk_manager")
    
    # All loggers are automatically managed by coordinator lifecycle
    return {
        'data': data_logger,
        'strategy': strategy_logger,
        'risk': risk_logger
    }

# Usage in container
async def run_container_with_lifecycle_logging():
    # Container creation automatically sets up logging
    coordinator = WorkflowCoordinator(config)
    container = await coordinator.create_container("strategy_001", container_config)
    
    # Get loggers for components
    loggers = setup_production_container_logging("strategy_001", coordinator)
    
    # Use loggers normally - lifecycle is handled automatically
    loggers['strategy'].log_info(
        "Strategy initialized", 
        strategy_type="momentum",
        parameters={'fast': 10, 'slow': 30}
    )
    
    # On shutdown, all cleanup is automatic
    await coordinator.shutdown_container("strategy_001")
```

### Cross-Container Event Tracing with Lifecycle Awareness

```python
# Trace events across containers with automatic lifecycle management
class ProductionHybridContainer:
    def __init__(self, container_id: str, coordinator: WorkflowCoordinator):
        self.container_id = container_id
        self.coordinator = coordinator
        
        # Get logging registry from coordinator
        self.log_registry = coordinator.container_log_registries[container_id]
        
        # Create component logger
        self.logger = self.log_registry.create_component_logger("container_manager")
        
        # Add event tracing capability
        self = EventTracingCapability.add_to_component(self, coordinator.coordinator_id)
    
    def publish_external(self, event, tier="standard"):
        """Publish event with automatic tracing and lifecycle awareness"""
        
        # Log the external publication
        self.logger.log_info(
            "Publishing external event", 
            event_type=event.type,
            tier=tier,
            publish_tier=tier,  # This sets event scope automatically
            lifecycle_operation="event_publish"
        )
        
        # Trace the cross-container event
        self.trace_external_event(
            event_id=event.id,
            source_container=self.container_id,
            target_container="unknown",  # Will be filled by router
            tier=tier,
            event_type=event.type
        )
        
        # Actual event publishing logic...
    
    async def shutdown(self):
        """Shutdown with automatic cleanup"""
        self.logger.log_info(
            "Container shutting down",
            lifecycle_operation="container_shutdown"
        )
        
        # Coordinator handles all log cleanup automatically
        await self.coordinator.shutdown_container(self.container_id)
```

### Production Operations Example

```python
# Production monitoring and maintenance
async def production_logging_operations():
    """Example of production logging operations"""
    
    # Initialize coordinator with production config
    config = {
        'coordinator_id': 'prod_coordinator_001',
        'log_dir': '/var/logs/admf-pc',
        'logging': {
            'retention_policy': {
                'max_age_days': 30,
                'archive_after_days': 7,
                'max_total_size_gb': 10.0,
                'compression_enabled': True
            },
            'performance': {
                'async_writing': True,
                'batch_size': 1000,
                'flush_interval_seconds': 5
            },
            'monitoring': {
                'disk_usage_alert_threshold_gb': 8.0,
                'error_rate_alert_threshold': 10
            }
        }
    }
    
    coordinator = WorkflowCoordinator(config)
    
    # Create multiple containers (simulating 50+ container environment)
    containers = []
    for i in range(50):
        container_id = f"strategy_container_{i:03d}"
        container = await coordinator.create_container(container_id, {
            'type': 'strategy',
            'strategy': 'momentum',
            'allocation': 0.02  # 2% each
        })
        containers.append(container)
    
    # All logging is automatically managed by coordinator
    # - Log files created automatically
    # - Background cleanup runs every hour
    # - Disk usage monitored continuously
    # - Retention policies applied automatically
    
    # Get system status including logging
    status = await coordinator.get_system_status()
    print(f"System running with {status['coordinator']['active_containers']} containers")
    print(f"Total log disk usage: {status['logging']['disk_usage_mb']:.1f} MB")
    print(f"Oldest log: {status['logging']['oldest_log']}")
    
    # Simulate running for a while, then graceful shutdown
    await asyncio.sleep(3600)  # Run for 1 hour
    
    # Graceful shutdown with automatic cleanup
    await coordinator.shutdown()
    print("All containers and logs cleaned up automatically")
```

### Debugging 50+ Containers with Lifecycle Management

```python
# Debug specific container performance with lifecycle awareness
async def debug_production_containers():
    """Debug performance issues in production containers"""
    
    coordinator = WorkflowCoordinator(production_config)
    
    # Wait for system to be running
    await asyncio.sleep(10)
    
    # Get logging dashboard
    dashboard = LoggingDashboard(coordinator.log_manager)
    
    # Check overall system health
    metrics = dashboard.get_real_time_metrics()
    
    if metrics['system_health']['status'] == 'warning':
        print(f"⚠️  System warning: {metrics['system_health']['disk_usage_percentage']:.1f}% disk usage")
        
        # Trigger emergency cleanup
        await coordinator.log_manager.cleanup_and_archive_logs()
    
    # Check individual container health
    for container_id, container_info in metrics['container_breakdown'].items():
        if container_info['total_size_mb'] > 100:  # Large log files
            print(f"📊 Container {container_id} has large logs: {container_info['total_size_mb']:.1f} MB")
            
            # Get detailed container logs
            registry = coordinator.container_log_registries.get(container_id)
            if registry:
                stats = registry.get_registry_stats()
                print(f"   Components: {stats['components']}")
                print(f"   Created: {stats['creation_time']}")
    
    # Check cross-container flows
    debugger = ContainerDebugger(coordinator.coordinator_id)
    flows = debugger.get_cross_container_flows(time_window_minutes=5)
    
    # Analyze communication patterns
    fast_tier_count = len([f for f in flows if 'fast' in f.get('tier', '')])
    standard_tier_count = len([f for f in flows if 'standard' in f.get('tier', '')])
    reliable_tier_count = len([f for f in flows if 'reliable' in f.get('tier', '')])
    
    print(f"📡 Cross-container communication (last 5 min):")
    print(f"   Fast tier: {fast_tier_count} events")
    print(f"   Standard tier: {standard_tier_count} events")
    print(f"   Reliable tier: {reliable_tier_count} events")
    
    # All logging lifecycle is managed automatically
    # - No manual cleanup needed
    # - Automatic archiving and compression
    # - Resource monitoring and alerting
    # - Graceful degradation on errors
```

---

## Benefits and Advantages

### 1. Perfect Container Isolation with Lifecycle Management

**Problem Solved**: No more searching through mixed logs from 50+ containers, with automatic cleanup
```
Before: Single log with mixed container output + manual maintenance (impossible to manage)
After:  logs/containers/strategy_001/momentum.log + automatic lifecycle management
```

### 2. Zero Inheritance Flexibility

**Problem Solved**: Can add logging to ANY component regardless of origin
```python
# ALL of these work identically with full lifecycle support:
custom_strategy = add_logging_to_any_component(MyStrategy(), "c1", "strategy")
sklearn_model = add_logging_to_any_component(RandomForestClassifier(), "c2", "ml")
simple_function = add_logging_to_any_component(my_function, "c3", "func")
external_lib = add_logging_to_any_component(talib.RSI, "c4", "talib")
```

### 3. Communication Pattern Awareness

**Problem Solved**: Understand exactly how events flow through the hybrid architecture
```
Event Scope Classification:
• internal_bus: Strategy coordination within container
• external_standard_tier: Strategy → Risk communication  
• external_reliable_tier: Risk → Execution communication
• external_fast_tier: Data → Strategy communication
• lifecycle_management: Log system operations
```

### 4. Cross-Container Correlation with Lifecycle Tracking

**Problem Solved**: Trace signals end-to-end across all containers with automatic management
```
Correlation Flow:
trade_abc123: data_container → strategy_015 → risk_container → execution
            : (2ms)          → (5ms)       → (15ms)        → (45ms)
Total latency: 67ms for complete signal processing
Lifecycle: All logs automatically archived after 7 days, deleted after 30 days
```

### 5. Production-Grade Performance and Reliability

**Problem Solved**: Enterprise-grade logging that scales and manages itself
- **Async Batch Writing**: 1000+ events buffered and written efficiently
- **Automatic Compression**: Large logs compressed to save 60-80% space
- **Resource Monitoring**: Continuous disk usage and performance tracking
- **Graceful Degradation**: System continues operating even with logging failures
- **Zero Maintenance**: Complete automation of log lifecycle

### 6. Operational Excellence

**Development Velocity**:
- Add logging to any component instantly
- No framework learning curve
- Test components with or without logging
- Easy to mock logging for tests

**Operations Excellence**:
- **Zero Manual Maintenance**: All lifecycle automated
- **Predictable Resource Usage**: Retention policies prevent disk exhaustion
- **Real-time Monitoring**: Dashboard metrics for operational visibility
- **Automated Recovery**: Self-healing from common logging issues
- **Performance Optimization**: Async writing and batching for high throughput

### 7. Complete Lifecycle Management

**Problem Solved**: Production logging without operational burden

| Lifecycle Stage | Automated Management | Benefit |
|----------------|---------------------|---------|
| **Creation** | Coordinator creates log structure | Consistent organization |
| **Writing** | Async batching and compression | High performance |
| **Rotation** | Size-based automatic rotation | Prevents large files |
| **Archiving** | Time-based compression and storage | Space efficiency |
| **Cleanup** | Policy-based deletion | Prevents disk exhaustion |
| **Monitoring** | Resource tracking and alerting | Operational visibility |

### 8. Architectural Consistency

The logging system perfectly mirrors the hybrid communication architecture:

| Communication Layer | Logging Layer | Lifecycle Management |
|-------------------|---------------|---------------------|
| Internal Event Bus | Container-Isolated Logs | Per-container cleanup |
| External Fast Tier | Flow Tracing | High-frequency archiving |
| External Standard Tier | Correlation Tracking | Standard retention |
| External Reliable Tier | Error Correlation | Long-term retention |
| Coordinator | System Logs | Complete lifecycle orchestration |

### 9. Future-Proof Design with Operational Maturity

The protocol + composition approach with lifecycle management ensures the logging system can evolve:

- **New Component Types**: Any future component can gain logging capability with automatic lifecycle
- **New Communication Patterns**: Event scope detection can be extended with new lifecycle rules
- **New Debugging Tools**: Compose new capabilities from existing components
- **External Tool Integration**: No framework lock-in, easy to integrate with monitoring systems
- **Scaling Requirements**: Lifecycle management scales linearly with container count
- **Compliance Needs**: Retention policies can be configured for regulatory requirements

### 10. Complete Log Lifecycle Management

**What the Coordinator Now Manages**:

1. **✅ Creation**: 
   - Creates standardized log directory structure
   - Registers each container for logging
   - Sets up component loggers automatically

2. **✅ Runtime Management**:
   - Tracks all active container loggers
   - Monitors disk usage continuously
   - Provides system status with log metrics
   - Handles performance optimization

3. **✅ Cleanup and Retention**:
   - Archives old logs (compressed)
   - Deletes expired logs based on policy
   - Cleanup on container shutdown
   - Emergency cleanup on disk space alerts

4. **✅ Monitoring and Alerting**:
   - Disk usage tracking with alerts
   - Error rate monitoring
   - Performance metrics collection
   - Operational dashboard integration

**Benefits**:
- 🎯 **Centralized**: All log management in one place
- 🧹 **Automatic Cleanup**: No manual log maintenance needed
- 📊 **Visibility**: Full logging system status and metrics
- ⚡ **Performance**: Async writing, batching, and compression
- 🔒 **Reliable**: Proper shutdown, cleanup, and error recovery
- 🏭 **Production Ready**: Enterprise-grade operational features

---

## Conclusion

The Container-Aware Logging and Debugging System v3 solves the critical observability and operational challenges of the hybrid tiered communication architecture while maintaining perfect consistency with the protocol + composition philosophy.

**Key Achievements**:
- ✅ **Container Isolation**: Each of 50+ containers has separate, focused log streams
- ✅ **Zero Inheritance**: Any component can gain logging capability through composition
- ✅ **Communication Awareness**: Logs are classified by communication pattern (internal bus, external tiers)
- ✅ **Cross-Container Tracing**: End-to-end signal tracking with correlation IDs
- ✅ **Performance Scaling**: Linear scalability with container count and async optimization
- ✅ **Architectural Consistency**: Logging boundaries match communication boundaries
- ✅ **Complete Lifecycle Management**: Automated creation, rotation, archiving, and cleanup
- ✅ **Production Operations**: Resource monitoring, alerting, and automated maintenance
- ✅ **Operational Excellence**: Zero-maintenance logging with enterprise-grade reliability

The system demonstrates that sophisticated observability AND complete operational lifecycle management can be achieved through **protocol + composition** rather than inheritance, enabling debugging capabilities that scale naturally with the hybrid container architecture while maintaining the flexibility to enhance any component regardless of its origin.

**Result: Debugging 50+ isolated containers becomes manageable, traceable, performant, and completely automated for production operations.**

This architecture provides the foundation for enterprise-grade trading systems that can scale to hundreds of containers while maintaining full observability and requiring zero manual log management.