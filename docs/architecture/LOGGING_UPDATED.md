# Container-Aware Logging and Debugging System
# Protocol + Composition Design for Hybrid Tiered Communication Architecture

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Protocol, runtime_checkable
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from collections import defaultdict, deque

# ================================
# PROTOCOLS (Zero Inheritance!)
# ================================

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

# ================================
# DATA STRUCTURES
# ================================

class EventScope(Enum):
    INTERNAL_BUS = "internal_bus"
    EXTERNAL_FAST = "external_fast_tier"
    EXTERNAL_STANDARD = "external_standard_tier"
    EXTERNAL_RELIABLE = "external_reliable_tier"
    COMPONENT_INTERNAL = "component_internal"

@dataclass
class LogEntry:
    timestamp: str
    container_id: str
    component_name: str
    level: str
    message: str
    event_scope: str
    correlation_id: Optional[str] = None
    event_id: Optional[str] = None
    context: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class FlowEvent:
    timestamp: str
    event_id: str
    event_type: str
    source: str
    target: str
    flow_type: str  # 'internal' or 'external'
    container_id: Optional[str] = None
    tier: Optional[str] = None
    latency_ms: Optional[float] = None
    correlation_id: Optional[str] = None

# ================================
# COMPOSABLE LOGGING COMPONENTS
# ================================

class LogWriter:
    """Writes logs to files - composable component"""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._handle = None
        self._init_handle()
    
    def _init_handle(self):
        try:
            self._handle = open(self.log_file, 'a', encoding='utf-8')
        except Exception as e:
            print(f"Failed to open log file {self.log_file}: {e}")
    
    def write(self, entry: Dict[str, Any]) -> None:
        if self._handle:
            try:
                json.dump(entry, self._handle)
                self._handle.write('\n')
                self._handle.flush()
            except Exception as e:
                print(f"Failed to write log entry: {e}")
    
    def close(self):
        if self._handle:
            self._handle.close()

class ContainerContext:
    """Container context - composable component"""
    
    def __init__(self, container_id: str, component_name: str):
        self.container_id = container_id
        self.component_name = component_name

class CorrelationTracker:
    """Correlation tracking - composable component"""
    
    def __init__(self):
        self.context = threading.local()
    
    def set_correlation_id(self, correlation_id: str) -> None:
        self.context.correlation_id = correlation_id
    
    def get_correlation_id(self) -> Optional[str]:
        return getattr(self.context, 'correlation_id', None)

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
        else:
            return EventScope.COMPONENT_INTERNAL.value

# ================================
# MAIN LOGGING COMPONENTS
# ================================

class ContainerLogger:
    """
    Container-aware logger built through composition.
    No inheritance - just protocols and composition!
    """
    
    def __init__(self, container_id: str, component_name: str, 
                 log_level: str = 'INFO', base_log_dir: str = "logs"):
        
        # Compose with various components
        self.container_context = ContainerContext(container_id, component_name)
        self.correlation_tracker = CorrelationTracker()
        self.scope_detector = EventScopeDetector()
        
        # Setup log writers
        base_path = Path(base_log_dir)
        container_log_path = base_path / "containers" / container_id / f"{component_name}.log"
        master_log_path = base_path / "master.log"
        
        self.container_writer = LogWriter(container_log_path)
        self.master_writer = LogWriter(master_log_path)
        
        self.log_level = log_level
    
    # Implement Loggable protocol
    def log(self, level: str, message: str, **context) -> None:
        """Main logging method - implements Loggable protocol"""
        # Use composed components
        correlation_id = self.correlation_tracker.get_correlation_id()
        event_scope = self.scope_detector.detect_scope(context)
        
        log_entry = LogEntry(
            timestamp=datetime.utcnow().isoformat(),
            container_id=self.container_context.container_id,
            component_name=self.container_context.component_name,
            level=level,
            message=message,
            event_scope=event_scope,
            correlation_id=correlation_id,
            event_id=context.get('event_id'),
            context=context
        )
        
        # Write to both logs using composed writers
        self.container_writer.write(log_entry.to_dict())
        
        # Only important messages to master log
        if level in ['ERROR', 'WARNING', 'INFO']:
            master_entry = {
                'timestamp': log_entry.timestamp,
                'container': log_entry.container_id,
                'component': log_entry.component_name,
                'level': log_entry.level,
                'message': log_entry.message,
                'scope': log_entry.event_scope
            }
            self.master_writer.write(master_entry)
    
    # Implement ContainerAware protocol
    @property
    def container_id(self) -> str:
        return self.container_context.container_id
    
    @property
    def component_name(self) -> str:
        return self.container_context.component_name
    
    # Implement CorrelationAware protocol
    def set_correlation_id(self, correlation_id: str) -> None:
        self.correlation_tracker.set_correlation_id(correlation_id)
    
    def get_correlation_id(self) -> Optional[str]:
        return self.correlation_tracker.get_correlation_id()
    
    # Convenience methods
    def log_info(self, message: str, **context):
        self.log('INFO', message, **context)
    
    def log_error(self, message: str, **context):
        self.log('ERROR', message, **context)
    
    def log_debug(self, message: str, **context):
        self.log('DEBUG', message, **context)
    
    def log_warning(self, message: str, **context):
        self.log('WARNING', message, **context)
    
    def with_correlation_id(self, correlation_id: str):
        """Context manager for correlation tracking"""
        return CorrelationContext(self, correlation_id)
    
    def close(self):
        self.container_writer.close()
        self.master_writer.close()

class EventFlowTracer:
    """
    Event flow tracer built through composition.
    Tracks events across container boundaries.
    """
    
    def __init__(self, coordinator_id: str, base_log_dir: str = "logs"):
        self.coordinator_id = coordinator_id
        
        # Compose with log writer
        flow_log_path = Path(base_log_dir) / "flows" / f"{coordinator_id}_event_flows.log"
        self.flow_writer = LogWriter(flow_log_path)
        
        # Compose with correlation tracker
        self.correlation_tracker = CorrelationTracker()
    
    # Implement EventTrackable protocol
    def trace_event(self, event_id: str, source: str, target: str, **context) -> None:
        """Main event tracing method"""
        flow_event = FlowEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_id=event_id,
            event_type=context.get('event_type', 'unknown'),
            source=source,
            target=target,
            flow_type=context.get('flow_type', 'unknown'),
            container_id=context.get('container_id'),
            tier=context.get('tier'),
            latency_ms=context.get('latency_ms'),
            correlation_id=self.correlation_tracker.get_correlation_id()
        )
        
        self.flow_writer.write(asdict(flow_event))
    
    def trace_internal_event(self, container_id: str, event_id: str, 
                           source: str, target: str, **context):
        """Trace internal container events"""
        self.trace_event(
            event_id=event_id,
            source=source,
            target=target,
            flow_type='internal',
            container_id=container_id,
            **context
        )
    
    def trace_external_event(self, event_id: str, source_container: str, 
                           target_container: str, tier: str, **context):
        """Trace cross-container events"""
        self.trace_event(
            event_id=event_id,
            source=source_container,
            target=target_container,
            flow_type='external',
            tier=tier,
            **context
        )

class ContainerDebugger:
    """
    Debugging tools composed from smaller components.
    No inheritance - just protocols!
    """
    
    def __init__(self, coordinator_id: str, base_log_dir: str = "logs"):
        self.coordinator_id = coordinator_id
        self.base_log_dir = Path(base_log_dir)
        
        # Compose with various components
        self.flow_tracer = EventFlowTracer(coordinator_id, base_log_dir)
        self.container_loggers: Dict[str, ContainerLogger] = {}
        self.event_correlator = EventCorrelator()
    
    def create_container_logger(self, container_id: str, component_name: str) -> ContainerLogger:
        """Factory method for container loggers"""
        logger = ContainerLogger(container_id, component_name, base_log_dir=str(self.base_log_dir))
        logger_key = f"{container_id}.{component_name}"
        self.container_loggers[logger_key] = logger
        return logger
    
    def trace_signal_flow(self, start_event_id: str) -> List[Dict]:
        """Trace a signal from start event through entire system"""
        flow_events = self._read_flow_events_for_signal(start_event_id)
        return self._reconstruct_signal_path(flow_events)
    
    def debug_container_isolation(self, container_id: str) -> Dict:
        """Debug what's happening inside a specific container"""
        container_logs = self._read_container_logs(container_id)
        return {
            'log_count': len(container_logs),
            'error_count': len([log for log in container_logs if log.get('level') == 'ERROR']),
            'components': list(set(log.get('component_name') for log in container_logs)),
            'event_scopes': list(set(log.get('event_scope') for log in container_logs)),
            'recent_activity': container_logs[-10:] if container_logs else []
        }
    
    def get_cross_container_flows(self, time_window_minutes: int = 5) -> List[Dict]:
        """Get all cross-container communication in time window"""
        # Read flow logs and filter by time
        cutoff_time = datetime.utcnow().timestamp() - (time_window_minutes * 60)
        
        flows = []
        flow_log_path = self.base_log_dir / "flows" / f"{self.coordinator_id}_event_flows.log"
        
        if flow_log_path.exists():
            with open(flow_log_path, 'r') as f:
                for line in f:
                    try:
                        flow_event = json.loads(line.strip())
                        event_time = datetime.fromisoformat(flow_event['timestamp']).timestamp()
                        if event_time > cutoff_time and flow_event['flow_type'] == 'external':
                            flows.append(flow_event)
                    except:
                        continue
        
        return flows
    
    def _read_flow_events_for_signal(self, event_id: str) -> List[Dict]:
        """Read all flow events related to a signal"""
        # Implementation would read from flow logs
        # This is a simplified version
        return []
    
    def _reconstruct_signal_path(self, flow_events: List[Dict]) -> List[Dict]:
        """Reconstruct the path a signal took through the system"""
        # Implementation would analyze flow events and build path
        return []
    
    def _read_container_logs(self, container_id: str) -> List[Dict]:
        """Read logs for a specific container"""
        logs = []
        container_log_dir = self.base_log_dir / "containers" / container_id
        
        if container_log_dir.exists():
            for log_file in container_log_dir.glob("*.log"):
                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            log_entry = json.loads(line.strip())
                            logs.append(log_entry)
                        except:
                            continue
        
        return logs

class EventCorrelator:
    """
    Correlates events across containers - composable component.
    """
    
    def __init__(self):
        self.event_chains: Dict[str, List[str]] = defaultdict(list)
    
    def add_event_to_chain(self, correlation_id: str, event_id: str):
        """Add event to correlation chain"""
        self.event_chains[correlation_id].append(event_id)
    
    def get_event_chain(self, correlation_id: str) -> List[str]:
        """Get full event chain for correlation ID"""
        return self.event_chains.get(correlation_id, [])

class CorrelationContext:
    """Context manager for correlation tracking"""
    
    def __init__(self, logger_or_tracer: Any, correlation_id: str):
        self.target = logger_or_tracer
        self.correlation_id = correlation_id
        self.previous_id = None
    
    def __enter__(self):
        if hasattr(self.target, 'get_correlation_id'):
            self.previous_id = self.target.get_correlation_id()
        if hasattr(self.target, 'set_correlation_id'):
            self.target.set_correlation_id(self.correlation_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.target, 'set_correlation_id'):
            self.target.set_correlation_id(self.previous_id)

# ================================
# CAPABILITY COMPONENTS
# ================================

class LoggingCapability:
    """
    Adds logging capability to any component.
    Pure composition - no inheritance!
    """
    
    @staticmethod
    def add_to_component(component: Any, container_id: str, component_name: str, 
                        log_level: str = 'INFO') -> Any:
        """Add logging capability to any component"""
        
        # Create logger and attach it
        logger = ContainerLogger(container_id, component_name, log_level)
        component.logger = logger
        
        # Add convenience methods
        component.log = logger.log
        component.log_info = logger.log_info
        component.log_error = logger.log_error
        component.log_debug = logger.log_debug
        component.log_warning = logger.log_warning
        
        # Add correlation tracking
        component.set_correlation_id = logger.set_correlation_id
        component.get_correlation_id = logger.get_correlation_id
        component.with_correlation_id = logger.with_correlation_id
        
        return component

class EventTracingCapability:
    """
    Adds event tracing capability to any component.
    """
    
    @staticmethod
    def add_to_component(component: Any, coordinator_id: str) -> Any:
        """Add event tracing capability to any component"""
        
        tracer = EventFlowTracer(coordinator_id)
        component.event_tracer = tracer
        
        # Add tracing methods
        component.trace_event = tracer.trace_event
        component.trace_internal_event = tracer.trace_internal_event
        component.trace_external_event = tracer.trace_external_event
        
        return component

class DebuggingCapability:
    """
    Adds debugging capability to any component.
    """
    
    @staticmethod
    def add_to_component(component: Any) -> Any:
        """Add debugging capability to any component"""
        
        # Add state capture
        def capture_state() -> Dict[str, Any]:
            state = {
                'timestamp': datetime.utcnow().isoformat(),
                'component_type': type(component).__name__,
            }
            
            # Capture public attributes
            for attr in dir(component):
                if not attr.startswith('_'):
                    try:
                        value = getattr(component, attr)
                        if not callable(value):
                            # Try to serialize - if it fails, skip
                            try:
                                json.dumps(value)
                                state[attr] = value
                            except:
                                state[attr] = str(value)
                    except:
                        pass
            
            return state
        
        component.capture_state = capture_state
        
        # Add tracing toggle
        component._tracing_enabled = False
        component.enable_tracing = lambda enabled: setattr(component, '_tracing_enabled', enabled)
        
        return component

# ================================
# USAGE EXAMPLES
# ================================

def create_logging_components_for_container(container_id: str, coordinator_id: str):
    """
    Factory function to create all logging components for a container.
    Pure composition approach.
    """
    
    # Create debugger for the container
    debugger = ContainerDebugger(coordinator_id)
    
    # Create loggers for different components
    data_logger = debugger.create_container_logger(container_id, "data_handler")
    strategy_logger = debugger.create_container_logger(container_id, "strategy")
    risk_logger = debugger.create_container_logger(container_id, "risk_manager")
    
    return {
        'debugger': debugger,
        'loggers': {
            'data': data_logger,
            'strategy': strategy_logger,
            'risk': risk_logger
        }
    }

def add_logging_to_any_component(component: Any, container_id: str, component_name: str):
    """
    Add logging capability to ANY component using pure composition.
    No inheritance required!
    """
    
    # Add logging capability
    component = LoggingCapability.add_to_component(component, container_id, component_name)
    
    # Add event tracing capability  
    component = EventTracingCapability.add_to_component(component, "main_coordinator")
    
    # Add debugging capability
    component = DebuggingCapability.add_to_component(component)
    
    return component

# Example usage with any component type:
class SimpleStrategy:
    def generate_signal(self, data):
        return {"action": "BUY", "strength": 0.8}

# Add logging to any component - no inheritance needed!
strategy = SimpleStrategy()
strategy = add_logging_to_any_component(strategy, "container_001", "momentum_strategy")

# Now it can log!
# strategy.log_info("Generating signal", data_size=len(data))
# strategy.trace_internal_event("evt_123", "strategy", "risk_manager")

# Works with functions too!
def my_signal_function(data):
    return {"action": "SELL", "strength": 0.6}

# Add logging capability to a function
my_signal_function = add_logging_to_any_component(my_signal_function, "container_002", "custom_signal")

# Even works with external library components!
# sklearn_model = add_logging_to_any_component(RandomForestClassifier(), "container_003", "ml_model")