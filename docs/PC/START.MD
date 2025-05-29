# Protocol + Scoped Container Architecture Implementation
# Combines the best of both architectural patterns

from typing import Protocol, runtime_checkable, Dict, Any, Optional, List, Callable, Union
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import threading
import importlib
import uuid
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import gc

# ====================
# Core Protocols
# ====================

@runtime_checkable
class Component(Protocol):
    """Lifecycle management protocol for system components"""
    
    @abstractmethod
    def initialize(self, context: 'SystemContext') -> None:
        """Initialize component with system dependencies"""
        ...
        
    @abstractmethod
    def start(self) -> None:
        """Begin component operation"""
        ...
        
    @abstractmethod
    def stop(self) -> None:
        """Stop component operation"""
        ...
        
    @abstractmethod
    def reset(self) -> None:
        """Reset component state for reuse"""
        ...
        
    @abstractmethod
    def teardown(self) -> None:
        """Release all resources"""
        ...

@runtime_checkable
class EventSubscriber(Protocol):
    """Protocol for components that handle events"""
    
    @abstractmethod
    def initialize_event_subscriptions(self) -> None:
        """Set up event subscriptions"""
        ...

@runtime_checkable
class Optimizable(Protocol):
    """Protocol for components that can be optimized"""
    
    @abstractmethod
    def get_parameter_space(self) -> Dict[str, Any]:
        """Return parameter space for optimization"""
        ...
        
    @abstractmethod
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Apply parameter values"""
        ...
        
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameter values"""
        ...

@runtime_checkable
class Configurable(Protocol):
    """Protocol for components with configuration"""
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """Apply configuration settings"""
        ...
        
    @abstractmethod
    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration"""
        ...

# ====================
# System Context
# ====================

class RunMode(Enum):
    BACKTEST_SINGLE = "backtest_single"
    BACKTEST_PARALLEL = "backtest_parallel"
    OPTIMIZATION = "optimization"
    LIVE_TRADING = "live_trading"
    PAPER_TRADING = "paper_trading"

@dataclass
class SystemContext:
    """Shared context passed to all components during initialization"""
    config: 'Config'
    event_bus: 'EventBus'
    container: 'Container'
    logger: 'Logger'
    run_mode: RunMode
    metadata: Optional[Dict[str, Any]] = None
    
    def resolve(self, service_name: str) -> Any:
        """Convenience method to resolve services from container"""
        return self.container.resolve(service_name)

# ====================
# Scoped Container Implementation
# ====================

class ScopedContainer:
    """Scoped container with protocol-based component management"""
    
    def __init__(self, container_id: str, shared_services: Dict[str, Any] = None):
        self.container_id = container_id
        self.shared_services = shared_services or {}
        self.local_components: Dict[str, Any] = {}
        self.component_factory = ComponentFactory()
        self.dependency_graph = DependencyGraph()
        self._lock = threading.RLock()
        
        # Container-specific services
        self.event_bus = EventBus()
        self.logger = StructuredLogger(f"container.{container_id}")
        
        # Capability registry for this container
        self.capability_registry = CapabilityRegistry()
    
    def register_shared_service(self, name: str, service: Any) -> None:
        """Register a shared read-only service"""
        with self._lock:
            self.shared_services[name] = service
    
    def create_component(self, spec: Dict[str, Any]) -> Any:
        """Create component with automatic protocol detection"""
        component = self.component_factory.create_component(spec)
        component_name = spec.get('name', component.__class__.__name__)
        
        with self._lock:
            self.local_components[component_name] = component
            
            # Track dependencies
            dependencies = spec.get('dependencies', [])
            self.dependency_graph.add_component(component_name, spec)
            for dep in dependencies:
                self.dependency_graph.add_dependency(component_name, dep)
        
        return component
    
    def resolve(self, name: str) -> Any:
        """Resolve from local components first, then shared services"""
        with self._lock:
            # Check local components first
            if name in self.local_components:
                return self.local_components[name]
            
            # Check container services
            if name == "event_bus":
                return self.event_bus
            if name == "logger":
                return self.logger
            
            # Fall back to shared services
            if name in self.shared_services:
                return self.shared_services[name]
            
            raise ValueError(f"Component '{name}' not found in container {self.container_id}")
    
    def initialize_scope(self) -> None:
        """Initialize all components respecting dependencies"""
        init_order = self.dependency_graph.get_initialization_order()
        
        context = SystemContext(
            config=self.resolve("config") if "config" in self.shared_services else Config(),
            event_bus=self.event_bus,
            container=self,
            logger=self.logger,
            run_mode=self._determine_run_mode()
        )
        
        for component_name in init_order:
            component = self.local_components[component_name]
            
            # Initialize if component supports it (duck typing)
            if hasattr(component, 'initialize') and callable(getattr(component, 'initialize')):
                component.initialize(context)
            
            # Set up event subscriptions if supported
            if hasattr(component, 'initialize_event_subscriptions'):
                component.initialize_event_subscriptions()
    
    def start_scope(self) -> None:
        """Start all components"""
        start_order = self.dependency_graph.get_initialization_order()
        
        for component_name in start_order:
            component = self.local_components[component_name]
            if hasattr(component, 'start') and callable(getattr(component, 'start')):
                component.start()
    
    def stop_scope(self) -> None:
        """Stop all components in reverse order"""
        stop_order = reversed(self.dependency_graph.get_initialization_order())
        
        for component_name in stop_order:
            component = self.local_components.get(component_name)
            if component and hasattr(component, 'stop'):
                try:
                    component.stop()
                except Exception as e:
                    self.logger.error(f"Error stopping {component_name}: {e}")
    
    def teardown_scope(self) -> None:
        """Teardown all components in reverse order"""
        teardown_order = reversed(self.dependency_graph.get_initialization_order())
        
        for component_name in teardown_order:
            component = self.local_components.get(component_name)
            if component and hasattr(component, 'teardown'):
                try:
                    component.teardown()
                except Exception as e:
                    self.logger.error(f"Error tearing down {component_name}: {e}")
        
        # Clear local components
        self.local_components.clear()
        self.dependency_graph = DependencyGraph()
    
    def _determine_run_mode(self) -> RunMode:
        """Determine run mode from context"""
        # Could be enhanced to check shared config or other sources
        return RunMode.BACKTEST_SINGLE

# ====================
# Component Factory with Capabilities
# ====================

class ComponentFactory:
    """Creates and configures components based on specifications"""
    
    def __init__(self, capability_registry: Optional['CapabilityRegistry'] = None):
        self.capability_registry = capability_registry or CapabilityRegistry()
    
    def create_component(self, spec: Dict[str, Any]) -> Any:
        """Create component from specification"""
        # Create base component
        component = self._create_base_component(spec)
        
        # Apply capabilities
        capabilities = self._resolve_capabilities(spec)
        for capability_name in capabilities:
            capability = self.capability_registry.get(capability_name)
            component = capability.apply(component, spec)
        
        return component
    
    def _create_base_component(self, spec: Dict[str, Any]) -> Any:
        """Create the base component instance"""
        if 'class' in spec:
            return self._create_from_class(spec)
        elif 'function' in spec:
            return self._create_from_function(spec)
        elif 'instance' in spec:
            return spec['instance']  # Pre-created instance
        else:
            raise ValueError("Component spec must include 'class', 'function', or 'instance'")
    
    def _create_from_class(self, spec: Dict[str, Any]) -> Any:
        """Create component from class specification"""
        class_path = spec['class']
        params = spec.get('params', {})
        
        # Handle built-in classes
        if class_path in globals():
            cls = globals()[class_path]
        # Dynamic import
        elif '.' in class_path:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
        else:
            raise ValueError(f"Cannot resolve class: {class_path}")
        
        return cls(**params)
    
    def _create_from_function(self, spec: Dict[str, Any]) -> Any:
        """Wrap function as component"""
        func = spec['function']
        params = spec.get('params', {})
        return FunctionWrapper(func, params, spec.get('name', 'function'))
    
    def _resolve_capabilities(self, spec: Dict[str, Any]) -> List[str]:
        """Resolve capabilities from spec or profile"""
        if 'capabilities' in spec:
            return spec['capabilities']
        
        if 'profile' in spec:
            # Look up profile capabilities
            profiles = {
                'minimal': [],
                'basic': ['lifecycle'],
                'trading': ['lifecycle', 'events', 'reset'],
                'optimizable': ['lifecycle', 'events', 'optimization', 'reset']
            }
            return profiles.get(spec['profile'], [])
        
        return []

# ====================
# Container Factory & Management
# ====================

class ContainerFactory:
    """Factory for creating specialized containers"""
    
    def __init__(self, shared_services: Dict[str, Any]):
        self.shared_services = shared_services
        self.container_count = 0
        self._lock = threading.Lock()
    
    def create_backtest_container(self, config: Dict[str, Any]) -> ScopedContainer:
        """Create container for backtesting"""
        container_id = self._generate_container_id("backtest")
        container = ScopedContainer(container_id, self.shared_services.copy())
        
        # Create components based on config
        component_specs = config.get('components', {})
        for name, spec in component_specs.items():
            spec['name'] = name
            container.create_component(spec)
        
        return container
    
    def create_optimization_container(self, parameters: Dict[str, Any]) -> ScopedContainer:
        """Create container for optimization trial"""
        container_id = self._generate_container_id("opt")
        container = ScopedContainer(container_id, self.shared_services.copy())
        
        # Create components with specific parameters
        container.create_component({
            'name': 'strategy',
            'class': parameters['strategy_class'],
            'params': parameters['strategy_params'],
            'capabilities': ['lifecycle', 'events', 'optimization']
        })
        
        container.create_component({
            'name': 'portfolio',
            'class': 'Portfolio',
            'params': {'initial_cash': 100000},
            'capabilities': ['lifecycle', 'events', 'reset']
        })
        
        # Add other required components...
        
        return container
    
    def create_live_trading_container(self, strategy_spec: Dict[str, Any]) -> ScopedContainer:
        """Create container for live trading strategy"""
        container_id = self._generate_container_id("live")
        container = ScopedContainer(container_id, self.shared_services.copy())
        
        # Create live trading components
        container.create_component({
            'name': 'strategy',
            'class': strategy_spec['class'],
            'params': strategy_spec['params'],
            'capabilities': ['lifecycle', 'events', 'monitoring']
        })
        
        # Virtual portfolio for tracking
        container.create_component({
            'name': 'portfolio',
            'class': 'VirtualPortfolio',
            'params': {
                'strategy_id': container_id,
                'initial_capital': strategy_spec.get('allocated_capital', 100000)
            }
        })
        
        return container
    
    def _generate_container_id(self, prefix: str) -> str:
        """Generate unique container ID"""
        with self._lock:
            self.container_count += 1
            return f"{prefix}_{self.container_count}_{uuid.uuid4().hex[:8]}"

# ====================
# Container Lifecycle Manager
# ====================

class ContainerLifecycleManager:
    """Manages lifecycle of multiple containers"""
    
    def __init__(self, shared_services: Dict[str, Any]):
        self.shared_services = shared_services
        self.container_factory = ContainerFactory(shared_services)
        self.active_containers: Dict[str, ScopedContainer] = {}
        self._lock = threading.RLock()
    
    def create_and_start_container(self, container_type: str, 
                                  spec: Dict[str, Any]) -> str:
        """Create, initialize, and start a container"""
        # Create appropriate container
        if container_type == "backtest":
            container = self.container_factory.create_backtest_container(spec)
        elif container_type == "optimization":
            container = self.container_factory.create_optimization_container(spec)
        elif container_type == "live_trading":
            container = self.container_factory.create_live_trading_container(spec)
        else:
            raise ValueError(f"Unknown container type: {container_type}")
        
        # Initialize container
        container.initialize_scope()
        
        # Start components
        container.start_scope()
        
        # Track active container
        with self._lock:
            self.active_containers[container.container_id] = container
        
        return container.container_id
    
    def stop_and_destroy_container(self, container_id: str) -> None:
        """Stop and destroy a container"""
        with self._lock:
            if container_id not in self.active_containers:
                return
            
            container = self.active_containers[container_id]
        
        # Stop components
        container.stop_scope()
        
        # Teardown
        container.teardown_scope()
        
        # Remove from tracking
        with self._lock:
            del self.active_containers[container_id]

# ====================
# Capabilities System
# ====================

from abc import ABC

class Capability(ABC):
    """Base class for composable capabilities"""
    
    @abstractmethod
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        """Apply this capability to a component"""
        ...
    
    @abstractmethod
    def get_name(self) -> str:
        """Get capability name"""
        ...

class LifecycleCapability(Capability):
    """Adds lifecycle management to components"""
    
    def get_name(self) -> str:
        return "lifecycle"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        if not hasattr(component, '_lifecycle_state'):
            component._lifecycle_state = {
                'initialized': False,
                'running': False,
                'name': spec.get('name', component.__class__.__name__)
            }
            
            # Add lifecycle methods if they don't exist
            if not hasattr(component, 'initialize'):
                def initialize(ctx):
                    component._lifecycle_state['initialized'] = True
                component.initialize = initialize
            
            if not hasattr(component, 'start'):
                def start():
                    component._lifecycle_state['running'] = True
                component.start = start
            
            if not hasattr(component, 'stop'):
                def stop():
                    component._lifecycle_state['running'] = False
                component.stop = stop
            
            if not hasattr(component, 'teardown'):
                def teardown():
                    component._lifecycle_state['initialized'] = False
                component.teardown = teardown
        
        return component

class EventCapability(Capability):
    """Adds event handling to components"""
    
    def get_name(self) -> str:
        return "events"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        if not hasattr(component, '_events'):
            component._events = EventSubscriptions()
            
            # Enhance initialize method to set up events
            original_init = getattr(component, 'initialize', lambda ctx: None)
            def enhanced_init(ctx):
                component._events.initialize_with_bus(ctx.event_bus)
                original_init(ctx)
                if hasattr(component, 'setup_subscriptions'):
                    component.setup_subscriptions()
            component.initialize = enhanced_init
            
            # Add subscription helper
            component.subscribe = component._events.subscribe
            component.publish = component._events.publish
        
        return component

class OptimizationCapability(Capability):
    """Adds optimization support to components"""
    
    def get_name(self) -> str:
        return "optimization"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        if not hasattr(component, '_parameters'):
            component._parameters = {}
            component._parameter_space = {}
            
            # Add optimization methods if they don't exist
            if not hasattr(component, 'get_parameter_space'):
                component.get_parameter_space = lambda: component._parameter_space
            
            if not hasattr(component, 'set_parameters'):
                def set_parameters(params):
                    component._parameters.update(params)
                component.set_parameters = set_parameters
            
            if not hasattr(component, 'get_parameters'):
                component.get_parameters = lambda: component._parameters.copy()
        
        return component

class ResetCapability(Capability):
    """Adds reset functionality to components"""
    
    def get_name(self) -> str:
        return "reset"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        if not hasattr(component, 'reset'):
            # Try to find state attributes to reset
            state_attrs = spec.get('reset_attributes', [])
            
            def reset_method():
                for attr in state_attrs:
                    if hasattr(component, attr):
                        value = getattr(component, attr)
                        if isinstance(value, list):
                            value.clear()
                        elif isinstance(value, dict):
                            value.clear()
                        elif hasattr(value, 'reset'):
                            value.reset()
                        else:
                            setattr(component, attr, None)
                
                # Call original reset if it exists
                if hasattr(component, '_original_reset'):
                    component._original_reset()
            
            # Save original reset if it exists
            if hasattr(component, 'reset'):
                component._original_reset = component.reset
            
            component.reset = reset_method
        
        return component

class CapabilityRegistry:
    """Registry for all available capabilities"""
    
    def __init__(self):
        self.capabilities: Dict[str, Capability] = {}
        self._register_standard_capabilities()
    
    def register(self, capability: Capability) -> None:
        """Register a capability"""
        self.capabilities[capability.get_name()] = capability
    
    def get(self, name: str) -> Capability:
        """Get capability by name"""
        if name not in self.capabilities:
            raise ValueError(f"Unknown capability: {name}")
        return self.capabilities[name]
    
    def _register_standard_capabilities(self) -> None:
        """Register standard capabilities"""
        self.register(LifecycleCapability())
        self.register(EventCapability())
        self.register(OptimizationCapability())
        self.register(ResetCapability())

# ====================
# Supporting Classes
# ====================

class EventBus:
    """Simple event bus implementation"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.RLock()
    
    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to events"""
        with self._lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(handler)
    
    def publish(self, event: 'Event') -> None:
        """Publish event to all subscribers"""
        event_type = event.event_type
        
        with self._lock:
            handlers = self.subscribers.get(event_type, [])
        
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"Error in event handler {handler}: {e}")

class Event:
    """Event message structure"""
    
    def __init__(self, event_type: str, payload: Dict[str, Any] = None):
        self.event_type = event_type
        self.payload = payload or {}
        self.timestamp = datetime.now()

class EventSubscriptions:
    """Composable event handling"""
    
    def __init__(self):
        self.event_bus: Optional[EventBus] = None
        self.subscriptions: List[tuple] = []
    
    def initialize_with_bus(self, event_bus: EventBus) -> None:
        """Initialize with event bus from system context"""
        self.event_bus = event_bus
        
        # Register stored subscriptions
        for event_type, handler in self.subscriptions:
            self.event_bus.subscribe(event_type, handler)
    
    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to event type"""
        if not self.event_bus:
            # Store for later when bus is available
            self.subscriptions.append((event_type, handler))
        else:
            self.event_bus.subscribe(event_type, handler)
    
    def publish(self, event: Event) -> None:
        """Publish event"""
        if self.event_bus:
            self.event_bus.publish(event)

class DependencyGraph:
    """Manages component dependencies"""
    
    def __init__(self):
        self.components: Dict[str, Dict[str, Any]] = {}
        self.dependencies: Dict[str, List[str]] = {}
    
    def add_component(self, name: str, metadata: Dict[str, Any]) -> None:
        """Add component to graph"""
        self.components[name] = metadata
        if name not in self.dependencies:
            self.dependencies[name] = []
    
    def add_dependency(self, component: str, dependency: str) -> None:
        """Add dependency relationship"""
        if component not in self.dependencies:
            self.dependencies[component] = []
        self.dependencies[component].append(dependency)
    
    def get_initialization_order(self) -> List[str]:
        """Get components in initialization order"""
        # Simple topological sort
        visited = set()
        order = []
        
        def visit(component: str):
            if component in visited:
                return
            visited.add(component)
            
            for dep in self.dependencies.get(component, []):
                if dep in self.components:
                    visit(dep)
            
            order.append(component)
        
        for component in self.components:
            visit(component)
        
        return order

class FunctionWrapper:
    """Wraps a function to make it behave like a component"""
    
    def __init__(self, func: Callable, params: Dict[str, Any], name: str):
        self.func = func
        self.params = params
        self.name = name
        self._result = None
    
    def __call__(self, *args, **kwargs):
        """Make the wrapper callable"""
        merged_kwargs = {**self.params, **kwargs}
        self._result = self.func(*args, **merged_kwargs)
        return self._result
    
    def get_result(self):
        """Get last function result"""
        return self._result

class StructuredLogger:
    """Simple structured logger"""
    
    def __init__(self, name: str):
        self.name = name
    
    def info(self, message: str, **kwargs):
        print(f"[INFO] {self.name}: {message}", kwargs)
    
    def error(self, message: str, **kwargs):
        print(f"[ERROR] {self.name}: {message}", kwargs)
    
    def debug(self, message: str, **kwargs):
        print(f"[DEBUG] {self.name}: {message}", kwargs)

class Config:
    """Simple configuration holder"""
    
    def __init__(self, data: Dict[str, Any] = None):
        self.data = data or {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        keys = key.split('.')
        value = self.data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value

class Container:
    """Simple dependency injection container"""
    
    def __init__(self):
        self.instances: Dict[str, Any] = {}
        self._lock = threading.RLock()
    
    def register_instance(self, name: str, instance: Any) -> None:
        """Register an instance"""
        with self._lock:
            self.instances[name] = instance
    
    def resolve(self, name: str) -> Any:
        """Resolve a dependency"""
        with self._lock:
            if name in self.instances:
                return self.instances[name]
            raise ValueError(f"Dependency '{name}' not found")

# ====================
# Orchestration Examples
# ====================

class BacktestCoordinator:
    """Orchestrates backtesting using scoped containers"""
    
    def __init__(self, shared_services: Dict[str, Any]):
        self.lifecycle_manager = ContainerLifecycleManager(shared_services)
        self.results_collector = {}
    
    def run_backtest(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single backtest in an isolated container"""
        container_id = None
        
        try:
            # Create and start container
            container_id = self.lifecycle_manager.create_and_start_container(
                "backtest", config
            )
            
            # Get container
            container = self.lifecycle_manager.active_containers[container_id]
            
            # Run backtest logic
            results = self._execute_backtest(container)
            
            return results
            
        finally:
            # Always cleanup
            if container_id:
                self.lifecycle_manager.stop_and_destroy_container(container_id)
    
    def _execute_backtest(self, container: ScopedContainer) -> Dict[str, Any]:
        """Execute backtest within container"""
        # Get components
        data_handler = container.resolve("data_handler")
        strategy = container.resolve("strategy")
        portfolio = container.resolve("portfolio")
        
        # Run backtest loop
        events_processed = 0
        
        # Main event loop would go here...
        # This is simplified for example
        
        return {
            'container_id': container.container_id,
            'events_processed': events_processed,
            'final_value': 100000,  # Would get from portfolio
            'sharpe_ratio': 1.5
        }

class OptimizationOrchestrator:
    """Orchestrates parallel optimization using containers"""
    
    def __init__(self, shared_services: Dict[str, Any]):
        self.lifecycle_manager = ContainerLifecycleManager(shared_services)
        self.max_workers = 4
    
    def run_optimization(self, base_config: Dict[str, Any], 
                        parameter_space: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run parallel optimization with isolated containers"""
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for params in parameter_space:
                # Merge base config with specific parameters
                trial_config = {
                    'strategy_class': base_config['strategy_class'],
                    'strategy_params': params
                }
                
                # Submit to executor
                future = executor.submit(self._run_single_trial, trial_config)
                futures.append((params, future))
            
            # Collect results
            results = []
            for params, future in futures:
                try:
                    result = future.result()
                    result['parameters'] = params
                    results.append(result)
                except Exception as e:
                    print(f"Trial failed for {params}: {e}")
        
        # Find best result
        best_result = max(results, key=lambda x: x.get('sharpe_ratio', 0))
        
        return {
            'best_parameters': best_result['parameters'],
            'best_sharpe': best_result['sharpe_ratio'],
            'all_results': results
        }
    
    def _run_single_trial(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run single optimization trial in container"""
        container_id = None
        
        try:
            # Create container for this trial
            container_id = self.lifecycle_manager.create_and_start_container(
                "optimization", config
            )
            
            # Get container
            container = self.lifecycle_manager.active_containers[container_id]
            
            # Run backtest
            return self._execute_trial(container)
            
        finally:
            if container_id:
                self.lifecycle_manager.stop_and_destroy_container(container_id)
    
    def _execute_trial(self, container: ScopedContainer) -> Dict[str, Any]:
        """Execute optimization trial"""
        # Simplified - would run actual backtest
        return {
            'sharpe_ratio': 1.5,
            'total_return': 0.15,
            'max_drawdown': -0.08
        }

# ====================
# Example Components
# ====================

class Portfolio:
    """Example portfolio component"""
    
    def __init__(self, initial_cash: float = 100000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}
        self.trades = []
    
    def get_portfolio_value(self) -> float:
        """Get total portfolio value"""
        return self.cash + sum(pos['value'] for pos in self.positions.values())
    
    def reset(self) -> None:
        """Reset portfolio state"""
        self.cash = self.initial_cash
        self.positions.clear()
        self.trades.clear()

class TrendFollowingStrategy:
    """Example strategy component"""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.prices = []
        self.last_signal = None
    
    def setup_subscriptions(self) -> None:
        """Set up event subscriptions"""
        self.subscribe("BAR", self.on_bar)
    
    def on_bar(self, event: Event) -> None:
        """Handle price bar"""
        price = event.payload['close']
        self.prices.append(price)
        
        if len(self.prices) >= self.slow_period:
            fast_ma = sum(self.prices[-self.fast_period:]) / self.fast_period
            slow_ma = sum(self.prices[-self.slow_period:]) / self.slow_period
            
            if fast_ma > slow_ma and self.last_signal != "BUY":
                self.last_signal = "BUY"
                signal_event = Event("SIGNAL", {
                    'direction': 'BUY',
                    'strength': (fast_ma - slow_ma) / slow_ma
                })
                self.publish(signal_event)
            
            elif fast_ma < slow_ma and self.last_signal != "SELL":
                self.last_signal = "SELL"
                signal_event = Event("SIGNAL", {
                    'direction': 'SELL',
                    'strength': (slow_ma - fast_ma) / slow_ma
                })
                self.publish(signal_event)
    
    def get_parameter_space(self) -> Dict[str, Any]:
        """Define optimizable parameters"""
        return {
            'fast_period': [5, 10, 15, 20],
            'slow_period': [20, 30, 40, 50]
        }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Apply parameters"""
        if 'fast_period' in params:
            self.fast_period = params['fast_period']
        if 'slow_period' in params:
            self.slow_period = params['slow_period']
    
    def reset(self) -> None:
        """Reset strategy state"""
        self.prices.clear()
        self.last_signal = None

# ====================
# Usage Example
# ====================

def main():
    """Example usage of the system"""
    
    # Set up shared services
    shared_services = {
        'config': Config({
            'system': {'name': 'ADMF-Trader'},
            'data': {'source': 'historical'}
        }),
        'market_data': None,  # Would be actual market data service
    }
    
    # Example 1: Single backtest
    print("Running single backtest...")
    backtest_config = {
        'components': {
            'strategy': {
                'class': 'TrendFollowingStrategy',
                'params': {'fast_period': 10, 'slow_period': 30},
                'capabilities': ['lifecycle', 'events', 'optimization', 'reset']
            },
            'portfolio': {
                'class': 'Portfolio',
                'params': {'initial_cash': 100000},
                'capabilities': ['lifecycle', 'reset']
            }
        }
    }
    
    coordinator = BacktestCoordinator(shared_services)
    results = coordinator.run_backtest(backtest_config)
    print(f"Backtest results: {results}")
    
    # Example 2: Optimization
    print("\nRunning optimization...")
    base_config = {
        'strategy_class': 'TrendFollowingStrategy'
    }
    
    parameter_space = [
        {'fast_period': 5, 'slow_period': 20},
        {'fast_period': 10, 'slow_period': 30},
        {'fast_period': 15, 'slow_period': 40},
        {'fast_period': 20, 'slow_period': 50}
    ]
    
    optimizer = OptimizationOrchestrator(shared_services)
    opt_results = optimizer.run_optimization(base_config, parameter_space)
    print(f"Best parameters: {opt_results['best_parameters']}")
    print(f"Best Sharpe: {opt_results['best_sharpe']}")

if __name__ == "__main__":
    main()
