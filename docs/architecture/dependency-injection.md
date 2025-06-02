# Dependency Injection Architecture

## Overview

The protocol-based composition is enabled by a sophisticated dependency injection system that manages component relationships without requiring explicit coupling. Components declare their dependencies through protocol interfaces rather than concrete class references, allowing the injection system to wire together arbitrary combinations of compatible components.

This approach differs significantly from traditional dependency injection frameworks that rely on class hierarchies or framework-specific annotations. Instead, ADMF-PC uses protocol inspection to determine compatibility, enabling components from different libraries, frameworks, or implementation approaches to be combined seamlessly.

## Core Concepts

### Protocol-Based Dependencies

```python
from typing import Protocol

class DataProvider(Protocol):
    """Protocol defining data provider interface"""
    def get_bar_data(self, symbol: str, timestamp: datetime) -> BarData: ...
    def subscribe(self, callback: Callable[[BarData], None]) -> None: ...

class IndicatorCalculator(Protocol):
    """Protocol for indicator calculation"""
    def calculate(self, data: pd.Series) -> float: ...
    def get_required_periods(self) -> int: ...

class SignalGenerator(Protocol):
    """Protocol for signal generation"""
    def generate_signal(self, indicators: Dict[str, float]) -> Signal: ...
```

### Component Registration

```python
class ComponentRegistry:
    """Central registry for all available components"""
    def __init__(self):
        self.components = {}
        self.protocols = {}
        self.instances = {}
    
    def register_component(self, 
                          component_class: Type,
                          protocols: List[Type[Protocol]],
                          name: Optional[str] = None) -> None:
        """Register a component with its implemented protocols"""
        name = name or component_class.__name__
        
        self.components[name] = {
            'class': component_class,
            'protocols': protocols,
            'metadata': self._extract_metadata(component_class)
        }
        
        # Index by protocol for fast lookup
        for protocol in protocols:
            if protocol not in self.protocols:
                self.protocols[protocol] = []
            self.protocols[protocol].append(name)
    
    def get_components_for_protocol(self, protocol: Type[Protocol]) -> List[str]:
        """Get all components implementing a protocol"""
        return self.protocols.get(protocol, [])
```

## Dependency Resolution

### Automatic Wiring

```python
class DependencyResolver:
    """Resolves and wires component dependencies"""
    def __init__(self, registry: ComponentRegistry):
        self.registry = registry
        self.resolution_cache = {}
    
    def resolve_dependencies(self, component_name: str) -> Dict[str, Any]:
        """Resolve all dependencies for a component"""
        if component_name in self.resolution_cache:
            return self.resolution_cache[component_name]
        
        component_info = self.registry.components[component_name]
        component_class = component_info['class']
        
        # Inspect constructor
        signature = inspect.signature(component_class.__init__)
        dependencies = {}
        
        for param_name, param in signature.parameters.items():
            if param_name == 'self':
                continue
                
            # Check if parameter has a protocol type hint
            if hasattr(param.annotation, '__protocol__'):
                protocol = param.annotation
                # Find compatible component
                compatible = self.registry.get_components_for_protocol(protocol)
                
                if compatible:
                    # Use configuration or defaults to select
                    selected = self._select_component(compatible, param_name)
                    dependencies[param_name] = self._create_instance(selected)
                elif param.default is param.empty:
                    raise DependencyError(
                        f"No component found for required protocol {protocol}"
                    )
        
        self.resolution_cache[component_name] = dependencies
        return dependencies
```

### Circular Dependency Detection

```python
class DependencyGraph:
    """Manages dependency graph and detects cycles"""
    def __init__(self):
        self.graph = {}
        self.visited = set()
        self.rec_stack = set()
    
    def add_dependency(self, component: str, dependency: str) -> None:
        """Add a dependency relationship"""
        if component not in self.graph:
            self.graph[component] = []
        self.graph[component].append(dependency)
        
        # Check for cycles
        if self._has_cycle():
            raise CircularDependencyError(
                f"Circular dependency detected: {component} -> {dependency}"
            )
    
    def _has_cycle(self) -> bool:
        """Detect cycles using DFS"""
        self.visited.clear()
        self.rec_stack.clear()
        
        for node in self.graph:
            if node not in self.visited:
                if self._has_cycle_util(node):
                    return True
        return False
    
    def _has_cycle_util(self, node: str) -> bool:
        self.visited.add(node)
        self.rec_stack.add(node)
        
        for neighbor in self.graph.get(node, []):
            if neighbor not in self.visited:
                if self._has_cycle_util(neighbor):
                    return True
            elif neighbor in self.rec_stack:
                return True
        
        self.rec_stack.remove(node)
        return False
```

## Component Enhancement Through Capabilities

The architecture supports dynamic component enhancement through a capability system that can augment any component with cross-cutting concerns without modifying the original implementation:

### Capability System

```python
class CapabilityProvider:
    """Base class for capability providers"""
    def enhance(self, component: Any) -> Any:
        """Enhance component with capability"""
        raise NotImplementedError

class LoggingCapability(CapabilityProvider):
    """Add logging to any component"""
    def enhance(self, component: Any) -> Any:
        class LoggingWrapper:
            def __init__(self, wrapped):
                self._wrapped = wrapped
                self._logger = logging.getLogger(wrapped.__class__.__name__)
            
            def __getattr__(self, name):
                attr = getattr(self._wrapped, name)
                if callable(attr):
                    def logged_method(*args, **kwargs):
                        self._logger.debug(f"Calling {name} with args={args}")
                        try:
                            result = attr(*args, **kwargs)
                            self._logger.debug(f"{name} returned {result}")
                            return result
                        except Exception as e:
                            self._logger.error(f"{name} failed: {e}")
                            raise
                    return logged_method
                return attr
        
        return LoggingWrapper(component)

class MonitoringCapability(CapabilityProvider):
    """Add performance monitoring"""
    def enhance(self, component: Any) -> Any:
        class MonitoringWrapper:
            def __init__(self, wrapped):
                self._wrapped = wrapped
                self._metrics = {}
            
            def __getattr__(self, name):
                attr = getattr(self._wrapped, name)
                if callable(attr):
                    def monitored_method(*args, **kwargs):
                        start_time = time.time()
                        result = attr(*args, **kwargs)
                        duration = time.time() - start_time
                        
                        if name not in self._metrics:
                            self._metrics[name] = []
                        self._metrics[name].append(duration)
                        
                        return result
                    return monitored_method
                return attr
            
            def get_metrics(self):
                return {
                    name: {
                        'count': len(times),
                        'avg': np.mean(times),
                        'max': max(times),
                        'min': min(times)
                    }
                    for name, times in self._metrics.items()
                }
        
        return MonitoringWrapper(component)
```

### Capability Composition

```python
class CapabilityComposer:
    """Compose multiple capabilities onto components"""
    def __init__(self):
        self.capabilities = {
            'logging': LoggingCapability(),
            'monitoring': MonitoringCapability(),
            'validation': ValidationCapability(),
            'caching': CachingCapability(),
            'retry': RetryCapability()
        }
    
    def enhance_component(self, 
                         component: Any, 
                         capabilities: List[str]) -> Any:
        """Apply multiple capabilities to a component"""
        enhanced = component
        
        for capability_name in capabilities:
            if capability_name in self.capabilities:
                provider = self.capabilities[capability_name]
                enhanced = provider.enhance(enhanced)
            else:
                raise ValueError(f"Unknown capability: {capability_name}")
        
        return enhanced
```

## Container Injection

### Container-Aware Injection

```python
class ContainerInjector:
    """Inject dependencies with container context"""
    def __init__(self, registry: ComponentRegistry):
        self.registry = registry
        self.resolver = DependencyResolver(registry)
        self.containers = {}
    
    def create_container(self, 
                        container_type: str,
                        config: Dict[str, Any]) -> Container:
        """Create container with injected dependencies"""
        container = Container(container_id=generate_id())
        
        # Resolve container components
        components = config.get('components', [])
        for component_config in components:
            component_name = component_config['type']
            capabilities = component_config.get('capabilities', [])
            
            # Create component with dependencies
            instance = self._create_component(
                component_name,
                component_config.get('parameters', {}),
                container
            )
            
            # Enhance with capabilities
            if capabilities:
                composer = CapabilityComposer()
                instance = composer.enhance_component(instance, capabilities)
            
            # Register in container
            container.add_component(instance)
        
        # Wire event connections
        self._wire_container_events(container, config)
        
        return container
    
    def _create_component(self, 
                         component_name: str,
                         parameters: Dict,
                         container: Container) -> Any:
        """Create component with resolved dependencies"""
        # Resolve dependencies
        dependencies = self.resolver.resolve_dependencies(component_name)
        
        # Add container context
        dependencies['container'] = container
        dependencies.update(parameters)
        
        # Create instance
        component_class = self.registry.components[component_name]['class']
        return component_class(**dependencies)
```

## Protocol Inspection

### Runtime Protocol Checking

```python
class ProtocolInspector:
    """Inspect and validate protocol implementations"""
    
    @staticmethod
    def implements_protocol(obj: Any, protocol: Type[Protocol]) -> bool:
        """Check if object implements protocol"""
        # Get protocol methods
        protocol_methods = {
            name for name, value in inspect.getmembers(protocol)
            if not name.startswith('_') and callable(value)
        }
        
        # Check object has all methods
        obj_methods = {
            name for name, value in inspect.getmembers(obj)
            if not name.startswith('_') and callable(value)
        }
        
        return protocol_methods.issubset(obj_methods)
    
    @staticmethod
    def validate_protocol_signature(obj: Any, protocol: Type[Protocol]) -> List[str]:
        """Validate method signatures match protocol"""
        errors = []
        
        for name, protocol_method in inspect.getmembers(protocol):
            if name.startswith('_') or not callable(protocol_method):
                continue
                
            obj_method = getattr(obj, name, None)
            if obj_method is None:
                errors.append(f"Missing method: {name}")
                continue
            
            # Compare signatures
            protocol_sig = inspect.signature(protocol_method)
            obj_sig = inspect.signature(obj_method)
            
            # Check parameters (excluding 'self')
            protocol_params = list(protocol_sig.parameters.values())[1:]
            obj_params = list(obj_sig.parameters.values())[1:]
            
            if len(protocol_params) != len(obj_params):
                errors.append(
                    f"Method {name}: parameter count mismatch"
                )
                continue
            
            # Check parameter types
            for p1, p2 in zip(protocol_params, obj_params):
                if p1.annotation != p2.annotation:
                    errors.append(
                        f"Method {name}: parameter {p1.name} type mismatch"
                    )
        
        return errors
```

## Advanced Injection Patterns

### Lazy Injection

```python
class LazyInjector:
    """Defer component creation until first use"""
    def __init__(self, resolver: DependencyResolver):
        self.resolver = resolver
    
    def create_lazy_proxy(self, component_name: str) -> Any:
        """Create lazy proxy for component"""
        class LazyProxy:
            def __init__(self, name: str, resolver: DependencyResolver):
                self._component_name = name
                self._resolver = resolver
                self._instance = None
            
            def _ensure_instance(self):
                if self._instance is None:
                    dependencies = self._resolver.resolve_dependencies(
                        self._component_name
                    )
                    component_class = registry.components[
                        self._component_name
                    ]['class']
                    self._instance = component_class(**dependencies)
            
            def __getattr__(self, name):
                self._ensure_instance()
                return getattr(self._instance, name)
        
        return LazyProxy(component_name, self.resolver)
```

### Scoped Injection

```python
class ScopedInjector:
    """Manage component lifecycle scopes"""
    def __init__(self):
        self.scopes = {
            'singleton': {},      # One instance globally
            'container': {},      # One per container
            'transient': None     # New instance each time
        }
    
    def get_instance(self, 
                    component_name: str, 
                    scope: str = 'transient',
                    context: Optional[str] = None) -> Any:
        """Get instance based on scope"""
        if scope == 'singleton':
            if component_name not in self.scopes['singleton']:
                self.scopes['singleton'][component_name] = (
                    self._create_instance(component_name)
                )
            return self.scopes['singleton'][component_name]
            
        elif scope == 'container':
            if context is None:
                raise ValueError("Container scope requires context")
            
            if context not in self.scopes['container']:
                self.scopes['container'][context] = {}
            
            if component_name not in self.scopes['container'][context]:
                self.scopes['container'][context][component_name] = (
                    self._create_instance(component_name)
                )
            return self.scopes['container'][context][component_name]
            
        else:  # transient
            return self._create_instance(component_name)
```

## Configuration-Driven Injection

### YAML Configuration

```yaml
components:
  - name: "data_provider"
    type: "CSVDataProvider"
    parameters:
      file_path: "data/market_data.csv"
    capabilities: ["logging", "monitoring"]
    
  - name: "indicator_hub"
    type: "IndicatorHub"
    dependencies:
      data_provider: "data_provider"  # Reference by name
    parameters:
      cache_size: 1000
      
  - name: "strategy"
    type: "MomentumStrategy"
    dependencies:
      indicator_calculator: "indicator_hub"
    parameters:
      lookback_period: 20
    capabilities: ["logging", "validation"]
    scope: "container"  # One per container
```

### Configuration Parser

```python
class InjectionConfigParser:
    """Parse configuration for dependency injection"""
    def __init__(self, registry: ComponentRegistry):
        self.registry = registry
        self.injector = ContainerInjector(registry)
    
    def parse_and_create(self, config: Dict) -> Container:
        """Parse config and create wired container"""
        # Build dependency graph
        graph = DependencyGraph()
        components = {}
        
        # First pass: register all components
        for comp_config in config['components']:
            name = comp_config['name']
            comp_type = comp_config['type']
            
            # Validate component exists
            if comp_type not in self.registry.components:
                raise ConfigError(f"Unknown component type: {comp_type}")
            
            components[name] = comp_config
            
            # Add dependencies to graph
            for dep_name in comp_config.get('dependencies', {}).values():
                graph.add_dependency(name, dep_name)
        
        # Second pass: create in dependency order
        creation_order = graph.topological_sort()
        instances = {}
        
        for name in creation_order:
            comp_config = components[name]
            instance = self._create_component_from_config(
                comp_config,
                instances
            )
            instances[name] = instance
        
        # Create container with all components
        return self._assemble_container(instances, config)
```

## Testing Dependency Injection

### Mock Injection

```python
class MockInjector:
    """Inject mocks for testing"""
    def __init__(self, registry: ComponentRegistry):
        self.registry = registry
        self.mocks = {}
    
    def register_mock(self, protocol: Type[Protocol], mock: Any) -> None:
        """Register mock for protocol"""
        self.mocks[protocol] = mock
    
    def create_with_mocks(self, component_name: str) -> Any:
        """Create component with mocked dependencies"""
        component_class = self.registry.components[component_name]['class']
        signature = inspect.signature(component_class.__init__)
        
        dependencies = {}
        for param_name, param in signature.parameters.items():
            if param_name == 'self':
                continue
                
            if hasattr(param.annotation, '__protocol__'):
                protocol = param.annotation
                if protocol in self.mocks:
                    dependencies[param_name] = self.mocks[protocol]
                else:
                    # Create auto-mock
                    dependencies[param_name] = create_autospec(protocol)
        
        return component_class(**dependencies)
```

## Best Practices

### DO:
- Define clear protocol interfaces
- Use type hints for dependency declaration
- Validate protocol implementations at runtime
- Keep dependency graphs shallow
- Use appropriate scopes for components

### DON'T:
- Create circular dependencies
- Use concrete classes as dependencies
- Bypass the injection system
- Share mutable state between components
- Over-use singleton scope

## Summary

The dependency injection architecture enables:

1. **Flexibility**: Mix components from any source
2. **Testability**: Easy mock injection
3. **Maintainability**: Clear dependency relationships
4. **Extensibility**: Add capabilities without modification
5. **Safety**: Compile-time and runtime validation

This approach allows ADMF-PC to achieve true composability while maintaining type safety and clear architectural boundaries.