"""
Component Discovery System for ADMF-PC

This module provides automatic discovery and registration of containers,
strategies, classifiers, and other components using decorators and module scanning.

Key Features:
- Auto-discovery via decorators
- Plugin-based architecture
- Configuration-driven registration
- Type-safe component resolution
"""

import importlib
import inspect
import pkgutil
from typing import Dict, Any, Type, Callable, Set, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import logging
from enum import Enum

from ..containers.protocols import ContainerRole, ContainerProtocol
from ..events import EventBusProtocol

logger = logging.getLogger(__name__)


@dataclass
class ComponentInfo:
    """Information about a discoverable component."""
    name: str
    component_type: str  # 'container', 'strategy', 'classifier', 'risk_validator'
    role: Optional[ContainerRole] = None
    factory: Optional[Callable] = None
    capabilities: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComponentRegistry:
    """
    Registry for auto-discovered components.
    
    Supports:
    - Decorator-based registration
    - Module scanning
    - Configuration-driven discovery
    - Type-safe resolution
    """
    
    def __init__(self):
        self._components: Dict[str, ComponentInfo] = {}
        self._factories: Dict[str, Callable] = {}
        self._by_type: Dict[str, List[str]] = {}
        self._by_role: Dict[ContainerRole, List[str]] = {}
    
    def register_component(self, info: ComponentInfo) -> None:
        """Register a component."""
        self._components[info.name] = info
        
        if info.factory:
            self._factories[info.name] = info.factory
        
        # Index by type
        if info.component_type not in self._by_type:
            self._by_type[info.component_type] = []
        self._by_type[info.component_type].append(info.name)
        
        # Index by role (for containers)
        if info.role:
            if info.role not in self._by_role:
                self._by_role[info.role] = []
            self._by_role[info.role].append(info.name)
        
        logger.debug(f"Registered component: {info.name} ({info.component_type})")
    
    def get_component(self, name: str) -> Optional[ComponentInfo]:
        """Get component info by name."""
        return self._components.get(name)
    
    def get_factory(self, name: str) -> Optional[Callable]:
        """Get factory function by name."""
        return self._factories.get(name)
    
    def get_components_by_type(self, component_type: str) -> List[ComponentInfo]:
        """Get all components of a specific type."""
        names = self._by_type.get(component_type, [])
        return [self._components[name] for name in names]
    
    def get_containers_by_role(self, role: ContainerRole) -> List[ComponentInfo]:
        """Get all containers for a specific role."""
        names = self._by_role.get(role, [])
        return [self._components[name] for name in names]
    
    def list_components(self) -> List[str]:
        """List all registered component names."""
        return list(self._components.keys())


# Global registry instance
_global_registry = ComponentRegistry()


def get_component_registry() -> ComponentRegistry:
    """Get the global component registry."""
    return _global_registry


# Decorator-based registration

def container(
    role: ContainerRole,
    name: Optional[str] = None,
    capabilities: Optional[Set[str]] = None,
    dependencies: Optional[Set[str]] = None,
    **metadata
):
    """
    Decorator to register a container class.
    
    Example:
        @container(ContainerRole.STRATEGY, capabilities={'signal.generation'})
        class MomentumContainer:
            def __init__(self, config): ...
    """
    def decorator(cls):
        component_name = name or cls.__name__
        
        def factory(config: Dict[str, Any], **kwargs):
            return cls(config, **kwargs)
        
        info = ComponentInfo(
            name=component_name,
            component_type='container',
            role=role,
            factory=factory,
            capabilities=capabilities or set(),
            dependencies=dependencies or set(),
            metadata=metadata
        )
        
        _global_registry.register_component(info)
        
        # Add registry info to class for debugging
        cls._component_info = info
        
        return cls
    
    return decorator


def strategy(
    name: Optional[str] = None,
    features: Optional[List[str]] = None,
    feature_config: Optional[Dict[str, Any]] = None,
    required_features: Optional[List['FeatureSpec']] = None,  # Static feature requirements
    feature_discovery: Optional[Callable[[Dict[str, Any]], List['FeatureSpec']]] = None,  # Dynamic discovery
    parameter_space: Optional[Dict[str, Any]] = None,  # Parameter space for optimization
    validate_features: bool = True,
    **metadata
):
    """
    Decorator to register a strategy function or class.
    
    Supports both static and dynamic feature requirements:
    
    Example (static):
        @strategy(
            name='rsi_oversold',
            required_features=[
                FeatureSpec('rsi', {'period': 14})
            ]
        )
        def rsi_oversold(features, bar, params):
            # Strategy logic
            return signal
            
    Example (dynamic):
        @strategy(
            name='adaptive_sma',
            feature_discovery=lambda params: [
                FeatureSpec('sma', {'period': params['fast_period']}),
                FeatureSpec('sma', {'period': params['slow_period']})
            ],
            parameter_space={
                'fast_period': {'type': 'int', 'range': [5, 50], 'default': 10},
                'slow_period': {'type': 'int', 'range': [20, 200], 'default': 30}
            }
        )
        def adaptive_sma(features, bar, params):
            # Strategy logic
            return signal
    """
    def decorator(func_or_class):
        component_name = name or getattr(func_or_class, '__name__', str(func_or_class))
        
        # Process feature config metadata
        features_meta = feature_config or {}
        
        # Auto-generate feature requirements from feature config
        if features_meta and not features:
            features_list = []
            if isinstance(features_meta, list):
                # New simplified format: ['sma', 'rsi', 'bollinger_bands']
                features_list = features_meta[:]
            elif isinstance(features_meta, dict):
                # Old complex format: {'sma': {'params': [...], 'defaults': {...}}}
                for feat_name, feat_config in features_meta.items():
                    features_list.append(feat_name)
        else:
            features_list = features or []
        
        # Apply feature validation if enabled
        original_func = func_or_class
        if validate_features and features_list:
            from ...strategy.validation import validate_strategy_features
            
            # Store required features on function for validator
            func_or_class.required_features = features_list
            
            # Apply validation decorator
            func_or_class = validate_strategy_features(func_or_class)
            
            # Preserve original function reference
            func_or_class.__wrapped__ = original_func
        
        factory = func_or_class if callable(func_or_class) else lambda: func_or_class
        
        # Get module name for categorization
        module_name = getattr(func_or_class, '__module__', '')
        
        info = ComponentInfo(
            name=component_name,
            component_type='strategy',
            factory=factory,
            capabilities={'signal.generation'},
            metadata={
                'features': features_list,
                'feature_config': features_meta,
                'required_features': required_features,  # Static requirements
                'feature_discovery': feature_discovery,  # Dynamic discovery function
                'parameter_space': parameter_space,      # Parameter space for optimization
                'validate_features': validate_features,
                'module': module_name,  # Add module name for wildcard discovery
                **metadata
            }
        )
        
        _global_registry.register_component(info)
        
        # Add registry info for debugging
        if hasattr(func_or_class, '__dict__'):
            func_or_class._component_info = info
            # Also add strategy-specific metadata for strategy state to use
            func_or_class._strategy_metadata = {
                'name': component_name,  # Ensure name is included
                'features': features_list,
                'feature_config': features_meta,
                'required_features': required_features,
                'feature_discovery': feature_discovery,
                'parameter_space': parameter_space,
                'validate_features': validate_features,
                **metadata
            }
        
        # Ensure required_features is accessible
        if not hasattr(func_or_class, 'required_features'):
            func_or_class.required_features = features_list
        
        return func_or_class
    
    return decorator


def classifier(
    name: Optional[str] = None,
    regime_types: Optional[List[str]] = None,
    features: Optional[List[str]] = None,
    feature_config: Optional[List[str]] = None,
    param_feature_mapping: Optional[Dict[str, str]] = None,
    parameter_space: Optional[Dict[str, Any]] = None,  # Parameter space for optimization
    validate_features: bool = False,  # Disabled by default - ComponentState handles validation
    **metadata
):
    """
    Decorator to register a classifier.
    
    Example:
        @classifier(
            regime_types=['bull', 'bear', 'sideways'],
            features=['sma', 'volume']
        )
        def trend_classifier(features, params):
            return regime
    """
    def decorator(func_or_class):
        component_name = name or getattr(func_or_class, '__name__', str(func_or_class))
        
        # Support both legacy 'features' and new 'feature_config' parameters
        features_list = feature_config or features or []
        
        # Apply feature validation if enabled
        original_func = func_or_class
        if validate_features and features_list:
            from ...strategy.validation import validate_classifier_features
            
            # Store required features on function for validator
            func_or_class.required_features = features_list
            
            # Apply validation decorator
            func_or_class = validate_classifier_features(func_or_class)
            
            # Preserve original function reference
            func_or_class.__wrapped__ = original_func
        
        factory = func_or_class if callable(func_or_class) else lambda: func_or_class
        
        # Get module name for categorization
        module_name = getattr(func_or_class, '__module__', '')
        
        info = ComponentInfo(
            name=component_name,
            component_type='classifier',
            factory=factory,
            capabilities={'regime.classification'},
            metadata={
                'regime_types': regime_types or [],
                'features': features_list,
                'feature_config': feature_config,
                'param_feature_mapping': param_feature_mapping,
                'parameter_space': parameter_space,  # Add parameter space for optimization
                'validate_features': validate_features,
                'module': module_name,  # Add module name for wildcard discovery
                **metadata
            }
        )
        
        _global_registry.register_component(info)
        
        if hasattr(func_or_class, '__dict__'):
            func_or_class._component_info = info
            # Also add classifier-specific metadata for component state to use
            func_or_class._classifier_metadata = {
                'features': features_list,
                'feature_config': feature_config,
                'param_feature_mapping': param_feature_mapping,
                'validate_features': validate_features,
                **metadata
            }
        
        # Ensure required_features is accessible
        if not hasattr(func_or_class, 'required_features'):
            func_or_class.required_features = features_list
        
        return func_or_class
    
    return decorator


def risk_validator(
    name: Optional[str] = None,
    validation_types: Optional[List[str]] = None,
    **metadata
):
    """
    Decorator to register a risk validator.
    
    Example:
        @risk_validator(validation_types=['position_size', 'drawdown'])
        def position_risk_validator(order, portfolio_state, params):
            return validation_result
    """
    def decorator(func_or_class):
        component_name = name or getattr(func_or_class, '__name__', str(func_or_class))
        
        factory = func_or_class if callable(func_or_class) else lambda: func_or_class
        
        info = ComponentInfo(
            name=component_name,
            component_type='risk_validator',
            factory=factory,
            capabilities={'risk.validation'},
            metadata={
                'validation_types': validation_types or [],
                **metadata
            }
        )
        
        _global_registry.register_component(info)
        
        if hasattr(func_or_class, '__dict__'):
            func_or_class._component_info = info
        
        return func_or_class
    
    return decorator


def feature(
    name: Optional[str] = None,
    params: Optional[List[str]] = None,
    min_history: Optional[int] = None,
    dependencies: Optional[List[str]] = None,
    **metadata
):
    """
    Decorator to register a feature/indicator function.
    
    Example:
        @feature(params=['period'], min_history=20)
        def sma(data: pd.Series, period: int) -> float:
            return data.rolling(period).mean().iloc[-1]
            
        @feature(
            params=['fast_period', 'slow_period'], 
            dependencies=['sma'],
            min_history=50
        )
        def sma_crossover(data: pd.Series, fast_period: int, slow_period: int) -> Dict[str, float]:
            fast_sma = sma(data, fast_period)
            slow_sma = sma(data, slow_period)
            return {'fast': fast_sma, 'slow': slow_sma, 'signal': fast_sma > slow_sma}
    """
    def decorator(func_or_class):
        component_name = name or getattr(func_or_class, '__name__', str(func_or_class))
        
        factory = func_or_class if callable(func_or_class) else lambda: func_or_class
        
        # Calculate min_history if not provided but params are
        if min_history is None and params:
            # Try to infer from parameter names
            period_params = [p for p in params if 'period' in p.lower()]
            if period_params:
                # Use the first period parameter as a hint
                metadata['auto_min_history'] = True
        
        info = ComponentInfo(
            name=component_name,
            component_type='feature',
            factory=factory,
            capabilities={'feature.calculation'},
            dependencies=set(dependencies or []),
            metadata={
                'params': params or [],
                'min_history': min_history,
                **metadata
            }
        )
        
        _global_registry.register_component(info)
        
        if hasattr(func_or_class, '__dict__'):
            func_or_class._component_info = info
        
        return func_or_class
    
    return decorator


def execution_model(
    name: Optional[str] = None,
    model_type: str = 'slippage',  # 'slippage', 'commission', 'liquidity'
    params: Optional[Dict[str, Any]] = None,
    **metadata
):
    """
    Decorator for execution model classes/functions.
    
    Enables dynamic discovery and configuration of execution models for backtesting
    different market conditions (slippage, commission structures, etc.).
    
    Args:
        name: Optional custom name for the model
        model_type: Type of execution model ('slippage', 'commission', 'liquidity')
        params: Default parameters for the model
        **metadata: Additional metadata (e.g., description, tags)
    
    Example:
        @execution_model(model_type='slippage', params={'base_pct': 0.001})
        class PercentageSlippageModel:
            def __init__(self, base_pct: float = 0.001):
                self.base_pct = base_pct
            
            def calculate_slippage(self, order, market_price, market_data):
                return market_price * self.base_pct
                
        @execution_model(model_type='commission')
        def zero_commission(order, fill_price, fill_quantity):
            return 0.0
    """
    def decorator(cls_or_func):
        component_name = name or getattr(cls_or_func, '__name__', str(cls_or_func))
        
        # Create factory based on whether it's a class or function
        if inspect.isclass(cls_or_func):
            # For classes, factory instantiates with params
            def factory(**kwargs):
                merged_params = {**(params or {}), **kwargs}
                return cls_or_func(**merged_params)
        else:
            # For functions, factory returns the function itself
            factory = cls_or_func
        
        # Extract parameter info
        extracted_params = params or {}
        if not extracted_params:
            # Try to extract from signature
            if inspect.isclass(cls_or_func) and hasattr(cls_or_func, '__init__'):
                sig = inspect.signature(cls_or_func.__init__)
                for param_name, param in sig.parameters.items():
                    if param_name not in ['self', 'args', 'kwargs'] and param.default != inspect.Parameter.empty:
                        extracted_params[param_name] = param.default
        
        info = ComponentInfo(
            name=component_name,
            component_type='execution_model',
            factory=factory,
            capabilities={f'execution.{model_type}'},
            dependencies=set(),
            metadata={
                'model_type': model_type,
                'params': extracted_params,
                **metadata
            }
        )
        
        _global_registry.register_component(info)
        
        # Add metadata to the object for runtime introspection
        if hasattr(cls_or_func, '__dict__'):
            cls_or_func._component_info = info
            cls_or_func._execution_model_metadata = {
                'name': component_name,
                'model_type': model_type,
                'params': extracted_params,
                **metadata
            }
        
        return cls_or_func
    
    return decorator


def workflow(
    name: Optional[str] = None,
    description: str = "",
    tags: Optional[List[str]] = None,
    composable: bool = True,
    **metadata
):
    """
    Decorator for registering workflows.
    
    The decorated function should return a workflow definition dict with:
    - phases: List of phase definitions
    - description: Optional workflow description
    - tags: Optional tags for categorization
    
    Usage:
        @workflow(
            name='adaptive_ensemble',
            description='Multi-phase adaptive ensemble workflow',
            tags=['optimization', 'ensemble', 'multi-phase']
        )
        def adaptive_ensemble_workflow():
            return {
                'phases': [
                    {'name': 'grid_search', 'topology': 'signal_generation'},
                    {'name': 'regime_analysis', 'topology': 'analysis'},
                ]
            }
    """
    def decorator(func):
        workflow_name = name or func.__name__.replace('_workflow', '')
        
        # Store metadata on the function
        func._workflow_metadata = {
            'name': workflow_name,
            'description': description,
            'tags': tags or [],
            'composable': composable,
            'module': func.__module__,
            'function': func.__name__
        }
        
        info = ComponentInfo(
            name=workflow_name,
            component_type='workflow',
            factory=func,
            capabilities={'workflow.definition'},
            metadata={
                'description': description,
                'tags': tags or [],
                'composable': composable,
                **metadata
            }
        )
        
        _global_registry.register_component(info)
        
        logger.debug(f"Registered workflow: {workflow_name}")
        
        return func
    
    return decorator


# Workflow composition helper
class WorkflowComposer:
    """
    Helper class for composing workflows from components.
    
    Example:
        composer = WorkflowComposer()
        
        # Create walk-forward by repeating backtest
        walk_forward = composer.repeat('backtest', times=12, config={
            'window_size': 30,
            'step_size': 30
        })
        
        # Create pipeline from components
        pipeline = composer.sequence([
            'data_validation',
            'feature_engineering',
            'parameter_optimization',
            'backtest',
            'report_generation'
        ])
    """
    
    @staticmethod
    def repeat(
        component: str,
        times: int,
        config: Optional[Dict[str, Any]] = None,
        vary_param: Optional[Dict[str, List[Any]]] = None
    ) -> Dict[str, Any]:
        """
        Repeat a workflow component multiple times.
        
        Args:
            component: Base workflow or topology to repeat
            times: Number of repetitions
            config: Static config for all repetitions
            vary_param: Parameters to vary across repetitions
            
        Returns:
            Workflow definition with repeated phases
        """
        phases = []
        
        for i in range(times):
            phase_config = (config or {}).copy()
            
            # Vary parameters if specified
            if vary_param:
                for param_name, param_values in vary_param.items():
                    if i < len(param_values):
                        phase_config[param_name] = param_values[i]
                    else:
                        # Cycle through values if not enough
                        phase_config[param_name] = param_values[i % len(param_values)]
            
            # Add time window for walk-forward style
            phase_config['repetition_index'] = i
            
            phases.append({
                'name': f'{component}_{i}',
                'topology': component,
                'config_override': phase_config
            })
        
        return {'phases': phases}
    
    @staticmethod
    def sequence(
        components: List[str],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a sequence of workflow components."""
        phases = []
        
        for i, component in enumerate(components):
            phases.append({
                'name': f'{component}_{i}',
                'topology': component,
                'config_override': config or {}
            })
        
        return {'phases': phases}
    
    @staticmethod
    def parallel(
        components: List[str],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create parallel workflow components."""
        phases = []
        
        for component in components:
            phases.append({
                'name': component,
                'topology': component,
                'config_override': config or {},
                'parallel': True  # Mark for parallel execution
            })
        
        return {'phases': phases}


# Auto-discovery functions

def discover_components_in_module(module_name: str) -> int:
    """
    Discover and register components in a module.
    
    Args:
        module_name: Module to scan (e.g., 'src.strategy.strategies')
        
    Returns:
        Number of components discovered
    """
    try:
        module = importlib.import_module(module_name)
        count = 0
        
        for name, obj in inspect.getmembers(module):
            # Check if object has component info (was decorated)
            if hasattr(obj, '_component_info'):
                count += 1
                logger.debug(f"Discovered component: {name} in {module_name}")
        
        return count
        
    except ImportError as e:
        logger.warning(f"Could not import module {module_name}: {e}")
        return 0


def discover_components_in_package(package_name: str) -> int:
    """
    Recursively discover components in a package.
    
    Args:
        package_name: Package to scan (e.g., 'src.strategy')
        
    Returns:
        Number of components discovered
    """
    try:
        package = importlib.import_module(package_name)
        total_count = 0
        
        # First, discover in the package itself
        total_count += discover_components_in_module(package_name)
        
        # Then recursively discover in submodules
        if hasattr(package, '__path__'):
            for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
                full_module_name = f"{package_name}.{modname}"
                
                if ispkg:
                    # Recursively scan subpackages
                    total_count += discover_components_in_package(full_module_name)
                else:
                    # Scan module
                    total_count += discover_components_in_module(full_module_name)
        
        return total_count
        
    except ImportError as e:
        logger.warning(f"Could not import package {package_name}: {e}")
        return 0


def discover_components() -> Dict[str, int]:
    """Simple wrapper for auto-discovery to match expected import."""
    return auto_discover_all_components()


def auto_discover_all_components() -> Dict[str, int]:
    """
    Auto-discover all components in the ADMF-PC system.
    
    Returns:
        Dict with component counts by package
    """
    packages_to_scan = [
        'src.strategy.strategies',
        'src.strategy.classifiers', 
        'src.risk',
        'src.execution',
        'src.data',
        'src.analytics'
    ]
    
    results = {}
    total_discovered = 0
    
    for package in packages_to_scan:
        count = discover_components_in_package(package)
        results[package] = count
        total_discovered += count
        
        if count > 0:
            logger.info(f"Discovered {count} components in {package}")
    
    logger.info(f"Total components discovered: {total_discovered}")
    return results


# Configuration-driven discovery

def load_components_from_config(config_path: str) -> int:
    """
    Load component registrations from a YAML/JSON config file.
    
    Example config:
    ```yaml
    components:
      containers:
        - name: "advanced_momentum_container"
          role: "strategy"
          module: "src.custom.containers.momentum"
          class: "AdvancedMomentumContainer"
          capabilities: ["signal.generation", "risk.aware"]
      
      strategies:
        - name: "ml_momentum"
          module: "src.custom.strategies.ml"
          function: "ml_momentum_strategy"
          features: ["sma", "rsi", "volume"]
    ```
    """
    import yaml
    from pathlib import Path
    
    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning(f"Config file not found: {config_path}")
        return 0
    
    with open(config_file, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        else:
            import json
            config = json.load(f)
    
    components_config = config.get('components', {})
    total_loaded = 0
    
    # Load containers
    for container_config in components_config.get('containers', []):
        try:
            module = importlib.import_module(container_config['module'])
            cls = getattr(module, container_config['class'])
            
            role = ContainerRole(container_config['role'])
            capabilities = set(container_config.get('capabilities', []))
            
            def factory(config: Dict[str, Any], **kwargs):
                return cls(config, **kwargs)
            
            info = ComponentInfo(
                name=container_config['name'],
                component_type='container',
                role=role,
                factory=factory,
                capabilities=capabilities
            )
            
            _global_registry.register_component(info)
            total_loaded += 1
            
        except Exception as e:
            logger.error(f"Failed to load container {container_config.get('name')}: {e}")
    
    # Load strategies
    for strategy_config in components_config.get('strategies', []):
        try:
            module = importlib.import_module(strategy_config['module'])
            func = getattr(module, strategy_config['function'])
            
            info = ComponentInfo(
                name=strategy_config['name'],
                component_type='strategy',
                factory=func,
                capabilities={'signal.generation'},
                metadata={
                    'features': strategy_config.get('features', [])
                }
            )
            
            _global_registry.register_component(info)
            total_loaded += 1
            
        except Exception as e:
            logger.error(f"Failed to load strategy {strategy_config.get('name')}: {e}")
    
    logger.info(f"Loaded {total_loaded} components from config: {config_path}")
    return total_loaded