"""
Refactored TopologyBuilder Implementation

Handles tracing configuration without importing event system components.
The topology builder creates topologies, and if tracing is requested,
it configures containers to create their own tracers.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib
import json

logger = logging.getLogger(__name__)


class TopologyBuilder:
    """
    Builds topologies from topology definitions.
    
    Key changes:
    - Accepts tracing_config instead of event_tracer
    - Passes configuration to containers, not event objects
    - Containers create their own tracers if needed
    """
    
    def __init__(self):
        """Initialize topology builder."""
        logger.info("TopologyBuilder initialized")
    
    def build_topology(self, topology_definition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a topology from a topology definition.
        
        Args:
            topology_definition: Complete topology definition including:
                - mode: The topology mode (backtest, signal_generation, etc.)
                - config: Configuration for the topology
                - tracing_config: Optional tracing configuration (not tracer object!)
                - metadata: Optional metadata about the execution
            
        Returns:
            Dict containing the topology structure:
            - containers: Dict of container instances
            - adapters: List of configured adapters
            - metadata: Topology metadata
        """
        mode = topology_definition.get('mode')
        if not mode:
            raise ValueError("Topology definition must include 'mode'")
            
        config = topology_definition.get('config', {})
        tracing_config = topology_definition.get('tracing_config', {})
        metadata = topology_definition.get('metadata', {})
        
        logger.info(f"Building {mode} topology")
        
        # Add tracing configuration to config if enabled
        if tracing_config.get('enabled', False):
            # Ensure execution config exists
            if 'execution' not in config:
                config['execution'] = {}
            
            # Merge trace settings into execution config for containers
            config['execution']['enable_event_tracing'] = True
            if 'trace_settings' not in config['execution']:
                config['execution']['trace_settings'] = {}
            
            trace_settings = config['execution']['trace_settings']
            trace_settings['trace_id'] = tracing_config.get('trace_id')
            trace_settings['trace_dir'] = tracing_config.get('trace_dir', './traces')
            trace_settings['max_events'] = tracing_config.get('max_events', 10000)
            
            # Pass through container-specific settings
            if 'container_settings' in tracing_config:
                trace_settings['container_settings'] = tracing_config['container_settings']
            
            logger.info(f"Tracing enabled with trace_id: {tracing_config.get('trace_id')}")
        
        # Add metadata to config
        config['execution_metadata'] = metadata
        
        # Import the appropriate topology creation function
        topology_module = self._get_topology_module(mode)
        
        # Create the topology (containers will handle their own tracing)
        topology = topology_module(config)
        
        # Add metadata to topology
        topology['metadata'] = {
            'mode': mode,
            'created_at': str(datetime.now()),
            'config_hash': self._hash_config(config),
            'tracing_enabled': tracing_config.get('enabled', False),
            **metadata
        }
        
        logger.info(f"Built {mode} topology with {len(topology.get('containers', {}))} containers")
        
        return topology
    
    def _get_topology_module(self, mode: str):
        """Get the topology creation function for the given mode."""
        try:
            if mode == 'backtest':
                from .topologies import create_backtest_topology
                return create_backtest_topology
            elif mode == 'signal_generation':
                from .topologies import create_signal_generation_topology
                return create_signal_generation_topology
            elif mode == 'signal_replay':
                from .topologies import create_signal_replay_topology
                return create_signal_replay_topology
            elif mode == 'optimization':
                # Optimization uses backtest topology with special config
                from .topologies import create_backtest_topology
                return create_backtest_topology
            else:
                raise ValueError(f"Unknown topology mode: {mode}")
        except ImportError as e:
            raise ImportError(f"Failed to import topology module for mode '{mode}': {e}")
    
    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Generate a hash of the configuration for tracking."""
        # Remove non-serializable items before hashing
        config_copy = config.copy()
        config_copy.pop('tracing', None)  # Remove tracing config
        config_copy.pop('execution_metadata', None)  # Remove metadata
        
        # Sort keys for consistent hashing
        config_str = json.dumps(config_copy, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def get_supported_modes(self) -> List[str]:
        """Get list of supported topology modes."""
        return ['backtest', 'signal_generation', 'signal_replay', 'optimization', 'analysis']


# Example of how containers would handle tracing
class ContainerTracingMixin:
    """
    Mixin showing how containers handle their own tracing.
    
    This would be used by actual container implementations.
    """
    
    def _setup_tracing(self, config: Dict[str, Any]):
        """
        Setup tracing if configured.
        
        Containers create their own tracers based on configuration.
        """
        tracing_config = config.get('tracing', {})
        if tracing_config.get('enabled', False):
            # Container creates its own tracer
            # This way, orchestration never touches event system
            from ...events.tracing import EventTracer
            
            trace_id = tracing_config.get('trace_id', str(self.container_id))
            self.event_tracer = EventTracer(
                correlation_id=trace_id,
                max_events=tracing_config.get('max_events', 10000)
            )
            
            # Subscribe tracer to container's event bus
            if hasattr(self, 'event_bus'):
                self.event_bus.subscribe_all(self.event_tracer.trace_event)
            
            logger.debug(f"Container {self.container_id} tracing enabled")
    
    def _get_trace_summary(self) -> Optional[Dict[str, Any]]:
        """Get trace summary if tracing is enabled."""
        if hasattr(self, 'event_tracer'):
            return self.event_tracer.get_summary()
        return None


# Component builder helper functions (moved from helpers/component_builder.py)

def create_stateless_components(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create all stateless components from configuration."""
    components = {
        'strategies': {},
        'classifiers': {},
        'risk_validators': {},
        'execution_models': {}
    }
    
    # Create strategy instances from config
    for strat_config in config.get('strategies', []):
        if 'type' not in strat_config:
            logger.error(f"Strategy config missing 'type': {strat_config}")
            continue
        strat_type = strat_config['type']
        strategy = _create_strategy(strat_type, strat_config)
        if strategy:
            components['strategies'][strat_type] = strategy
    
    # Create classifier instances from config
    for class_config in config.get('classifiers', []):
        if 'type' not in class_config:
            logger.error(f"Classifier config missing 'type': {class_config}")
            continue
        class_type = class_config['type']
        classifier = _create_classifier(class_type, class_config)
        if classifier:
            components['classifiers'][class_type] = classifier
    
    # Create risk validator instances from config
    for risk_config in config.get('risk_profiles', []):
        if 'type' not in risk_config:
            logger.error(f"Risk profile config missing 'type': {risk_config}")
            continue
        risk_type = risk_config['type']
        validator = _create_risk_validator(risk_type, risk_config)
        if validator:
            components['risk_validators'][risk_type] = validator
    
    # Create execution model instances from config
    for exec_config in config.get('execution_models', []):
        if 'type' not in exec_config:
            logger.error(f"Execution model config missing 'type': {exec_config}")
            continue
        exec_type = exec_config['type']
        exec_models = _create_execution_models(exec_type, exec_config)
        if exec_models:
            components['execution_models'][exec_type] = exec_models
    
    return components


def _create_strategy(strategy_type: str, config: Dict[str, Any]) -> Any:
    """Create a stateless strategy using discovery system."""
    from ..components.discovery import get_component_registry
    registry = get_component_registry()
    
    # Get strategy from registry
    strategy_info = registry.get_component(strategy_type)
    
    if strategy_info:
        logger.info(f"Found strategy '{strategy_type}' in registry")
        return strategy_info.factory
    
    # If not in registry, log error
    logger.error(f"Strategy type '{strategy_type}' not found in registry. Available strategies: {list(registry.list_by_type('strategy'))}")
    return None


def _create_classifier(classifier_type: str, config: Dict[str, Any]) -> Any:
    """Create a stateless classifier using discovery system."""
    from ..components.discovery import get_component_registry
    registry = get_component_registry()
    
    # Try exact match first
    classifier_info = registry.get_component(classifier_type)
    
    # If not found, try with _classifier suffix
    if not classifier_info:
        classifier_info = registry.get_component(f"{classifier_type}_classifier")
    
    if classifier_info:
        logger.info(f"Found classifier '{classifier_type}' in registry")
        return classifier_info.factory
    
    # If not in registry, log error
    logger.error(f"Classifier type '{classifier_type}' not found in registry. Available classifiers: {list(registry.list_by_type('classifier'))}")
    return None


def _create_risk_validator(risk_type: str, config: Dict[str, Any]) -> Any:
    """Create a stateless risk validator."""
    from ..components.discovery import get_component_registry
    # First check registry
    registry = get_component_registry()
    
    # Try exact match
    validator_info = registry.get_component(risk_type)
    
    # Try with _validator suffix
    if not validator_info:
        validator_info = registry.get_component(f"{risk_type}_validator")
    
    # Try with validate_ prefix
    if not validator_info:
        validator_info = registry.get_component(f"validate_{risk_type}")
    
    if validator_info:
        logger.info(f"Found risk validator '{risk_type}' in registry")
        return validator_info.factory
    
    # If not in registry, try direct import as fallback
    try:
        from ...risk import validators
        validator_func = getattr(validators, f"validate_{risk_type}", None)
        if validator_func:
            logger.warning(f"Risk validator '{risk_type}' not in registry but found via import")
            return validator_func
    except (ImportError, AttributeError):
        pass
    
    logger.error(f"Risk validator type '{risk_type}' not found")
    return None


def _create_execution_models(exec_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Create stateless execution models from config."""
    from ..components.discovery import get_component_registry
    models = {}
    registry = get_component_registry()
    
    # Slippage model
    slippage_config = config.get('slippage', {})
    if 'type' in slippage_config:
        slippage_type = slippage_config['type']
        slippage_info = registry.get_component(f'{slippage_type}_slippage')
        
        if slippage_info:
            params = slippage_config.get('params', {})
            models['slippage'] = slippage_info.factory(**params)
            logger.info(f"Created slippage model '{slippage_type}' with params: {params}")
        else:
            logger.error(f"Slippage model '{slippage_type}' not found in registry")
    
    # Commission model  
    commission_config = config.get('commission', {})
    if 'type' in commission_config:
        commission_type = commission_config['type']
        commission_info = registry.get_component(f'{commission_type}_commission')
        
        if commission_info:
            params = commission_config.get('params', {})
            models['commission'] = commission_info.factory(**params)
            logger.info(f"Created commission model '{commission_type}' with params: {params}")
        else:
            logger.error(f"Commission model '{commission_type}' not found in registry")
    
    # Other execution models from config
    for model_key in ['liquidity', 'market_impact', 'latency']:
        if model_key in config:
            model_config = config[model_key]
            if 'type' in model_config:
                model_type = model_config['type']
                model_info = registry.get_component(f'{model_type}_{model_key}')
                
                if model_info:
                    params = model_config.get('params', {})
                    models[model_key] = model_info.factory(**params)
                    logger.info(f"Created {model_key} model '{model_type}' with params: {params}")
    
    return models


def _get_strategy_feature_requirements(strategy_type: str, config: Dict[str, Any]) -> List[str]:
    """Get required features for a strategy from registry metadata."""
    from ..components.discovery import get_component_registry
    registry = get_component_registry()
    strategy_info = registry.get_component(strategy_type)
    
    if strategy_info:
        # Get features from metadata
        base_features = strategy_info.metadata.get('features', [])
        feature_config = strategy_info.metadata.get('feature_config', {})
        if feature_config:
            base_features.extend(feature_config.keys())
        
        # Add any custom features from config
        custom_features = config.get('required_features', [])
        all_features = list(set(base_features + custom_features))
        
        logger.info(f"Strategy '{strategy_type}' requires features: {all_features}")
        return all_features
    else:
        logger.error(f"Strategy '{strategy_type}' not found in registry, cannot determine feature requirements")
        # Return any explicitly configured features
        return config.get('required_features', [])