"""
Bootstrap system for container initialization.

This module provides bootstrap functionality to initialize containers
with all required components and services in the correct order.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Type
import logging
from pathlib import Path
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    
from .universal import UniversalScopedContainer
from .factory import ContainerFactory
from ..components import get_registry, auto_discover_components
from ..dependencies import DependencyValidator


logger = logging.getLogger(__name__)


class ContainerBootstrap:
    """
    Bootstrap system for container initialization.
    
    This class handles the initialization sequence for containers,
    ensuring all components are properly configured and dependencies
    are satisfied.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        component_paths: Optional[List[str]] = None
    ):
        """
        Initialize the bootstrap system.
        
        Args:
            config_path: Path to bootstrap configuration
            component_paths: Paths to scan for components
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.component_paths = component_paths or []
        
        # Core services
        self.factory = ContainerFactory()
        self.registry = get_registry()
        self.validator = DependencyValidator()
        
        # Shared services that will be available to all containers
        self.shared_services: Dict[str, Any] = {}
        
        logger.info("ContainerBootstrap initialized")
    
    def initialize(self) -> None:
        """
        Initialize the bootstrap system.
        
        This performs one-time initialization tasks:
        - Component discovery
        - Shared service creation
        - Validation
        """
        # Discover components
        if self.component_paths:
            self._discover_components()
        
        # Initialize shared services
        if 'shared_services' in self.config:
            self._initialize_shared_services()
        
        # Validate configuration
        self._validate_configuration()
        
        logger.info("Bootstrap initialization complete")
    
    def create_container(
        self,
        container_spec: Dict[str, Any],
        container_type: Optional[str] = None,
        container_id: Optional[str] = None
    ) -> str:
        """
        Create a container from specification.
        
        Args:
            container_spec: Container specification
            container_type: Optional container type override
            container_id: Optional container ID
            
        Returns:
            Container ID
        """
        container_type = container_type or container_spec.get('type', 'generic')
        
        # Get container-specific configuration
        container_config = container_spec.get('config', {})
        
        # Merge with shared services
        services = self.shared_services.copy()
        if 'services' in container_spec:
            services.update(container_spec['services'])
        
        # Create appropriate container type
        if container_type == 'backtest':
            return self._create_backtest_container(
                container_spec,
                services,
                container_id
            )
        elif container_type == 'optimization':
            return self._create_optimization_container(
                container_spec,
                services,
                container_id
            )
        elif container_type == 'live_trading':
            return self._create_live_trading_container(
                container_spec,
                services,
                container_id
            )
        else:
            # Generic container
            return self._create_generic_container(
                container_spec,
                services,
                container_id,
                container_type
            )
    
    def create_batch(
        self,
        batch_spec: Dict[str, Any]
    ) -> List[str]:
        """
        Create multiple containers from batch specification.
        
        Args:
            batch_spec: Batch specification
            
        Returns:
            List of container IDs
        """
        container_ids = []
        
        # Extract common configuration
        base_spec = batch_spec.get('base', {})
        variations = batch_spec.get('variations', [])
        
        for i, variation in enumerate(variations):
            # Merge base and variation
            container_spec = self._merge_specs(base_spec, variation)
            
            # Generate container ID if not provided
            container_id = variation.get('id', f"batch_{i}")
            
            # Create container
            container_id = self.create_container(
                container_spec,
                container_id=container_id
            )
            container_ids.append(container_id)
        
        logger.info(f"Created batch of {len(container_ids)} containers")
        return container_ids
    
    def add_shared_service(self, name: str, service: Any) -> None:
        """Add a shared service."""
        self.shared_services[name] = service
        logger.debug(f"Added shared service: {name}")
    
    # Private methods
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load bootstrap configuration."""
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path, 'r') as f:
            if path.suffix in ('.yaml', '.yml'):
                if not HAS_YAML:
                    raise ImportError("PyYAML is required for YAML config files")
                return yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
    
    def _discover_components(self) -> None:
        """Discover components in specified paths."""
        for path in self.component_paths:
            try:
                auto_discover_components(path)
                logger.info(f"Discovered components in: {path}")
            except Exception as e:
                logger.error(f"Component discovery failed for {path}: {e}")
    
    def _initialize_shared_services(self) -> None:
        """Initialize shared services from configuration."""
        services_config = self.config['shared_services']
        
        for service_name, service_spec in services_config.items():
            try:
                service = self._create_service(service_spec)
                self.shared_services[service_name] = service
                logger.info(f"Initialized shared service: {service_name}")
            except Exception as e:
                logger.error(f"Failed to create service {service_name}: {e}")
                raise
    
    def _create_service(self, service_spec: Dict[str, Any]) -> Any:
        """Create a service from specification."""
        service_type = service_spec['type']
        params = service_spec.get('params', {})
        
        # Get service class from registry
        service_class = self.registry.get_class(service_type)
        if not service_class:
            raise ValueError(f"Service type not found: {service_type}")
        
        # Create service instance
        return service_class(**params)
    
    def _validate_configuration(self) -> None:
        """Validate bootstrap configuration."""
        # Validate component registry
        components = self.registry.list_all()
        logger.info(f"Registry contains {len(components)} components")
        
        # Validate shared services
        for name, service in self.shared_services.items():
            logger.debug(f"Shared service '{name}': {type(service).__name__}")
    
    def _create_backtest_container(
        self,
        spec: Dict[str, Any],
        services: Dict[str, Any],
        container_id: Optional[str]
    ) -> str:
        """Create a backtest container."""
        strategy_spec = spec['strategy']
        additional = spec.get('components', [])
        
        return self.factory.create_backtest_container(
            strategy_spec=strategy_spec,
            shared_services=services,
            container_id=container_id,
            additional_components=additional
        )
    
    def _create_optimization_container(
        self,
        spec: Dict[str, Any],
        services: Dict[str, Any],
        container_id: Optional[str]
    ) -> str:
        """Create an optimization container."""
        strategy_spec = spec['strategy']
        trial_id = spec.get('trial_id', container_id or 'default')
        
        return self.factory.create_optimization_container(
            strategy_spec=strategy_spec,
            trial_id=trial_id,
            shared_services=services
        )
    
    def _create_live_trading_container(
        self,
        spec: Dict[str, Any],
        services: Dict[str, Any],
        container_id: Optional[str]
    ) -> str:
        """Create a live trading container."""
        strategy_spec = spec['strategy']
        broker_config = spec['broker']
        
        return self.factory.create_live_trading_container(
            strategy_spec=strategy_spec,
            broker_config=broker_config,
            shared_services=services,
            container_id=container_id
        )
    
    def _create_generic_container(
        self,
        spec: Dict[str, Any],
        services: Dict[str, Any],
        container_id: Optional[str],
        container_type: str
    ) -> str:
        """Create a generic container."""
        components = spec.get('components', [])
        
        return self.factory.create_custom_container(
            container_type=container_type,
            components=components,
            shared_services=services,
            container_id=container_id
        )
    
    def _merge_specs(
        self,
        base: Dict[str, Any],
        override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge two specifications, with override taking precedence."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursive merge for nested dicts
                result[key] = self._merge_specs(result[key], value)
            else:
                result[key] = value
        
        return result


# Example bootstrap configuration
EXAMPLE_BOOTSTRAP_CONFIG = """
# Bootstrap configuration example
shared_services:
  DataProvider:
    type: HistoricalDataProvider
    params:
      data_path: "data/historical"
      
  IndicatorHub:
    type: SharedIndicatorHub
    params:
      indicators:
        - type: SMA
          params: {periods: [10, 20, 50]}
        - type: RSI
          params: {period: 14}

component_paths:
  - "src/strategies"
  - "src/indicators"
  - "src/risk"

validation:
  strict_mode: true
  check_dependencies: true
"""