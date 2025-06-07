"""
Communication routing factory using protocol-based design.

This module provides factory functions for creating communication routes
following ADMF-PC's protocol-based architecture.
"""

from typing import Dict, Any, Type, List, Optional, Callable
import logging

from ..containers.protocols import Container
from ..events.type_flow_integration import TypeFlowValidator, validate_route_network
from .protocols import CommunicationRoute
from .composition import compose_route_with_infrastructure

# Import protocol-based routes
from .pipe import PipelineRoute
from .broadcast import BroadcastRoute
from .filter import FilterRoute, create_feature_filter


class RoutingFactory:
    """Factory for creating communication routes.
    
    This factory creates protocol-based routes without inheritance.
    All routes follow the CommunicationRoute protocol but don't
    inherit from any base class.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, 
                 enable_type_validation: bool = True):
        """Initialize the routing factory.
        
        Args:
            logger: Logger for factory operations
            enable_type_validation: Enable type flow validation
        """
        self.logger = logger or logging.getLogger(__name__)
        self.enable_type_validation = enable_type_validation
        self.type_validator = TypeFlowValidator() if enable_type_validation else None
        
        # Registry of route types and their factory functions
        self.route_registry: Dict[str, Callable] = {
            # Core patterns
            'pipeline': lambda n, c: compose_route_with_infrastructure(PipelineRoute, n, c),
            'broadcast': lambda n, c: compose_route_with_infrastructure(BroadcastRoute, n, c),
            'filter': lambda n, c: compose_route_with_infrastructure(FilterRoute, n, c),
            
            # Convenience aliases
            'pipe': lambda n, c: compose_route_with_infrastructure(PipelineRoute, n, c),
            'fanout': lambda n, c: compose_route_with_infrastructure(BroadcastRoute, n, c),
            'feature_filter': create_feature_filter,
        }
        
        # Active route instances
        self.active_routes: List[Any] = []
        
    def create_route(self, name: str, config: Dict[str, Any]) -> Any:
        """Create a communication route instance.
        
        Args:
            name: Unique route name
            config: Route configuration including 'type'
            
        Returns:
            Configured route instance
            
        Raises:
            ValueError: If route type is unknown or invalid configuration
        """
        route_type = config.get('type')
        if not route_type:
            raise ValueError("Route configuration must specify 'type'")
            
        if route_type not in self.route_registry:
            raise ValueError(f"Unknown route type: {route_type}")
        
        # Add type validation settings to config if enabled
        if self.enable_type_validation:
            config = config.copy()  # Don't modify original
            config.setdefault('enable_type_validation', True)
            
        self.logger.info(f"Creating {route_type} route: {name}")
        
        # Create route using registered factory
        factory_fn = self.route_registry[route_type]
        route = factory_fn(name, config)
        
        # Track active route
        self.active_routes.append(route)
        
        return route
        
    def register_route_type(self, route_type: str, factory_fn: Callable):
        """Register a custom route type.
        
        Args:
            route_type: Type identifier for the route
            factory_fn: Factory function that creates route instances
        """
        self.route_registry[route_type] = factory_fn
        self.logger.info(f"Registered route type: {route_type}")
        
    def create_routes_from_config(self, 
                                   routes_config: List[Dict[str, Any]],
                                   containers: Dict[str, Container]) -> List[Any]:
        """Create multiple routes from configuration with validation.
        
        Args:
            routes_config: List of route configurations
            containers: Available containers
            
        Returns:
            List of configured route instances
            
        Raises:
            ValueError: If configuration validation fails
        """
        # Validate route network configuration if type validation is enabled
        if self.enable_type_validation and self.type_validator:
            validation_result = validate_route_network(
                routes_config, containers, strict_mode=False
            )
            
            if not validation_result.valid:
                error_msg = f"Route configuration validation failed: {'; '.join(validation_result.errors)}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Log warnings
            for warning in validation_result.warnings:
                self.logger.warning(f"Route configuration warning: {warning}")
        
        routes = []
        
        for route_config in routes_config:
            name = route_config.get('name', f"route_{len(routes)}")
            
            try:
                # Create route
                route = self.create_route(name, route_config)
                
                # Setup with containers
                if hasattr(route, 'setup'):
                    route.setup(containers)
                    
                routes.append(route)
                
            except Exception as e:
                self.logger.error(f"Failed to create route {name}: {e}")
                raise
                
        return routes
        
    def start_all(self) -> None:
        """Start all active routes."""
        for route in self.active_routes:
            if hasattr(route, 'start'):
                # Skip pipeline routes that have no connections configured
                # They will be started later by the workflow manager after configuration
                if (hasattr(route, 'connections') and 
                    hasattr(route, 'config') and 
                    route.config.get('type') == 'pipeline' and 
                    not route.connections):
                    self.logger.info(f"Skipping start of pipeline route '{route.name}' - will be started after configuration")
                    continue
                    
                route.start()
                self.logger.info(f"Started route: {route.name}")
                
    def stop_all(self) -> None:
        """Stop all active routes."""
        for route in self.active_routes:
            if hasattr(route, 'stop'):
                route.stop()
                self.logger.info(f"Stopped route: {route.name}")
                
        self.active_routes.clear()
    
    def validate_configuration(self, 
                              routes_config: List[Dict[str, Any]],
                              containers: Dict[str, Container],
                              strict_mode: bool = False) -> bool:
        """Validate route configuration without creating routes.
        
        Args:
            routes_config: List of route configurations
            containers: Available containers
            strict_mode: If True, treat warnings as errors
            
        Returns:
            True if configuration is valid
        """
        if not self.enable_type_validation or not self.type_validator:
            self.logger.info("Type validation disabled - skipping configuration validation")
            return True
            
        try:
            validation_result = validate_route_network(
                routes_config, containers, strict_mode=strict_mode
            )
            
            if validation_result.valid:
                self.logger.info("Route configuration validation passed")
                
                # Log warnings even if validation passed
                for warning in validation_result.warnings:
                    self.logger.warning(f"Configuration warning: {warning}")
            else:
                self.logger.error("Route configuration validation failed:")
                for error in validation_result.errors:
                    self.logger.error(f"  • {error}")
                    
            return validation_result.valid
            
        except Exception as e:
            self.logger.error(f"Error during configuration validation: {e}")
            return False
    
    def get_configuration_report(self, 
                               routes_config: List[Dict[str, Any]],
                               containers: Dict[str, Container]) -> str:
        """Generate detailed configuration analysis report.
        
        Args:
            routes_config: List of route configurations
            containers: Available containers
            
        Returns:
            Formatted analysis report
        """
        if not self.enable_type_validation or not self.type_validator:
            return "Type validation disabled - no report available"
            
        try:
            from ..events.type_flow_integration import create_type_flow_report
            return create_type_flow_report(containers, routes_config)
        except Exception as e:
            return f"Error generating configuration report: {e}"


def create_route_network(config: Dict[str, Any],
                         containers: Dict[str, Container],
                         logger: Optional[logging.Logger] = None) -> RoutingFactory:
    """Create a complete route network from configuration.
    
    This is a convenience function that creates all routes and
    wires them up according to the configuration.
    
    Args:
        config: Network configuration with 'routes' list
        containers: Available containers
        logger: Optional logger
        
    Returns:
        Configured RoutingFactory with all routes created
    """
    factory = RoutingFactory(logger)
    
    # Create routes
    routes_config = config.get('routes', [])
    routes = factory.create_routes_from_config(routes_config, containers)
    
    # Start all routes
    factory.start_all()
    
    return factory


# Convenience functions for common patterns

def create_simple_pipeline(containers: List[Container], 
                          name: str = "main_pipeline") -> Any:
    """Create a simple pipeline route.
    
    Args:
        containers: List of containers in pipeline order
        name: Pipeline name
        
    Returns:
        Configured pipeline route
    """
    config = {
        'type': 'pipeline',
        'containers': [c.name for c in containers]
    }
    
    route = compose_route_with_infrastructure(PipelineRoute, name, config)
    
    # Setup with container mapping
    container_map = {c.name: c for c in containers}
    route.setup(container_map)
    
    return route


def create_event_bus(source: Container,
                    targets: List[Container],
                    name: str = "event_bus") -> Any:
    """Create a broadcast route acting as an event bus.
    
    Args:
        source: Source container
        targets: Target containers
        name: Bus name
        
    Returns:
        Configured broadcast route
    """
    config = {
        'type': 'broadcast',
        'source': source.name,
        'targets': [t.name for t in targets]
    }
    
    route = compose_route_with_infrastructure(BroadcastRoute, name, config)
    
    # Setup with container mapping
    all_containers = [source] + targets
    container_map = {c.name: c for c in all_containers}
    route.setup(container_map)
    
    return route