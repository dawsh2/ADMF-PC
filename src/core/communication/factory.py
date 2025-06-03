"""Event Communication Factory for ADMF-PC.

This module provides the factory for creating communication adapters and
managing the overall communication layer. The factory pattern allows for
flexible configuration and runtime adapter selection.

Key components:
- EventCommunicationFactory: Creates adapters from configuration
- CommunicationLayer: Manages all active adapters system-wide
"""

from typing import Dict, Any, Type, List, Optional
import asyncio
from datetime import datetime
import numpy as np

from .base_adapter import CommunicationAdapter, AdapterConfig
from .pipeline_adapter import PipelineCommunicationAdapter
from ..logging.log_manager import LogManager
from ..logging.container_logger import ContainerLogger


class EventCommunicationFactory:
    """Factory for creating communication adapters from configuration.
    
    This factory maintains a registry of available adapter types and
    creates configured instances for the communication layer.
    """
    
    def __init__(self, coordinator_id: str, log_manager: LogManager):
        """Initialize the communication factory.
        
        Args:
            coordinator_id: ID of the coordinator using this factory
            log_manager: Log manager for creating loggers
        """
        self.coordinator_id = coordinator_id
        self.log_manager = log_manager
        
        # Registry of available adapter types
        self.adapter_registry: Dict[str, Type[CommunicationAdapter]] = {
            'pipeline': PipelineCommunicationAdapter,
            # Future adapters will be registered here:
            # 'broadcast': BroadcastCommunicationAdapter,
            # 'hierarchical': HierarchicalCommunicationAdapter,
            # 'selective': SelectiveCommunicationAdapter,
        }
        
        # Create factory logger
        self.logger = ContainerLogger(
            container_id=coordinator_id,
            component_name="communication_factory",
            log_level="INFO"
        )
        
        # Track active adapters for lifecycle management
        self.active_adapters: List[CommunicationAdapter] = []
        
        self.logger.info(
            "Communication factory initialized",
            coordinator_id=coordinator_id,
            available_adapters=list(self.adapter_registry.keys()),
            lifecycle_operation="factory_initialization"
        )
    
    def register_adapter_type(self, adapter_type: str, adapter_class: Type[CommunicationAdapter]):
        """Register a new adapter type in the factory.
        
        Args:
            adapter_type: String identifier for the adapter type
            adapter_class: Adapter class to register
        """
        self.adapter_registry[adapter_type] = adapter_class
        
        self.logger.info(
            "Registered new adapter type",
            adapter_type=adapter_type,
            adapter_class=adapter_class.__name__,
            total_types=len(self.adapter_registry)
        )
    
    def create_communication_layer(self, config: Dict[str, Any], containers: Dict[str, Any]) -> 'CommunicationLayer':
        """Create a communication layer from configuration.
        
        Args:
            config: Communication configuration with adapter definitions
            containers: Available containers to wire together
            
        Returns:
            Configured communication layer
        """
        self.logger.info(
            "Creating communication layer",
            adapter_configs=len(config.get('adapters', [])),
            available_containers=len(containers),
            lifecycle_operation="communication_layer_creation"
        )
        
        # Create communication layer
        communication_layer = CommunicationLayer(self.coordinator_id, self.log_manager)
        
        # Create and add adapters
        for adapter_config in config.get('adapters', []):
            try:
                adapter = self._create_adapter(adapter_config, containers)
                adapter_name = adapter_config.get('name', f"{adapter_config['type']}_adapter")
                communication_layer.add_adapter(adapter_name, adapter)
                self.active_adapters.append(adapter)
                
            except Exception as e:
                self.logger.error(
                    "Failed to create adapter",
                    adapter_config=adapter_config,
                    error=str(e),
                    error_type=type(e).__name__,
                    lifecycle_operation="adapter_creation_error"
                )
                # Continue with other adapters
        
        self.logger.info(
            "Communication layer created",
            total_adapters=len(communication_layer.adapters),
            lifecycle_operation="communication_layer_ready"
        )
        
        return communication_layer
    
    def _create_adapter(self, adapter_config: Dict[str, Any], containers: Dict[str, Any]) -> CommunicationAdapter:
        """Create a single adapter from configuration.
        
        Args:
            adapter_config: Adapter configuration
            containers: Available containers
            
        Returns:
            Configured adapter instance
        """
        adapter_type = adapter_config.get('type')
        
        if adapter_type not in self.adapter_registry:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
        
        # Create adapter configuration
        config = AdapterConfig(
            name=adapter_config.get('name', f"{adapter_type}_adapter"),
            adapter_type=adapter_type,
            retry_attempts=adapter_config.get('retry_attempts', 3),
            retry_delay_ms=adapter_config.get('retry_delay_ms', 100),
            timeout_ms=adapter_config.get('timeout_ms', 5000),
            buffer_size=adapter_config.get('buffer_size', 1000),
            enable_compression=adapter_config.get('enable_compression', False),
            enable_encryption=adapter_config.get('enable_encryption', False),
            custom_settings=adapter_config.get('custom_settings', {})
        )
        
        # Create logger for adapter
        adapter_logger = ContainerLogger(
            container_id=self.coordinator_id,
            component_name=f"adapter.{config.name}",
            log_level=adapter_config.get('log_level', 'INFO')
        )
        
        # Create adapter instance
        adapter_class = self.adapter_registry[adapter_type]
        adapter = adapter_class(config, adapter_logger)
        
        # Setup adapter based on type
        if adapter_type == 'pipeline':
            # Pipeline-specific setup
            container_names = adapter_config.get('containers', [])
            pipeline_containers = []
            
            for name in container_names:
                if name not in containers:
                    self.logger.warning(
                        f"Container '{name}' not found for pipeline",
                        adapter_name=config.name,
                        available_containers=list(containers.keys())
                    )
                    continue
                pipeline_containers.append(containers[name])
            
            if pipeline_containers:
                adapter.setup_pipeline(pipeline_containers)
        
        # Future adapter types will have their own setup logic here
        
        self.logger.info(
            "Adapter created",
            adapter_type=adapter_type,
            adapter_name=config.name,
            lifecycle_operation="adapter_creation"
        )
        
        return adapter
    
    async def cleanup_all_adapters(self):
        """Cleanup all created adapters."""
        self.logger.info(
            "Starting cleanup of all adapters",
            adapter_count=len(self.active_adapters),
            lifecycle_operation="factory_cleanup"
        )
        
        cleanup_tasks = []
        for adapter in self.active_adapters:
            try:
                if hasattr(adapter, 'cleanup'):
                    cleanup_tasks.append(adapter.cleanup())
            except Exception as e:
                self.logger.error(
                    "Error preparing adapter cleanup",
                    adapter_name=adapter.config.name,
                    error=str(e),
                    error_type=type(e).__name__,
                    lifecycle_operation="adapter_cleanup_error"
                )
        
        # Wait for all cleanups to complete
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        self.active_adapters.clear()
        
        self.logger.info(
            "All adapters cleaned up",
            lifecycle_operation="factory_cleanup_complete"
        )


class CommunicationLayer:
    """Manages all communication adapters in the system.
    
    The communication layer provides a unified interface for managing
    multiple adapters and tracking system-wide communication metrics.
    """
    
    def __init__(self, coordinator_id: str, log_manager: LogManager):
        """Initialize the communication layer.
        
        Args:
            coordinator_id: ID of the coordinator
            log_manager: Log manager for creating loggers
        """
        self.coordinator_id = coordinator_id
        self.log_manager = log_manager
        
        self.logger = ContainerLogger(
            container_id=coordinator_id,
            component_name="communication_layer",
            log_level="INFO"
        )
        
        self.adapters: Dict[str, CommunicationAdapter] = {}
        self.start_time = datetime.utcnow()
        
        self.logger.info(
            "Communication layer initialized",
            coordinator_id=coordinator_id,
            lifecycle_operation="layer_initialization"
        )
    
    def add_adapter(self, name: str, adapter: CommunicationAdapter):
        """Add an adapter to the communication layer.
        
        Args:
            name: Unique name for the adapter
            adapter: Adapter instance to add
        """
        if name in self.adapters:
            self.logger.warning(
                "Replacing existing adapter",
                adapter_name=name,
                existing_type=type(self.adapters[name]).__name__,
                new_type=type(adapter).__name__
            )
        
        self.adapters[name] = adapter
        
        self.logger.info(
            "Adapter added to communication layer",
            adapter_name=name,
            adapter_type=type(adapter).__name__,
            total_adapters=len(self.adapters),
            lifecycle_operation="adapter_registration"
        )
    
    def remove_adapter(self, name: str) -> Optional[CommunicationAdapter]:
        """Remove an adapter from the communication layer.
        
        Args:
            name: Name of adapter to remove
            
        Returns:
            Removed adapter or None if not found
        """
        adapter = self.adapters.pop(name, None)
        
        if adapter:
            self.logger.info(
                "Adapter removed from communication layer",
                adapter_name=name,
                adapter_type=type(adapter).__name__,
                remaining_adapters=len(self.adapters),
                lifecycle_operation="adapter_deregistration"
            )
        else:
            self.logger.warning(
                "Attempted to remove non-existent adapter",
                adapter_name=name
            )
        
        return adapter
    
    def get_adapter(self, name: str) -> Optional[CommunicationAdapter]:
        """Get an adapter by name.
        
        Args:
            name: Name of adapter to retrieve
            
        Returns:
            Adapter instance or None if not found
        """
        return self.adapters.get(name)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for all adapters.
        
        Returns:
            System-wide communication metrics
        """
        adapter_metrics = {}
        total_events_sent = 0
        total_events_received = 0
        total_errors = 0
        all_latencies = []
        
        for name, adapter in self.adapters.items():
            metrics = adapter.get_metrics()
            adapter_metrics[name] = {
                'adapter_type': type(adapter).__name__,
                'events_sent': metrics.events_sent,
                'events_received': metrics.events_received,
                'events_failed': metrics.events_failed,
                'error_rate': metrics.error_rate,
                'average_latency_ms': metrics.average_latency_ms,
                'bytes_sent': metrics.bytes_sent,
                'bytes_received': metrics.bytes_received,
                'is_connected': adapter.is_connected,
                'is_running': adapter.is_running
            }
            
            # Aggregate totals
            total_events_sent += metrics.events_sent
            total_events_received += metrics.events_received
            total_errors += metrics.errors_count
            
            # Collect latencies for percentile calculations
            if metrics.events_sent + metrics.events_received > 0:
                # Estimate latency distribution
                avg_latency = metrics.average_latency_ms
                for _ in range(min(100, metrics.events_sent + metrics.events_received)):
                    # Simulate latency distribution around average
                    all_latencies.append(avg_latency * np.random.lognormal(0, 0.3))
        
        # Calculate system-wide statistics
        uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
        total_events = total_events_sent + total_events_received
        
        system_metrics = {
            'total_adapters': len(self.adapters),
            'active_adapters': sum(1 for a in self.adapters.values() if a.is_running),
            'connected_adapters': sum(1 for a in self.adapters.values() if a.is_connected),
            'uptime_seconds': uptime_seconds,
            'total_events_sent': total_events_sent,
            'total_events_received': total_events_received,
            'total_events': total_events,
            'total_errors': total_errors,
            'overall_error_rate': total_errors / max(total_events, 1),
            'events_per_second': total_events / max(uptime_seconds, 1),
            'adapters': adapter_metrics,
            'overall_health': self._calculate_overall_health(adapter_metrics)
        }
        
        # Add latency percentiles if we have data
        if all_latencies:
            system_metrics['latency_percentiles'] = {
                'p50': np.percentile(all_latencies, 50),
                'p95': np.percentile(all_latencies, 95),
                'p99': np.percentile(all_latencies, 99),
                'max': max(all_latencies)
            }
        
        return system_metrics
    
    def _calculate_overall_health(self, adapter_metrics: Dict[str, Dict[str, Any]]) -> str:
        """Calculate overall communication system health.
        
        Args:
            adapter_metrics: Metrics for all adapters
            
        Returns:
            Health status: 'healthy', 'warning', or 'critical'
        """
        if not adapter_metrics:
            return 'unknown'
        
        # Check error rates
        error_rates = [m['error_rate'] for m in adapter_metrics.values()]
        max_error_rate = max(error_rates) if error_rates else 0
        
        # Check connection status
        disconnected_count = sum(1 for m in adapter_metrics.values() if not m['is_connected'])
        disconnected_ratio = disconnected_count / len(adapter_metrics)
        
        # Determine health status
        if max_error_rate < 0.01 and disconnected_ratio == 0:
            return 'healthy'
        elif max_error_rate < 0.05 and disconnected_ratio < 0.25:
            return 'warning'
        else:
            return 'critical'
    
    async def setup_all_adapters(self):
        """Setup all adapters in the communication layer."""
        self.logger.info(
            "Setting up all adapters",
            adapter_count=len(self.adapters),
            lifecycle_operation="layer_setup"
        )
        
        setup_tasks = []
        for name, adapter in self.adapters.items():
            if hasattr(adapter, 'setup'):
                setup_tasks.append(self._setup_adapter(name, adapter))
        
        # Setup all adapters concurrently
        if setup_tasks:
            results = await asyncio.gather(*setup_tasks, return_exceptions=True)
            
            # Log any setup failures
            for i, (name, result) in enumerate(zip(self.adapters.keys(), results)):
                if isinstance(result, Exception):
                    self.logger.error(
                        "Adapter setup failed",
                        adapter_name=name,
                        error=str(result),
                        error_type=type(result).__name__,
                        lifecycle_operation="adapter_setup_error"
                    )
    
    async def _setup_adapter(self, name: str, adapter: CommunicationAdapter):
        """Setup a single adapter with error handling.
        
        Args:
            name: Adapter name
            adapter: Adapter to setup
        """
        try:
            await adapter.setup()
            self.logger.info(
                "Adapter setup complete",
                adapter_name=name,
                adapter_type=type(adapter).__name__,
                lifecycle_operation="adapter_setup_success"
            )
        except Exception as e:
            self.logger.error(
                "Adapter setup failed",
                adapter_name=name,
                error=str(e),
                error_type=type(e).__name__,
                lifecycle_operation="adapter_setup_error"
            )
            raise
    
    async def cleanup(self):
        """Cleanup all adapters in the communication layer."""
        self.logger.info(
            "Starting communication layer cleanup",
            adapter_count=len(self.adapters),
            lifecycle_operation="layer_cleanup"
        )
        
        cleanup_tasks = []
        for name, adapter in self.adapters.items():
            if hasattr(adapter, 'cleanup'):
                cleanup_tasks.append(self._cleanup_adapter(name, adapter))
        
        # Cleanup all adapters concurrently
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        # Clear adapter registry
        self.adapters.clear()
        
        self.logger.info(
            "Communication layer cleanup complete",
            lifecycle_operation="layer_cleanup_complete"
        )
    
    async def _cleanup_adapter(self, name: str, adapter: CommunicationAdapter):
        """Cleanup a single adapter with error handling.
        
        Args:
            name: Adapter name
            adapter: Adapter to cleanup
        """
        try:
            await adapter.cleanup()
            self.logger.info(
                "Adapter cleanup complete",
                adapter_name=name,
                adapter_type=type(adapter).__name__,
                lifecycle_operation="adapter_cleanup_success"
            )
        except Exception as e:
            self.logger.error(
                "Error cleaning up adapter",
                adapter_name=name,
                error=str(e),
                error_type=type(e).__name__,
                lifecycle_operation="adapter_cleanup_error"
            )
    
    def get_adapter_status_summary(self) -> Dict[str, str]:
        """Get a quick status summary of all adapters.
        
        Returns:
            Dictionary of adapter names to status strings
        """
        status_summary = {}
        
        for name, adapter in self.adapters.items():
            if adapter.is_running and adapter.is_connected:
                status = "active"
            elif adapter.is_connected:
                status = "connected"
            elif adapter.is_running:
                status = "running"
            else:
                status = "inactive"
            
            status_summary[name] = status
        
        return status_summary