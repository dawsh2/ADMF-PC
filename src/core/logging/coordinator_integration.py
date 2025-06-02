"""
Coordinator Integration Example for Logging System v3
Shows how to integrate the new logging system with the workflow coordinator
"""

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from .log_manager import LogManager
from .container_logger import ContainerLogger
from .capabilities import add_logging_to_any_component


class LoggingEnabledCoordinator:
    """
    Example coordinator integration with automatic logging lifecycle management.
    
    This shows how to integrate the new v3 logging system with any coordinator
    to provide automatic container logging setup and cleanup.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize coordinator with integrated logging.
        
        Args:
            config: Coordinator configuration including logging settings
        """
        self.coordinator_id = config.get('coordinator_id', f"coord_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.config = config
        
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
        self.container_log_registries = {}
        
        # Start background maintenance
        self._background_tasks = []
        if config.get('logging', {}).get('performance', {}).get('async_writing', True):
            self._start_background_maintenance()
        
        self.logger.info(
            "Logging-enabled coordinator initialized",
            coordinator_id=self.coordinator_id,
            config_summary=self._get_config_summary(),
            lifecycle_operation="coordinator_initialization"
        )
    
    async def create_container(self, container_id: str, container_config: Dict) -> Any:
        """
        Create container with automatic log setup.
        
        Args:
            container_id: Unique container identifier
            container_config: Container configuration
            
        Returns:
            Created container with logging capability
        """
        # Register logging for this container FIRST
        log_registry = self.log_manager.register_container(container_id)
        self.container_log_registries[container_id] = log_registry
        
        # Create the actual container (this would be your container creation logic)
        container = await self._instantiate_container(container_id, container_config)
        
        # Add logging capability to container automatically
        enhanced_container = add_logging_to_any_component(
            container,
            container_id,
            "container_manager",
            use_production_logger=True
        )
        
        # Store container
        self.containers[container_id] = enhanced_container
        
        self.logger.info(
            "Container created with automatic logging",
            container_id=container_id,
            container_type=container_config.get('type'),
            components_enabled=container_config.get('logging', {}).get('components', []),
            lifecycle_operation="container_creation"
        )
        
        # Log initial container state
        enhanced_container.log_state_change(
            "NONE", "INITIALIZING", 
            f"Container {container_id} created by coordinator"
        )
        
        return enhanced_container
    
    async def shutdown_container(self, container_id: str):
        """
        Shutdown container and cleanup logs automatically.
        
        Args:
            container_id: Container to shutdown
        """
        if container_id in self.containers:
            container = self.containers[container_id]
            
            # Log container shutdown
            if hasattr(container, 'log_state_change'):
                container.log_state_change(
                    "RUNNING", "SHUTTING_DOWN",
                    f"Container {container_id} shutdown requested"
                )
            
            # Shutdown container
            if hasattr(container, 'shutdown'):
                await container.shutdown()
            
            # Cleanup logging automatically
            if container_id in self.container_log_registries:
                registry = self.container_log_registries[container_id]
                del self.container_log_registries[container_id]
            
            self.log_manager.unregister_container(container_id)
            del self.containers[container_id]
            
            self.logger.info(
                "Container shutdown with automatic log cleanup",
                container_id=container_id,
                lifecycle_operation="container_shutdown"
            )
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status including logging.
        
        Returns:
            System status with logging information
        """
        return {
            "coordinator": {
                "id": self.coordinator_id,
                "active_containers": len(self.containers),
                "config_summary": self._get_config_summary()
            },
            "logging": self.log_manager.get_log_summary(),
            "containers": {
                container_id: {
                    "type": type(container).__name__,
                    "has_logging": hasattr(container, 'logger'),
                    "log_registry": self.container_log_registries[container_id].get_registry_stats()
                    if container_id in self.container_log_registries else None
                }
                for container_id, container in self.containers.items()
            }
        }
    
    async def shutdown(self):
        """
        Graceful shutdown with automatic log cleanup.
        """
        self.logger.info(
            "Starting coordinator shutdown with automatic log cleanup",
            active_containers=len(self.containers),
            lifecycle_operation="coordinator_shutdown_start"
        )
        
        # Shutdown all containers (logging cleanup is automatic)
        for container_id in list(self.containers.keys()):
            await self.shutdown_container(container_id)
        
        # Stop background tasks
        for task in self._background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Shutdown log manager (final cleanup)
        await self.log_manager.shutdown()
        
        print("✅ Coordinator shutdown complete - all logs cleaned up automatically")
    
    def _get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for logging."""
        return {
            "log_dir": self.config.get('log_dir', 'logs'),
            "async_writing": self.config.get('logging', {}).get('performance', {}).get('async_writing', True),
            "retention_days": self.config.get('logging', {}).get('retention_policy', {}).get('max_age_days', 30),
            "compression_enabled": self.config.get('logging', {}).get('retention_policy', {}).get('compression_enabled', True)
        }
    
    def _start_background_maintenance(self):
        """Start background maintenance tasks."""
        # Log maintenance task
        maintenance_task = asyncio.create_task(self._periodic_log_reporting())
        self._background_tasks.append(maintenance_task)
    
    async def _periodic_log_reporting(self):
        """Background task for periodic log status reporting."""
        while True:
            try:
                await asyncio.sleep(1800)  # Report every 30 minutes
                
                status = await self.get_system_status()
                logging_status = status['logging']
                
                self.logger.info(
                    "Periodic logging system status",
                    disk_usage_mb=logging_status['disk_usage_mb'],
                    active_containers=logging_status['active_containers'],
                    total_log_directories=logging_status['total_log_directories'],
                    lifecycle_operation="periodic_status_report"
                )
                
                # Alert if disk usage is high
                if logging_status['disk_usage_mb'] > 8000:  # 8GB threshold
                    self.logger.warning(
                        "High log disk usage detected - triggering cleanup",
                        disk_usage_mb=logging_status['disk_usage_mb'],
                        lifecycle_operation="high_disk_usage_alert"
                    )
                    await self.log_manager.cleanup_and_archive_logs()
                
            except Exception as e:
                self.logger.error(
                    "Error in periodic log reporting",
                    error=str(e),
                    lifecycle_operation="periodic_reporting_error"
                )
                await asyncio.sleep(300)  # Retry in 5 minutes
    
    async def _instantiate_container(self, container_id: str, container_config: Dict) -> Any:
        """
        Create the actual container instance.
        
        This is where you would put your actual container creation logic.
        For this example, we'll create a simple mock container.
        """
        
        class MockContainer:
            def __init__(self, container_id: str, config: Dict):
                self.container_id = container_id
                self.config = config
                self.status = "RUNNING"
            
            async def shutdown(self):
                self.status = "SHUTDOWN"
        
        return MockContainer(container_id, container_config)


# Example usage function
async def example_coordinator_with_logging():
    """
    Example of using the logging-enabled coordinator.
    
    This demonstrates the complete lifecycle with automatic logging management.
    """
    
    # Configuration with logging settings
    config = {
        'coordinator_id': 'example_coordinator',
        'log_dir': '/tmp/example_logs',
        'logging': {
            'retention_policy': {
                'max_age_days': 30,
                'archive_after_days': 7,
                'compression_enabled': True
            },
            'performance': {
                'async_writing': True,
                'batch_size': 1000,
                'flush_interval_seconds': 5
            }
        }
    }
    
    # Create coordinator with automatic logging
    coordinator = LoggingEnabledCoordinator(config)
    
    try:
        # Create multiple containers - logging setup is automatic
        containers = []
        for i in range(5):
            container_id = f"strategy_container_{i:03d}"
            container = await coordinator.create_container(container_id, {
                'type': 'strategy',
                'strategy': 'momentum',
                'allocation': 0.2
            })
            containers.append(container)
            
            # Containers can log immediately after creation
            container.log_info(
                "Strategy container operational",
                container_id=container_id,
                strategy_type="momentum"
            )
        
        # Get system status (includes comprehensive logging info)
        status = await coordinator.get_system_status()
        print(f"✅ System running with {status['coordinator']['active_containers']} containers")
        print(f"📊 Log disk usage: {status['logging']['disk_usage_mb']:.1f} MB")
        print(f"📁 Log directories: {status['logging']['total_log_directories']}")
        
        # Simulate running for a short time
        await asyncio.sleep(2)
        
        # Example: Shutdown specific container
        await coordinator.shutdown_container("strategy_container_002")
        print("✅ Container shutdown completed with automatic log cleanup")
        
    finally:
        # Graceful shutdown with automatic log cleanup
        await coordinator.shutdown()


if __name__ == "__main__":
    # Run the example
    asyncio.run(example_coordinator_with_logging())