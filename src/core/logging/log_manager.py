"""
LogManager - Centralized Log Lifecycle Management for Logging System v3
Complete lifecycle management including creation, rotation, archiving, and cleanup
"""

import asyncio
import gzip
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Set, Optional, List
from collections import defaultdict

from .container_logger import ContainerLogger, ProductionContainerLogger
from .protocols import LifecycleManaged


class LogRetentionPolicy:
    """
    Manages log retention and cleanup policies.
    
    This component handles automated log lifecycle including archiving,
    compression, and deletion based on configurable policies.
    """
    
    def __init__(self, 
                 max_age_days: int = 30,
                 max_size_gb: float = 10.0,
                 archive_after_days: int = 7,
                 compression_enabled: bool = True):
        """
        Initialize log retention policy.
        
        Args:
            max_age_days: Maximum age before deletion
            max_size_gb: Maximum total size before cleanup
            archive_after_days: Days before archiving logs
            compression_enabled: Whether to compress archived logs
        """
        self.max_age_days = max_age_days
        self.max_size_gb = max_size_gb
        self.archive_after_days = archive_after_days
        self.compression_enabled = compression_enabled
    
    async def apply_retention_rules(self, base_log_dir: Path) -> Dict[str, Any]:
        """
        Apply all retention policies and return statistics.
        
        Args:
            base_log_dir: Base logging directory
            
        Returns:
            Dictionary with cleanup statistics
        """
        stats = {
            'archived_files': 0,
            'deleted_files': 0,
            'compressed_files': 0,
            'space_freed_mb': 0,
            'errors': []
        }
        
        try:
            # Apply policies in order
            archived_stats = await self._archive_old_logs(base_log_dir)
            deleted_stats = await self._delete_expired_logs(base_log_dir)
            compressed_stats = await self._compress_large_logs(base_log_dir)
            
            # Combine statistics
            for key in stats:
                if key in archived_stats:
                    if isinstance(archived_stats[key], (int, float)):
                        stats[key] += archived_stats[key]
                    elif isinstance(archived_stats[key], list):
                        stats[key].extend(archived_stats[key])
                        
                if key in deleted_stats:
                    if isinstance(deleted_stats[key], (int, float)):
                        stats[key] += deleted_stats[key]
                    elif isinstance(deleted_stats[key], list):
                        stats[key].extend(deleted_stats[key])
                        
                if key in compressed_stats:
                    if isinstance(compressed_stats[key], (int, float)):
                        stats[key] += compressed_stats[key]
                    elif isinstance(compressed_stats[key], list):
                        stats[key].extend(compressed_stats[key])
                        
        except Exception as e:
            stats['errors'].append(f"Policy application error: {e}")
        
        return stats
    
    async def _archive_old_logs(self, base_log_dir: Path) -> Dict[str, Any]:
        """Archive logs older than archive threshold."""
        cutoff_time = datetime.now().timestamp() - (self.archive_after_days * 24 * 3600)
        archive_dir = base_log_dir / "archived"
        stats = {'archived_files': 0, 'space_freed_mb': 0, 'errors': []}
        
        for log_file in base_log_dir.rglob("*.log"):
            try:
                if log_file.stat().st_mtime < cutoff_time:
                    # Don't archive system logs or already archived logs
                    if "system" in log_file.parts or "archived" in log_file.parts:
                        continue
                    
                    # Move to archive with date organization
                    relative_path = log_file.relative_to(base_log_dir)
                    file_date = datetime.fromtimestamp(log_file.stat().st_mtime)
                    date_dir = file_date.strftime("%Y-%m-%d")
                    archive_path = archive_dir / date_dir / relative_path
                    archive_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    original_size = log_file.stat().st_size
                    
                    # Compress and move if enabled
                    if self.compression_enabled:
                        with open(log_file, 'rb') as f_in:
                            with gzip.open(f"{archive_path}.gz", 'wb') as f_out:
                                f_out.writelines(f_in)
                        log_file.unlink()  # Delete original
                    else:
                        log_file.rename(archive_path)
                    
                    stats['archived_files'] += 1
                    stats['space_freed_mb'] += original_size / (1024 * 1024)
                    
            except Exception as e:
                stats['errors'].append(f"Error archiving {log_file}: {e}")
        
        return stats
    
    async def _delete_expired_logs(self, base_log_dir: Path) -> Dict[str, Any]:
        """Delete logs older than max age."""
        cutoff_time = datetime.now().timestamp() - (self.max_age_days * 24 * 3600)
        stats = {'deleted_files': 0, 'space_freed_mb': 0, 'errors': []}
        
        # Delete old archived logs
        archive_dir = base_log_dir / "archived"
        if archive_dir.exists():
            for archived_file in archive_dir.rglob("*"):
                try:
                    if archived_file.is_file() and archived_file.stat().st_mtime < cutoff_time:
                        original_size = archived_file.stat().st_size
                        archived_file.unlink()
                        stats['deleted_files'] += 1
                        stats['space_freed_mb'] += original_size / (1024 * 1024)
                        
                except Exception as e:
                    stats['errors'].append(f"Error deleting {archived_file}: {e}")
        
        return stats
    
    async def _compress_large_logs(self, base_log_dir: Path) -> Dict[str, Any]:
        """Compress large log files to save space."""
        stats = {'compressed_files': 0, 'space_freed_mb': 0, 'errors': []}
        
        if not self.compression_enabled:
            return stats
        
        # Compress logs larger than 50MB
        size_threshold = 50 * 1024 * 1024
        
        for log_file in base_log_dir.rglob("*.log"):
            try:
                if log_file.stat().st_size > size_threshold:
                    # Don't compress if already compressed recently
                    compressed_version = log_file.with_suffix('.log.gz')
                    if compressed_version.exists():
                        continue
                    
                    original_size = log_file.stat().st_size
                    
                    # Compress the file
                    with open(log_file, 'rb') as f_in:
                        with gzip.open(compressed_version, 'wb') as f_out:
                            f_out.writelines(f_in)
                    
                    # Remove original
                    log_file.unlink()
                    
                    compressed_size = compressed_version.stat().st_size
                    space_saved = (original_size - compressed_size) / (1024 * 1024)
                    
                    stats['compressed_files'] += 1
                    stats['space_freed_mb'] += space_saved
                    
            except Exception as e:
                stats['errors'].append(f"Error compressing {log_file}: {e}")
        
        return stats


class ContainerLogRegistry:
    """
    Manages loggers for a specific container with lifecycle support.
    
    This registry tracks all loggers within a container and provides
    centralized management for container-specific logging.
    """
    
    def __init__(self, container_id: str, log_dir: Path, log_manager: 'LogManager'):
        """
        Initialize container log registry.
        
        Args:
            container_id: Unique container identifier
            log_dir: Log directory for this container
            log_manager: Parent log manager
        """
        self.container_id = container_id
        self.log_dir = log_dir
        self.log_manager = log_manager
        self.component_loggers: Dict[str, ContainerLogger] = {}
        self.creation_time = datetime.utcnow()
    
    def create_component_logger(self, component_name: str, 
                               use_production_logger: bool = False) -> ContainerLogger:
        """
        Create logger for a component within this container.
        
        Args:
            component_name: Name of the component
            use_production_logger: Whether to use production-optimized logger
            
        Returns:
            ContainerLogger instance for the component
        """
        if component_name in self.component_loggers:
            return self.component_loggers[component_name]
        
        # Create appropriate logger type
        if use_production_logger:
            logger = ProductionContainerLogger(
                self.container_id,
                component_name,
                base_log_dir=str(self.log_dir.parent.parent)  # Back to base logs dir
            )
        else:
            logger = ContainerLogger(
                self.container_id,
                component_name,
                base_log_dir=str(self.log_dir.parent.parent)
            )
        
        self.component_loggers[component_name] = logger
        
        # Register with log manager
        writer_key = f"{self.container_id}.{component_name}"
        self.log_manager.log_writers[writer_key] = logger.container_writer
        
        return logger
    
    def close_all_loggers(self):
        """Close all loggers for this container."""
        for logger in self.component_loggers.values():
            logger.close()
        self.component_loggers.clear()
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            'container_id': self.container_id,
            'component_count': len(self.component_loggers),
            'components': list(self.component_loggers.keys()),
            'creation_time': self.creation_time.isoformat(),
            'log_directory': str(self.log_dir),
            'logger_summaries': {
                name: logger.get_summary() 
                for name, logger in self.component_loggers.items()
            }
        }


class LogManager:
    """
    Centralized log lifecycle management for coordinator.
    
    This manager handles the complete lifecycle of logging including
    container registration, log cleanup, retention policies, and
    performance monitoring.
    """
    
    def __init__(self, coordinator_id: str, base_log_dir: str = "logs", 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize log manager.
        
        Args:
            coordinator_id: Unique coordinator identifier
            base_log_dir: Base directory for logs
            config: Optional configuration dictionary
        """
        self.coordinator_id = coordinator_id
        self.base_log_dir = Path(base_log_dir)
        self.active_containers: Set[str] = set()
        self.log_writers: Dict[str, Any] = {}  # Track all writers
        self.container_registries: Dict[str, ContainerLogRegistry] = {}
        
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
        
        # Initialize background task placeholder
        self._background_task = None
        self._background_enabled = False
    
    def _initialize_log_structure(self):
        """Create standardized log directory structure."""
        directories = [
            self.base_log_dir / "containers",
            self.base_log_dir / "flows",
            self.base_log_dir / "system", 
            self.base_log_dir / "correlations",
            self.base_log_dir / "archived"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create coordinator system logger
        self.system_logger = ContainerLogger(
            self.coordinator_id,
            "log_manager",
            base_log_dir=str(self.base_log_dir)
        )
        
        self.system_logger.info(
            "Log management system initialized",
            base_log_dir=str(self.base_log_dir),
            async_writing=self.async_writing,
            lifecycle_operation="initialization"
        )
    
    def register_container(self, container_id: str) -> ContainerLogRegistry:
        """
        Register new container and setup its logging.
        
        Args:
            container_id: Unique container identifier
            
        Returns:
            ContainerLogRegistry for the container
        """
        self.active_containers.add(container_id)
        
        # Create container log directory
        container_log_dir = self.base_log_dir / "containers" / container_id
        container_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create registry for this container's loggers
        registry = ContainerLogRegistry(container_id, container_log_dir, self)
        self.container_registries[container_id] = registry
        
        self.system_logger.info(
            "Registered container for logging",
            container_id=container_id,
            log_dir=str(container_log_dir),
            lifecycle_operation="container_registration"
        )
        
        return registry
    
    def unregister_container(self, container_id: str):
        """Cleanup container logging on shutdown."""
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
                if hasattr(writer, 'close'):
                    writer.close()
            
            self.system_logger.info(
                "Unregistered container logging",
                container_id=container_id,
                lifecycle_operation="container_cleanup"
            )
    
    async def cleanup_and_archive_logs(self):
        """Periodic cleanup and archiving."""
        self.system_logger.info(
            "Starting log cleanup and archiving",
            lifecycle_operation="maintenance_start"
        )
        
        try:
            stats = await self.retention_policy.apply_retention_rules(self.base_log_dir)
            
            self.system_logger.info(
                "Log cleanup completed",
                **stats,
                lifecycle_operation="maintenance_complete"
            )
            
        except Exception as e:
            self.system_logger.error(
                "Error during log cleanup",
                error=str(e),
                lifecycle_operation="maintenance_error"
            )
    
    def get_log_summary(self) -> Dict[str, Any]:
        """Get logging system status."""
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
                "max_size_gb": self.retention_policy.max_size_gb,
                "compression_enabled": self.retention_policy.compression_enabled
            }
        }
    
    def _calculate_disk_usage(self) -> float:
        """Calculate total disk usage of logs in MB."""
        total_size = 0
        for file_path in self.base_log_dir.rglob("*.log*"):
            try:
                total_size += file_path.stat().st_size
            except:
                pass
        return total_size / (1024 * 1024)
    
    def _get_oldest_log_date(self) -> Optional[str]:
        """Get oldest log file date."""
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
        """Get current log directory structure."""
        structure = {}
        
        containers_dir = self.base_log_dir / "containers"
        if containers_dir.exists():
            for container_dir in containers_dir.iterdir():
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
    
    def start_background_tasks(self):
        """Start background maintenance tasks (call this in async context)."""
        if self.async_writing and not self._background_enabled:
            # Create the background task
            self._background_task = asyncio.create_task(self._periodic_maintenance())
            self._background_enabled = True
    
    async def _periodic_maintenance(self):
        """Background task for periodic maintenance."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self.cleanup_and_archive_logs()
            except Exception as e:
                self.system_logger.error(
                    "Error in periodic maintenance",
                    error=str(e),
                    lifecycle_operation="maintenance_error"
                )
                await asyncio.sleep(600)  # Retry in 10 minutes
    
    async def shutdown(self):
        """Graceful shutdown with cleanup."""
        self.system_logger.info(
            "Shutting down log manager",
            lifecycle_operation="shutdown_start"
        )
        
        # Stop background maintenance
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup all containers
        for container_id in list(self.active_containers):
            self.unregister_container(container_id)
        
        # Final cleanup
        await self.cleanup_and_archive_logs()
        
        # Close system logger
        self.system_logger.close()
        
        self.system_logger.info(
            "Log manager shutdown complete",
            lifecycle_operation="shutdown_complete"
        )