"""
ContainerLogger - Main Container-Aware Logger for Logging System v3
Built through composition with lifecycle management and zero inheritance
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
from contextlib import contextmanager

from .protocols import Loggable, ContainerAware, CorrelationAware, LifecycleManaged
from .log_writer import LogWriter
from .container_context import ContainerContext, EnhancedContainerContext
from .correlation_tracker import CorrelationTracker, CorrelationContext
from .scope_detector import EventScopeDetector, EventScope


class ContainerLogger:
    """
    Container-aware logger built through composition.
    No inheritance - just protocols and composition!
    
    This logger composes various specialized components to provide:
    - Container-isolated logging
    - Cross-container correlation tracking
    - Event scope classification
    - Lifecycle management
    - Performance optimization
    
    Inspired by existing StructuredLogger but enhanced for multi-container architectures.
    """
    
    def __init__(self, container_id: str, component_name: str, 
                 log_level: str = 'INFO', base_log_dir: str = "logs",
                 enable_performance_tracking: bool = False):
        """
        Initialize container logger through composition.
        
        Args:
            container_id: Unique container identifier
            component_name: Component name within container
            log_level: Minimum log level
            base_log_dir: Base directory for log files
            enable_performance_tracking: Whether to track performance metrics
        """
        # Compose with various components - NO INHERITANCE!
        if enable_performance_tracking:
            self.container_context = EnhancedContainerContext(container_id, component_name, True)
        else:
            self.container_context = ContainerContext(container_id, component_name)
            
        self.correlation_tracker = CorrelationTracker()
        self.scope_detector = EventScopeDetector()
        
        # Setup composed log writers
        base_path = Path(base_log_dir)
        container_log_path = base_path / "containers" / container_id / f"{component_name}.log"
        master_log_path = base_path / "master.log"
        flow_log_path = base_path / "flows" / "cross_container_flows.log"
        
        self.container_writer = LogWriter(container_log_path)
        self.master_writer = LogWriter(master_log_path)
        self.flow_writer = LogWriter(flow_log_path)
        
        self.log_level = log_level
        self._closed = False
        
        # Thread safety
        self._lock = threading.Lock()
    
    # Implement Loggable protocol through composition
    def log(self, level: str, message: str, **context) -> None:
        """
        Main logging method - implements Loggable protocol.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Log message
            **context: Additional context fields
        """
        if self._closed:
            return
            
        start_time = datetime.utcnow()
        
        # Use composed components to build functionality
        correlation_id = self.correlation_tracker.get_correlation_id()
        event_scope = self.scope_detector.detect_scope(context)
        
        # Create structured log entry
        log_entry = {
            'timestamp': start_time.isoformat() + 'Z',
            'level': level,
            'message': message,
            'container_id': self.container_context.container_id,
            'component_name': self.container_context.component_name,
            'event_scope': event_scope,
            'correlation_id': correlation_id,
            'thread_id': threading.get_ident(),
            'thread_name': threading.current_thread().name,
            'context': context
        }
        
        # Add event ID if provided for tracing
        if 'event_id' in context:
            log_entry['event_id'] = context['event_id']
            # Track event in correlation chain
            self.correlation_tracker.track_event(
                context['event_id'], 
                f"{self.container_context.container_id}.{self.container_context.component_name}"
            )
        
        # Write using composed writers
        with self._lock:
            self.container_writer.write(log_entry)
            
            # Write to master log for important messages
            if level in ['ERROR', 'WARNING', 'INFO']:
                master_entry = {
                    'timestamp': log_entry['timestamp'],
                    'level': level,
                    'container': self.container_context.container_id,
                    'component': self.container_context.component_name,
                    'message': message,
                    'scope': event_scope,
                    'correlation_id': correlation_id
                }
                self.master_writer.write(master_entry)
            
            # Write cross-container flow events
            if event_scope.startswith('external_'):
                flow_entry = {
                    'timestamp': log_entry['timestamp'],
                    'event_type': context.get('event_type', 'unknown'),
                    'source': f"{self.container_context.container_id}.{self.container_context.component_name}",
                    'target': context.get('target', 'unknown'),
                    'tier': event_scope.replace('external_', '').replace('_tier', ''),
                    'correlation_id': correlation_id,
                    'message': message
                }
                self.flow_writer.write(flow_entry)
        
        # Update context metrics
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        if hasattr(self.container_context, 'update_metrics'):
            # Check if it's the enhanced version with more parameters
            if isinstance(self.container_context, EnhancedContainerContext):
                self.container_context.update_metrics(
                    level, start_time, processing_time, context.get('event_type')
                )
            else:
                self.container_context.update_metrics(level, start_time)
    
    # Implement ContainerAware protocol
    @property
    def container_id(self) -> str:
        """Get container ID."""
        return self.container_context.container_id
    
    @property 
    def component_name(self) -> str:
        """Get component name."""
        return self.container_context.component_name
    
    # Implement CorrelationAware protocol
    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for tracking."""
        self.correlation_tracker.set_correlation_id(correlation_id)
    
    def get_correlation_id(self) -> Optional[str]:
        """Get current correlation ID."""
        return self.correlation_tracker.get_correlation_id()
    
    @contextmanager
    def with_correlation_id(self, correlation_id: str):
        """Context manager for correlation tracking."""
        with CorrelationContext(self.correlation_tracker, correlation_id):
            yield
    
    # Convenience methods (inspired by existing StructuredLogger)
    def trace(self, message: str, **context):
        """Log at TRACE level for ultra-detailed debugging."""
        self.log('TRACE', message, **context)
    
    def debug(self, message: str, **context):
        """Log at DEBUG level for development debugging."""
        self.log('DEBUG', message, **context)
    
    def info(self, message: str, **context):
        """Log at INFO level for general information."""
        self.log('INFO', message, **context)
        
    def warning(self, message: str, **context):
        """Log at WARNING level for concerning conditions."""
        self.log('WARNING', message, **context)
    
    def error(self, message: str, **context):
        """Log at ERROR level for error conditions.""" 
        self.log('ERROR', message, **context)
    
    def critical(self, message: str, **context):
        """Log at CRITICAL level for critical system failures."""
        self.log('CRITICAL', message, **context)
    
    # ComponentLogger pattern methods (inspired by existing event_logger.py)
    def log_event_flow(self, event_type: str, source: str, destination: str, payload_summary: str) -> None:
        """
        Log event flow for debugging and validation.
        
        Enhanced version of ComponentLogger pattern for container architectures.
        """
        self.info(
            f"EVENT_FLOW | {source} → {destination} | {event_type}",
            event_type=event_type,
            source=source,
            destination=destination,
            payload_summary=payload_summary,
            log_category="event_flow",
            target=destination  # For scope detection
        )
    
    def log_state_change(self, old_state: str, new_state: str, trigger: str) -> None:
        """Log component state changes for debugging."""
        self.info(
            f"STATE_CHANGE | {old_state} → {new_state}",
            old_state=old_state,
            new_state=new_state,
            trigger=trigger,
            log_category="state_change"
        )
    
    def log_performance_metric(self, metric_name: str, value: float, context: Dict[str, Any]) -> None:
        """Log performance metrics with context for monitoring."""
        self.info(
            f"PERFORMANCE | {metric_name} | {value}",
            metric_name=metric_name,
            metric_value=value,
            log_category="performance_metric",
            **context
        )
    
    def log_validation_result(self, test_name: str, passed: bool, details: str) -> None:
        """Log validation test results for compliance tracking."""
        level_method = self.info if passed else self.error
        status = "PASS" if passed else "FAIL"
        level_method(
            f"VALIDATION | {test_name} | {status}",
            test_name=test_name,
            passed=passed,
            details=details,
            log_category="validation_result"
        )
    
    # Event-specific logging methods (inspired by event_logger.py)
    def log_bar_event(self, symbol: str, timestamp: Any, price: float, bar_num: Optional[int] = None):
        """Log a BAR event for market data processing."""
        bar_info = f" ({bar_num})" if bar_num else ""
        self.debug(
            f"BAR | {symbol} @ {timestamp}{bar_info} - Price: {price:.4f}",
            event_type="BAR",
            symbol=symbol,
            price=price,
            bar_num=bar_num,
            publish_tier="fast"  # BAR events use fast tier
        )
    
    def log_signal_event(self, signal):
        """Log a SIGNAL event for trading strategy output."""
        # Handle both Signal objects and dictionaries
        if hasattr(signal, 'symbol'):  # Signal dataclass object
            symbol = signal.symbol
            side = getattr(signal, 'side', getattr(signal, 'direction', 'UNKNOWN'))
            strength = float(signal.strength)
            reason = getattr(signal, 'metadata', {}).get('reason', 'No reason')
        else:  # Dictionary
            symbol = signal.get('symbol', 'UNKNOWN')
            side = signal.get('side', signal.get('direction', 'UNKNOWN'))
            strength = signal.get('strength', 0)
            reason = signal.get('metadata', {}).get('reason', 'No reason')
        
        self.info(
            f"SIGNAL | {symbol} {side} (strength: {strength:.2f}) - {reason}",
            event_type="SIGNAL",
            symbol=symbol,
            side=side,
            strength=strength,
            reason=reason,
            publish_tier="standard"  # SIGNAL events use standard tier
        )
    
    def log_order_event(self, order):
        """Log an ORDER event for trade execution.""" 
        # Handle both Order objects and dictionaries
        if hasattr(order, 'symbol'):  # Order object
            symbol = order.symbol
            side = order.side.value if hasattr(order.side, 'value') else str(order.side)
            quantity = float(order.quantity)
            price = float(order.price) if getattr(order, 'price', None) else 0.0
        else:  # Dictionary
            symbol = order.get('symbol', 'UNKNOWN')
            side = order.get('side', 'UNKNOWN')
            quantity = order.get('quantity', 0)
            price = order.get('price', 0)
        
        self.info(
            f"ORDER | {symbol} {side} {quantity:.2f} @ {price:.4f}",
            event_type="ORDER",
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            publish_tier="reliable"  # ORDER events use reliable tier
        )
    
    def log_fill_event(self, fill):
        """Log a FILL event for trade execution completion."""
        # Handle both Fill objects and dictionaries
        if hasattr(fill, 'symbol'):  # Fill object
            symbol = fill.symbol
            side = fill.side.value if hasattr(fill.side, 'value') else str(fill.side)
            quantity = float(fill.quantity)
            price = float(fill.price)
        else:  # Dictionary
            symbol = fill.get('symbol', 'UNKNOWN')
            side = fill.get('side', 'UNKNOWN')
            quantity = fill.get('quantity', 0)
            price = fill.get('price', 0)
        
        self.info(
            f"FILL | {symbol} {side} {quantity:.2f} @ {price:.4f}",
            event_type="FILL",
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            publish_tier="reliable"  # FILL events use reliable tier
        )
    
    # Lifecycle management
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive logger summary."""
        return {
            'container_context': self.container_context.get_summary(),
            'correlation_stats': self.correlation_tracker.get_statistics(),
            'scope_detector_stats': self.scope_detector.get_statistics(),
            'writer_metrics': {
                'container': self.container_writer.get_metrics(),
                'master': self.master_writer.get_metrics(),
                'flow': self.flow_writer.get_metrics()
            },
            'log_level': self.log_level,
            'closed': self._closed
        }
    
    def close(self):
        """Close logger and clean up resources."""
        if not self._closed:
            with self._lock:
                self.container_writer.close()
                self.master_writer.close() 
                self.flow_writer.close()
                self._closed = True


class ProductionContainerLogger(ContainerLogger):
    """
    Production-optimized container logger with advanced features.
    
    This version includes performance optimizations, advanced metrics,
    and production-ready features for high-throughput environments.
    """
    
    def __init__(self, container_id: str, component_name: str,
                 log_level: str = 'INFO', base_log_dir: str = "logs",
                 enable_async_writing: bool = True):
        """
        Initialize production logger with async writing support.
        
        Args:
            container_id: Unique container identifier
            component_name: Component name within container
            log_level: Minimum log level
            base_log_dir: Base directory for log files
            enable_async_writing: Whether to use async batch writing
        """
        super().__init__(
            container_id, 
            component_name, 
            log_level, 
            base_log_dir,
            enable_performance_tracking=True
        )
        
        # Replace standard writers with async versions if enabled
        if enable_async_writing:
            from .log_writer import AsyncBatchLogWriter
            
            base_path = Path(base_log_dir)
            container_log_path = base_path / "containers" / container_id / f"{component_name}.log"
            master_log_path = base_path / "master.log"
            flow_log_path = base_path / "flows" / "cross_container_flows.log"
            
            self.async_container_writer = AsyncBatchLogWriter(container_log_path)
            self.async_master_writer = AsyncBatchLogWriter(master_log_path)
            self.async_flow_writer = AsyncBatchLogWriter(flow_log_path)
            self.async_enabled = True
        else:
            self.async_enabled = False
    
    async def log_async(self, level: str, message: str, **context):
        """Async version of log method for high-performance environments."""
        if self._closed or not self.async_enabled:
            # Fall back to sync logging
            self.log(level, message, **context)
            return
            
        start_time = datetime.utcnow()
        
        # Build log entry (same as sync version)
        correlation_id = self.correlation_tracker.get_correlation_id()
        event_scope = self.scope_detector.detect_scope(context)
        
        log_entry = {
            'timestamp': start_time.isoformat() + 'Z',
            'level': level,
            'message': message,
            'container_id': self.container_context.container_id,
            'component_name': self.container_context.component_name,
            'event_scope': event_scope,
            'correlation_id': correlation_id,
            'thread_id': threading.get_ident(),
            'thread_name': threading.current_thread().name,
            'context': context
        }
        
        # Async write operations
        await self.async_container_writer.write_async(log_entry)
        
        if level in ['ERROR', 'WARNING', 'INFO']:
            master_entry = {
                'timestamp': log_entry['timestamp'],
                'level': level,
                'container': self.container_context.container_id,
                'component': self.container_context.component_name,
                'message': message,
                'scope': event_scope,
                'correlation_id': correlation_id
            }
            await self.async_master_writer.write_async(master_entry)
        
        if event_scope.startswith('external_'):
            flow_entry = {
                'timestamp': log_entry['timestamp'],
                'event_type': context.get('event_type', 'unknown'),
                'source': f"{self.container_context.container_id}.{self.container_context.component_name}",
                'target': context.get('target', 'unknown'),
                'tier': event_scope.replace('external_', '').replace('_tier', ''),
                'correlation_id': correlation_id,
                'message': message
            }
            await self.async_flow_writer.write_async(flow_entry)
        
        # Update metrics
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        self.container_context.update_metrics(level, start_time, processing_time, context.get('event_type'))
    
    async def close_async(self):
        """Async close with proper flush of batched writes."""
        if self.async_enabled:
            # Flush any remaining batched writes
            await self.async_container_writer._flush_batch()
            await self.async_master_writer._flush_batch()
            await self.async_flow_writer._flush_batch()
        
        self.close()