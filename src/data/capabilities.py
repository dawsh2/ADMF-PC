"""
Data module capabilities for Protocol+Composition enhancement.

These capabilities add functionality to simple data classes through composition,
not inheritance.
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import logging
import time


class DataLoggingCapability:
    """Adds logging functionality to data components."""
    
    def get_name(self) -> str:
        return "data_logging"
    
    def apply(self, component: Any, config: Dict[str, Any]) -> Any:
        """Apply logging capability to component."""
        # Set up logger
        logger_name = config.get('logger_name', f"data.{component.name}")
        logger = logging.getLogger(logger_name)
        
        # Add logging methods
        def log_info(message: str, **kwargs):
            """Log info message with context."""
            extra_info = f" | {', '.join(f'{k}={v}' for k, v in kwargs.items())}" if kwargs else ""
            logger.info(f"[{component.name}] {message}{extra_info}")
        
        def log_error(message: str, **kwargs):
            """Log error message with context."""
            extra_info = f" | {', '.join(f'{k}={v}' for k, v in kwargs.items())}" if kwargs else ""
            logger.error(f"[{component.name}] {message}{extra_info}")
        
        def log_warning(message: str, **kwargs):
            """Log warning message with context."""
            extra_info = f" | {', '.join(f'{k}={v}' for k, v in kwargs.items())}" if kwargs else ""
            logger.warning(f"[{component.name}] {message}{extra_info}")
        
        # Add methods to component
        component.log_info = log_info
        component.log_error = log_error
        component.log_warning = log_warning
        component.logger = logger
        
        return component


class DataMonitoringCapability:
    """Adds monitoring and metrics to data components."""
    
    def get_name(self) -> str:
        return "data_monitoring"
    
    def apply(self, component: Any, config: Dict[str, Any]) -> Any:
        """Apply monitoring capability to component."""
        # Initialize metrics storage
        component._metrics = {
            'load_times': [],
            'bar_count': 0,
            'error_count': 0,
            'start_time': time.time()
        }
        
        # Track performance if configured
        track_methods = config.get('track_performance', [])
        
        def get_metrics() -> Dict[str, Any]:
            """Get current metrics."""
            uptime = time.time() - component._metrics['start_time']
            return {
                'uptime_seconds': uptime,
                'bars_processed': component._metrics['bar_count'],
                'errors': component._metrics['error_count'],
                'average_load_time': (
                    sum(component._metrics['load_times']) / len(component._metrics['load_times'])
                    if component._metrics['load_times'] else 0
                )
            }
        
        def record_metric(name: str, value: Any) -> None:
            """Record a metric value."""
            if name not in component._metrics:
                component._metrics[name] = []
            
            if isinstance(component._metrics[name], list):
                component._metrics[name].append(value)
            else:
                component._metrics[name] = value
        
        # Add methods to component
        component.get_metrics = get_metrics
        component.record_metric = record_metric
        
        # Wrap tracked methods
        for method_name in track_methods:
            if hasattr(component, method_name):
                original_method = getattr(component, method_name)
                
                def create_wrapped_method(orig_method, metric_name):
                    def wrapped_method(*args, **kwargs):
                        start_time = time.time()
                        try:
                            result = orig_method(*args, **kwargs)
                            duration = time.time() - start_time
                            component.record_metric(f"{metric_name}_duration", duration)
                            return result
                        except Exception as e:
                            component._metrics['error_count'] += 1
                            raise
                    return wrapped_method
                
                setattr(component, method_name, create_wrapped_method(original_method, method_name))
        
        return component


class DataEventCapability:
    """Adds event emission capability to data components."""
    
    def get_name(self) -> str:
        return "data_events"
    
    def apply(self, component: Any, config: Dict[str, Any]) -> Any:
        """Apply event capability to component."""
        # Event handlers
        component._event_handlers = {}
        
        def emit_event(event_type: str, payload: Dict[str, Any]) -> None:
            """Emit an event."""
            # Create event with timestamp
            event = {
                'type': event_type,
                'timestamp': datetime.now(),
                'source': component.name,
                'payload': payload
            }
            
            # Call registered handlers
            handlers = component._event_handlers.get(event_type, [])
            for handler in handlers:
                try:
                    handler(event)
                except Exception as e:
                    if hasattr(component, 'log_error'):
                        component.log_error(f"Event handler error: {e}")
        
        def subscribe_to_event(event_type: str, handler: Callable) -> None:
            """Subscribe to an event type."""
            if event_type not in component._event_handlers:
                component._event_handlers[event_type] = []
            component._event_handlers[event_type].append(handler)
        
        def unsubscribe_from_event(event_type: str, handler: Callable) -> None:
            """Unsubscribe from an event type."""
            if event_type in component._event_handlers:
                if handler in component._event_handlers[event_type]:
                    component._event_handlers[event_type].remove(handler)
        
        # Add methods to component
        component.emit_event = emit_event
        component.subscribe_to_event = subscribe_to_event
        component.unsubscribe_from_event = unsubscribe_from_event
        
        return component


class DataValidationCapability:
    """Adds comprehensive data validation to data components."""
    
    def get_name(self) -> str:
        return "data_validation"
    
    def apply(self, component: Any, config: Dict[str, Any]) -> Any:
        """Apply validation capability to component."""
        from .handlers import SimpleDataValidator
        
        # Create validator
        component._validator = SimpleDataValidator()
        
        # Validation settings
        validate_on_load = config.get('validate_on_load', True)
        strict_validation = config.get('strict_validation', False)
        
        def validate_data(data=None, symbol: str = None) -> Dict[str, Any]:
            """Validate data."""
            if data is None and symbol and hasattr(component, 'data'):
                data = component.data.get(symbol)
            
            if data is None:
                return {
                    'passed': False,
                    'errors': ['No data to validate'],
                    'warnings': [],
                    'metadata': {}
                }
            
            return component._validator.validate_data(data)
        
        def get_validation_rules() -> List[str]:
            """Get validation rules."""
            return component._validator.get_validation_rules()
        
        # Add methods to component
        component.validate_data = validate_data
        component.get_validation_rules = get_validation_rules
        
        return component


class DataSplittingCapability:
    """Adds train/test splitting functionality to data components."""
    
    def get_name(self) -> str:
        return "data_splitting"
    
    def apply(self, component: Any, config: Dict[str, Any]) -> Any:
        """Apply splitting capability to component."""
        # Initialize split storage if not exists
        if not hasattr(component, 'splits'):
            component.splits = {}
            component.active_split = None
        
        # Default split configuration
        default_method = config.get('default_method', 'ratio')
        default_ratio = config.get('default_ratio', 0.7)
        
        def setup_split(method: str = default_method, **kwargs) -> None:
            """Set up train/test split."""
            # Use existing method if component has it
            if hasattr(component, 'setup_split'):
                component.setup_split(method, **kwargs)
            else:
                raise NotImplementedError("Component doesn't support splitting")
        
        def set_active_split(split_name: Optional[str]) -> None:
            """Set active data split."""
            if hasattr(component, 'set_active_split'):
                component.set_active_split(split_name)
            else:
                component.active_split = split_name
        
        def get_split_info() -> Dict[str, Any]:
            """Get information about splits."""
            if hasattr(component, 'get_split_info'):
                return component.get_split_info()
            else:
                return {'active_split': getattr(component, 'active_split', None)}
        
        # Add methods to component (if they don't exist)
        if not hasattr(component, 'setup_split'):
            component.setup_split = setup_split
        if not hasattr(component, 'set_active_split'):
            component.set_active_split = set_active_split
        if not hasattr(component, 'get_split_info'):
            component.get_split_info = get_split_info
        
        return component


class MemoryOptimizationCapability:
    """Adds memory optimization to data components."""
    
    def get_name(self) -> str:
        return "memory_optimization"
    
    def apply(self, component: Any, config: Dict[str, Any]) -> Any:
        """Apply memory optimization capability."""
        import gc
        import numpy as np
        
        def _optimize_dtypes(self, df):
            """Optimize DataFrame dtypes for memory efficiency."""
            # Price data - use float32
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = df[col].astype(np.float32)
            
            # Volume - use appropriate integer type
            if "volume" in df.columns:
                max_vol = df["volume"].max()
                if max_vol < np.iinfo(np.uint32).max:
                    df["volume"] = df["volume"].astype(np.uint32)
                else:
                    df["volume"] = df["volume"].astype(np.uint64)
            
            return df
        
        def optimize_memory() -> Dict[str, Any]:
            """Optimize memory usage of loaded data."""
            optimization_results = {}
            
            if hasattr(component, 'data'):
                for symbol, data in component.data.items():
                    if hasattr(data, 'memory_usage'):  # pandas DataFrame
                        original_memory = data.memory_usage(deep=True).sum()
                        
                        # Optimize dtypes
                        optimized_data = _optimize_dtypes(None, data)
                        optimized_memory = optimized_data.memory_usage(deep=True).sum()
                        
                        component.data[symbol] = optimized_data
                        
                        optimization_results[symbol] = {
                            'original_mb': original_memory / (1024**2),
                            'optimized_mb': optimized_memory / (1024**2),
                            'reduction_pct': (1 - optimized_memory/original_memory) * 100
                        }
            
            # Force garbage collection
            gc.collect()
            
            return optimization_results
        
        def get_memory_usage() -> Dict[str, Any]:
            """Get current memory usage."""
            usage = {}
            
            if hasattr(component, 'data'):
                for symbol, data in component.data.items():
                    if hasattr(data, 'memory_usage'):
                        usage[symbol] = {
                            'memory_mb': data.memory_usage(deep=True).sum() / (1024**2),
                            'rows': len(data),
                            'columns': len(data.columns)
                        }
            
            return usage
        
        # Add methods to component
        component.optimize_memory = optimize_memory
        component.get_memory_usage = get_memory_usage
        
        return component


# Capability registry for easy access
DATA_CAPABILITIES = {
    'logging': DataLoggingCapability,
    'monitoring': DataMonitoringCapability,
    'events': DataEventCapability,
    'validation': DataValidationCapability,
    'splitting': DataSplittingCapability,
    'memory_optimization': MemoryOptimizationCapability
}


def apply_capabilities(component: Any, capabilities: List[str], config: Dict[str, Any] = None) -> Any:
    """
    Apply multiple capabilities to a component.
    
    Args:
        component: Component to enhance
        capabilities: List of capability names
        config: Configuration for capabilities
        
    Returns:
        Enhanced component
    """
    config = config or {}
    
    for capability_name in capabilities:
        if capability_name in DATA_CAPABILITIES:
            capability_class = DATA_CAPABILITIES[capability_name]
            capability = capability_class()
            
            # Get capability-specific config
            capability_config = config.get(capability_name, {})
            
            # Apply capability
            component = capability.apply(component, capability_config)
        else:
            raise ValueError(f"Unknown capability: {capability_name}")
    
    return component
