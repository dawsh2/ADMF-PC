"""
Type definitions for ADMF-PC.

This package contains all type definitions organized by domain:
- trading: Order, Signal, Position types
- workflow: Workflow and coordination types  
- events: Event system types
- duck_types: Flexible duck-typed interfaces
- decimal: Financial calculation utilities
"""

# Re-export commonly used types for convenience
from .trading import (
    Bar, Position, Order, Fill, Signal,
    OrderType, OrderSide, SignalType
)

from .events import (
    EventType, Event, EventHandler, EventPublisher, EventSubscriber,
    EventBusProtocol, EventCapable,
    create_market_event, create_signal_event, create_system_event, create_error_event
)

from .workflow import (
    WorkflowType, WorkflowPhase, ExecutionContext, WorkflowConfig,
    PhaseResult, WorkflowResult
)

from .decimal import (
    ensure_decimal, round_price, round_quantity,
    calculate_value, calculate_commission, calculate_slippage,
    safe_divide, format_currency, format_percentage,
    DecimalEncoder, validate_price, validate_quantity, validate_percentage
)

# Duck types available but not re-exported by default to avoid confusion
# Import explicitly: from src.core.types.duck_types import ComponentLike

__all__ = [
    # Trading types
    'OrderSide', 'OrderType', 'SignalType', 'FillType', 'FillStatus', 'OrderStatus',
    'Signal', 'Order', 'Position',
    
    # Event types  
    'EventType', 'Event', 'EventHandler', 'EventPublisher', 'EventSubscriber',
    'EventBusProtocol', 'EventCapable',
    'create_market_event', 'create_signal_event', 'create_system_event', 'create_error_event',
    
    # Workflow types
    'WorkflowType', 'WorkflowPhase', 'ExecutionContext', 'WorkflowConfig',
    'PhaseResult', 'WorkflowResult',
    
    # Decimal utilities
    'ensure_decimal', 'round_price', 'round_quantity',
    'calculate_value', 'calculate_commission', 'calculate_slippage',
    'safe_divide', 'format_currency', 'format_percentage',
    'DecimalEncoder', 'validate_price', 'validate_quantity', 'validate_percentage'
]