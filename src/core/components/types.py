"""
Component types for the component system.

Minimal types needed by component protocols.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum


class ComponentType(str, Enum):
    """Types of components in the system."""
    STRATEGY = "strategy"
    INDICATOR = "indicator"
    CLASSIFIER = "classifier"
    RISK_VALIDATOR = "risk_validator"
    DATA_PROVIDER = "data_provider"
    ORDER_EXECUTOR = "order_executor"
    PORTFOLIO = "portfolio"


@dataclass
class ComponentMetadata:
    """Metadata about a component."""
    component_type: ComponentType
    version: str
    author: Optional[str] = None
    description: Optional[str] = None
    capabilities: list[str] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []


@dataclass
class PositionPayload:
    """Position data used in Portfolio protocol."""
    symbol: str
    quantity: float
    avg_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass 
class HealthStatus:
    """Component health status."""
    healthy: bool
    message: str = "OK"
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}