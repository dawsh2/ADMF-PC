"""
Risk management types.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from decimal import Decimal


class RiskLimitType(str, Enum):
    """Types of risk limits."""
    POSITION_SIZE = "position_size"
    PORTFOLIO_EXPOSURE = "portfolio_exposure"
    DRAWDOWN = "drawdown"
    CORRELATION = "correlation"
    SECTOR_EXPOSURE = "sector_exposure"
    VAR = "value_at_risk"
    CONCENTRATION = "concentration"


class RiskAction(str, Enum):
    """Actions to take when risk limit is breached."""
    REJECT = "reject"  # Reject the order
    SCALE_DOWN = "scale_down"  # Reduce position size
    WARN = "warn"  # Allow but warn
    HALT = "halt"  # Stop all trading


@dataclass
class RiskLimit:
    """Individual risk limit configuration."""
    limit_type: RiskLimitType
    threshold: float
    action: RiskAction
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskCheck:
    """Result of a risk validation check."""
    order_id: str
    timestamp: datetime
    passed: bool
    checks_performed: Dict[str, bool] = field(default_factory=dict)
    violations: List[str] = field(default_factory=list)
    adjusted_quantity: Optional[float] = None
    message: Optional[str] = None


@dataclass
class PortfolioRisk:
    """Current portfolio risk metrics."""
    timestamp: datetime
    total_exposure: Decimal
    net_exposure: Decimal
    gross_exposure: Decimal
    
    # Drawdown metrics
    current_drawdown: float
    max_drawdown: float
    
    # Position metrics
    largest_position_pct: float
    position_count: int
    concentration_score: float  # 0-1, higher = more concentrated
    
    # Optional drawdown metric
    drawdown_duration_days: Optional[int] = None
    
    # Greeks (if applicable)
    portfolio_beta: Optional[float] = None
    portfolio_delta: Optional[Decimal] = None
    
    # VaR metrics
    var_95: Optional[Decimal] = None  # 95% Value at Risk
    var_99: Optional[Decimal] = None  # 99% Value at Risk
    
    # Correlation metrics
    avg_correlation: Optional[float] = None
    max_correlation: Optional[float] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionRisk:
    """Risk metrics for a single position."""
    symbol: str
    position_size: Decimal
    position_value: Decimal
    pct_of_portfolio: float
    
    # Position-specific risk
    position_var_95: Optional[Decimal] = None
    position_volatility: Optional[float] = None
    beta: Optional[float] = None
    
    # Stop loss info
    stop_loss_price: Optional[Decimal] = None
    stop_loss_risk: Optional[Decimal] = None  # $ at risk if stop hit
    
    # Time in position
    entry_time: Optional[datetime] = None
    holding_period_days: Optional[int] = None


@dataclass
class RiskConfig:
    """Configuration for risk management."""
    # Position limits
    max_position_size_pct: float = 0.05  # 5% max per position
    max_sector_exposure_pct: float = 0.30  # 30% max per sector
    
    # Portfolio limits
    max_gross_exposure: float = 1.0  # 100% gross exposure
    max_net_exposure: float = 1.0  # 100% net exposure
    
    # Drawdown limits
    max_drawdown_pct: float = 0.20  # 20% max drawdown
    drawdown_halt_pct: float = 0.25  # Halt at 25% drawdown
    
    # Correlation limits
    max_correlation: float = 0.70  # Max correlation between positions
    
    # VaR limits
    max_var_95_pct: float = 0.10  # 10% VaR limit
    
    # Risk actions
    risk_limits: List[RiskLimit] = field(default_factory=list)
    
    # Feature flags
    enable_position_sizing: bool = True
    enable_correlation_checks: bool = True
    enable_var_calculation: bool = False
    enable_stop_losses: bool = True
    
    metadata: Dict[str, Any] = field(default_factory=dict)