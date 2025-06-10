"""
Pydantic Configuration Models

Modern, type-safe configuration validation for ADMF-PC.
Provides clear error messages and automatic documentation generation.
"""

from typing import Dict, Any, List, Optional, Union, Literal
from datetime import datetime, date
from pathlib import Path

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
    from pydantic import ValidationError, ConfigDict
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Create stubs for graceful degradation
    BaseModel = object
    Field = lambda *args, **kwargs: None
    field_validator = lambda *args, **kwargs: lambda func: func
    model_validator = lambda *args, **kwargs: lambda func: func
    ValidationError = Exception
    ConfigDict = dict

import logging

logger = logging.getLogger(__name__)


class BaseConfig(BaseModel):
    """Base configuration with common settings."""
    
    model_config = ConfigDict(
        # Allow extra fields for backward compatibility
        extra="allow",
        # Use enum values instead of names
        use_enum_values=True,
        # Validate field default values
        validate_default=True,
        # Validate on assignment
        validate_assignment=True
    )


class DataConfig(BaseConfig):
    """Data source configuration."""
    
    symbols: List[str] = Field(..., min_items=1, description="Asset symbols to trade")
    start_date: str = Field(..., description="Backtest start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="Backtest end date (YYYY-MM-DD)")
    frequency: Literal["1min", "5min", "15min", "30min", "1h", "1d"] = Field(
        "1d", description="Data frequency"
    )
    source: str = Field("csv", description="Data source type")
    file_path: Optional[str] = Field(None, description="Path to data file (for CSV source)")
    timezone: str = Field("UTC", description="Data timezone")
    
    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_date_format(cls, v):
        """Validate date format."""
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")
    
    @model_validator(mode='after')
    def validate_date_order(self):
        """Validate that start_date is before end_date."""
        if self.start_date and self.end_date:
            start_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(self.end_date, '%Y-%m-%d')
            
            if start_dt >= end_dt:
                raise ValueError("start_date must be before end_date")
        
        return self
    
    @field_validator('file_path')
    @classmethod
    def validate_file_path(cls, v):
        """Validate file path exists for CSV source."""
        if v and not Path(v).exists():
            logger.warning(f"Data file not found: {v}")
            # Don't fail validation - file might be created later
        return v


class PortfolioConfig(BaseConfig):
    """Portfolio configuration."""
    
    initial_capital: float = Field(..., gt=0, description="Initial portfolio capital")
    currency: str = Field("USD", description="Portfolio base currency")
    commission_type: Literal["fixed", "percentage"] = Field("percentage", description="Commission type")
    commission_value: float = Field(0.001, ge=0, description="Commission value")
    slippage_type: Literal["fixed", "percentage"] = Field("percentage", description="Slippage type")
    slippage_value: float = Field(0.001, ge=0, description="Slippage value")
    
    @field_validator('commission_value', 'slippage_value')
    @classmethod
    def validate_cost_values(cls, v):
        """Validate commission and slippage values are reasonable."""
        if v > 0.1:  # 10% seems unreasonably high
            logger.warning(f"Value {v} seems very high (>10%)")
        return v


class StrategyConfig(BaseConfig):
    """Strategy configuration."""
    
    name: str = Field(..., description="Strategy instance name")
    type: str = Field(..., description="Strategy algorithm type")
    enabled: bool = Field(True, description="Whether strategy is enabled")
    allocation: float = Field(1.0, ge=0, le=1, description="Strategy allocation (0-1)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")
    capabilities: List[str] = Field(default_factory=list, description="Required capabilities")
    
    @field_validator('type')
    @classmethod
    def validate_strategy_type(cls, v):
        """Validate strategy type against known types."""
        # Import here to avoid circular imports
        try:
            from ..component_schemas import list_components_for_type
            known_types = list_components_for_type('strategies')
            if known_types and v not in known_types:
                available = ', '.join(known_types)
                raise ValueError(f"Unknown strategy type '{v}'. Available: {available}")
        except ImportError:
            # Fallback for common types
            common_types = ['momentum', 'mean_reversion', 'trend_following', 'moving_average_crossover']
            if v not in common_types:
                logger.warning(f"Strategy type '{v}' not in common types: {common_types}")
        
        return v
    
    @field_validator('parameters')
    @classmethod
    def validate_parameters(cls, v):
        """Validate strategy parameters."""
        # Basic parameter validation - type checking is handled by Pydantic
        if isinstance(v, dict):
            if 'lookback_period' in v and v['lookback_period'] < 1:
                raise ValueError("lookback_period must be positive")
            if 'period' in v and v['period'] < 1:
                raise ValueError("period must be positive")
        
        return v


class RiskLimitConfig(BaseConfig):
    """Risk limit configuration."""
    
    type: Literal[
        "position", "exposure", "drawdown", "var", 
        "concentration", "leverage", "daily_loss", "symbol_restriction"
    ] = Field(..., description="Risk limit type")
    enabled: bool = Field(True, description="Whether limit is enabled")
    
    # Position limits
    max_position: Optional[float] = Field(None, gt=0, description="Maximum position size")
    max_position_pct: Optional[float] = Field(None, gt=0, le=1, description="Max position as % of portfolio")
    
    # Exposure limits  
    max_exposure_pct: Optional[float] = Field(None, gt=0, le=1, description="Maximum exposure percentage")
    
    # Drawdown limits
    max_drawdown_pct: Optional[float] = Field(None, gt=0, le=1, description="Maximum drawdown percentage")
    reduce_at_pct: Optional[float] = Field(None, gt=0, le=1, description="Reduce positions at drawdown %")
    
    # VaR limits
    confidence_level: Optional[float] = Field(None, gt=0, lt=1, description="VaR confidence level")
    max_var_pct: Optional[float] = Field(None, gt=0, le=1, description="Maximum VaR percentage")
    
    # Leverage limits
    max_leverage: Optional[float] = Field(None, gt=0, description="Maximum leverage ratio")
    
    # Daily loss limits
    max_daily_loss: Optional[float] = Field(None, gt=0, description="Maximum daily loss amount")
    max_daily_loss_pct: Optional[float] = Field(None, gt=0, le=1, description="Maximum daily loss percentage")
    
    # Symbol restrictions
    allowed_symbols: Optional[List[str]] = Field(None, description="Allowed trading symbols")
    blocked_symbols: Optional[List[str]] = Field(None, description="Blocked trading symbols")


class PositionSizerConfig(BaseConfig):
    """Position sizing configuration."""
    
    name: str = Field(..., description="Position sizer name")
    type: Literal["fixed", "percentage", "volatility", "kelly", "atr"] = Field(
        ..., description="Position sizing algorithm"
    )
    
    # Type-specific parameters
    size: Optional[float] = Field(None, gt=0, description="Fixed position size")
    percentage: Optional[float] = Field(None, gt=0, le=1, description="Percentage of portfolio")
    risk_per_trade: Optional[float] = Field(None, gt=0, description="Risk per trade")
    lookback_period: Optional[int] = Field(None, gt=0, description="Lookback period for calculations")
    kelly_fraction: Optional[float] = Field(None, gt=0, le=1, description="Kelly criterion fraction")
    max_leverage: Optional[float] = Field(None, gt=0, description="Maximum leverage")
    risk_amount: Optional[float] = Field(None, gt=0, description="Risk amount per trade")
    atr_multiplier: Optional[float] = Field(None, gt=0, description="ATR multiplier")
    
    @field_validator('percentage', 'kelly_fraction')
    @classmethod
    def validate_percentages(cls, v):
        """Ensure percentages are reasonable."""
        if v and v > 0.5:
            logger.warning(f"Percentage {v} is greater than 50% - this seems high")
        return v


class RiskConfig(BaseConfig):
    """Risk management configuration."""
    
    position_sizers: List[PositionSizerConfig] = Field(
        default_factory=list, description="Position sizing algorithms"
    )
    limits: List[RiskLimitConfig] = Field(
        default_factory=list, description="Risk limits"
    )


class ExecutionConfig(BaseConfig):
    """Execution configuration."""
    
    type: Literal["simulated", "live", "paper"] = Field("simulated", description="Execution mode")
    enable_event_tracing: bool = Field(False, description="Enable event tracing")
    
    # Tracing configuration
    trace_settings: Optional[Dict[str, Any]] = Field(None, description="Event tracing settings")
    
    # Simulated execution settings
    slippage: float = Field(0.001, ge=0, description="Simulated slippage")
    commission: float = Field(0.005, ge=0, description="Simulated commission")
    latency_ms: int = Field(0, ge=0, description="Simulated latency in milliseconds")
    
    @field_validator('trace_settings')
    @classmethod
    def validate_trace_settings(cls, v):
        """Validate tracing settings when tracing is enabled."""
        if v and 'trace_dir' in v:
            trace_dir = Path(v['trace_dir'])
            if not trace_dir.parent.exists():
                logger.warning(f"Trace directory parent does not exist: {trace_dir.parent}")
        return v


class WorkflowConfig(BaseConfig):
    """Complete workflow configuration."""
    
    name: str = Field(..., description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    workflow_type: Literal["backtest", "optimization", "walk_forward", "live"] = Field(
        "backtest", description="Workflow type"
    )
    
    # Core configuration sections
    data: DataConfig = Field(..., description="Data configuration")
    portfolio: PortfolioConfig = Field(..., description="Portfolio configuration")
    strategies: List[StrategyConfig] = Field(..., min_items=1, description="Strategy configurations")
    risk: Optional[RiskConfig] = Field(None, description="Risk management configuration")
    execution: Optional[ExecutionConfig] = Field(None, description="Execution configuration")
    
    # Optional metadata
    tags: List[str] = Field(default_factory=list, description="Workflow tags")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    version: str = Field("1.0.0", description="Configuration version")
    
    @field_validator('strategies')
    @classmethod
    def validate_strategies_allocation(cls, v):
        """Validate strategy allocations sum to reasonable total."""
        total_allocation = sum(s.allocation for s in v if s.enabled)
        
        if total_allocation > 1.0:
            logger.warning(f"Total strategy allocation {total_allocation} exceeds 1.0")
        elif total_allocation < 0.1:
            logger.warning(f"Total strategy allocation {total_allocation} seems very low")
        
        return v
    
    @model_validator(mode='after')
    def validate_consistency(self):
        """Cross-field validation."""
        # Live workflows should not use simulated execution
        if self.workflow_type == 'live' and self.execution and self.execution.type == 'simulated':
            logger.warning("Live workflow using simulated execution - this may not be intended")
        
        return self


# Convenience functions for validation
def validate_workflow_dict(config_dict: Dict[str, Any]) -> WorkflowConfig:
    """Validate a workflow configuration dictionary."""
    try:
        return WorkflowConfig(**config_dict)
    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise


def get_validation_errors(config_dict: Dict[str, Any]) -> List[str]:
    """Get validation errors as list of strings."""
    try:
        WorkflowConfig(**config_dict)
        return []
    except ValidationError as e:
        return [f"{'.'.join(str(x) for x in error['loc'])}: {error['msg']}" for error in e.errors()]


def validate_partial_config(config_dict: Dict[str, Any], model_class: BaseModel) -> List[str]:
    """Validate partial configuration against specific model."""
    try:
        model_class(**config_dict)
        return []
    except ValidationError as e:
        return [f"{'.'.join(str(x) for x in error['loc'])}: {error['msg']}" for error in e.errors()]


# Schema documentation generation
def generate_schema_docs() -> str:
    """Generate markdown documentation for all schemas."""
    docs = []
    docs.append("# Configuration Schema Reference\n")
    docs.append("Auto-generated from Pydantic models.\n")
    
    models = [
        ("Workflow", WorkflowConfig),
        ("Data", DataConfig), 
        ("Portfolio", PortfolioConfig),
        ("Strategy", StrategyConfig),
        ("Risk Management", RiskConfig),
        ("Risk Limits", RiskLimitConfig),
        ("Position Sizing", PositionSizerConfig),
        ("Execution", ExecutionConfig)
    ]
    
    for name, model in models:
        docs.append(f"## {name} Configuration\n")
        
        # Get field descriptions
        if hasattr(model, 'model_fields'):
            # Pydantic v2
            for field_name, field_info in model.model_fields.items():
                field_type = field_info.annotation
                description = field_info.description or "No description"
                required = field_info.is_required()
                default = field_info.default if not required else "Required"
                
                type_name = getattr(field_type, '__name__', str(field_type))
                docs.append(f"- **{field_name}** (`{type_name}`) - {description}")
                if not required:
                    docs.append(f"  - Default: `{default}`")
        else:
            # Fallback for when Pydantic is not available
            docs.append("- Configuration validation unavailable (install pydantic>=2.0.0)")
        
        docs.append("")
    
    return "\n".join(docs)


def get_example_config() -> Dict[str, Any]:
    """Get a complete example configuration."""
    return {
        "name": "Example Momentum Strategy",
        "description": "Simple momentum strategy backtest",
        "workflow_type": "backtest",
        "data": {
            "symbols": ["SPY", "QQQ"],
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "frequency": "1d",
            "source": "csv",
            "file_path": "./data/SPY.csv"
        },
        "portfolio": {
            "initial_capital": 100000,
            "currency": "USD",
            "commission_type": "percentage",
            "commission_value": 0.001
        },
        "strategies": [
            {
                "name": "momentum_1",
                "type": "momentum",
                "allocation": 0.8,
                "parameters": {
                    "lookback_period": 20,
                    "threshold": 0.02
                }
            }
        ],
        "risk": {
            "limits": [
                {
                    "type": "position",
                    "max_position_pct": 0.1
                }
            ]
        },
        "execution": {
            "type": "simulated",
            "enable_event_tracing": False
        }
    }