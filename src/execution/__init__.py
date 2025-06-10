"""
Execution module with separated synchronous and asynchronous trading components.

This module provides:
- Synchronous execution for high-performance simulation
- Asynchronous execution for real broker integration
- Shared types and protocols
- Factory functions for easy creation

Architecture:
- synchronous/ - Synchronous components for simulation
- asynchronous/ - Asynchronous components for real trading
- types.py - Common data structures
- factory.py - Easy creation functions
"""

# Core types and protocols
from .types import (
    Order, Fill, Position, Bar, ExecutionStats,
    OrderType, OrderSide, OrderStatus, FillStatus
)
from .sync_protocols import (
    SyncBroker, SyncEngine,
    SlippageModel, CommissionModel, LiquidityModel
)
from .async_protocols import (
    AsyncBroker, AsyncEngine, MarketDataFeed, OrderMonitor
)

# Sync components (synchronous)
from .synchronous import (
    SyncExecutionEngine,
    SimulatedBroker,
    SyncOrderManager,
    PercentageSlippageModel,
    TieredCommissionModel,
    ZeroCommissionModel,
    UnlimitedLiquidityModel
)

# Async components (asynchronous) - optional, requires aiohttp
try:
    from .asynchronous import (
        AsyncExecutionEngine,
        AsyncOrderManager,
        BrokerConfig,
        RateLimiter,
        CacheManager,
        AsyncMarketDataFeed
    )
    from .asynchronous.brokers import AlpacaBroker
    _ASYNC_AVAILABLE = True
except ImportError:
    # Async components not available (missing dependencies)
    _ASYNC_AVAILABLE = False

# Factory functions
from .factory import (
    create_sync_engine,
    # create_alpaca_engine,  # Temporarily disabled
    create_execution_engine,
    create_zero_cost_sync,
    create_realistic_sync,
    # create_paper_trading_engine,  # Temporarily disabled
    # create_live_trading_engine    # Temporarily disabled
)

# Keep calc functions from original module
from .calc import (
    ensure_decimal,
    round_price,
    round_quantity,
    calculate_value,
    calculate_commission,
    calculate_slippage,
    calculate_pnl,
    calculate_return_pct,
    safe_divide,
    format_currency,
    format_percentage,
    DecimalEncoder,
    validate_price,
    validate_quantity,
    validate_percentage
)

__all__ = [
    # Core types
    "Order", "Fill", "Position", "Bar", "ExecutionStats",
    "OrderType", "OrderSide", "OrderStatus", "FillStatus",
    
    # Protocols
    "SyncBroker", "SyncEngine", "AsyncBroker", "AsyncEngine",
    "SlippageModel", "CommissionModel", "LiquidityModel",
    "MarketDataFeed", "OrderMonitor",
    
    # Sync components (synchronous)
    "SyncExecutionEngine", "SimulatedBroker", "SyncOrderManager",
    "PercentageSlippageModel", "TieredCommissionModel", "ZeroCommissionModel",
    "UnlimitedLiquidityModel",
    
    # Async components (asynchronous) - available if dependencies installed
    "AsyncExecutionEngine", "AsyncOrderManager", "BrokerConfig",
    "RateLimiter", "CacheManager", "AsyncMarketDataFeed", "AlpacaBroker",
    
    # Factory functions
    "create_sync_engine", "create_execution_engine",
    "create_zero_cost_sync", "create_realistic_sync",
    # "create_alpaca_engine", "create_paper_trading_engine", "create_live_trading_engine",  # Temporarily disabled
    
    # Financial calculations
    "ensure_decimal", "round_price", "round_quantity", "calculate_value",
    "calculate_commission", "calculate_slippage", "calculate_pnl",
    "calculate_return_pct", "safe_divide", "format_currency", 
    "format_percentage", "DecimalEncoder", "validate_price",
    "validate_quantity", "validate_percentage"
]


# Backward compatibility aliases for existing code
DefaultExecutionEngine = SyncExecutionEngine
BacktestExecutionEngine = SyncExecutionEngine  # Backward compatibility
ExecutionEngine = SyncExecutionEngine          # Default to sync
Broker = SyncBroker
# Additional backward compatibility
BacktestOrderManager = SyncOrderManager        # Backward compatibility
if _ASYNC_AVAILABLE:
    LiveOrderManager = AsyncOrderManager       # Backward compatibility
    LiveMarketDataFeed = AsyncMarketDataFeed   # Backward compatibility
    LiveExecutionEngine = AsyncExecutionEngine # Backward compatibility
else:
    # Provide None for async components when not available
    AsyncExecutionEngine = None
    AsyncOrderManager = None
    AsyncMarketDataFeed = None
    AlpacaBroker = None
    LiveOrderManager = None
    LiveMarketDataFeed = None
    LiveExecutionEngine = None
    
create_backtest_engine = create_sync_engine    # Backward compatibility
create_zero_cost_backtest = create_zero_cost_sync  # Backward compatibility
create_realistic_backtest = create_realistic_sync  # Backward compatibility