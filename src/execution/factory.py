"""
Factory functions for creating execution engines.

Provides easy-to-use factory functions for both sync and async execution.
"""

from typing import Dict, Any, Optional
from decimal import Decimal

from .types import ExecutionStats
from .synchronous import (
    SyncExecutionEngine,
    SimulatedBroker,
    SyncOrderManager,
    PercentageSlippageModel,
    ZeroCommissionModel,
    TieredCommissionModel,
    UnlimitedLiquidityModel
)
# Async components - optional, requires aiohttp
try:
    from .asynchronous import (
        AsyncExecutionEngine,
        AsyncOrderManager
    )
    from .asynchronous.brokers import AlpacaBroker
    _ASYNC_AVAILABLE = True
except ImportError:
    # Async components not available (missing dependencies)
    _ASYNC_AVAILABLE = False


def create_sync_engine(
    component_id: str,
    portfolio_state = None,
    slippage_pct: float = 0.001,
    commission_model: str = "zero",
    commission_params: Optional[Dict[str, Any]] = None
) -> SyncExecutionEngine:
    """
    Create a synchronous execution engine.
    
    Args:
        component_id: Unique identifier for the engine
        portfolio_state: Portfolio state manager (optional)
        slippage_pct: Base slippage percentage (default 0.1%)
        commission_model: Commission model type ("zero", "percentage", "tiered")
        commission_params: Parameters for commission model
        
    Returns:
        Configured SyncExecutionEngine
    """
    # Create slippage model
    slippage_model = PercentageSlippageModel(base_slippage_pct=slippage_pct)
    
    # Create commission model
    commission_params = commission_params or {}
    
    if commission_model == "zero":
        commission = ZeroCommissionModel()
    elif commission_model == "percentage":
        commission = ZeroCommissionModel()  # Use percentage model if implemented
    elif commission_model == "tiered":
        commission = TieredCommissionModel(**commission_params)
    else:
        commission = ZeroCommissionModel()
    
    # Create liquidity model
    liquidity_model = UnlimitedLiquidityModel()
    
    # Create broker
    broker = SimulatedBroker(
        broker_id=f"{component_id}_broker",
        slippage_model=slippage_model,
        commission_model=commission,
        liquidity_model=liquidity_model,
        portfolio_state=portfolio_state
    )
    
    # Create order manager
    order_manager = SyncOrderManager(f"{component_id}_orders")
    
    # Create execution engine
    engine = SyncExecutionEngine(
        component_id=component_id,
        broker=broker,
        order_manager=order_manager
    )
    
    return engine


if _ASYNC_AVAILABLE:
    def create_alpaca_engine(
        component_id: str,
        api_key: str,
        secret_key: str,
        paper_trading: bool = True
    ) -> AsyncExecutionEngine:
        """
        Create an asynchronous Alpaca trading engine.
        
        Args:
            component_id: Unique identifier for the engine
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper_trading: Use paper trading (default True)
            
        Returns:
            Configured AsyncExecutionEngine with Alpaca broker
        """
        # Create Alpaca broker
        broker = AlpacaBroker(
            api_key=api_key,
            secret_key=secret_key,
            paper_trading=paper_trading
        )
        
        # Create order manager
        order_manager = AsyncOrderManager(f"{component_id}_orders")
        
        # Create execution engine
        engine = AsyncExecutionEngine(
            component_id=component_id,
            broker=broker,
            order_manager=order_manager
        )
        
        return engine


def create_execution_engine(
    component_id: str,
    mode: str = "backtest",
    **kwargs
) -> SyncExecutionEngine:
    """
    Create execution engine based on mode.
    
    Args:
        component_id: Unique identifier for the engine
        mode: Execution mode ("backtest" or "sync")
        **kwargs: Mode-specific parameters
        
    Returns:
        Configured execution engine
    """
    if mode == "backtest" or mode == "sync":
        return create_sync_engine(component_id, **kwargs)
    elif mode == "live" or mode == "async":
        if not _ASYNC_AVAILABLE:
            raise ImportError("Async execution requires aiohttp. Install with: pip install aiohttp")
        
        broker_type = kwargs.get("broker_type", "alpaca")
        
        if broker_type == "alpaca":
            return create_alpaca_engine(component_id, **kwargs)
        else:
            raise ValueError(f"Unsupported broker type: {broker_type}")
    else:
        raise ValueError(f"Unsupported execution mode: {mode}. Use 'sync'/'backtest' or 'async'/'live'.")


# Pre-configured factory functions for common scenarios
def create_zero_cost_sync(component_id: str, portfolio_state = None) -> SyncExecutionEngine:
    """Create sync engine with zero slippage and commission."""
    return create_sync_engine(
        component_id=component_id,
        portfolio_state=portfolio_state,
        slippage_pct=0.0,
        commission_model="zero"
    )


def create_realistic_sync(component_id: str, portfolio_state = None) -> SyncExecutionEngine:
    """Create sync engine with realistic costs."""
    return create_sync_engine(
        component_id=component_id,
        portfolio_state=portfolio_state,
        slippage_pct=0.001,  # 0.1% slippage
        commission_model="tiered",
        commission_params={
            "tiers": [
                (0.0, 0.005),      # $0+: 0.5%
                (10000.0, 0.003),  # $10k+: 0.3%
                (100000.0, 0.001)  # $100k+: 0.1%
            ],
            "minimum_commission": 1.0
        }
    )


if _ASYNC_AVAILABLE:
    def create_paper_trading_engine(
        component_id: str,
        api_key: str,
        secret_key: str
    ) -> AsyncExecutionEngine:
        """Create Alpaca paper trading engine."""
        return create_alpaca_engine(
            component_id=component_id,
            api_key=api_key,
            secret_key=secret_key,
            paper_trading=True
        )


    def create_live_trading_engine(
        component_id: str,
        api_key: str,
        secret_key: str
    ) -> AsyncExecutionEngine:
        """Create Alpaca live trading engine."""
        return create_alpaca_engine(
            component_id=component_id,
            api_key=api_key,
            secret_key=secret_key,
            paper_trading=False
        )


# Backward compatibility aliases
create_backtest_engine = create_sync_engine
create_zero_cost_backtest = create_zero_cost_sync
create_realistic_backtest = create_realistic_sync