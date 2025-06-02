"""
File: src/core/events/event_flow.py
Status: ACTIVE
Architecture Ref: SYSTEM_ARCHITECTURE_v5.md#event-flow
Step: 1 - Core Pipeline Test
Dependencies: core.events, data.models, strategy, risk, execution

Event flow setup for Step 1 core pipeline validation.
Creates complete event-driven architecture with feedback loops.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from datetime import datetime

from .enhanced_isolation import get_enhanced_isolation_manager
from ..logging.structured import ContainerLogger
from ...data.models import Bar
from ...strategy.indicators import SimpleMovingAverage
from ...strategy.strategies.simple_trend import SimpleTrendStrategy


class DataSource:
    """
    Simple data source for Step 1 testing.
    
    Emits market data events to kick off the pipeline.
    """
    
    def __init__(self, container_id: str):
        self.container_id = container_id
        self.event_bus = None  # Injected by setup
        self.logger = ContainerLogger("DataSource", container_id, "data_source")
    
    def emit_bar(self, bar: Bar) -> None:
        """Emit a market data bar event."""
        self.logger.log_event_flow(
            "BAR_EVENT",
            "data_source",
            "pipeline",
            f"{bar.symbol} {bar.close}"
        )
        
        if self.event_bus:
            self.event_bus.publish("BAR", bar)


class MockRiskManager:
    """
    Mock risk manager for Step 1 testing.
    
    Transforms signals into orders and handles fill events.
    """
    
    def __init__(self, container_id: str):
        self.container_id = container_id
        self.event_bus = None  # Injected by setup
        self.logger = ContainerLogger("RiskManager", container_id, "risk_manager")
        self.processed_signals = []
        self.portfolio_updates = []
    
    def on_signal(self, signal) -> None:
        """Process trading signal and create order."""
        self.logger.log_event_flow(
            "ORDER_EVENT",
            "risk_manager",
            "execution",
            f"{signal.side.value} {signal.symbol}"
        )
        
        self.processed_signals.append(signal)
        
        # Create mock order
        order = {
            'order_id': f"order_{len(self.processed_signals)}",
            'signal_id': signal.signal_id,
            'symbol': signal.symbol,
            'side': signal.side.value,
            'quantity': 100,  # Fixed for testing
            'order_type': 'MARKET',
            'timestamp': signal.timestamp
        }
        
        if self.event_bus:
            self.event_bus.publish("ORDER", order)
    
    def on_fill(self, fill) -> None:
        """Process fill event and update portfolio."""
        self.logger.log_event_flow(
            "PORTFOLIO_UPDATE",
            "risk_manager",
            "portfolio",
            f"Fill {fill['symbol']} {fill['quantity']}"
        )
        
        # Create portfolio update
        portfolio_update = {
            'fill_id': fill['fill_id'],
            'symbol': fill['symbol'],
            'quantity': fill['quantity'],
            'price': fill['price'],
            'timestamp': fill['timestamp'],
            'portfolio_value': 100000 + len(self.portfolio_updates) * 100  # Mock value
        }
        
        self.portfolio_updates.append(portfolio_update)
        
        if self.event_bus:
            self.event_bus.publish("PORTFOLIO_UPDATE", portfolio_update)


class MockExecutionEngine:
    """
    Mock execution engine for Step 1 testing.
    
    Processes orders and generates fill events.
    """
    
    def __init__(self, container_id: str):
        self.container_id = container_id
        self.event_bus = None  # Injected by setup
        self.logger = ContainerLogger("ExecutionEngine", container_id, "execution_engine")
        self.processed_orders = []
    
    def on_order(self, order) -> None:
        """Process order and generate fill."""
        self.logger.log_event_flow(
            "FILL_EVENT",
            "execution",
            "risk_manager",
            f"Fill {order['symbol']} {order['quantity']}"
        )
        
        self.processed_orders.append(order)
        
        # Create mock fill (immediate execution for testing)
        fill = {
            'fill_id': f"fill_{len(self.processed_orders)}",
            'order_id': order['order_id'],
            'symbol': order['symbol'],
            'side': order['side'],
            'quantity': order['quantity'],
            'price': 100.0 + len(self.processed_orders),  # Mock price
            'timestamp': datetime.now(),
            'execution_fees': 1.0
        }
        
        if self.event_bus:
            self.event_bus.publish("FILL", fill)


class MockPortfolioState:
    """
    Mock portfolio state for Step 1 testing.
    
    Tracks portfolio updates from fills.
    """
    
    def __init__(self, container_id: str):
        self.container_id = container_id
        self.logger = ContainerLogger("PortfolioState", container_id, "portfolio_state")
        self.updates = []
    
    def update(self, portfolio_update) -> None:
        """Update portfolio state."""
        self.logger.info(
            "Portfolio updated",
            symbol=portfolio_update['symbol'],
            quantity=portfolio_update['quantity'],
            portfolio_value=portfolio_update['portfolio_value']
        )
        
        self.updates.append(portfolio_update)


def setup_core_pipeline(container_id: str) -> Dict[str, Any]:
    """
    Wire up the core event pipeline with complete cycle.
    
    This function creates all components and sets up the event flow
    as specified in Step 1 requirements.
    
    Args:
        container_id: Unique container ID for isolation
        
    Returns:
        Dictionary containing event bus and all components
        
    Architecture Context:
        - Part of: Core Pipeline Test (step-01-core-pipeline.md)
        - Creates: Complete event-driven pipeline with feedback loops
        - Enables: End-to-end validation of event flow
        - Supports: Container isolation testing
    """
    logger = ContainerLogger("EventFlow", container_id, "event_flow_setup")
    logger.info("Setting up core pipeline", container_id=container_id)
    
    # Create isolated event bus
    isolation_manager = get_enhanced_isolation_manager()
    event_bus = isolation_manager.create_container_bus(container_id)
    
    # Create components
    data_source = DataSource(container_id)
    indicator = SimpleMovingAverage(20, container_id)
    strategy = SimpleTrendStrategy(10, 20, container_id)
    risk_manager = MockRiskManager(container_id)
    execution = MockExecutionEngine(container_id)
    portfolio_state = MockPortfolioState(container_id)
    
    # Wire up event flow (including feedback loop)
    event_bus.subscribe("BAR", indicator.on_bar)
    event_bus.subscribe("BAR", strategy.on_bar)
    event_bus.subscribe("SIGNAL", risk_manager.on_signal)
    event_bus.subscribe("ORDER", execution.on_order)
    event_bus.subscribe("FILL", risk_manager.on_fill)  # Critical feedback
    event_bus.subscribe("PORTFOLIO_UPDATE", portfolio_state.update)
    
    # Inject event bus into components
    data_source.event_bus = event_bus
    strategy.event_bus = event_bus
    risk_manager.event_bus = event_bus
    execution.event_bus = event_bus  # Execution needs to emit fills
    
    logger.info(
        "Core pipeline setup complete",
        components=6,
        event_subscriptions=6
    )
    
    return {
        'event_bus': event_bus,
        'components': {
            'data_source': data_source,
            'indicator': indicator,
            'strategy': strategy,
            'risk_manager': risk_manager,
            'execution': execution,
            'portfolio_state': portfolio_state
        },
        'container_id': container_id
    }


def run_simple_pipeline_test(container_id: str, test_bars: list) -> Dict[str, Any]:
    """
    Run a simple pipeline test with provided bar data.
    
    Args:
        container_id: Container ID for isolation
        test_bars: List of Bar objects to process
        
    Returns:
        Test results and pipeline state
    """
    logger = ContainerLogger("PipelineTest", container_id, "pipeline_test")
    logger.info("Running simple pipeline test", bar_count=len(test_bars))
    
    # Setup pipeline
    pipeline = setup_core_pipeline(container_id)
    data_source = pipeline['components']['data_source']
    
    # Process bars
    for i, bar in enumerate(test_bars):
        logger.trace(f"Processing bar {i+1}/{len(test_bars)}", symbol=bar.symbol)
        data_source.emit_bar(bar)
    
    # Collect results
    risk_manager = pipeline['components']['risk_manager']
    execution = pipeline['components']['execution']
    portfolio_state = pipeline['components']['portfolio_state']
    
    results = {
        'container_id': container_id,
        'bars_processed': len(test_bars),
        'signals_generated': len(risk_manager.processed_signals),
        'orders_created': len(execution.processed_orders),
        'portfolio_updates': len(portfolio_state.updates),
        'final_portfolio_value': portfolio_state.updates[-1]['portfolio_value'] if portfolio_state.updates else 100000
    }
    
    logger.info("Pipeline test complete", **results)
    
    # Cleanup
    isolation_manager = get_enhanced_isolation_manager()
    isolation_manager.remove_container_bus(container_id)
    
    return results