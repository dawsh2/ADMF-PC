#!/usr/bin/env python3
"""
Test portfolio receiving and tracing signals.

This tests the simplified approach where the portfolio's tracer
captures incoming events directly without republishing.
"""

import logging
from pathlib import Path
import json
from typing import Dict, Any, Optional

from src.core.events import EventBus, Event, EventType
from src.core.containers import Container, ContainerConfig
from src.core.coordinator import TopologyBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Test portfolio signal tracing."""
    logger.info("=== Testing Portfolio Signal Tracing ===")
    
    # Create root container
    root_config = ContainerConfig(
        name="root",
        components=[],
        config={}
    )
    root = Container(root_config)
    
    # Create data container and add components directly
    data_config = ContainerConfig(
        name="data_container", 
        components=[],  # We'll add manually
        config={
            "symbol": "SPY",
            "data_dir": "./data",
            "max_bars": 20  # Small test
        }
    )
    data_container = Container(data_config)
    root.add_child_container(data_container)
    
    # Add data handler component
    from src.data.handlers import SimpleHistoricalDataHandler
    data_handler = SimpleHistoricalDataHandler(
        handler_id="data_SPY",
        data_dir="./data"
    )
    data_handler.load_data(["SPY"])
    data_handler.max_bars = 20
    data_container.add_component("data_streamer", data_handler)
    
    # Create strategy container and add components directly
    strategy_config = ContainerConfig(
        name="strategy_container",
        components=[],  # We'll add manually
        config={
            "strategies": [{
                "name": "momentum_1",
                "type": "simple_momentum",
                "params": {
                    "sma_fast": 5,
                    "sma_slow": 10
                }
            }],
            "feature_configs": {
                "sma_fast": {"feature": "sma", "period": 5},
                "sma_slow": {"feature": "sma", "period": 10}
            },
            "symbols": ["SPY"]
        }
    )
    strategy_container = Container(strategy_config)
    root.add_child_container(strategy_container)
    
    # Add strategy state component
    from src.strategy.state import StrategyState
    
    # Create a simple test strategy that only uses SMAs
    def test_sma_strategy(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Simple test strategy using only SMAs."""
        sma_fast = features.get('sma_fast')
        sma_slow = features.get('sma_slow')
        price = bar.get('close', 0)
        symbol = bar.get('symbol', 'UNKNOWN')
        
        if sma_fast is None or sma_slow is None:
            return None
        
        # Simple crossover logic
        if sma_fast > sma_slow:
            return {
                'symbol': symbol,
                'direction': 'long',
                'strength': min(1.0, (sma_fast - sma_slow) / sma_slow),
                'price': price,
                'signal_type': 'entry',
                'reason': 'Fast SMA above slow SMA'
            }
        else:
            return {
                'symbol': symbol,
                'direction': 'flat',
                'strength': 1.0,
                'price': price,
                'signal_type': 'exit',
                'reason': 'Fast SMA below slow SMA'
            }
    
    # First, add the strategy function to container config BEFORE creating StrategyState
    strategy_container.config.config['stateless_components'] = {
        'strategies': {
            'simple_momentum': test_sma_strategy  # Using our test strategy
        }
    }
    
    strategy_state = StrategyState(
        symbols=["SPY"],
        feature_configs=strategy_config.config["feature_configs"]
    )
    strategy_container.add_component("strategy_state", strategy_state)
    
    # Create portfolio container with tracing enabled
    portfolio_config = ContainerConfig(
        name="portfolio_container",
        components=[],  # We'll add manually
        config={
            "initial_capital": 100000,
            "managed_strategies": ["momentum_1"],
            # Enable event tracing
            "execution": {
                "enable_event_tracing": True,
                "trace_settings": {
                    "trace_id": "portfolio_trace",
                    "max_events": 1000,
                    "storage_backend": "memory"
                }
            }
        }
    )
    portfolio_container = Container(portfolio_config)
    root.add_child_container(portfolio_container)
    
    # Add portfolio state component
    from src.portfolio import PortfolioState
    portfolio_state = PortfolioState(
        initial_capital=100000
    )
    portfolio_container.add_component("portfolio_manager", portfolio_state)
    
    # Enable tracing on portfolio container
    portfolio_container._setup_tracing()
    
    # Subscribe portfolio to SIGNAL events on root bus
    # The portfolio will trace these when they arrive via receive_event
    portfolio_manager = portfolio_container.get_component("portfolio_manager")
    if portfolio_manager and hasattr(portfolio_manager, 'process_event'):
        # Subscribe to root bus (where strategies publish)
        root.event_bus.subscribe(
            EventType.SIGNAL.value,
            portfolio_container.receive_event,  # Container receives and traces
            filter_func=lambda e: e.payload.get('strategy_id') == 'momentum_1'
        )
        logger.info("Portfolio subscribed to SIGNAL events on root bus")
    
    # Initialize containers
    root.initialize()
    data_container.initialize()
    strategy_container.initialize()
    portfolio_container.initialize()
    
    # Start containers
    root.start()
    data_container.start()
    strategy_container.start()
    portfolio_container.start()
    
    # Run data streaming
    logger.info("\n=== Streaming Data ===")
    data_container.execute()
    
    # Check traced events
    logger.info("\n=== Checking Traced Events ===")
    if hasattr(portfolio_container.event_bus, '_tracer'):
        tracer = portfolio_container.event_bus._tracer
        summary = tracer.get_summary()
        logger.info(f"Tracer summary: {summary}")
        
        # Get traced events
        events = list(tracer.recent_events)
        logger.info(f"Total events traced: {len(events)}")
        
        # Count by type
        event_types = {}
        signal_count = 0
        for event in events:
            event_type = event.event_type
            event_types[event_type] = event_types.get(event_type, 0) + 1
            if event_type == 'SIGNAL':
                signal_count += 1
                # Show first few signals
                if signal_count <= 3:
                    logger.info(f"Signal {signal_count}: {event.payload}")
        
        logger.info(f"Event types traced: {event_types}")
        
        # Verify we captured incoming signals
        if 'SIGNAL' in event_types:
            logger.info(f"✅ Successfully traced {event_types['SIGNAL']} incoming signals!")
        else:
            logger.warning("❌ No signals were traced")
    else:
        logger.warning("❌ Portfolio tracer not found")
    
    # Cleanup
    portfolio_container.stop()
    strategy_container.stop()
    data_container.stop()
    root.stop()
    
    portfolio_container.cleanup()
    strategy_container.cleanup()
    data_container.cleanup()
    root.cleanup()
    
    logger.info("\n✅ Test complete!")

if __name__ == "__main__":
    main()