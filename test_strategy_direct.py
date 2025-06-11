#!/usr/bin/env python3
"""
Direct test of strategy state without topology patterns.
"""

import logging
from src.core.containers.container import Container, ContainerConfig
from src.data.handlers import SimpleHistoricalDataHandler
from src.strategy.state import StrategyState
from src.strategy.strategies.momentum import MomentumStrategy
from src.core.events.types import EventType

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    # Create root container for event propagation
    root_config = ContainerConfig(
        name="root",
        components=[],
        config={},
        container_type="root"
    )
    root_container = Container(root_config)
    
    # Create data container as child of root
    data_config = ContainerConfig(
        name="data_container",
        components=["data_streamer"],
        config={
            "symbol": "SPY",
            "data_dir": "./data",
            "max_bars": 5
        },
        container_type="data"
    )
    data_container = Container(data_config)
    root_container.add_child_container(data_container)
    
    # Create data handler component
    data_handler = SimpleHistoricalDataHandler(
        handler_id="data_SPY",
        data_dir="./data"
    )
    data_handler.load_data(["SPY"])
    data_handler.max_bars = 5
    data_container.add_component("data_streamer", data_handler)
    
    # Create strategy container as child of root
    strategy_config = ContainerConfig(
        name="strategy_container",
        components=["strategy_state"],
        config={
            "symbols": ["SPY"],
            "features": {
                "sma_fast": {
                    "feature": "sma",
                    "period": 10
                },
                "sma_slow": {
                    "feature": "sma",
                    "period": 20
                },
                "rsi": {
                    "feature": "rsi",
                    "period": 14
                }
            }
        },
        container_type="strategy"
    )
    strategy_container = Container(strategy_config)
    root_container.add_child_container(strategy_container)
    
    # Create strategy state component
    strategy_state = StrategyState(
        symbols=["SPY"],
        feature_configs={
            "sma_fast": {"feature": "sma", "period": 10},
            "sma_slow": {"feature": "sma", "period": 20},
            "rsi": {"feature": "rsi", "period": 14}
        }
    )
    strategy_container.add_component("strategy_state", strategy_state)
    
    # Add a momentum strategy to the state
    momentum_strategy = MomentumStrategy(
        strategy_id="momentum_1",
        params={"fast_period": 10, "slow_period": 20}
    )
    strategy_state.add_strategy("momentum_1", momentum_strategy.generate_signal)
    
    # Subscribe to events for debugging
    def log_event(event):
        logger.info(f"Event: {event.event_type} - {event.payload.get('symbol', 'N/A')}")
    
    root_container.event_bus.subscribe(EventType.BAR.value, log_event)
    root_container.event_bus.subscribe(EventType.SIGNAL.value, log_event)
    
    # Initialize and start containers
    logger.info("Initializing containers...")
    root_container.initialize()
    data_container.initialize()
    strategy_container.initialize()
    
    logger.info("Starting containers...")
    root_container.start()
    data_container.start()
    strategy_container.start()
    
    # Stream data
    logger.info("Streaming 5 bars of data...")
    for i in range(5):
        logger.info(f"Streaming bar {i+1}")
        data_handler.stream_next()
    
    # Check metrics
    metrics = strategy_state.get_metrics()
    logger.info(f"Strategy metrics: {metrics}")
    
    # Cleanup
    logger.info("Cleaning up...")
    strategy_container.stop()
    data_container.stop()
    root_container.stop()
    
    strategy_container.cleanup()
    data_container.cleanup()
    root_container.cleanup()
    
    logger.info("Test complete!")

if __name__ == "__main__":
    main()