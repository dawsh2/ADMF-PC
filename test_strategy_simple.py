#!/usr/bin/env python3
"""
Simple test of strategy state with features.
"""

import logging
from src.core.containers.container import Container, ContainerConfig
from src.data.handlers import SimpleHistoricalDataHandler
from src.strategy.state import StrategyState
from src.core.events.types import EventType

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def simple_momentum_strategy(features, bar, params):
    """Simple momentum strategy function."""
    sma_fast = features.get('sma_fast')
    sma_slow = features.get('sma_slow')
    
    if sma_fast is None or sma_slow is None:
        return None
    
    # Simple crossover logic
    if sma_fast > sma_slow:
        return {
            'signal_type': 'entry',
            'direction': 'long',
            'strength': min((sma_fast - sma_slow) / sma_slow, 1.0),
            'metadata': {'reason': 'fast_above_slow'}
        }
    else:
        return {
            'signal_type': 'exit',
            'direction': 'flat',
            'strength': 1.0,
            'metadata': {'reason': 'fast_below_slow'}
        }

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
            "max_bars": 50  # Need enough for features
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
    data_handler.max_bars = 50
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
            "sma_slow": {"feature": "sma", "period": 20}
        }
    )
    strategy_container.add_component("strategy_state", strategy_state)
    
    # Add our simple momentum strategy
    strategy_state.add_strategy("momentum_1", simple_momentum_strategy)
    
    # Subscribe to events for debugging
    signal_count = [0]
    
    def log_bar(event):
        logger.info(f"BAR: {event.payload.get('symbol')} - Close: {event.payload.get('bar', {}).close}")
    
    def log_signal(event):
        signal_count[0] += 1
        payload = event.payload
        logger.info(f"SIGNAL #{signal_count[0]}: {payload.get('direction')} {payload.get('symbol')} "
                   f"strength={payload.get('strength', 0):.2f} from {payload.get('strategy_id')}")
    
    root_container.event_bus.subscribe(EventType.BAR.value, log_bar)
    # SIGNAL events require a filter - accept all signals for debugging
    root_container.event_bus.subscribe(
        EventType.SIGNAL.value, 
        log_signal,
        filter_func=lambda e: True  # Accept all signals for debugging
    )
    
    # Initialize and start containers
    logger.info("Initializing containers...")
    root_container.initialize()
    data_container.initialize()
    strategy_container.initialize()
    
    logger.info("Starting containers...")
    root_container.start()
    data_container.start()
    strategy_container.start()
    
    # Execute containers - this will stream data
    logger.info("Executing containers (streaming 50 bars)...")
    data_container.execute()  # This will stream all bars
    
    # Check metrics
    metrics = strategy_state.get_metrics()
    logger.info(f"\nStrategy metrics:")
    logger.info(f"  Bars processed: {metrics['bars_processed']}")
    logger.info(f"  Signals generated: {metrics['signals_generated']}")
    logger.info(f"  Features computed: {metrics['feature_summary']['configured_features']}")
    
    # Cleanup
    logger.info("\nCleaning up...")
    strategy_container.stop()
    data_container.stop()
    root_container.stop()
    
    strategy_container.cleanup()
    data_container.cleanup()
    root_container.cleanup()
    
    logger.info("Test complete!")

if __name__ == "__main__":
    main()