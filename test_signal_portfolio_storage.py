#!/usr/bin/env python3
"""
Test that signals are properly stored in portfolio containers.

This tests the full flow:
1. Data streaming -> Strategy generation -> Portfolio reception -> Storage
"""

import logging
import json
from pathlib import Path
from src.core.containers.container import Container, ContainerConfig
from src.core.events.bus import EventBus
from src.core.events.types import Event, EventType
from src.data.handlers import SimpleHistoricalDataHandler
from src.strategy.state import StrategyState
from src.portfolio.state import PortfolioState
from src.strategy.strategies.ma_crossover import ma_crossover_strategy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_signal_storage():
    """Test signal generation and storage in portfolio containers."""
    
    # Create root container with shared event bus
    root_config = ContainerConfig(
        name="root",
        container_id="root_test",
        config={
            'execution': {
                'enable_event_tracing': True,
                'trace_settings': {
                    'storage_backend': 'hierarchical',
                    'enable_console_output': True,
                    'console_filter': ['SIGNAL'],
                    'container_settings': {
                        '*': {'enabled': True, 'max_events': 1000}
                    }
                }
            },
            'metadata': {
                'workflow_id': 'test_signal_storage'
            }
        }
    )
    root_container = Container(root_config, parent_event_bus=None)
    
    # Create strategy container as child
    strategy_config = ContainerConfig(
        name="strategy",
        container_id="strategy_test",
        components=['strategy_state'],
        config={
            'symbols': ['SPY'],
            'feature_configs': {
                'sma_5': {'feature': 'sma', 'period': 5},
                'sma_10': {'feature': 'sma', 'period': 10}
            },
            'strategies': [{
                'name': 'ma_crossover',
                'type': 'ma_crossover',
                'params': {'fast_period': 5, 'slow_period': 10}
            }],
            'stateless_components': {
                'strategies': {
                    'ma_crossover': ma_crossover_strategy
                }
            }
        }
    )
    strategy_container = root_container.create_child(strategy_config)
    
    # Add strategy state component
    strategy_state = StrategyState(
        symbols=['SPY'],
        feature_configs={
            'sma_5': {'feature': 'sma', 'period': 5},
            'sma_10': {'feature': 'sma', 'period': 10}
        }
    )
    strategy_container.add_component('strategy_state', strategy_state)
    
    # Create portfolio container as child
    portfolio_config = ContainerConfig(
        name="portfolio",
        container_id="portfolio_test",
        components=['portfolio_manager'],
        config={
            'initial_capital': 100000,
            'managed_strategies': ['ma_crossover'],
            'execution': root_config.config['execution'],  # Pass execution config to child
            'metadata': root_config.config['metadata']    # Pass metadata to child
        }
    )
    portfolio_container = root_container.create_child(portfolio_config)
    
    # Add portfolio manager
    portfolio_manager = PortfolioState()
    portfolio_container.add_component('portfolio_manager', portfolio_manager)
    
    # Create data container as child
    data_config = ContainerConfig(
        name="data",
        container_id="data_test",
        components=['data_streamer'],
        config={
            'symbol': 'SPY',
            'max_bars': 15  # Just enough to generate signals
        }
    )
    data_container = root_container.create_child(data_config)
    
    # Add data handler
    data_handler = SimpleHistoricalDataHandler()
    data_handler.load_data(['SPY'])
    data_handler.max_bars = 15
    data_container.add_component('data_streamer', data_handler)
    
    # Initialize and start all containers
    for container in [root_container, data_container, strategy_container, portfolio_container]:
        container.initialize()
        container.start()
    
    # Set up portfolio subscription to signals
    def signal_filter(event):
        """Filter signals for this portfolio."""
        if hasattr(event, 'payload'):
            strategy_id = event.payload.get('strategy_id', '')
            # Match signals from ma_crossover strategy
            return 'ma_crossover' in strategy_id
        return False
    
    # Subscribe portfolio to SIGNAL events
    root_container.event_bus.subscribe(
        EventType.SIGNAL.value,
        portfolio_manager.process_event,
        filter_func=signal_filter
    )
    
    logger.info("=== Starting data streaming ===")
    
    # Execute data streaming (which triggers the whole flow)
    data_container.execute()
    
    # Give events time to propagate
    import time
    time.sleep(0.1)
    
    logger.info("=== Checking results ===")
    
    # Check if signals were generated
    strategy_metrics = strategy_state.get_metrics()
    logger.info(f"Strategy metrics: {strategy_metrics}")
    logger.info(f"Signals generated: {strategy_metrics['signals_generated']}")
    
    # Portfolio signals are shown in console output above
    
    # Clean up
    for container in [portfolio_container, strategy_container, data_container, root_container]:
        container.stop()
        container.cleanup()
    
    # Check what was stored
    workspace_path = Path('workspaces/test_signal_storage')
    if workspace_path.exists():
        logger.info("\n=== Storage check ===")
        for item in sorted(workspace_path.iterdir()):
            if item.is_dir():
                logger.info(f"\nContainer: {item.name}")
                for file in sorted(item.iterdir()):
                    logger.info(f"  {file.name} ({file.stat().st_size} bytes)")
                    
                    # Show signal events if found
                    if file.name == 'events.jsonl':
                        with open(file, 'r') as f:
                            lines = f.readlines()
                            signal_count = sum(1 for line in lines if '"event_type": "SIGNAL"' in line)
                            logger.info(f"    Found {signal_count} SIGNAL events out of {len(lines)} total")
    else:
        logger.warning("No workspace directory found!")
    
    return strategy_metrics['signals_generated'] > 0

if __name__ == '__main__':
    success = test_signal_storage()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")