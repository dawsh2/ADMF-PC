#!/usr/bin/env python3
"""Simple test of strategy tracing using existing infrastructure."""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.containers.container import Container, ContainerConfig
from src.core.events.types import Event, EventType
from src.strategy.state import StrategyState


def main():
    """Test strategy container with signal tracing."""
    
    # Create container config for strategy container
    config = ContainerConfig(
        name="test_strategy_container",
        container_type="strategy",
        components=["strategy_state"],
        config={
            'symbols': ['SPY'],
            'features': {
                'fast_ma': {'feature': 'sma', 'period': 5},
                'slow_ma': {'feature': 'sma', 'period': 20}
            },
            'strategies': [
                {
                    'name': 'test_ma_crossover',
                    'type': 'ma_crossover',
                    'params': {'fast_period': 5, 'slow_period': 20}
                }
            ],
            'execution': {
                'enable_event_tracing': True,
                'trace_settings': {
                    'use_sparse_storage': True,
                    'container_settings': {
                        'strategy*': {'enabled': True}
                    }
                }
            },
            'metadata': {
                'workflow_id': 'strategy_trace_test',
                'experiment_type': 'signal_generation'
            }
        }
    )
    
    logger.info("Creating strategy container with tracing...")
    
    # Create container (no parent event bus = root container)
    container = Container(config)
    
    # Add strategy state component
    strategy_state = StrategyState(
        symbols=['SPY'],
        feature_configs=config.config['features']
    )
    container.add_component('strategy_state', strategy_state)
    
    # Add a test strategy manually
    def test_strategy(features, bar, params):
        """Simple test strategy that generates signals."""
        if 'fast_ma' in features and 'slow_ma' in features:
            if features['fast_ma'] > features['slow_ma']:
                return {'signal_type': 'entry', 'direction': 'long', 'strength': 1.0}
            else:
                return {'signal_type': 'entry', 'direction': 'short', 'strength': 1.0}
        return None
    
    strategy_state.add_strategy('SPY_test_ma_crossover', test_strategy, {'fast_period': 5, 'slow_period': 20})
    
    # Initialize and start container
    container.initialize()
    container.start()
    
    logger.info(f"Container initialized. Strategy tracer enabled: {hasattr(container, '_strategy_tracer')}")
    
    # Simulate some BAR events
    bars = [
        {'open': 100, 'high': 101, 'low': 99, 'close': 100.5, 'volume': 1000},
        {'open': 100.5, 'high': 102, 'low': 100, 'close': 101.5, 'volume': 1100},
        {'open': 101.5, 'high': 103, 'low': 101, 'close': 102.5, 'volume': 1200},
        {'open': 102.5, 'high': 104, 'low': 102, 'close': 103.5, 'volume': 1300},
        {'open': 103.5, 'high': 105, 'low': 103, 'close': 104.5, 'volume': 1400},
        {'open': 104.5, 'high': 105, 'low': 102, 'close': 102.5, 'volume': 1500},
        {'open': 102.5, 'high': 103, 'low': 101, 'close': 101.5, 'volume': 1600},
        {'open': 101.5, 'high': 102, 'low': 100, 'close': 100.5, 'volume': 1700},
    ]
    
    for i, bar_data in enumerate(bars):
        event = Event(
            event_type=EventType.BAR.value,
            timestamp=datetime.now(),
            payload={
                'symbol': 'SPY',
                'bar': type('Bar', (), bar_data)()  # Create simple bar object
            },
            source_id='test_data',
            container_id='test_data_container'
        )
        
        logger.info(f"Publishing BAR {i}: close={bar_data['close']}")
        container.event_bus.publish(event)
    
    # Stop and cleanup
    container.stop()
    container.cleanup()
    
    logger.info("Test completed. Check workspaces/strategy_trace_test/ for output files.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())