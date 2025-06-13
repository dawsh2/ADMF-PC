#!/usr/bin/env python3
"""Direct test of strategy signal tracing."""

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
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.events.bus import EventBus
from src.core.events.types import Event, EventType
from src.core.events.observers.strategy_signal_tracer import StrategySignalTracer
from src.strategy.types import SignalDirection


def main():
    """Test strategy signal tracer directly."""
    
    # Create event bus
    event_bus = EventBus(bus_id="test_bus")
    
    # Create strategy signal tracer
    tracer = StrategySignalTracer(
        strategy_id="test_ma_crossover",
        workflow_id="direct_test",
        storage_config={'base_dir': './workspaces'},
        strategy_params={'fast_period': 5, 'slow_period': 20}
    )
    
    # Attach tracer to event bus
    event_bus.attach_observer(tracer)
    
    logger.info("Testing strategy signal tracer...")
    
    # Simulate some signals
    signals = [
        {'bar': 0, 'direction': 'long', 'symbol': 'SPY'},
        {'bar': 5, 'direction': 'flat', 'symbol': 'SPY'},
        {'bar': 10, 'direction': 'short', 'symbol': 'SPY'},
        {'bar': 15, 'direction': 'flat', 'symbol': 'SPY'},
        {'bar': 20, 'direction': 'long', 'symbol': 'SPY'},
    ]
    
    for signal in signals:
        event = Event(
            event_type=EventType.SIGNAL.value,
            timestamp=datetime.now(),
            payload={
                'symbol': signal['symbol'],
                'direction': signal['direction'],
                'strategy_id': 'test_ma_crossover',
                'price': 100.0 + signal['bar'],  # Fake price
                'strength': 1.0
            },
            source_id='test_strategy',
            container_id='test_container'
        )
        
        logger.info(f"Publishing signal at bar {signal['bar']}: {signal['direction']}")
        event_bus.publish(event)
        
        # Advance bar index in storage
        tracer.storage._bar_index = signal['bar']
    
    # Get statistics
    stats = tracer.get_statistics()
    logger.info(f"\nTracer statistics: {stats}")
    
    # Flush to disk
    filepath = tracer.flush()
    if filepath:
        logger.info(f"\nSignals saved to: {filepath}")
        
        # Read and display the file
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
            logger.info(f"\nSaved data:")
            logger.info(f"  Metadata: {data['metadata']}")
            logger.info(f"  Changes: {len(data['changes'])} signal changes")
            for change in data['changes']:
                logger.info(f"    Bar {change['idx']}: {change['val']} ({change['sym']})")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())