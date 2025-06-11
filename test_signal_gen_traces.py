#!/usr/bin/env python3
"""
Test signal generation with trace storage.
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from src.core.containers.container import Container, ContainerConfig
from src.data.handlers import SimpleHistoricalDataHandler
from src.strategy.state import StrategyState
from src.core.events.types import EventType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
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
    # Create workspace for traces
    workspace_dir = Path("./workspaces/signal_gen_test")
    workspace_dir.mkdir(parents=True, exist_ok=True)
    
    # Create containers with trace storage
    root_config = ContainerConfig(
        name="root",
        components=[],
        config={
            "trace_enabled": True,
            "trace_dir": str(workspace_dir / "root"),
            "trace_level": "DEBUG"
        },
        container_type="root"
    )
    root_container = Container(root_config)
    
    # Track all events
    all_events = []
    
    def capture_event(event):
        """Capture all events for storage."""
        # Convert payload for JSON serialization
        payload = event.payload.copy() if hasattr(event.payload, 'copy') else event.payload
        
        # Handle special types in payload
        if 'bar' in payload and hasattr(payload['bar'], '__dict__'):
            # Convert Bar object to dict
            bar = payload['bar']
            payload['bar'] = {
                'symbol': bar.symbol,
                'timestamp': str(bar.timestamp),
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': float(bar.volume)
            }
        
        all_events.append({
            'timestamp': event.timestamp.isoformat(),
            'event_type': event.event_type,
            'source_id': event.source_id,
            'container_id': getattr(event, 'container_id', None),
            'payload': payload
        })
    
    # Subscribe to capture all events on root
    root_container.event_bus.subscribe(EventType.BAR.value, capture_event)
    root_container.event_bus.subscribe(
        EventType.SIGNAL.value, 
        capture_event,
        filter_func=lambda e: True
    )
    
    # Create data container
    data_config = ContainerConfig(
        name="data_container",
        components=["data_streamer"],
        config={
            "symbol": "SPY",
            "data_dir": "./data",
            "max_bars": 50,  # Process 50 bars for signals
            "trace_dir": str(workspace_dir / "data")
        },
        container_type="data"
    )
    data_container = Container(data_config)
    root_container.add_child_container(data_container)
    
    # Create data handler
    data_handler = SimpleHistoricalDataHandler(
        handler_id="data_SPY",
        data_dir="./data"
    )
    data_handler.load_data(["SPY"])
    data_handler.max_bars = 50
    data_container.add_component("data_streamer", data_handler)
    
    # Create strategy container
    strategy_config = ContainerConfig(
        name="strategy_container",
        components=["strategy_state"],
        config={
            "symbols": ["SPY"],
            "features": {
                "sma_fast": {"feature": "sma", "period": 5},
                "sma_slow": {"feature": "sma", "period": 10}
            },
            "trace_dir": str(workspace_dir / "strategy")
        },
        container_type="strategy"
    )
    strategy_container = Container(strategy_config)
    root_container.add_child_container(strategy_container)
    
    # Create strategy state
    strategy_state = StrategyState(
        symbols=["SPY"],
        feature_configs={
            "sma_fast": {"feature": "sma", "period": 5},
            "sma_slow": {"feature": "sma", "period": 10}
        }
    )
    strategy_container.add_component("strategy_state", strategy_state)
    
    # Add strategy
    strategy_state.add_strategy("momentum_1", simple_momentum_strategy)
    
    # Initialize and start
    logger.info("=== Initializing Signal Generation ===")
    root_container.initialize()
    data_container.initialize()
    strategy_container.initialize()
    
    root_container.start()
    data_container.start()
    strategy_container.start()
    
    # Execute - stream data
    logger.info("\n=== Streaming Data ===")
    data_container.execute()
    
    # Save captured events
    events_file = workspace_dir / "events.jsonl"
    logger.info(f"\n=== Saving {len(all_events)} events to {events_file} ===")
    
    # Custom JSON encoder for pandas timestamps
    def json_encoder(obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        elif hasattr(obj, '__float__'):
            return float(obj)
        elif hasattr(obj, '__int__'):
            return int(obj)
        else:
            return str(obj)
    
    with open(events_file, 'w') as f:
        for event in all_events:
            f.write(json.dumps(event, default=json_encoder) + '\n')
    
    # Create summary
    summary = {
        'run_id': 'signal_gen_test',
        'timestamp': datetime.now().isoformat(),
        'containers': {
            'root': root_container.container_id,
            'data': data_container.container_id,
            'strategy': strategy_container.container_id
        },
        'events_captured': len(all_events),
        'event_types': {
            'BAR': len([e for e in all_events if e['event_type'] == 'BAR']),
            'SIGNAL': len([e for e in all_events if e['event_type'] == 'SIGNAL'])
        },
        'metrics': strategy_state.get_metrics()
    }
    
    summary_file = workspace_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=json_encoder)
    
    logger.info(f"Summary saved to {summary_file}")
    logger.info(f"Event types: {summary['event_types']}")
    
    # Show sample events
    logger.info("\n=== Sample Events ===")
    for event in all_events[:3]:
        logger.info(f"{event['event_type']}: {event['source_id']} at {event['timestamp']}")
    
    if all_events:
        logger.info("...")
        for event in all_events[-3:]:
            logger.info(f"{event['event_type']}: {event['source_id']} at {event['timestamp']}")
    
    # Cleanup
    strategy_container.stop()
    data_container.stop()
    root_container.stop()
    
    strategy_container.cleanup()
    data_container.cleanup()
    root_container.cleanup()
    
    logger.info("\n✅ Signal generation complete!")

if __name__ == "__main__":
    main()