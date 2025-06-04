"""Test signal to order conversion in the risk container."""

import yaml
from datetime import datetime
from decimal import Decimal
from src.core.logging.structured import setup_logging
from src.core.events.types import EventType
from src.core.coordinator.yaml_coordinator import YAMLCoordinator
from src.core.logging.event_logger import get_event_logger

# Setup logging
setup_logging(level="DEBUG")
logger = get_event_logger(__name__)

# Test configuration with clear signal generation
config = {
    "name": "Test Signal to Order",
    "type": "backtest",
    "data": {
        "source": "csv",
        "file_path": "data/SPY.csv",
        "max_bars": 5  # Just 5 bars for testing
    },
    "strategies": {
        "type": "mean_reversion_simple",  # Use simple mean reversion
        "parameters": {
            "ma_period": 2,  # Very short MA to trigger signals quickly
            "entry_threshold": 0.001,  # Very sensitive
            "exit_threshold": 0.0005
        }
    },
    "risk": {
        "initial_capital": 100000,
        "position_size_method": "fixed",
        "fixed_position_size": 1000
    },
    "execution": {
        "mode": "backtest"
    }
}

# Create coordinator
coordinator = YAMLCoordinator()

# Set up event monitoring
signal_count = 0
order_count = 0
monitored_events = []

def monitor_signals(event):
    global signal_count
    signal_count += 1
    signal = event.payload
    logger.info(f"SIGNAL CREATED: {signal.signal_id} for {signal.symbol} - {signal.side} @ strength {signal.strength}")
    monitored_events.append(("SIGNAL", signal))

def monitor_orders(event):
    global order_count
    order_count += 1
    order = event.payload
    logger.info(f"ORDER CREATED: {order.order_id} for {order.symbol} - {order.side} {order.quantity} shares")
    monitored_events.append(("ORDER", order))

# Add monitoring to the workflow config
config["monitoring"] = {
    "signal_callback": monitor_signals,
    "order_callback": monitor_orders
}

# Run the backtest
logger.info("Starting test...")
try:
    result = coordinator.run_workflow(config)
except Exception as e:
    logger.error(f"Workflow failed: {e}")
    # Try a simpler approach - just run and parse logs
    import subprocess
    subprocess.run(["python", "main.py", "config/simple_backtest.yaml"])

# Print results
logger.info(f"\n{'='*50}")
logger.info(f"Test Complete!")
logger.info(f"Signals generated: {signal_count}")
logger.info(f"Orders created: {order_count}")
logger.info(f"Signal→Order conversion rate: {order_count/signal_count*100:.1f}%" if signal_count > 0 else "No signals generated")
logger.info(f"{'='*50}\n")

if signal_count == 0:
    logger.warning("No signals were generated. Check strategy parameters.")
elif order_count == 0:
    logger.error("Signals were generated but no orders were created! Signal→Order conversion is broken.")
else:
    logger.info("Signal→Order conversion is working correctly!")