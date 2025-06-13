#!/usr/bin/env python3
"""Debug event bus sharing issue."""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.containers.container import Container, ContainerConfig
from src.core.events.bus import EventBus


def main():
    """Test event bus sharing between containers."""
    
    # Create root container
    root_config = ContainerConfig(
        name="root",
        container_type="root",
        components=[],
        config={}
    )
    root = Container(root_config)
    logger.info(f"Root container event bus: {root.event_bus}")
    
    # Create child containers
    data_config = ContainerConfig(
        name="data",
        container_type="data",
        components=[],
        config={}
    )
    
    strategy_config = ContainerConfig(
        name="strategy",
        container_type="strategy",
        components=[],
        config={}
    )
    
    # Create children using create_child
    data_child = root.create_child(data_config)
    strategy_child = root.create_child(strategy_config)
    
    logger.info(f"Data child event bus: {data_child.event_bus}")
    logger.info(f"Strategy child event bus: {strategy_child.event_bus}")
    logger.info(f"Same event bus? {data_child.event_bus is strategy_child.event_bus}")
    logger.info(f"Same as root? {data_child.event_bus is root.event_bus}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())