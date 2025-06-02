#!/usr/bin/env python3
"""
Test script for the Hybrid Tiered Communication implementation.

This script tests the basic functionality of our hybrid communication system
and verifies that sub-containers can properly publish events.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add src to path for imports
src_path = str(Path(__file__).parent / 'src')
sys.path.insert(0, src_path)

# Change to src directory to fix relative imports
os.chdir(src_path)

try:
    from core.events.bootstrap import setup_hybrid_communication, validate_communication_setup
    from core.events.tiered_router import TieredEventRouter
    from core.events.types import Event, EventType
    from execution.containers import StrategyContainer, DataContainer
    from datetime import datetime
    logger = logging.getLogger(__name__)
    logger.info("✅ Successfully imported hybrid communication modules")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"❌ Import error: {e}")
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_basic_hybrid_communication():
    """Test basic hybrid communication functionality"""
    logger.info("🧪 Testing Basic Hybrid Communication")
    
    try:
        # Create a simple multi-strategy container
        strategy_config = {
            'strategies': [
                {
                    'name': 'momentum_strategy',
                    'type': 'momentum',
                    'parameters': {
                        'lookback_period': 20,
                        'momentum_threshold': 0.0001
                    }
                },
                {
                    'name': 'mean_reversion_strategy', 
                    'type': 'mean_reversion',
                    'parameters': {
                        'lookback_period': 15,
                        'entry_threshold': 0.5
                    }
                }
            ]
        }
        
        # Create strategy container (this will create sub-containers)
        strategy_container = StrategyContainer(strategy_config, container_id="test_strategy_ensemble")
        
        # Initialize the strategy container
        await strategy_container.initialize()
        
        # Set up hybrid communication
        router = await setup_hybrid_communication(strategy_container)
        
        # Validate the setup
        validation_result = await validate_communication_setup(strategy_container, router)
        
        logger.info(f"📊 Validation Result: {validation_result['status']}")
        logger.info(f"📊 Containers Checked: {validation_result['containers_checked']}")
        
        if validation_result['issues']:
            logger.error("❌ Issues found:")
            for issue in validation_result['issues']:
                logger.error(f"  • {issue}")
        
        if validation_result['warnings']:
            logger.warning("⚠️ Warnings:")
            for warning in validation_result['warnings']:
                logger.warning(f"  • {warning}")
        
        # Test sub-container signal publishing
        await test_sub_container_signal_publishing(strategy_container)
        
        # Display router statistics
        logger.info("📊 Final Router Statistics:")
        logger.info(router.debug_routing())
        
        return validation_result['status'] == 'valid'
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_sub_container_signal_publishing(strategy_container):
    """Test that sub-containers can publish signals properly"""
    logger.info("🧪 Testing Sub-Container Signal Publishing")
    
    # Create a test signal event
    test_signal_event = Event(
        event_type=EventType.SIGNAL,
        payload={
            'timestamp': datetime.now(),
            'signals': [
                {
                    'symbol': 'TEST',
                    'direction': 'BUY',
                    'strength': 0.8,
                    'signal_type': 'entry'
                }
            ],
            'market_data': {'TEST': {'close': 100.0}},
            'source': 'test_sub_strategy'
        },
        timestamp=datetime.now()
    )
    
    # Get sub-containers and test their signal publishing
    sub_containers = strategy_container.child_containers
    logger.info(f"📊 Found {len(sub_containers)} sub-containers")
    
    for i, sub_container in enumerate(sub_containers):
        logger.info(f"🔬 Testing sub-container {i+1}: {sub_container.metadata.name}")
        
        # Check if sub-container has hybrid interface
        if hasattr(sub_container, 'publish_external'):
            logger.info(f"✅ Sub-container has hybrid interface")
            
            # Check if registered with router
            if hasattr(sub_container, 'external_router') and sub_container.external_router:
                logger.info(f"✅ Sub-container registered with router")
                
                # Test signal publishing
                try:
                    from core.events.hybrid_interface import CommunicationTier
                    sub_container.publish_external(test_signal_event, tier=CommunicationTier.STANDARD)
                    logger.info(f"✅ Sub-container successfully published signal via Event Router")
                except Exception as e:
                    logger.error(f"❌ Sub-container failed to publish signal: {e}")
            else:
                logger.error(f"❌ Sub-container not registered with router")
        else:
            logger.error(f"❌ Sub-container missing hybrid interface")


async def test_data_broadcasting():
    """Test data container broadcasting functionality"""
    logger.info("🧪 Testing Data Broadcasting")
    
    try:
        # Create data container
        data_config = {
            'source': 'csv',
            'symbols': ['TEST'],
            'max_bars': 5  # Small number for testing
        }
        
        data_container = DataContainer(data_config, container_id="test_data_container")
        
        # Create strategy container that subscribes to data
        strategy_config = {
            'type': 'momentum',
            'parameters': {'lookback_period': 5}
        }
        
        strategy_container = StrategyContainer(strategy_config, container_id="test_strategy_subscriber")
        
        # Set up communication
        router = await setup_hybrid_communication(data_container)
        
        # Register strategy container
        if hasattr(strategy_container, 'register_with_router'):
            strategy_container.register_with_router(router)
        
        # Test broadcasting a BAR event
        test_bar_event = Event(
            event_type=EventType.BAR,
            payload={
                'timestamp': datetime.now(),
                'market_data': {'TEST': {'close': 100.0, 'volume': 1000}},
                'symbol': 'TEST'
            },
            timestamp=datetime.now()
        )
        
        logger.info("📤 Broadcasting test BAR event via Fast Tier")
        from core.events.hybrid_interface import CommunicationTier
        data_container.publish_external(test_bar_event, tier=CommunicationTier.FAST)
        
        # Wait for event processing
        await asyncio.sleep(0.1)
        
        logger.info("📊 Data broadcasting test completed")
        
    except Exception as e:
        logger.error(f"❌ Data broadcasting test failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all tests"""
    logger.info("🚀 Starting Hybrid Tiered Communication Tests")
    
    success = True
    
    # Test 1: Basic hybrid communication
    success &= await test_basic_hybrid_communication()
    
    # Test 2: Data broadcasting
    await test_data_broadcasting()
    
    if success:
        logger.info("✅ All tests passed!")
        return 0
    else:
        logger.error("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)