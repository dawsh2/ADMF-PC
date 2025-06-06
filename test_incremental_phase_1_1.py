#!/usr/bin/env python3
"""
Test script for Phase 1.1 of the incremental testing plan.

This script tests the basic container lifecycle without going through
the full Coordinator/TopologyBuilder flow.
"""

import asyncio
import logging

from src.core.containers.container import Container, ContainerConfig
from src.core.containers.protocols import ContainerRole
from src.core.events import EventBus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_phase_1_1():
    """
    Phase 1.1: Test root container lifecycle only.
    
    Tests:
    1. Container creation
    2. Container initialization 
    3. Container start
    4. Event bus creation
    5. Container stop
    """
    logger.info("=== Phase 1.1: Root Container Only ===")
    
    tests_passed = []
    tests_failed = []
    
    try:
        # Test 1: Create root container
        logger.info("Test 1: Creating root container...")
        root_config = ContainerConfig(
            role=ContainerRole.BACKTEST,
            name="root_backtest_container",
            container_id="root_container",
            config={
                'description': 'Root container for incremental testing',
                'phase': '1.1'
            },
            capabilities=set()
        )
        
        root_container = Container(root_config)
        
        if root_container and root_container.container_id == "root_container":
            tests_passed.append("Container creation")
            logger.info("✓ Container created successfully")
        else:
            tests_failed.append("Container creation")
            logger.error("✗ Container creation failed")
        
        # Test 2: Initialize container
        logger.info("Test 2: Initializing container...")
        await root_container.initialize()
        
        if hasattr(root_container, '_initialized') and root_container._initialized:
            tests_passed.append("Container initialization")
            logger.info("✓ Container initialized successfully")
        else:
            tests_failed.append("Container initialization")
            logger.error("✗ Container initialization failed")
        
        # Test 3: Start container
        logger.info("Test 3: Starting container...")
        await root_container.start()
        
        if hasattr(root_container, '_running') and root_container._running:
            tests_passed.append("Container start")
            logger.info("✓ Container started successfully")
        else:
            tests_failed.append("Container start")
            logger.error("✗ Container start failed")
        
        # Test 4: Event bus exists
        logger.info("Test 4: Checking event bus...")
        if hasattr(root_container, 'event_bus') and isinstance(root_container.event_bus, EventBus):
            tests_passed.append("Event bus creation")
            logger.info("✓ Event bus created successfully")
        else:
            tests_failed.append("Event bus creation")
            logger.error("✗ Event bus creation failed")
        
        # Test 5: Get container state
        logger.info("Test 5: Getting container state...")
        state = root_container.get_state_info()
        
        if state and 'role' in state and state['role'] == ContainerRole.BACKTEST:
            tests_passed.append("Container state")
            logger.info(f"✓ Container state retrieved: {state}")
        else:
            tests_failed.append("Container state")
            logger.error("✗ Container state retrieval failed")
        
        # Test 6: Stop container
        logger.info("Test 6: Stopping container...")
        await root_container.stop()
        
        if hasattr(root_container, '_running') and not root_container._running:
            tests_passed.append("Container stop")
            logger.info("✓ Container stopped successfully")
        else:
            tests_failed.append("Container stop")
            logger.error("✗ Container stop failed")
            
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        tests_failed.append(f"Exception: {str(e)}")
    
    # Summary
    total_tests = len(tests_passed) + len(tests_failed)
    logger.info("\n=== Phase 1.1 Test Summary ===")
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Passed: {len(tests_passed)}")
    logger.info(f"Failed: {len(tests_failed)}")
    
    if tests_passed:
        logger.info("\nPassed tests:")
        for test in tests_passed:
            logger.info(f"  ✓ {test}")
    
    if tests_failed:
        logger.info("\nFailed tests:")
        for test in tests_failed:
            logger.info(f"  ✗ {test}")
    
    success_rate = len(tests_passed) / total_tests if total_tests > 0 else 0
    logger.info(f"\nSuccess rate: {success_rate:.1%}")
    
    # Human checkpoint
    logger.info("\n=== Human Checkpoint ===")
    logger.info("Please review:")
    logger.info("  - Container creation sequence in logs")
    logger.info("  - Canonical behavior patterns confirmed")
    logger.info("  - Clean shutdown verified")
    
    return len(tests_failed) == 0


if __name__ == "__main__":
    success = asyncio.run(test_phase_1_1())
    exit(0 if success else 1)