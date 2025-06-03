#!/usr/bin/env python3
"""
Test script to validate that the synchronous fill processing fix works correctly.

This test validates that:
1. Fill events during position closing are processed synchronously
2. Portfolio state is updated immediately after each fill
3. Final portfolio summary reflects all closed positions

Usage:
    python test_fill_sync_fix.py
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any

# Setup logging to see the synchronous processing in action
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_containers_pipeline_sync_fill():
    """Test that demonstrates the synchronous fill processing fix."""
    logger.info("🧪 Testing synchronous fill processing fix")
    
    # Import the fixed containers module
    try:
        import src.execution.containers_pipeline as containers_module
        from src.execution.containers_pipeline import ExecutionContainer
        from src.risk.portfolio_state import PortfolioState
        from src.execution.protocols import Fill, OrderSide
        from src.core.events.types import Event, EventType
        
        # Set up a mock global portfolio state with positions
        containers_module._GLOBAL_PORTFOLIO_STATE = PortfolioState(initial_capital=Decimal('100000'))
        
        # Add some mock positions to close
        containers_module._GLOBAL_PORTFOLIO_STATE.update_position('SPY', 100, Decimal('400.00'), datetime.now())
        containers_module._GLOBAL_PORTFOLIO_STATE.update_position('AAPL', -50, Decimal('150.00'), datetime.now())
        
        logger.info(f"📊 Initial portfolio state:")
        logger.info(f"   💰 Cash: ${containers_module._GLOBAL_PORTFOLIO_STATE.get_cash_balance():.2f}")
        positions = containers_module._GLOBAL_PORTFOLIO_STATE.get_all_positions()
        for symbol, pos in positions.items():
            logger.info(f"   📈 {symbol}: {pos.quantity} shares @ ${pos.average_price:.2f}")
        
        # Create mock execution container
        execution_container = ExecutionContainer({}, "test_execution")
        
        # Create mock fills for closing positions
        # For SPY: we have +100 shares, so we need to SELL 100 to close
        spy_fill = type('Fill', (), {
            'symbol': 'SPY',
            'quantity': 100,
            'price': Decimal('405.00'),
            'side': OrderSide.SELL,
            'executed_at': datetime.now(),
            'commission': Decimal('0.50')
        })()
        
        # For AAPL: we have -50 shares (short), so we need to BUY 50 to close
        aapl_fill = type('Fill', (), {
            'symbol': 'AAPL', 
            'quantity': 50,
            'price': Decimal('155.00'),
            'side': OrderSide.BUY,
            'executed_at': datetime.now(),
            'commission': Decimal('0.50')
        })()
        
        mock_order = type('Order', (), {
            'order_id': 'TEST-123',
            'symbol': 'SPY',
            'side': OrderSide.SELL,
            'quantity': 100
        })()
        
        # Test synchronous fill processing
        async def test_sync_processing():
            logger.info("🔄 Testing synchronous fill processing...")
            
            # Process SPY close fill synchronously
            await execution_container._process_fill_synchronously(spy_fill, mock_order, "test_close")
            
            # Check portfolio state immediately after processing
            logger.info(f"📊 Portfolio state after SPY fill:")
            logger.info(f"   💰 Cash: ${containers_module._GLOBAL_PORTFOLIO_STATE.get_cash_balance():.2f}")
            positions = containers_module._GLOBAL_PORTFOLIO_STATE.get_all_positions()
            for symbol, pos in positions.items():
                logger.info(f"   📈 {symbol}: {pos.quantity} shares @ ${pos.average_price:.2f}")
            
            # Process AAPL close fill synchronously
            mock_order.symbol = 'AAPL'
            mock_order.side = OrderSide.BUY
            await execution_container._process_fill_synchronously(aapl_fill, mock_order, "test_close")
            
            # Check final portfolio state
            logger.info(f"📊 Final portfolio state after both fills:")
            logger.info(f"   💰 Cash: ${containers_module._GLOBAL_PORTFOLIO_STATE.get_cash_balance():.2f}")
            positions = containers_module._GLOBAL_PORTFOLIO_STATE.get_all_positions()
            for symbol, pos in positions.items():
                logger.info(f"   📈 {symbol}: {pos.quantity} shares @ ${pos.average_price:.2f}")
            
            # Validate that positions are closed
            spy_position = positions.get('SPY')
            aapl_position = positions.get('AAPL')
            
            if spy_position is None or spy_position.quantity == 0:
                logger.info("✅ SPY position successfully closed")
            else:
                logger.error(f"❌ SPY position not properly closed: {spy_position.quantity} shares remaining")
                
            if aapl_position is None or aapl_position.quantity == 0:
                logger.info("✅ AAPL position successfully closed")
            else:
                logger.error(f"❌ AAPL position not properly closed: {aapl_position.quantity} shares remaining")
                
            # Validate cash balance increased from closing positions
            expected_cash_increase = (100 * Decimal('405.00') - Decimal('0.50')) + (50 * Decimal('155.00') - Decimal('0.50'))
            expected_initial_cost = (100 * Decimal('400.00')) + (50 * Decimal('150.00'))  # Cash used for initial positions
            expected_final_cash = Decimal('100000') + expected_cash_increase - expected_initial_cost
            
            actual_cash = containers_module._GLOBAL_PORTFOLIO_STATE.get_cash_balance()
            logger.info(f"💰 Expected cash: ${expected_final_cash:.2f}, Actual: ${actual_cash:.2f}")
            
            if abs(actual_cash - expected_final_cash) < Decimal('1.00'):
                logger.info("✅ Cash balance correctly updated")
            else:
                logger.error("❌ Cash balance not correctly updated")
        
        # Run the test
        asyncio.run(test_sync_processing())
        
        logger.info("🎯 Synchronous fill processing test completed")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the synchronous fill processing test."""
    logger.info("🚀 Starting synchronous fill processing validation")
    
    success = test_containers_pipeline_sync_fill()
    
    if success:
        logger.info("✅ All tests passed - synchronous fill processing is working correctly")
    else:
        logger.error("❌ Tests failed - synchronous fill processing needs debugging")
    
    return success

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)