#!/usr/bin/env python3
"""Test script to isolate sync execution issues."""

import sys
import asyncio
sys.path.append('/Users/daws/ADMF-PC')

from src.execution.order_manager import OrderManager
from src.execution.execution_context import ExecutionContext
from src.execution.market_simulation import MarketSimulator
from src.execution.protocols import Order, OrderSide, OrderType
from decimal import Decimal

def test_sync_components():
    """Test if synchronous components work in isolation."""
    print("Testing synchronous components...")
    
    # Test OrderManager
    print("1. Testing OrderManager...")
    order_manager = OrderManager()
    order = order_manager.create_order(
        symbol="SPY",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET
    )
    print(f"   ✅ Created order: {order.order_id}")
    
    # Test ExecutionContext
    print("2. Testing ExecutionContext...")
    context = ExecutionContext()
    context.add_active_order(order.order_id)
    active_orders = context.get_active_orders()
    print(f"   ✅ Active orders: {len(active_orders)}")
    
    # Test MarketSimulator
    print("3. Testing MarketSimulator...")
    simulator = MarketSimulator()
    fill = simulator.simulate_fill(order, 100.0, 1000000, 0.01)
    print(f"   ✅ Simulated fill: {fill.fill_id if fill else 'None'}")
    
    print("✅ All synchronous components work!")

async def test_async_to_sync_call():
    """Test calling sync methods from async context."""
    print("Testing async to sync calls...")
    
    def sync_method():
        print("   Sync method called")
        return "sync_result"
    
    # This should work fine
    result = sync_method()
    print(f"   ✅ Result: {result}")

if __name__ == "__main__":
    print("🧪 Testing synchronous execution components...")
    
    try:
        test_sync_components()
        print()
        
        print("🧪 Testing async to sync calls...")
        asyncio.run(test_async_to_sync_call())
        
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()