"""
Script to verify and fix the exact exit price issue.

The problem: Stop losses and take profits are exiting at market price instead of 
the calculated stop/target prices.

The fix: Ensure the broker uses the order.price field for exit orders.
"""

import subprocess
import json

print("=== EXACT EXIT PRICE FIX ===")
print("\nCurrent issue: Stops and targets exit at market price, not calculated price")
print("Expected: Stop loss at -0.075%, Take profit at +0.15%")
print("\nAnalyzing the issue...")

# Show current broker logic
print("\n1. Current broker price logic (lines 186-211):")
broker_code = """
# For stop loss and take profit orders, use the specified price
# This ensures exits happen at the calculated stop/target levels
self.logger.info(f"🔍 Order metadata: {order.metadata}")
self.logger.info(f"🔍 Order price: {order.price}, type: {type(order.price)}, exit_type: {order.metadata.get('exit_type')}")

# Convert price to float for comparison (handles Decimal type)
try:
    price_value = float(order.price) if order.price is not None else 0
except:
    price_value = 0
    
if price_value > 0 and order.metadata.get('exit_type') in ['stop_loss', 'take_profit', 'trailing_stop']:
    fill_price = price_value
    self.logger.info(f"✅ Using specified exit price for {order.metadata.get('exit_type')}: ${fill_price} (market: ${market_price})")
else:
    # Calculate slippage for regular market orders
    slippage = self.slippage_model.calculate_slippage(
        order, market_price, market_data.get('volume', 0)
    )
    
    # Calculate fill price
    if order.side == OrderSide.BUY:
        fill_price = market_price + slippage
    else:
        fill_price = market_price - slippage
"""
print(broker_code)

print("\n2. Risk manager exit price calculation (lines 604-627):")
risk_code = """
if exit_signal.exit_type == 'stop_loss':
    # Calculate stop loss price
    stop_loss_pct = risk_rules.get('stop_loss', 0)
    if stop_loss_pct:
        if position.quantity > 0:  # Long position
            exit_price = position.average_price * (Decimal(1) - Decimal(str(stop_loss_pct)))
        else:  # Short position
            exit_price = position.average_price * (Decimal(1) + Decimal(str(stop_loss_pct)))

elif exit_signal.exit_type == 'take_profit':
    # Calculate take profit price
    take_profit_pct = risk_rules.get('take_profit', 0)
    if take_profit_pct:
        if position.quantity > 0:  # Long position
            exit_price = position.average_price * (Decimal(1) + Decimal(str(take_profit_pct)))
        else:  # Short position
            exit_price = position.average_price * (Decimal(1) - Decimal(str(take_profit_pct)))
"""
print(risk_code)

print("\n3. The issue:")
print("   - Risk manager correctly calculates exit prices")
print("   - Portfolio state passes price in decision (line 639)")
print("   - Order is created with price (line 791)")
print("   - But broker might not be receiving orders with price set")

print("\n4. Debugging steps:")
print("   a) Add logging to see what price the broker receives")
print("   b) Ensure order.price is properly set from decision")
print("   c) Force broker to use order.price for exits")

print("\n5. Next step: Run test with enhanced logging to trace the issue")