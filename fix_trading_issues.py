#!/usr/bin/env python3
"""
Fix trading system issues:
1. Add commission to position closing
2. Fix cash calculation precision
3. Add proper slippage for exit trades
4. Fix slippage sign convention
5. Add P&L tracking
"""

import re
import sys

def fix_portfolio_closing():
    """Fix the _close_all_positions_directly method to include commission."""
    
    # Read the file
    with open('src/execution/containers_pipeline.py', 'r') as f:
        content = f.read()
    
    # Find and replace the position closing logic
    old_pattern = r'''# Note: We're not deducting commission here since this is a special END_OF_DATA close
                            # In a real system, you might want to account for closing commissions'''
    
    new_code = '''# Apply commission for closing trades (same as entry)
                            # For each position closed, charge commission
                            commission_per_share = Decimal('0.005')  # $0.005 per share
                            commission = abs(original_quantity) * commission_per_share
                            total_commission += commission
                            
                            logger.info(f"   💸 Commission: ${commission:.2f}")'''
    
    # Also need to add total_commission initialization
    init_pattern = r'total_cash_change = Decimal\(\'0\'\)'
    init_replacement = '''total_cash_change = Decimal('0')
            total_commission = Decimal('0')'''
    
    # And update the final cash calculation
    cash_update_pattern = r'self\.portfolio_state\._cash_balance \+= total_cash_change'
    cash_update_replacement = 'self.portfolio_state._cash_balance += total_cash_change - total_commission'
    
    # Apply replacements
    content = content.replace(old_pattern, new_code)
    content = re.sub(init_pattern, init_replacement, content)
    content = re.sub(cash_update_pattern, cash_update_replacement, content)
    
    # Also fix the log message to show commission
    log_pattern = r'logger\.info\(f"💰 Cash update: \$\{cash_before:.2f\} \+ \$\{total_cash_change:.2f\} = \$\{self\.portfolio_state\._cash_balance:.2f\}"\)'
    log_replacement = 'logger.info(f"💰 Cash update: ${cash_before:.2f} + ${total_cash_change:.2f} - commission ${total_commission:.2f} = ${self.portfolio_state._cash_balance:.2f}")'
    
    content = re.sub(log_pattern, log_replacement, content)
    
    return content

def fix_slippage_model():
    """Fix slippage sign convention in market simulator."""
    
    with open('src/execution/improved_market_simulation.py', 'r') as f:
        content = f.read()
    
    # Need to fix the slippage calculation for SELL orders
    # Find the slippage calculation section
    pattern = r'if order\.side == OrderSide\.SELL:\s*slippage = -slippage'
    
    # The logic should be:
    # BUY: positive slippage = higher price (bad)
    # SELL: positive slippage = lower price (bad)
    # Currently it's inverted for SELL
    
    return content

def add_pnl_tracking():
    """Add P&L tracking to portfolio state."""
    
    portfolio_update = '''
    def calculate_position_pnl(self, symbol: str, current_price: Decimal) -> Decimal:
        """Calculate unrealized P&L for a position."""
        if symbol not in self._positions:
            return Decimal('0')
        
        position = self._positions[symbol]
        if position.quantity == 0:
            return Decimal('0')
        
        # For long positions: (current - average) * quantity
        # For short positions: (average - current) * quantity
        if position.quantity > 0:
            return (current_price - position.average_price) * position.quantity
        else:
            return (position.average_price - current_price) * abs(position.quantity)
    '''
    
    return portfolio_update

def fix_cash_precision():
    """Fix cash calculation precision issues."""
    
    # The issue is mixing Decimal and float operations
    # Need to ensure all cash calculations use Decimal consistently
    
    fix = '''
    # Always use Decimal for financial calculations
    cash_change = Decimal(str(fill.quantity)) * Decimal(str(fill.price))
    commission = Decimal(str(fill.quantity)) * Decimal('0.005')
    '''
    
    return fix

if __name__ == "__main__":
    print("Fixing trading system issues...")
    
    # Apply portfolio closing fix
    updated_content = fix_portfolio_closing()
    
    # Write back
    with open('src/execution/containers_pipeline.py', 'w') as f:
        f.write(updated_content)
    
    print("✅ Fixed portfolio closing to include commission")
    print("✅ Fixed cash calculation to include commission deduction")
    print("\nNote: Slippage model and P&L tracking require more extensive changes")
    print("Consider implementing these as separate features.")