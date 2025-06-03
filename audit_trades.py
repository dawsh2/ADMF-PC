#!/usr/bin/env python3
"""Enhanced trade audit script for ADMF-PC backtest logs."""

import sys
import re
from decimal import Decimal
from datetime import datetime
from collections import defaultdict

def extract_trades_from_log(log_text):
    """Extract all trading activity from log text."""
    trades = []
    fills = []
    portfolio_updates = []
    positions = {}
    
    # Updated patterns for current log format
    patterns = {
        # [FILL] FILLED: SPY -1 1.00 @ 520.9294
        'fill': r"\[FILL\] FILLED: (\S+) (-?\d+) ([\d.]+) @ ([\d.]+)",
        
        # 💼 PortfolioContainer updated: SPY SELL 1.0 @ 520.9294050000001
        'portfolio_update': r"💼 PortfolioContainer updated: (\S+) (BUY|SELL) ([\d.]+) @ ([\d.]+)",
        
        # 📈 Position: -1.0 (quantity_delta was -1.0)
        'position': r"📈 Position: ([\d.-]+)",
        
        # 💰 Cash change: 518.32, New balance: 100518.32
        'cash_change': r"💰 Cash change: ([\d.-]+), New balance: ([\d.-]+)",
        
        # ✅ Closed position: SPY -1.0 shares @ $521.25, cash change: $-521.25
        'close': r"✅ Closed position: (\S+) ([\d.-]+) shares @ \$([\d.]+), cash change: \$([\d.-]+)",
        
        # [ORDER] SPY -1 1.00 @ 0.0000
        'order': r"\[ORDER\] (\S+) (-?\d+) ([\d.]+) @ ([\d.]+)",
        
        # [SIGNAL] SPY OrderSide.SELL (strength: 1.00) - Price above upper band - overbought
        'signal': r"\[SIGNAL\] (\S+) OrderSide\.(\w+) \(strength: ([\d.]+)\)",
        
        # Risk limit messages
        'risk_limit': r"❌ Risk limit rejected signal for (\S+):",
        
        # Final portfolio summary
        'final_cash': r"💰 Cash Balance: \$([\d.]+)",
        'final_equity': r"🏆 Total Equity: \$([\d.]+)"
    }
    
    current_timestamp = None
    timestamp_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
    
    for line in log_text.split('\n'):
        # Extract timestamp
        ts_match = re.search(timestamp_pattern, line)
        if ts_match:
            current_timestamp = ts_match.group(1)
        
        # Check for fills
        fill_match = re.search(patterns['fill'], line)
        if fill_match:
            symbol = fill_match.group(1)
            side_int = int(fill_match.group(2))
            quantity = Decimal(fill_match.group(3))
            price = Decimal(fill_match.group(4))
            
            fills.append({
                'timestamp': current_timestamp,
                'type': 'FILL',
                'symbol': symbol,
                'side': 'SELL' if side_int < 0 else 'BUY',
                'quantity': abs(quantity),
                'price': price,
                'value': abs(quantity) * price
            })
        
        # Check for portfolio updates
        portfolio_match = re.search(patterns['portfolio_update'], line)
        if portfolio_match:
            symbol = portfolio_match.group(1)
            side = portfolio_match.group(2)
            quantity = Decimal(portfolio_match.group(3))
            price = Decimal(portfolio_match.group(4))
            
            portfolio_updates.append({
                'timestamp': current_timestamp,
                'type': 'PORTFOLIO_UPDATE',
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'value': quantity * price
            })
        
        # Check for position closures
        close_match = re.search(patterns['close'], line)
        if close_match:
            symbol = close_match.group(1)
            quantity = Decimal(close_match.group(2))
            price = Decimal(close_match.group(3))
            cash_change = Decimal(close_match.group(4))
            
            trades.append({
                'timestamp': current_timestamp,
                'type': 'CLOSE',
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'cash_change': cash_change
            })
        
        # Check for orders
        order_match = re.search(patterns['order'], line)
        if order_match:
            symbol = order_match.group(1)
            side_int = int(order_match.group(2))
            quantity = Decimal(order_match.group(3))
            price = Decimal(order_match.group(4))
            
            trades.append({
                'timestamp': current_timestamp,
                'type': 'ORDER',
                'symbol': symbol,
                'side': 'SELL' if side_int < 0 else 'BUY',
                'quantity': abs(quantity),
                'price': price
            })
    
    # Extract final portfolio state
    final_state = {}
    for line in log_text.split('\n'):
        cash_match = re.search(patterns['final_cash'], line)
        if cash_match:
            final_state['cash'] = Decimal(cash_match.group(1))
        
        equity_match = re.search(patterns['final_equity'], line)
        if equity_match:
            final_state['equity'] = Decimal(equity_match.group(1))
    
    return fills, portfolio_updates, trades, final_state

def calculate_trade_summary(fills, closes, initial_capital=100000):
    """Calculate P&L and trade statistics."""
    positions = defaultdict(lambda: {'quantity': 0, 'cost_basis': Decimal('0'), 'trades': []})
    realized_pnl = Decimal('0')
    total_commission = Decimal('0')
    
    # Process fills to build positions
    for fill in fills:
        symbol = fill['symbol']
        quantity = fill['quantity']
        price = fill['price']
        side = fill['side']
        
        if side == 'BUY':
            # Long position
            positions[symbol]['quantity'] += quantity
            positions[symbol]['cost_basis'] += quantity * price
        else:  # SELL
            # Short position (negative quantity)
            positions[symbol]['quantity'] -= quantity
            positions[symbol]['cost_basis'] -= quantity * price
        
        positions[symbol]['trades'].append(fill)
        
        # Estimate commission (if not in logs)
        commission = quantity * Decimal('0.005')  # $0.005 per share
        total_commission += commission
    
    # Process position closures
    for close in closes:
        if close['type'] == 'CLOSE':
            symbol = close['symbol']
            quantity = close['quantity']
            close_price = close['price']
            
            if symbol in positions:
                # Calculate P&L for this close
                avg_cost = positions[symbol]['cost_basis'] / positions[symbol]['quantity'] if positions[symbol]['quantity'] != 0 else 0
                
                # For short positions, P&L is inverted
                if quantity < 0:  # Short position being closed
                    pnl = quantity * (avg_cost - close_price)
                else:  # Long position being closed
                    pnl = quantity * (close_price - avg_cost)
                
                realized_pnl += pnl
    
    return {
        'realized_pnl': realized_pnl,
        'total_commission': total_commission,
        'net_pnl': realized_pnl - total_commission,
        'open_positions': {k: v for k, v in positions.items() if v['quantity'] != 0}
    }

def print_trade_audit(fills, portfolio_updates, trades, final_state):
    """Print formatted trade audit report."""
    print("=" * 80)
    print("TRADE AUDIT REPORT")
    print("=" * 80)
    
    # Trading Activity
    print("\n📊 TRADING ACTIVITY")
    print("-" * 40)
    
    if fills:
        print(f"\n📈 FILLS ({len(fills)} total):")
        for fill in fills:
            print(f"   {fill['timestamp']} - {fill['symbol']} {fill['side']} {fill['quantity']} @ ${fill['price']:.4f}")
    else:
        print("\n📈 FILLS: None")
    
    # Position Closures
    closes = [t for t in trades if t['type'] == 'CLOSE']
    if closes:
        print(f"\n🔄 POSITION CLOSURES ({len(closes)} total):")
        for close in closes:
            print(f"   {close['timestamp']} - {close['symbol']} closed {abs(close['quantity'])} shares @ ${close['price']:.4f}")
            print(f"      Cash impact: ${close['cash_change']:.2f}")
    
    # Calculate summary
    summary = calculate_trade_summary(fills, closes)
    
    print("\n💰 P&L SUMMARY")
    print("-" * 40)
    print(f"   Realized P&L: ${summary['realized_pnl']:.2f}")
    print(f"   Commissions: ${summary['total_commission']:.2f}")
    print(f"   Net P&L: ${summary['net_pnl']:.2f}")
    
    # Final Portfolio State
    if final_state:
        print("\n🏆 FINAL PORTFOLIO STATE")
        print("-" * 40)
        if 'cash' in final_state:
            print(f"   Cash Balance: ${final_state['cash']:.2f}")
        if 'equity' in final_state:
            print(f"   Total Equity: ${final_state['equity']:.2f}")
            
        # Calculate return
        initial_capital = Decimal('100000')
        if 'equity' in final_state:
            returns = ((final_state['equity'] - initial_capital) / initial_capital) * 100
            print(f"   Return: {returns:.3f}%")
    
    # Risk Metrics
    if fills:
        print("\n📊 RISK METRICS")
        print("-" * 40)
        total_trades = len(fills)
        total_volume = sum(f['value'] for f in fills)
        avg_trade_size = total_volume / total_trades if total_trades > 0 else 0
        print(f"   Total Trades: {total_trades}")
        print(f"   Total Volume: ${total_volume:.2f}")
        print(f"   Avg Trade Size: ${avg_trade_size:.2f}")

if __name__ == "__main__":
    # Read from file or stdin
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            log_text = f.read()
    else:
        log_text = sys.stdin.read()
    
    fills, portfolio_updates, trades, final_state = extract_trades_from_log(log_text)
    print_trade_audit(fills, portfolio_updates, trades, final_state)