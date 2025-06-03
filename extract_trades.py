#!/usr/bin/env python3
"""Extract trades from backtest logs."""

import sys
import re
from decimal import Decimal

def extract_trades_from_log(log_text):
    """Extract all trades and position updates from log text."""
    trades = []
    fills = []
    positions = []
    
    # Pattern for fills
    fill_pattern = r"\[FILL\] (.*?) @ \$([\d.]+) - (BUY|SELL) (\d+) shares"
    # Pattern for portfolio updates
    portfolio_pattern = r"💼 PortfolioContainer updated: (.*?) (BUY|SELL) (\d+) @ ([\d.]+)"
    # Pattern for position updates
    position_pattern = r"📈 Position: ([\d.-]+)"
    # Pattern for cash updates
    cash_pattern = r"💰 Cash change: \$([\d.-]+), New balance: \$([\d.-]+)"
    # Pattern for closed positions
    close_pattern = r"✅ Closed position: (.*?) ([\d.-]+) shares @ \$([\d.]+), cash change: \$([\d.-]+)"
    
    for line in log_text.split('\n'):
        # Extract fills
        fill_match = re.search(fill_pattern, line)
        if fill_match:
            fills.append({
                'type': 'FILL',
                'symbol': fill_match.group(1),
                'price': Decimal(fill_match.group(2)),
                'side': fill_match.group(3),
                'quantity': int(fill_match.group(4))
            })
        
        # Extract portfolio updates
        portfolio_match = re.search(portfolio_pattern, line)
        if portfolio_match:
            trades.append({
                'type': 'TRADE',
                'symbol': portfolio_match.group(1),
                'side': portfolio_match.group(2),
                'quantity': int(portfolio_match.group(3)),
                'price': Decimal(portfolio_match.group(4))
            })
        
        # Extract position closures
        close_match = re.search(close_pattern, line)
        if close_match:
            trades.append({
                'type': 'CLOSE',
                'symbol': close_match.group(1),
                'quantity': Decimal(close_match.group(2)),
                'price': Decimal(close_match.group(3)),
                'cash_change': Decimal(close_match.group(4))
            })
    
    return trades, fills

def calculate_pnl(trades):
    """Calculate P&L from trades."""
    positions = {}
    realized_pnl = Decimal('0')
    
    for trade in trades:
        if trade['type'] == 'TRADE':
            symbol = trade['symbol']
            if symbol not in positions:
                positions[symbol] = {'quantity': 0, 'cost_basis': Decimal('0')}
            
            if trade['side'] == 'BUY':
                # Add to position
                positions[symbol]['quantity'] += trade['quantity']
                positions[symbol]['cost_basis'] += trade['quantity'] * trade['price']
            else:  # SELL
                # Reduce position and calculate P&L
                if positions[symbol]['quantity'] > 0:
                    avg_cost = positions[symbol]['cost_basis'] / positions[symbol]['quantity']
                    pnl = trade['quantity'] * (trade['price'] - avg_cost)
                    realized_pnl += pnl
                    
                    positions[symbol]['quantity'] -= trade['quantity']
                    positions[symbol]['cost_basis'] -= trade['quantity'] * avg_cost
        
        elif trade['type'] == 'CLOSE':
            # Position closure
            realized_pnl += trade.get('cash_change', 0)
    
    return realized_pnl, positions

if __name__ == "__main__":
    # Read from stdin or file
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            log_text = f.read()
    else:
        log_text = sys.stdin.read()
    
    trades, fills = extract_trades_from_log(log_text)
    
    print("=== TRADES FOUND ===")
    for trade in trades:
        print(f"{trade['type']}: {trade}")
    
    print("\n=== FILLS FOUND ===")
    for fill in fills:
        print(f"{fill['type']}: {fill}")
    
    print("\n=== P&L CALCULATION ===")
    realized_pnl, final_positions = calculate_pnl(trades)
    print(f"Realized P&L: ${realized_pnl}")
    print(f"Final Positions: {final_positions}")