"""Simple trade audit by parsing execution logs."""

import subprocess
import re
from decimal import Decimal
from collections import defaultdict

def run_backtest_and_audit():
    """Run backtest and parse logs for audit."""
    
    print("=" * 80)
    print("SIMPLE TRADE AUDIT - 50 BARS")
    print("=" * 80)
    
    # Run the backtest and capture output
    cmd = ["python", "main.py", "--config", "config/multi_strategy_test.yaml", "--bars", "50"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    output = result.stderr  # Logs go to stderr
    
    # Parse the output
    bars = []
    signals = []
    orders = []
    fills = []
    portfolio_updates = []
    
    # Regular expressions for parsing
    bar_pattern = r'\[BAR\] (\w+) @ ([\d\-\s:+]+) - Price: ([\d.]+)'
    signal_pattern = r'Generated signal.*?: (BUY|SELL) (\w+)'
    order_pattern = r'Created order.*?: (BUY|SELL) (\d+) (\w+)'
    fill_pattern = r'Simulated fill: (BUY|SELL) ([\d.]+) (\w+) @ ([\d.]+) \(market: ([\d.]+), slippage: ([\-\d.]+), commission: ([\d.]+)\)'
    position_pattern = r'Position: ([\-\d.]+) \(quantity_delta was ([\-\d.]+)\)'
    cash_pattern = r'Cash change: ([\-\d.]+), New balance: ([\d.]+)'
    closing_pattern = r'Closed position: (\w+) ([\-\d.]+) shares @ \$([\d.]+), cash change: \$([\-\d.]+)'
    
    # Parse bars
    for match in re.finditer(bar_pattern, output):
        bars.append({
            'symbol': match.group(1),
            'timestamp': match.group(2),
            'price': float(match.group(3))
        })
    
    # Parse fills
    for match in re.finditer(fill_pattern, output):
        fills.append({
            'side': match.group(1),
            'quantity': float(match.group(2)),
            'symbol': match.group(3),
            'fill_price': float(match.group(4)),
            'market_price': float(match.group(5)),
            'slippage': float(match.group(6)),
            'commission': float(match.group(7))
        })
    
    # Parse position and cash updates
    position_matches = list(re.finditer(position_pattern, output))
    cash_matches = list(re.finditer(cash_pattern, output))
    
    for i, (pos_match, cash_match) in enumerate(zip(position_matches, cash_matches)):
        portfolio_updates.append({
            'position': float(pos_match.group(1)),
            'quantity_delta': float(pos_match.group(2)),
            'cash_change': float(cash_match.group(1)),
            'cash_balance': float(cash_match.group(2))
        })
    
    # Parse closing
    closing_matches = list(re.finditer(closing_pattern, output))
    
    # Print results
    print(f"\nMarket Data Summary:")
    print(f"  Total bars: {len(bars)}")
    if bars:
        prices = [b['price'] for b in bars]
        print(f"  Price range: ${min(prices):.4f} - ${max(prices):.4f}")
        print(f"  First bar: {bars[0]['timestamp']} @ ${bars[0]['price']:.4f}")
        print(f"  Last bar: {bars[-1]['timestamp']} @ ${bars[-1]['price']:.4f}")
    
    print(f"\nTrades Executed: {len(fills)}")
    
    for i, fill in enumerate(fills, 1):
        print(f"\nTrade #{i}:")
        print(f"  Side: {fill['side']}")
        print(f"  Quantity: {fill['quantity']}")
        print(f"  Symbol: {fill['symbol']}")
        print(f"  Market Price: ${fill['market_price']:.4f}")
        print(f"  Fill Price: ${fill['fill_price']:.4f}")
        print(f"  Slippage: ${fill['slippage']:.4f}")
        print(f"  Commission: ${fill['commission']:.2f}")
        
        # Find the bar at fill time
        fill_bar_idx = None
        for j, bar in enumerate(bars):
            if bar['symbol'] == fill['symbol']:
                fill_bar_idx = j
                # Keep going to find the exact bar at fill time
                # For now, just use the current bar
        
        if fill_bar_idx is not None and fill_bar_idx < len(bars):
            bar = bars[fill_bar_idx]
            print(f"  Bar Price at Fill: ${bar['price']:.4f}")
            print(f"  Fill vs Bar Price Diff: ${fill['fill_price'] - bar['price']:.4f}")
    
    print(f"\nPortfolio Updates: {len(portfolio_updates)}")
    for i, update in enumerate(portfolio_updates, 1):
        print(f"\nUpdate #{i}:")
        print(f"  Position: {update['position']} shares")
        print(f"  Quantity Delta: {update['quantity_delta']}")
        print(f"  Cash Change: ${update['cash_change']:.2f}")
        print(f"  Cash Balance: ${update['cash_balance']:.2f}")
    
    if closing_matches:
        print(f"\nPosition Closing (END_OF_DATA):")
        for match in closing_matches:
            symbol = match.group(1)
            quantity = float(match.group(2))
            close_price = float(match.group(3))
            cash_change = float(match.group(4))
            
            print(f"  Symbol: {symbol}")
            print(f"  Quantity: {quantity} shares")
            print(f"  Close Price: ${close_price:.4f}")
            print(f"  Cash Change: ${cash_change:.2f}")
            
            # Compare with last market price
            last_market_price = bars[-1]['price'] if bars else None
            if last_market_price:
                print(f"  Last Market Price: ${last_market_price:.4f}")
                print(f"  Close vs Market Diff: ${close_price - last_market_price:.4f}")
    
    # Extract final results
    final_equity_match = re.search(r'Total Equity: \$([\d.]+)', output)
    if final_equity_match:
        final_equity = float(final_equity_match.group(1))
        initial_capital = 100000
        returns = ((final_equity - initial_capital) / initial_capital) * 100
        
        print(f"\nFinal Results:")
        print(f"  Initial Capital: ${initial_capital:,.2f}")
        print(f"  Final Equity: ${final_equity:,.2f}")
        print(f"  P&L: ${final_equity - initial_capital:,.2f}")
        print(f"  Return: {returns:.4f}%")

if __name__ == "__main__":
    run_backtest_and_audit()