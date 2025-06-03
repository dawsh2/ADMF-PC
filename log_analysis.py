"""Analyze the detailed log for logical consistency."""

import re

def analyze_log(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    print("=" * 80)
    print("LOG ANALYSIS - LOGICAL CONSISTENCY CHECK")
    print("=" * 80)
    
    # Track key events
    events = {
        'bars': [],
        'signals': [],
        'orders': [],
        'fills': [],
        'positions': [],
        'cash_updates': [],
        'risk_checks': []
    }
    
    # Parse the log
    for i, line in enumerate(lines):
        # Bar events
        if '[BAR]' in line:
            match = re.search(r'\[BAR\] (\w+) @ ([\d\-\s:+]+) - Price: ([\d.]+)', line)
            if match:
                events['bars'].append({
                    'symbol': match.group(1),
                    'timestamp': match.group(2),
                    'price': float(match.group(3)),
                    'line': i
                })
        
        # Signal generation
        if 'Generated signal' in line or 'signal triggered' in line:
            events['signals'].append({'line': i, 'text': line.strip()})
        
        # Order creation
        if '[ORDER]' in line or 'Created order' in line:
            events['orders'].append({'line': i, 'text': line.strip()})
        
        # Fill execution
        if 'Simulated fill:' in line:
            match = re.search(r'Simulated fill: (\w+) ([\d.]+) (\w+) @ ([\d.]+) \(market: ([\d.]+)', line)
            if match:
                events['fills'].append({
                    'side': match.group(1),
                    'quantity': float(match.group(2)),
                    'symbol': match.group(3),
                    'fill_price': float(match.group(4)),
                    'market_price': float(match.group(5)),
                    'line': i,
                    'text': line.strip()
                })
        
        # Position updates
        if 'Position:' in line and 'quantity_delta' in line:
            match = re.search(r'Position: ([\-\d.]+) \(quantity_delta was ([\-\d.]+)\)', line)
            if match:
                events['positions'].append({
                    'position': float(match.group(1)),
                    'delta': float(match.group(2)),
                    'line': i
                })
        
        # Cash updates
        if 'Cash change:' in line:
            match = re.search(r'Cash change: ([\-\d.]+), New balance: ([\d.]+)', line)
            if match:
                events['cash_updates'].append({
                    'change': float(match.group(1)),
                    'balance': float(match.group(2)),
                    'line': i
                })
        
        # Risk checks
        if 'RiskContainer' in line and ('skipping' in line or 'Already have' in line):
            events['risk_checks'].append({'line': i, 'text': line.strip()})
    
    # Analyze consistency
    print(f"\n1. DATA FLOW:")
    print(f"   Bars processed: {len(events['bars'])}")
    if events['bars']:
        print(f"   Price range: ${events['bars'][0]['price']:.4f} - ${events['bars'][-1]['price']:.4f}")
        print(f"   First bar: {events['bars'][0]['timestamp']}")
        print(f"   Last bar: {events['bars'][-1]['timestamp']}")
    
    print(f"\n2. SIGNAL GENERATION:")
    print(f"   Signals generated: {len(events['signals'])}")
    for sig in events['signals'][:3]:
        print(f"   Line {sig['line']}: {sig['text'][:80]}...")
    
    print(f"\n3. ORDER MANAGEMENT:")
    print(f"   Orders created: {len(events['orders'])}")
    print(f"   Risk checks (position limits): {len(events['risk_checks'])}")
    
    # Check for logical issues
    print(f"\n4. TRADE EXECUTION:")
    print(f"   Fills executed: {len(events['fills'])}")
    for fill in events['fills']:
        print(f"\n   Fill at line {fill['line']}:")
        print(f"   - Side: {fill['side']}")
        print(f"   - Quantity: {fill['quantity']}")
        print(f"   - Market price: ${fill['market_price']:.4f}")
        print(f"   - Fill price: ${fill['fill_price']:.4f}")
        print(f"   - Slippage: ${fill['fill_price'] - fill['market_price']:.4f}")
        
        # Find the bar at fill time
        fill_line = fill['line']
        closest_bar = None
        for bar in reversed(events['bars']):
            if bar['line'] < fill_line:
                closest_bar = bar
                break
        
        if closest_bar:
            print(f"   - Closest bar price: ${closest_bar['price']:.4f}")
            print(f"   - Fill vs bar diff: ${fill['fill_price'] - closest_bar['price']:.4f}")
    
    print(f"\n5. POSITION TRACKING:")
    for i, pos in enumerate(events['positions']):
        print(f"\n   Position update {i+1}:")
        print(f"   - Position: {pos['position']}")
        print(f"   - Delta: {pos['delta']}")
        print(f"   - Consistency: {'✓' if pos['position'] == pos['delta'] else '✗'}")
    
    print(f"\n6. CASH FLOW:")
    initial_cash = 100000
    print(f"   Initial: ${initial_cash:,.2f}")
    
    for i, cash in enumerate(events['cash_updates']):
        print(f"\n   Update {i+1}:")
        print(f"   - Change: ${cash['change']:+.2f}")
        print(f"   - Balance: ${cash['balance']:,.2f}")
        
        if i == 0:
            expected = initial_cash + cash['change']
            print(f"   - Expected: ${expected:,.2f}")
            print(f"   - Match: {'✓' if abs(expected - cash['balance']) < 0.01 else '✗'}")
    
    # Check for logical inconsistencies
    print(f"\n7. LOGICAL CONSISTENCY CHECKS:")
    
    # Check 1: Position delta should match position for first trade
    if events['positions'] and events['positions'][0]['position'] != events['positions'][0]['delta']:
        print("   ✗ WARNING: First position doesn't match delta")
    else:
        print("   ✓ First position matches delta")
    
    # Check 2: Short position cash flow
    if events['fills'] and events['cash_updates']:
        first_fill = events['fills'][0]
        first_cash = events['cash_updates'][0]
        
        if first_fill['side'] == 'SELL':
            # For a short sale, we should receive cash
            expected_cash = first_fill['quantity'] * first_fill['fill_price']
            # Minus commission (assuming ~2.60)
            expected_change = expected_cash - 2.60
            
            print(f"\n   Short sale analysis:")
            print(f"   - Sold {first_fill['quantity']} @ ${first_fill['fill_price']:.4f}")
            print(f"   - Expected cash in: ~${expected_change:.2f}")
            print(f"   - Actual cash change: ${first_cash['change']:.2f}")
            print(f"   - Difference: ${abs(expected_change - first_cash['change']):.2f}")
    
    # Check for END_OF_DATA closing
    print(f"\n8. END_OF_DATA CLOSING:")
    for i, line in enumerate(lines):
        if 'Using market price' in line:
            print(f"   Line {i}: {line.strip()}")
        if 'Closed position:' in line and 'cash change:' in line:
            print(f"   Line {i}: {line.strip()}")

if __name__ == "__main__":
    analyze_log('detailed_audit.log')