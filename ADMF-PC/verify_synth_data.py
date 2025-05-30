#!/usr/bin/env python3
"""
Verify SYNTH data has correct price movements.
"""

import csv

# Check first 100 rows of SYNTH data
with open('data/SYNTH_1min.csv', 'r') as f:
    reader = csv.DictReader(f)
    
    buy_opportunities = 0
    sell_opportunities = 0
    
    for i, row in enumerate(reader):
        if i >= 1000:
            break
            
        close = float(row['close'])
        
        if close <= 90:
            buy_opportunities += 1
            print(f"Row {i}: BUY opportunity at ${close:.2f}")
            
        if close >= 100:
            sell_opportunities += 1
            if sell_opportunities <= 5:  # Only show first 5
                print(f"Row {i}: SELL opportunity at ${close:.2f}")
    
    print(f"\nIn first 1000 bars:")
    print(f"  Buy opportunities (price <= $90): {buy_opportunities}")
    print(f"  Sell opportunities (price >= $100): {sell_opportunities}")