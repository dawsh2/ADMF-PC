"""Generate synthetic market data without external dependencies."""

import csv
import random
import math
from datetime import datetime, timedelta


def generate_synthetic_data(num_bars=5000):
    """Generate synthetic price data with mean reversion."""
    
    # Parameters
    start_price = 100.0
    target_low = 85.0
    target_high = 115.0
    volatility = 0.02
    mean_reversion_strength = 0.1
    
    # Initialize
    current_price = start_price
    start_time = datetime(2024, 1, 1, 9, 30, 0)
    
    data = []
    
    print(f"Generating {num_bars} bars of synthetic data...")
    
    for i in range(num_bars):
        # Calculate mean reversion force
        center = (target_high + target_low) / 2
        distance_from_center = (current_price - center) / center
        mean_reversion_force = -mean_reversion_strength * distance_from_center
        
        # Add some trend based on position in range
        if current_price < 90:
            trend = 0.001  # Slight upward bias when low
        elif current_price > 110:
            trend = -0.001  # Slight downward bias when high  
        else:
            trend = 0.0
        
        # Random component
        random_change = random.gauss(0, volatility)
        
        # Calculate price change
        price_change = trend + mean_reversion_force + random_change
        
        # Update price
        new_price = current_price * (1 + price_change)
        
        # Hard boundaries
        new_price = max(target_low - 5, min(target_high + 5, new_price))
        
        # Create OHLC data
        high = new_price * (1 + abs(random.gauss(0, volatility/4)))
        low = new_price * (1 - abs(random.gauss(0, volatility/4)))
        open_price = current_price
        close_price = new_price
        
        # Volume (random)
        volume = int(1000000 + random.gauss(0, 200000))
        
        # Timestamp
        timestamp = start_time + timedelta(minutes=i)
        
        data.append({
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close_price, 2),
            'volume': max(0, volume)
        })
        
        current_price = new_price
        
        # Progress
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1} bars...")
    
    return data


def save_to_csv(data, filename):
    """Save data to CSV file."""
    print(f"\nSaving to {filename}...")
    
    # Ensure data directory exists
    import os
    os.makedirs('data', exist_ok=True)
    
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Saved {len(data)} bars to {filename}")


def analyze_data(data):
    """Analyze the generated data."""
    prices = [row['close'] for row in data]
    
    print("\nData Analysis:")
    print(f"  Total bars: {len(data)}")
    print(f"  Price range: ${min(prices):.2f} - ${max(prices):.2f}")
    print(f"  Average price: ${sum(prices)/len(prices):.2f}")
    
    # Count potential trades
    buy_signals = sum(1 for p in prices if p <= 90)
    sell_signals = sum(1 for p in prices if p >= 100)
    
    print(f"\nTrading signals (based on $90/$100 thresholds):")
    print(f"  Potential buy signals (price <= $90): {buy_signals}")
    print(f"  Potential sell signals (price >= $100): {sell_signals}")
    print(f"  Signal ratio: {buy_signals/len(prices)*100:.1f}% / {sell_signals/len(prices)*100:.1f}%")


def main():
    """Generate synthetic data for backtesting."""
    print("=" * 70)
    print("SYNTHETIC DATA GENERATOR")
    print("=" * 70)
    print("Generating data with mean-reverting price action")
    print("Target range: $85 - $115")
    print("Trading rule: Buy at $90, Sell at $100")
    print("=" * 70)
    
    # Generate data
    data = generate_synthetic_data(num_bars=5000)
    
    # Analyze
    analyze_data(data)
    
    # Save
    save_to_csv(data, 'data/SYNTH_1min.csv')
    
    print("\n" + "=" * 70)
    print("READY FOR BACKTESTING!")
    print("=" * 70)
    print("Data saved to: data/SYNTH_1min.csv")
    print("Now you can run backtests with the YAML configuration")
    print("")


if __name__ == "__main__":
    main()