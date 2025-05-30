"""Generate synthetic market data for testing."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def generate_synthetic_price_data(
    symbol: str = "SYNTH",
    start_price: float = 100.0,
    volatility: float = 0.02,
    drift: float = 0.0001,
    num_bars: int = 1000,
    frequency: str = "1min"
) -> pd.DataFrame:
    """Generate synthetic OHLCV data with realistic patterns.
    
    Creates data with a simple rule for testing:
    - Price oscillates between 90 and 110
    - Buy signal when price <= 90
    - Sell signal when price >= 100
    """
    # Generate timestamps
    start_date = datetime(2023, 1, 1, 9, 30)  # Market open
    if frequency == "1min":
        freq = "T"  # minute frequency
        market_hours_per_day = 390  # 6.5 hours
    else:
        freq = "D"
        market_hours_per_day = 1
    
    dates = pd.date_range(start=start_date, periods=num_bars, freq=freq)
    
    # Generate synthetic price with mean reversion
    prices = [start_price]
    
    for i in range(1, num_bars):
        # Add some mean reversion to keep price in range
        current_price = prices[-1]
        
        # Mean reversion force
        if current_price > 105:
            drift_adj = -0.001  # Drift down
        elif current_price < 95:
            drift_adj = 0.001   # Drift up
        else:
            drift_adj = drift
        
        # Random walk with drift
        change = np.random.normal(drift_adj, volatility)
        new_price = current_price * (1 + change)
        
        # Ensure price stays in reasonable range
        new_price = max(85, min(115, new_price))
        prices.append(new_price)
    
    prices = np.array(prices)
    
    # Generate OHLC from price series
    data = []
    for i in range(len(dates)):
        # Add some intrabar volatility
        noise = np.random.uniform(0.998, 1.002, size=4)
        
        open_price = prices[i] * noise[0]
        high_price = prices[i] * max(noise) * 1.001
        low_price = prices[i] * min(noise) * 0.999
        close_price = prices[i]
        
        # Ensure OHLC relationships
        high_price = max(open_price, high_price, low_price, close_price)
        low_price = min(open_price, high_price, low_price, close_price)
        
        # Generate volume (higher volume at extremes)
        base_volume = 10000
        if close_price <= 92 or close_price >= 108:
            volume = base_volume * np.random.uniform(1.5, 2.5)
        else:
            volume = base_volume * np.random.uniform(0.8, 1.2)
        
        data.append({
            'timestamp': dates[i],
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': int(volume)
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df


def generate_multiple_symbols(symbols: list, num_bars: int = 1000) -> dict:
    """Generate synthetic data for multiple symbols with correlations."""
    data = {}
    
    # Base parameters for each symbol
    params = {
        'SPY': {'start_price': 100, 'volatility': 0.015, 'drift': 0.0001},
        'QQQ': {'start_price': 95, 'volatility': 0.020, 'drift': 0.00015},
        'IWM': {'start_price': 90, 'volatility': 0.025, 'drift': 0.00005},
        'TLT': {'start_price': 105, 'volatility': 0.010, 'drift': -0.00005}  # Bonds - inverse correlation
    }
    
    for symbol in symbols:
        param = params.get(symbol, {'start_price': 100, 'volatility': 0.02, 'drift': 0.0001})
        data[symbol] = generate_synthetic_price_data(
            symbol=symbol,
            num_bars=num_bars,
            **param
        )
    
    return data


def save_synthetic_data():
    """Generate and save synthetic data files."""
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Generate data
    print("Generating synthetic market data...")
    
    # Single symbol with 5000 bars
    spy_data = generate_synthetic_price_data("SPY", num_bars=5000)
    spy_data.to_csv(data_dir / "SPY_1min.csv")
    print(f"Generated SPY_1min.csv with {len(spy_data)} bars")
    print(f"Price range: ${spy_data['close'].min():.2f} - ${spy_data['close'].max():.2f}")
    
    # Multiple symbols with 2000 bars each
    symbols = ["SPY", "QQQ", "IWM", "TLT"]
    multi_data = generate_multiple_symbols(symbols, num_bars=2000)
    
    for symbol, df in multi_data.items():
        filename = data_dir / f"{symbol}_1h.csv"
        df.to_csv(filename)
        print(f"Generated {filename} with {len(df)} bars")
    
    # Create a simple info file
    info = {
        "generated": datetime.now().isoformat(),
        "description": "Synthetic market data for testing ADMF-PC",
        "files": {
            "SPY_1min.csv": "5000 bars of 1-minute SPY data",
            "SPY_1h.csv": "2000 bars of hourly SPY data",
            "QQQ_1h.csv": "2000 bars of hourly QQQ data",
            "IWM_1h.csv": "2000 bars of hourly IWM data",
            "TLT_1h.csv": "2000 bars of hourly TLT data"
        },
        "trading_rule": "Buy when price <= 90, Sell when price >= 100"
    }
    
    import json
    with open(data_dir / "data_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print("\nSynthetic data generation complete!")
    print("Trading rule: Buy at $90 or below, Sell at $100 or above")
    

if __name__ == "__main__":
    save_synthetic_data()