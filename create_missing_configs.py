#!/usr/bin/env python3
"""Create missing config files for strategies without configs."""

from pathlib import Path

# Missing configs and their appropriate categories
missing_configs = {
    'crossover': ['ma_crossover'],
    'momentum': ['dual_momentum', 'momentum_strategy', 'price_momentum'],
    'structure': ['atr_channel_breakout', 'fibonacci_retracement', 'price_action_swing', 'support_resistance_breakout'],
    'trend': ['multi_indicator_voting', 'trend_momentum_composite'],
    'volatility': ['donchian_bands']
}

# Config templates for each strategy
config_templates = {
    'ma_crossover': '''# Test configuration for MA Crossover strategy
name: test_ma_crossover
mode: signal_generation
symbols: ["SPY"]
start_date: "2024-01-01"
end_date: "2024-01-31"

# Generic moving average crossover strategy
strategy:
  ma_crossover:
    params:
      fast_period: 10
      slow_period: 20
      ma_type: sma  # sma or ema

# Run with: python main.py --config config/indicators/crossover/test_ma_crossover.yaml --signal-generation --bars 100
''',

    'dual_momentum': '''# Test configuration for Dual Momentum strategy
name: test_dual_momentum
mode: signal_generation
symbols: ["SPY"]
start_date: "2024-01-01"
end_date: "2024-01-31"

# Dual momentum strategy (price and time-series momentum)
strategy:
  dual_momentum:
    params:
      lookback_period: 20
      momentum_threshold: 0.0

# Run with: python main.py --config config/indicators/momentum/test_dual_momentum.yaml --signal-generation --bars 100
''',

    'momentum_strategy': '''# Test configuration for Momentum strategy
name: test_momentum_strategy
mode: signal_generation
symbols: ["SPY"]
start_date: "2024-01-01"
end_date: "2024-01-31"

# Basic momentum strategy
strategy:
  momentum_strategy:
    params:
      momentum_period: 10
      threshold: 0.0

# Run with: python main.py --config config/indicators/momentum/test_momentum_strategy.yaml --signal-generation --bars 100
''',

    'price_momentum': '''# Test configuration for Price Momentum strategy
name: test_price_momentum
mode: signal_generation
symbols: ["SPY"]
start_date: "2024-01-01"
end_date: "2024-01-31"

# Price momentum strategy
strategy:
  price_momentum:
    params:
      lookback_period: 20
      momentum_threshold: 0.01  # 1% threshold

# Run with: python main.py --config config/indicators/momentum/test_price_momentum.yaml --signal-generation --bars 100
''',

    'atr_channel_breakout': '''# Test configuration for ATR Channel Breakout strategy
name: test_atr_channel_breakout
mode: signal_generation
symbols: ["SPY"]
start_date: "2024-01-01"
end_date: "2024-01-31"

# ATR-based channel breakout strategy
strategy:
  atr_channel_breakout:
    params:
      atr_period: 14
      channel_period: 20
      atr_multiplier: 2.0

# Run with: python main.py --config config/indicators/structure/test_atr_channel_breakout.yaml --signal-generation --bars 100

# Expected behavior:
# - Uses SMA for channel middle
# - Upper channel = SMA + (ATR * multiplier)
# - Lower channel = SMA - (ATR * multiplier)
# - BUY signal when price > upper channel
# - SELL signal when price < lower channel
''',

    'fibonacci_retracement': '''# Test configuration for Fibonacci Retracement strategy
name: test_fibonacci_retracement
mode: signal_generation
symbols: ["SPY"]
start_date: "2024-01-01"
end_date: "2024-01-31"

# Fibonacci retracement level strategy
strategy:
  fibonacci_retracement:
    params:
      period: 50  # Lookback for high/low

# Run with: python main.py --config config/indicators/structure/test_fibonacci_retracement.yaml --signal-generation --bars 100

# Expected behavior:
# - Identifies recent high/low over lookback period
# - Calculates Fibonacci levels (23.6%, 38.2%, 50%, 61.8%, 78.6%)
# - Uptrend: BUY above 38.2%, SELL below 61.8%
# - Downtrend: SELL below 61.8%, BUY above 38.2%
''',

    'price_action_swing': '''# Test configuration for Price Action Swing strategy
name: test_price_action_swing
mode: signal_generation
symbols: ["SPY"]
start_date: "2024-01-01"
end_date: "2024-01-31"

# Price action swing high/low strategy
strategy:
  price_action_swing:
    params:
      period: 10  # Swing detection lookback

# Run with: python main.py --config config/indicators/structure/test_price_action_swing.yaml --signal-generation --bars 100

# Expected behavior:
# - Detects swing highs and lows
# - Higher highs + higher lows = uptrend (BUY)
# - Lower highs + lower lows = downtrend (SELL)
# - Mixed patterns = ranging (FLAT)
''',

    'support_resistance_breakout': '''# Test configuration for Support/Resistance Breakout strategy
name: test_support_resistance_breakout
mode: signal_generation
symbols: ["SPY"]
start_date: "2024-01-01"
end_date: "2024-01-31"

# Support/Resistance breakout strategy
strategy:
  support_resistance_breakout:
    params:
      period: 20
      threshold: 0.02  # 2% breakout threshold

# Run with: python main.py --config config/indicators/structure/test_support_resistance_breakout.yaml --signal-generation --bars 100

# Expected behavior:
# - Identifies support/resistance levels
# - BUY signal when price breaks above resistance
# - SELL signal when price breaks below support
''',

    'multi_indicator_voting': '''# Test configuration for Multi-Indicator Voting strategy
name: test_multi_indicator_voting
mode: signal_generation
symbols: ["SPY"]
start_date: "2024-01-01"
end_date: "2024-01-31"

# Multi-indicator voting ensemble strategy
strategy:
  multi_indicator_voting:
    params:
      rsi_period: 14
      macd_fast: 12
      macd_slow: 26
      macd_signal: 9
      bb_period: 20
      bb_stddev: 2.0
      min_votes: 2  # Minimum votes required for signal

# Run with: python main.py --config config/indicators/trend/test_multi_indicator_voting.yaml --signal-generation --bars 100

# Expected behavior:
# - Combines RSI, MACD, and Bollinger Bands
# - Each indicator votes: bullish (+1), bearish (-1), or neutral (0)
# - Signal when votes >= min_votes in same direction
''',

    'trend_momentum_composite': '''# Test configuration for Trend Momentum Composite strategy
name: test_trend_momentum_composite
mode: signal_generation
symbols: ["SPY"]
start_date: "2024-01-01"
end_date: "2024-01-31"

# Trend and momentum composite strategy
strategy:
  trend_momentum_composite:
    params:
      trend_period: 50
      momentum_period: 14
      adx_period: 14
      trend_strength_threshold: 25

# Run with: python main.py --config config/indicators/trend/test_trend_momentum_composite.yaml --signal-generation --bars 100

# Expected behavior:
# - Combines trend (SMA) and momentum (ADX/DI)
# - BUY: Uptrend + strong momentum (ADX > threshold)
# - SELL: Downtrend + strong momentum
# - FLAT: Weak trend (low ADX)
''',

    'donchian_bands': '''# Test configuration for Donchian Bands strategy
name: test_donchian_bands
mode: signal_generation
symbols: ["SPY"]
start_date: "2024-01-01"
end_date: "2024-01-31"

# Donchian channel bands (mean reversion) strategy
strategy:
  donchian_bands:
    params:
      period: 20

# Run with: python main.py --config config/indicators/volatility/test_donchian_bands.yaml --signal-generation --bars 100

# Expected behavior (MEAN REVERSION):
# - Upper band = highest high over period
# - Lower band = lowest low over period
# - BUY signal when price touches lower band (oversold)
# - SELL signal when price touches upper band (overbought)
# - FLAT when price is between bands
#
# Note: This is opposite of donchian_breakout which trades breakouts
'''
}

def create_missing_configs():
    """Create config files for strategies without configs."""
    created = 0
    
    for category, strategies in missing_configs.items():
        category_path = Path(f"config/indicators/{category}")
        
        # Ensure directory exists
        category_path.mkdir(parents=True, exist_ok=True)
        
        for strategy in strategies:
            config_file = category_path / f"test_{strategy}.yaml"
            
            if config_file.exists():
                print(f"⚠️  Config already exists: {config_file}")
                continue
            
            # Get template
            template = config_templates.get(strategy)
            if not template:
                print(f"❌ No template for: {strategy}")
                continue
            
            # Write config
            with open(config_file, 'w') as f:
                f.write(template)
            
            print(f"✅ Created: {config_file}")
            created += 1
    
    print(f"\nCreated {created} config files")

if __name__ == "__main__":
    create_missing_configs()