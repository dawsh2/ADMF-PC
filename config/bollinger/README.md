# Bollinger Bands Configuration Examples

This directory contains various Bollinger Bands strategy configurations for different trading styles.

## Configuration Files

### 1. `simple_bollinger.yaml`
Basic Bollinger Bands configuration with standard parameters (20-period, 2 std dev).
- Good starting point for testing
- Includes commented sections for optimization and filters

### 2. `bollinger_mean_reversion.yaml`
Mean reversion strategy that trades when price touches the outer bands.
- Uses 2.5 standard deviations for more extreme levels
- Includes filters to ensure price actually touches the bands

### 3. `bollinger_breakout.yaml`
Breakout strategy that trades when price breaks beyond the bands with momentum.
- Uses 1.5 standard deviations for earlier signals
- Requires high volume (1.5x average) for confirmation

### 4. `bollinger_squeeze.yaml`
Volatility expansion strategy that identifies low volatility periods.
- Trades when Bollinger Bands narrow (squeeze) and then expand
- Uses band width calculation to identify squeeze conditions

### 5. `bollinger_with_eod.yaml`
Standard Bollinger strategy with end-of-day position closing enabled.
- Prevents overnight holding with `close_eod: true`
- Shows how to add additional time-based filters if needed

## Usage Examples

```bash
# Run a simple backtest
python main.py --config config/bollinger/simple_bollinger.yaml --backtest

# Run with optimization (uncomment parameter arrays in config)
python main.py --config config/bollinger/simple_bollinger.yaml --backtest --optimize

# Run with EOD closing from command line
python main.py --config config/bollinger/simple_bollinger.yaml --backtest --close-eod

# Run mean reversion strategy
python main.py --config config/bollinger/bollinger_mean_reversion.yaml --backtest
```

## Key Parameters

- **period**: Number of bars for the moving average (typically 20)
- **std_dev**: Number of standard deviations for bands (typically 2.0)
- **filter**: Optional expression to filter signals based on conditions

## Adding Filters

You can add filters to any strategy to improve signal quality:

```yaml
filter: |
  signal != 0 and (
    volume > volume_sma_20 * 1.2 and      # Volume confirmation
    atr_14 / atr_sma_50 >= 0.8 and       # Normal volatility
    bar_of_day < 72                       # No late day entries
  )
```

## Notes

- All configs use 5-minute SPY data by default
- Adjust `start_date` and `end_date` as needed
- The `close_eod` flag in execution config enables automatic EOD closing
- Band calculations assume features like `upper_band`, `lower_band`, and `middle_band` are available