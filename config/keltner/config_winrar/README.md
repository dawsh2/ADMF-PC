# Keltner Bands Winning Configuration

This configuration implements the best-performing Keltner Bands mean reversion strategy based on analysis of 2,750 parameter combinations.

## Key Parameters

### Primary Strategy (16.92% annual return)
- **Period**: 50 bars (4.2 hours)
- **Multiplier**: 1.0 (tight bands)
- **Filters**: Volume, volatility, trend alignment, and time-based

### Secondary Strategy (8.03% average return)
- **Period**: 30 bars (2.5 hours)
- **Multiplier**: 1.0 (tight bands)
- **Filters**: Similar pattern with tighter constraints

## Performance Expectations

Based on historical analysis:
- **Annual Return**: 15-17% (compound)
- **Win Rate**: 75-79%
- **Trades per Day**: 1.5-2.5
- **Average Trade Duration**: 3-5 hours

## Key Success Factors

1. **Tight Bands (1.0 multiplier)**: All top strategies use the tightest Keltner bands
2. **Longer Periods**: 30-50 bar periods outperform shorter ones
3. **Selective Filters**: Reduce false signals while maintaining profitability
4. **Mean Reversion**: Works best in range-bound conditions

## Risk Management

- Position sizing: 2% risk per trade
- Daily stop loss: 2% of capital
- Maximum 5 concurrent positions
- Force exit at market close

## Usage

```bash
# Run backtest
python main.py --config config/keltner/config_winrar/config.yaml

# Run with specific date range
python main.py --config config/keltner/config_winrar/config.yaml --start 2024-01-01 --end 2024-06-30
```

## Notes

- Returns are before transaction costs
- Actual performance may vary based on execution and market conditions
- Consider paper trading before live implementation