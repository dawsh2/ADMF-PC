# Bollinger Strategy #10 Test Results Summary

## Test Configuration
- **Strategy**: Bollinger Bands
- **Parameters**: period=11, std_dev=2.0 (Strategy #10)
- **Test Location**: `config/bollinger/bollinger_10/`

## Key Findings

### 1. Isolated Test Results (bollinger_10)
- **Total bars processed**: 20,768
- **Signals generated**: 0
- **Trades**: 0
- **Performance**: N/A (no trades)

### 2. Parent Test Results (Strategy #10 in larger sweep)
- **Location**: `config/bollinger/results/20250623_062931/traces/bollinger_bands/SPY_5m_compiled_strategy_10.parquet`
- **Total records**: 4,698
- **Signal distribution**:
  - Buy signals: 1,134 (24.1%)
  - Sell signals: 1,235 (26.3%)
  - Neutral: 2,329 (49.6%)
- **Number of trades**: 965
- **Performance**:
  - Gross return: +9.74%
  - Transaction costs (10 bps/trade): -96.50%
  - **Net return: -86.76%** (massive loss due to overtrading)

## Critical Insights

1. **Data Mismatch**: The parent test used 1-minute data (`SPY_5m_1m.csv`) while the isolated bollinger_10 test used 5-minute data (`SPY_5m`). This explains why the isolated test generated no signals.

2. **Overtrading Problem**: With 965 trades, the strategy is trading far too frequently. Even with a positive gross return of 9.74%, transaction costs completely destroy performance.

3. **Parameters Too Tight**: The period=11, std_dev=2.0 parameters appear to generate too many signals on 1-minute data, leading to excessive trading.

## Recommendations

1. **For the isolated test**: Try running with 1-minute data to match the parent test, or adjust parameters for 5-minute data (e.g., larger period or smaller std_dev).

2. **For Strategy #10**: This configuration is not viable for live trading due to:
   - Excessive trading frequency (965 trades)
   - Negative net returns after costs (-86.76%)
   - Transaction costs 10x larger than gross returns

3. **Parameter Optimization**: Consider:
   - Increasing the period (e.g., 20-30) to reduce signal frequency
   - Adjusting std_dev based on timeframe
   - Adding filters to reduce false signals
   - Implementing minimum holding periods

## Conclusion

Strategy #10 (period=11, std_dev=2.0) is a clear example of a strategy that looks profitable before costs but is completely unviable after accounting for transaction costs. The isolated test's failure to generate any signals on 5-minute data suggests the parameters are highly sensitive to the data timeframe.