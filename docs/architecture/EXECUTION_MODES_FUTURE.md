# Future Enhancement: Execution Modes for L2/Tick Data

## Current State (OHLCV Data)

The unified architecture uses a **shared execution engine** across all portfolio containers. This works perfectly for OHLCV data because:
- All orders execute at the same price (bar close/VWAP)
- No realistic partial fills or order book modeling
- Market impact is simple slippage percentage
- Order conflicts don't exist (everyone fills at bar price)

## Future Requirement (L2/Tick Data)

When upgrading to L2 order book or tick data, the shared execution engine will need configuration options:

### The Problem
With realistic market microstructure:
- **Order conflicts**: Portfolio A sells 1000 AAPL while Portfolio B buys 800 AAPL
- **Partial fills**: Limited liquidity means orders compete for fills
- **Market impact**: Large aggregate orders move the market
- **Timing matters**: Microsecond differences affect execution

### The Solution: Execution Modes

```yaml
execution:
  mode: "isolated"  # Default for strategy comparison
  # Each portfolio gets independent execution
  # No cross-portfolio coordination
  # Best for: A/B testing strategies, parameter optimization
  
execution:
  mode: "shared"  # For realistic multi-strategy portfolios  
  # Single execution engine coordinates across portfolios
  # Handles order netting, fill distribution, aggregate impact
  # Best for: Production trading, realistic simulation
```

## Implementation Checklist (When Needed)

When implementing execution modes for L2/tick data:

1. **Configuration Schema**
   ```yaml
   execution:
     mode: "shared" | "isolated"
     shared_config:
       enable_order_netting: true
       enable_fill_distribution: true
       market_impact_model: "linear" | "square_root"
     isolated_config:
       parallel_execution: true
       independent_fills: true
   ```

2. **Architecture Changes**
   - For `isolated`: Embed ExecutionEngine in each PortfolioContainer
   - For `shared`: Keep current architecture
   - Add mode detection in TopologyBuilder

3. **Testing Requirements**
   - Test order conflict resolution with shared mode
   - Test complete isolation with isolated mode
   - Verify no performance regression for OHLCV data
   - Add L2 data integration tests

4. **Documentation Updates**
   - Update UNIFIED_ARCHITECTURE.md
   - Add examples for both modes
   - Document when to use each mode

## Warning Signs You Need This

If you see these behaviors with L2/tick data:
- Unrealistic fill rates (every order fills completely)
- No market impact from large orders
- Strategies that trade against each other get perfect fills
- Backtests too optimistic compared to live trading

## Current Action: None Required

The shared execution engine works perfectly for OHLCV data. Implement execution modes only when upgrading to more sophisticated market data.

---

**Note**: This is a future enhancement. Do not implement until L2/tick data is added to the system.