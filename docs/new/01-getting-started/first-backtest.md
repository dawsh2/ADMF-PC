# Your First Backtest - Deep Dive

This tutorial provides a detailed walkthrough of running and understanding your first ADMF-PC backtest.

## 📊 What We'll Build

We'll run a simple momentum strategy that:
- Buys when price crosses above its 20-day moving average
- Sells when price crosses below its 20-day moving average
- Manages risk with position sizing limits
- Tracks performance with detailed metrics

## 🏃 Running the Backtest

### Step 1: Examine the Configuration

First, let's look at the configuration file:

```yaml
# config/simple_backtest.yaml
workflow:
  type: "backtest"
  name: "Simple Momentum Strategy"
  description: "Basic momentum strategy with 20-day moving average"

data:
  source:
    type: "csv"
    path: "data/SPY_1m.csv"
  symbols: ["SPY"]
  start_date: "2023-01-01"
  end_date: "2023-12-31"
  
strategies:
  - type: "momentum"
    name: "MA20_Momentum"
    params:
      fast_period: 10
      slow_period: 20
      signal_threshold: 0.01  # 1% threshold for signals
    
risk_management:
  type: "fixed"
  params:
    position_size_pct: 2.0      # 2% of capital per position
    max_positions: 5            # Maximum 5 concurrent positions
    max_exposure_pct: 10.0      # Maximum 10% total exposure
    stop_loss_pct: 2.0          # 2% stop loss
    
execution:
  type: "market"
  params:
    slippage_bps: 10           # 10 basis points slippage
    commission_per_share: 0.01  # $0.01 per share commission
    
portfolio:
  initial_capital: 100000       # $100,000 starting capital
  currency: "USD"
  
reporting:
  type: "html"
  output_path: "reports/simple_backtest.html"
  include_charts: true
```

### Step 2: Run the Backtest

```bash
python main.py config/simple_backtest.yaml
```

### Step 3: Understanding the Output

You'll see output like:

```
[INFO] Loading configuration from config/simple_backtest.yaml
[INFO] Initializing Coordinator in TRADITIONAL mode
[INFO] Creating BacktestContainer: simple_momentum_strategy
[INFO] Loading data for SPY from 2023-01-01 to 2023-12-31
[INFO] Loaded 250 trading days, 97,500 bars
[INFO] Initializing strategy: MA20_Momentum
[INFO] Starting backtest execution...

[SIGNAL] 2023-01-15 09:45:00 | SPY | BUY | Strength: 0.85
[ORDER] 2023-01-15 09:45:00 | SPY | BUY | 495 shares @ $401.25
[FILL] 2023-01-15 09:45:00 | SPY | BUY | 495 shares @ $401.30 | Commission: $4.95

...

[INFO] Backtest complete. Generating report...
[INFO] Report saved to: reports/simple_backtest.html

=== Performance Summary ===
Total Return: 12.5%
Sharpe Ratio: 1.85
Max Drawdown: -8.2%
Win Rate: 58%
Total Trades: 47
```

## 🔍 Understanding What Happened

### 1. **Coordinator Initialization**
The Coordinator (the "central brain") orchestrated the entire process:
- Loaded configuration
- Created containers
- Managed execution flow
- Ensured reproducibility

### 2. **Container Creation**
A `BacktestContainer` was created with isolated:
- Event bus (for internal communication)
- Data handler
- Strategy instance
- Risk manager
- Execution engine
- Portfolio tracker

### 3. **Event Flow**
The system processed events in this sequence:

```
Market Data (BAR events)
    ↓
Indicator Calculation (INDICATOR events)
    ↓
Strategy Logic (SIGNAL events)
    ↓
Risk Management (validated signals)
    ↓
Order Generation (ORDER events)
    ↓
Execution Simulation (FILL events)
    ↓
Portfolio Update (PORTFOLIO_UPDATE events)
```

### 4. **Component Interaction**
All components communicated through events, never directly:
- Strategy emitted SIGNAL events
- Risk manager validated and emitted ORDER events
- Execution engine simulated fills and emitted FILL events
- Portfolio tracked positions via PORTFOLIO_UPDATE events

## 🛠️ Modifying Parameters

### Try Different Moving Average Periods

```yaml
strategies:
  - type: "momentum"
    params:
      fast_period: 5    # Changed from 10
      slow_period: 50   # Changed from 20
```

### Adjust Risk Parameters

```yaml
risk_management:
  params:
    position_size_pct: 1.0      # More conservative
    max_positions: 3            # Fewer positions
    stop_loss_pct: 1.0          # Tighter stop loss
```

### Change Date Range

```yaml
data:
  start_date: "2022-01-01"      # Earlier start
  end_date: "2023-12-31"        # Two years of data
```

## 📈 Analyzing Results

### View the HTML Report
Open `reports/simple_backtest.html` in your browser to see:
- Equity curve
- Drawdown chart
- Trade distribution
- Performance metrics
- Trade log

### Key Metrics Explained

- **Total Return**: Overall profit/loss percentage
- **Sharpe Ratio**: Risk-adjusted returns (higher is better)
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profits / Gross losses

## 🔧 Common Modifications

### 1. Add Multiple Strategies

```yaml
strategies:
  - type: "momentum"
    name: "Fast_Momentum"
    params:
      fast_period: 5
      slow_period: 20
    allocation: 0.5
    
  - type: "mean_reversion"
    name: "MeanRev_RSI"
    params:
      period: 14
      oversold: 30
      overbought: 70
    allocation: 0.5
```

### 2. Enable Signal Capture

```yaml
workflow:
  type: "backtest"
  capture_signals: true  # Save signals for later analysis
  signal_output_path: "signals/momentum_signals.pkl"
```

### 3. Add Performance Constraints

```yaml
risk_management:
  params:
    max_drawdown_pct: 10.0     # Stop if drawdown exceeds 10%
    daily_loss_limit_pct: 2.0  # Stop trading if daily loss > 2%
```

## 🎯 Next Steps

Now that you understand the basics:

1. **[Try Different Strategies](../03-user-guide/configuring-strategies.md)** - Explore mean reversion, pairs trading, etc.
2. **[Optimize Parameters](../03-user-guide/optimization.md)** - Find the best parameter values
3. **[Build Multi-Phase Workflows](../03-user-guide/multi-phase-workflows.md)** - Combine optimization with validation
4. **[Understand Container Architecture](../02-core-concepts/container-architecture.md)** - Deep dive into isolation

## 💡 Key Takeaways

1. **Zero Code**: You configured a complete trading system without programming
2. **Event-Driven**: Components communicated through events, not method calls
3. **Isolated Containers**: Each component ran in isolation for safety
4. **Reproducible**: The Coordinator ensured identical results across runs
5. **Configurable**: Every aspect can be modified through YAML

## 🤔 Common Questions

**Q: Why didn't I see the actual price data?**  
A: The system processes data internally through events. Enable debug logging to see detailed data flow.

**Q: Can I use minute-by-minute data?**  
A: Yes! The system supports any timeframe. Just ensure your data has the correct format.

**Q: How do I add custom indicators?**  
A: See [Custom Components](../08-advanced-topics/custom-components.md) for adding your own indicators.

**Q: Is this realistic?**  
A: Yes! The backtest includes slippage, commissions, and realistic fill simulation.

---

Ready for more? Continue to [Understanding Workflows](understanding-workflows.md) →