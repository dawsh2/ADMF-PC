# ADMF-PC in 5 Minutes

Get your first trading strategy running in under 5 minutes. No programming required!

## 🚀 Prerequisites

You need:
- Python 3.11+
- Git
- 5 minutes

## 📥 Step 1: Install (1 minute)

```bash
# Clone the repository
git clone https://github.com/your-org/ADMF-PC.git
cd ADMF-PC

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Step 2: Create Your First Strategy (1 minute)

Create a file called `my_first_strategy.yaml`:

```yaml
# my_first_strategy.yaml
workflow:
  type: "backtest"
  name: "My First Momentum Strategy"

data:
  source: "csv"
  file_path: "data/SPY_1m.csv"
  symbols: ["SPY"]
  start_date: "2023-01-01"
  end_date: "2023-12-31"

strategies:
  - name: "simple_momentum"
    type: "momentum"
    parameters:
      fast_period: 10
      slow_period: 30
      signal_threshold: 0.02

risk:
  initial_capital: 100000
  position_size_pct: 2.0
  max_drawdown_pct: 10.0

output:
  generate_report: true
  save_signals: true
  plot_charts: true
```

## ▶️ Step 3: Run Your Strategy (1 minute)

```bash
python main.py my_first_strategy.yaml
```

## 📈 Step 4: View Results (1 minute)

Your results are saved in `output/backtest_[timestamp]/`:

```
output/backtest_2024_01_15_093000/
├── performance_report.html    # Open this in your browser!
├── signals.csv               # All trading signals
├── trades.csv                # Executed trades
├── equity_curve.png          # Visual performance
└── statistics.json           # Performance metrics
```

## 🎉 Success! What Just Happened?

You just:
1. **Configured** a momentum trading strategy (no code!)
2. **Backtested** it on real market data
3. **Generated** professional performance reports

### Your Results Show:
- **Total Return**: Your strategy's profit/loss
- **Sharpe Ratio**: Risk-adjusted performance
- **Max Drawdown**: Worst peak-to-trough loss
- **Trade Count**: Number of trades executed

## 🔄 Step 5: Quick Experiments (1 minute)

Try these instant modifications:

### Change Strategy Parameters
```yaml
strategies:
  - name: "faster_momentum"
    type: "momentum"
    parameters:
      fast_period: 5      # Was 10
      slow_period: 20     # Was 30
      signal_threshold: 0.01  # Was 0.02
```

### Add Risk Management
```yaml
risk:
  initial_capital: 100000
  position_size_pct: 1.0      # Smaller positions
  max_drawdown_pct: 5.0       # Tighter risk control
  stop_loss_pct: 2.0          # Add stop loss
  take_profit_pct: 5.0        # Add profit target
```

### Test Different Time Period
```yaml
data:
  start_date: "2020-01-01"    # Further back
  end_date: "2023-12-31"      # Longer test
```

## 🎯 Common Next Steps

### Run Parameter Optimization
```yaml
# optimization.yaml
workflow:
  type: "optimization"
  algorithm: "grid_search"

parameter_space:
  fast_period: [5, 10, 15, 20]
  slow_period: [20, 30, 40, 50]
  signal_threshold: [0.01, 0.02, 0.03]
```

### Combine Multiple Strategies
```yaml
strategies:
  - name: "momentum"
    type: "momentum"
    allocation: 0.5
    
  - name: "mean_reversion"
    type: "mean_reversion"
    allocation: 0.5
```

### Add Market Regime Detection
```yaml
classifiers:
  - type: "volatility_regime"
    name: "vol_regime"
    
strategies:
  - name: "adaptive_momentum"
    type: "regime_adaptive"
    regime_strategies:
      low_volatility: {type: "momentum", fast: 10}
      high_volatility: {type: "mean_reversion", period: 20}
```

## ❓ Quick Troubleshooting

### "No data file found"
→ Download sample data:
```bash
python scripts/download_sample_data.py
```

### "Module not found"
→ Activate virtual environment:
```bash
source venv/bin/activate
```

### "Invalid configuration"
→ Check YAML indentation (use spaces, not tabs!)

## 📚 What's Next?

**Congratulations!** You've run your first strategy. Now:

1. **Understand the magic**: Read [CONCEPTS.md](CONCEPTS.md) (15 min)
2. **Build something custom**: Follow [FIRST_TASK.md](FIRST_TASK.md) (30 min)
3. **Explore all features**: See [ONBOARDING.md](ONBOARDING.md) (2 hours)

## 💡 Quick Tips

- **No Code Needed**: Everything is configuration
- **Mix Anything**: Combine any strategies or indicators
- **Fast Iteration**: Change YAML, run again
- **Production Ready**: Same config works in live trading

---

**🚀 You're now ready to explore the full power of ADMF-PC!**

Return to [Onboarding Hub](README.md) | Jump to [Key Concepts](CONCEPTS.md)