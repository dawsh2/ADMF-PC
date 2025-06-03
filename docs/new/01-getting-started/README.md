# Getting Started with ADMF-PC

Welcome! This guide will have you running your first algorithmic trading backtest in under 30 minutes.

## 📋 Prerequisites

- Python 3.11 or higher
- Basic familiarity with YAML
- No trading system experience required!

## 🚀 Quick Start (5 minutes)

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd ADMF-PC

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Sample Data

```bash
python scripts/download_sample_data.py
```

This downloads SPY (S&P 500 ETF) data for testing.

### 3. Run Your First Backtest

```bash
python main.py config/simple_backtest.yaml
```

You should see output showing:
- Data loading progress
- Strategy signals
- Trade execution
- Performance metrics

**Congratulations!** You've just run a momentum strategy backtest without writing any code.

## 🎯 What Just Happened?

You ran a complete algorithmic trading backtest using only a YAML configuration file. The system:

1. **Loaded Data**: Historical price data for SPY
2. **Calculated Indicators**: Moving averages for the momentum strategy  
3. **Generated Signals**: Buy/sell decisions based on momentum
4. **Managed Risk**: Position sizing and exposure limits
5. **Executed Trades**: Simulated order execution with realistic costs
6. **Reported Results**: Performance metrics and statistics

All of this happened through the configuration in `config/simple_backtest.yaml` - no programming required!

## 📚 Next Steps

### Understanding the System
1. **[Understanding Workflows](understanding-workflows.md)** - Learn how ADMF-PC orchestrates complex operations
2. **[Core Concepts](../02-core-concepts/README.md)** - Understand the architecture that makes this possible
3. **[First Backtest Deep Dive](first-backtest.md)** - Detailed walkthrough of what you just ran

### Start Building
1. **[Modify the Simple Strategy](first-backtest.md#modifying-parameters)** - Change parameters and see results
2. **[Try Different Strategies](../03-user-guide/configuring-strategies.md)** - Explore built-in strategies
3. **[Add Risk Management](../03-user-guide/risk-management.md)** - Configure position sizing and limits

### Common Questions

**Q: How is this different from other trading systems?**  
A: ADMF-PC uses a revolutionary architecture where everything is configuration. No inheritance, no programming, just YAML files that describe what you want.

**Q: Can I add my own strategies?**  
A: Yes! You can integrate Python functions, scikit-learn models, or any code that generates signals. See [Custom Components](../08-advanced-topics/custom-components.md).

**Q: Does this work with real trading?**  
A: Yes! The same YAML configuration works for both backtesting and live trading. See [Live Trading Guide](../03-user-guide/live-trading.md).

**Q: How fast is it?**  
A: Very fast! Signal replay optimization runs 10-100x faster than traditional backtesting. See [Performance Benchmarks](../04-reference/performance-benchmarks.md).

## 🆘 Getting Help

- **Installation Issues**: See [Troubleshooting](troubleshooting.md)
- **Conceptual Questions**: Read [Core Concepts](../02-core-concepts/README.md)
- **Examples**: Browse [Examples](../09-examples/README.md)
- **Community**: File issues at the project repository

## 🎓 Learning Paths

Based on your background, choose a learning path:

- **[Strategy Developer](../07-learning-paths/strategy-developer.md)** - For traders and quants
- **[ML Engineer](../07-learning-paths/ml-engineer.md)** - For data scientists
- **[System Administrator](../07-learning-paths/system-administrator.md)** - For deployment and operations

---

Ready to dive deeper? Continue to [Understanding Workflows](understanding-workflows.md) →