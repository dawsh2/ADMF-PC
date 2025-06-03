# User Guide

This guide provides task-oriented documentation for common ADMF-PC activities. Each section focuses on accomplishing specific goals rather than explaining concepts.

## 🎯 Quick Task Index

| I want to... | Go to... |
|-------------|----------|
| Configure a trading strategy | [Configuring Strategies](configuring-strategies.md) |
| Set up risk management | [Risk Management](risk-management.md) |
| Run a backtest | [Backtesting](backtesting.md) |
| Find optimal parameters | [Optimization](optimization.md) |
| Validate strategy robustness | [Walk-Forward Analysis](walk-forward-analysis.md) |
| Build complex workflows | [Multi-Phase Workflows](multi-phase-workflows.md) |
| Deploy to production | [Live Trading](live-trading.md) |

## 📚 User Guide Sections

### [Configuring Strategies](configuring-strategies.md)
Learn how to configure different types of trading strategies:
- Built-in strategies (momentum, mean reversion, breakout)
- Machine learning models (sklearn, TensorFlow)
- Custom functions and external APIs
- Multi-strategy portfolios
- Strategy parameters and tuning

### [Risk Management](risk-management.md)
Set up comprehensive risk controls:
- Position sizing methods
- Exposure limits and constraints
- Stop-loss and take-profit rules
- Portfolio-level risk management
- Dynamic risk adjustment

### [Backtesting](backtesting.md)
Run realistic historical simulations:
- Single strategy backtests
- Multi-strategy portfolios
- Including transaction costs and slippage
- Performance analysis and reporting
- Debugging poor performance

### [Optimization](optimization.md)
Find optimal strategy parameters:
- Grid search optimization
- Random search and Bayesian optimization
- Multi-objective optimization
- Constraint handling
- Signal replay for speed

### [Walk-Forward Analysis](walk-forward-analysis.md)
Validate strategy robustness:
- Rolling window validation
- Parameter stability analysis
- Out-of-sample testing
- Avoiding overfitting
- Regime-aware validation

### [Multi-Phase Workflows](multi-phase-workflows.md)
Build sophisticated research pipelines:
- Sequential workflow phases
- Parallel execution patterns
- Conditional logic and branching
- Data flow between phases
- Workflow monitoring and debugging

### [Live Trading](live-trading.md)
Deploy strategies to production:
- Paper trading and validation
- Real-time data integration
- Risk monitoring and alerts
- Position management
- Performance tracking

## 🚀 Common Workflows

### Strategy Development Workflow
```
1. Configure Strategy → 2. Run Backtest → 3. Optimize Parameters → 4. Validate → 5. Deploy
```

**Files needed**:
- Strategy configuration: `strategies/my_strategy.yaml`
- Backtest configuration: `config/backtest.yaml`
- Optimization configuration: `config/optimization.yaml`
- Validation configuration: `config/validation.yaml`

### Research Workflow
```
1. Broad Search → 2. Regime Analysis → 3. Ensemble Building → 4. Walk-Forward → 5. Final Test
```

**Typical timeline**: 2-5 days for comprehensive research

### Production Deployment
```
1. Final Validation → 2. Paper Trading → 3. Shadow Mode → 4. Gradual Rollout → 5. Full Production
```

**Safety timeline**: 2-4 weeks for careful deployment

## 🛠️ Configuration Basics

### YAML Structure
All ADMF-PC configurations follow this structure:

```yaml
# Required sections
workflow:
  type: "backtest"  # or optimization, multi_phase, etc.
  
data:
  symbols: ["SPY"]
  start_date: "2023-01-01"
  
strategies:
  - type: "momentum"
    params:
      fast_period: 10
      slow_period: 20

# Optional but recommended sections      
risk_management:
  position_size_pct: 0.02
  max_drawdown_pct: 0.15
  
execution:
  slippage_bps: 10
  commission_per_share: 0.01
  
portfolio:
  initial_capital: 100000
  
reporting:
  output_path: "reports/"
```

### Configuration Validation

Before running, validate your configuration:

```bash
# Validate YAML syntax
python -c "
import yaml
with open('config/my_config.yaml') as f:
    config = yaml.safe_load(f)
print('Configuration valid!')
"

# Test with small dataset
python main.py config/my_config.yaml --bars 100 --dry-run
```

## 📊 Data Requirements

### Supported Data Formats

**CSV Format** (recommended):
```csv
timestamp,open,high,low,close,volume
2023-01-01 09:30:00,400.0,401.5,399.5,401.0,1000000
2023-01-01 09:31:00,401.0,402.0,400.5,401.5,950000
```

**Requirements**:
- Timestamp column in `YYYY-MM-DD HH:MM:SS` format
- OHLCV columns with numeric data
- No missing values in price data
- Chronological ordering

### Data Sources

1. **Built-in downloader**: 
   ```bash
   python scripts/download_sample_data.py --symbols SPY,QQQ,IWM
   ```

2. **Custom CSV files**: Place in `data/` directory

3. **Database connections**: Configure in data section

4. **Live data feeds**: For production trading

## 🔧 Troubleshooting Quick Reference

### Common Configuration Errors

```yaml
# ❌ Wrong: Tabs instead of spaces
strategies:
	- type: "momentum"  # Tab character

# ✅ Correct: Consistent spaces
strategies:
  - type: "momentum"  # 2 spaces

# ❌ Wrong: Missing quotes
start_date: 2023-01-01

# ✅ Correct: Quoted dates
start_date: "2023-01-01"

# ❌ Wrong: Percentage as whole number
position_size_pct: 2.0  # Means 200%!

# ✅ Correct: Percentage as decimal
position_size_pct: 0.02  # Means 2%
```

### Performance Issues

```yaml
# If backtest is slow:
data:
  start_date: "2023-06-01"  # Reduce date range
  end_date: "2023-12-31"
  
# If optimization is slow:
optimization:
  signal_replay: true  # 10-100x speedup
  
# If memory issues:
infrastructure:
  max_workers: 4       # Reduce from default 8
  memory_limit_gb: 8   # Set explicit limit
```

### Common Runtime Errors

| Error | Solution |
|-------|----------|
| `FileNotFoundError: data/SPY.csv` | Run data download script |
| `ValidationError: position_size_pct` | Use 0.02 not 2.0 for 2% |
| `MemoryError` | Reduce date range or container count |
| `yaml.scanner.ScannerError` | Check YAML syntax |

## 📖 Learning Path

### Beginner (Week 1)
1. [Run first backtest](../01-getting-started/first-backtest.md)
2. [Configure simple strategy](configuring-strategies.md#basic-strategies)
3. [Set up risk management](risk-management.md#basic-risk-controls)
4. [Understand performance reports](backtesting.md#analyzing-results)

### Intermediate (Week 2-3)
1. [Optimize strategy parameters](optimization.md)
2. [Build multi-strategy portfolio](configuring-strategies.md#multi-strategy-portfolios)
3. [Validate with walk-forward](walk-forward-analysis.md)
4. [Create basic workflows](multi-phase-workflows.md)

### Advanced (Week 4+)
1. [Build complex workflows](multi-phase-workflows.md)
2. [Integrate ML models](configuring-strategies.md#machine-learning-integration)
3. [Deploy to paper trading](live-trading.md)
4. [Develop custom components](../08-advanced-topics/custom-components.md)

## 🤝 Getting Help

### Self-Service Resources
1. **Examples**: Check `config/` directory for working examples
2. **Error Messages**: Most errors include helpful suggestions
3. **Logs**: Run with `--verbose` for detailed information
4. **Validation**: Use `--dry-run` to test configurations

### Community Support
1. **Documentation**: Search this guide and [Core Concepts](../02-core-concepts/README.md)
2. **GitHub Issues**: Report bugs and request features
3. **Discussions**: Ask questions and share strategies

### Best Practices
1. **Start Simple**: Begin with basic configurations
2. **Validate Early**: Test with small datasets first
3. **Iterate Quickly**: Make small changes and test frequently
4. **Document Everything**: Keep notes on what works

---

Ready to start building? Begin with [Configuring Strategies](configuring-strategies.md) →