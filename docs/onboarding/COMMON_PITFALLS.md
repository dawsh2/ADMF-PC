# Common Pitfalls and How to Avoid Them

Learn from others' mistakes! This guide covers the most common issues new users face with ADMF-PC.

## 🚫 Pitfall #1: Trying to Write Python Code

### ❌ Wrong Approach
```python
# DON'T DO THIS - Creating Python files
class MyCustomStrategy:
    def __init__(self):
        self.ma_fast = 10
        self.ma_slow = 30
        
    def calculate_signal(self, data):
        # Complex logic...
```

### ✅ Correct Approach
```yaml
# Just use configuration!
strategies:
  - type: momentum
    fast_period: 10
    slow_period: 30
```

**Why**: ADMF-PC is a zero-code system. All logic is already implemented and optimized. You just configure what you want.

---

## 🚫 Pitfall #2: Not Understanding Event Flow

### ❌ Wrong Mental Model
"My strategy directly calls the execution engine"

### ✅ Correct Mental Model
```
Data → Indicators → Strategy → Signal → Risk → Order → Execution
  ↓        ↓           ↓         ↓       ↓       ↓         ↓
[BAR]  [INDICATOR] [STRATEGY] [SIGNAL] [ORDER] [FILL] [POSITION]
```

**Why**: Everything is event-driven. Components don't know about each other, they only respond to events.

---

## 🚫 Pitfall #3: Ignoring Container Isolation

### ❌ Wrong Assumption
```yaml
strategies:
  - name: "strategy_a"
    type: "momentum"
    # Trying to access strategy_b's data - WON'T WORK!
    use_signals_from: "strategy_b"
```

### ✅ Correct Approach
```yaml
# Use ensemble or coordinator patterns
strategies:
  - type: "ensemble"
    strategies:
      - {type: "momentum", name: "strategy_a"}
      - {type: "mean_reversion", name: "strategy_b"}
    combination_method: "weighted_vote"
```

**Why**: Containers are completely isolated. Use composition patterns to combine strategies.

---

## 🚫 Pitfall #4: Over-Optimizing Parameters

### ❌ Wrong Approach
```yaml
parameter_space:
  fast_period: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  slow_period: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
  threshold: [0.01, 0.011, 0.012, 0.013, 0.014, 0.015]
  # 11 × 11 × 6 = 726 combinations!
```

### ✅ Correct Approach
```yaml
parameter_space:
  fast_period: [5, 10, 15]      # Coarse grid
  slow_period: [20, 30, 40]     
  threshold: [0.01, 0.02, 0.03]
  # 3 × 3 × 3 = 27 combinations

validation:
  method: "walk_forward"        # Always validate!
  out_of_sample_ratio: 0.3
```

**Why**: Overfitting is the enemy. Use coarse grids and always validate out-of-sample.

---

## 🚫 Pitfall #5: Wrong YAML Indentation

### ❌ Wrong (Using Tabs)
```yaml
strategies:
	- type: momentum    # TAB used here - ERROR!
		fast_period: 10
```

### ✅ Correct (Using Spaces)
```yaml
strategies:
  - type: momentum    # 2 spaces
    fast_period: 10   # 4 spaces
```

**Why**: YAML requires consistent spacing. Always use spaces, never tabs.

---

## 🚫 Pitfall #6: Forgetting Risk Management

### ❌ Wrong Approach
```yaml
strategies:
  - type: momentum
    # No risk management!
    
risk:
  initial_capital: 1000000
  # That's it? No limits?
```

### ✅ Correct Approach
```yaml
risk:
  initial_capital: 1000000
  position_size_pct: 2.0      # 2% per position
  max_positions: 10           # Diversification
  max_drawdown_pct: 15.0      # Stop if losing
  stop_loss_pct: 3.0          # Per-position stop
  max_exposure_pct: 80.0      # Keep some cash
```

**Why**: Risk management is crucial. Always define comprehensive risk parameters.

---

## 🚫 Pitfall #7: Testing on Limited Data

### ❌ Wrong Approach
```yaml
data:
  start_date: "2023-10-01"
  end_date: "2023-12-31"    # Only 3 months!
```

### ✅ Correct Approach
```yaml
data:
  start_date: "2018-01-01"
  end_date: "2023-12-31"    # 6 years

validation:
  method: "walk_forward"
  windows: 12               # Test multiple periods
```

**Why**: Strategies need diverse market conditions to be robust.

---

## 🚫 Pitfall #8: Missing Market Regimes

### ❌ Wrong Approach
```yaml
strategies:
  - type: momentum    # Works great in trends...
    # But what about sideways markets?
```

### ✅ Correct Approach
```yaml
classifiers:
  - type: "market_regime"
    
strategies:
  - type: "regime_adaptive"
    regimes:
      trending: {type: "momentum"}
      sideways: {type: "mean_reversion"}
      volatile: {type: "volatility_capture"}
```

**Why**: Markets change. Adaptive strategies survive.

---

## 🚫 Pitfall #9: Ignoring Transaction Costs

### ❌ Wrong Approach
```yaml
# No transaction costs defined
# Assumes free trading!
```

### ✅ Correct Approach
```yaml
execution:
  commission_per_share: 0.005
  min_commission: 1.0
  slippage_model: "linear"
  slippage_bps: 5          # 5 basis points
  market_impact_model: "square_root"
```

**Why**: Transaction costs can turn a profitable strategy into a loser.

---

## 🚫 Pitfall #10: Not Using Signal Analysis

### ❌ Wrong Approach
```yaml
# Just run backtest and look at returns
output:
  generate_report: true
```

### ✅ Correct Approach
```yaml
output:
  generate_report: true
  save_signals: true        # Analyze signals!
  
analysis:
  signal_quality:
    - mae_mfe_analysis      # How far do trades go?
    - time_to_profit        # How long to profit?
    - signal_correlation    # Are signals diverse?
    - regime_attribution    # When do we make money?
```

**Why**: Understanding WHY a strategy works is as important as the returns.

---

## 🚫 Pitfall #11: Assuming Live = Backtest

### ❌ Wrong Assumption
"My backtest works perfectly, so live trading will too!"

### ✅ Correct Approach
1. **Paper Trade First**: Test with simulated money
2. **Start Small**: Use minimal capital initially
3. **Monitor Differences**: Track live vs backtest performance
4. **Have Safeguards**: Emergency stops, alerts, limits

```yaml
live_trading:
  mode: "paper"              # Start here!
  emergency_stop: true
  max_daily_loss: 1000
  alert_on_deviation: 0.1    # Alert if 10% different from backtest
```

**Why**: Live markets have latency, slippage, and partial fills that backtests approximate.

---

## 🚫 Pitfall #12: Not Reading the Docs

### ❌ Wrong Approach
"I'll just figure it out as I go..."

### ✅ Correct Approach
1. Complete [2-hour onboarding](ONBOARDING.md)
2. Try the [guided tutorial](FIRST_TASK.md)
3. Reference [architecture docs](../SYSTEM_ARCHITECTURE_V5.MD)
4. Use [example configs](../../config/)

**Why**: ADMF-PC is powerful but different. Understanding the philosophy saves hours of confusion.

---

## 💡 Golden Rules to Remember

1. **Configuration > Code**: Never write Python for strategies
2. **Events > Methods**: Think in terms of event flow
3. **Isolation > Sharing**: Containers don't share state
4. **Validation > Optimization**: Always test out-of-sample
5. **Risk > Returns**: Define risk before seeking profits
6. **Simple > Complex**: Start simple, add complexity gradually
7. **Adaptive > Static**: Markets change, strategies should too
8. **Costs > Gross Returns**: Always include realistic costs
9. **Understanding > Performance**: Know why strategies work
10. **Documentation > Intuition**: Read the docs thoroughly

---

## 🆘 Still Stuck?

If you're still having issues:

1. Check [FAQ](FAQ.md) for common questions
2. Review [example configurations](../../config/)
3. Read relevant [architecture docs](../architecture/)
4. Search the [GitHub issues](#)

Remember: Every expert was once a beginner. The ADMF-PC way of thinking takes time to internalize, but it's worth it!

[← Back to Hub](README.md) | [Next: FAQ →](FAQ.md)