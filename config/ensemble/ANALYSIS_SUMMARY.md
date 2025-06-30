# Ensemble Configuration Analysis

## Current Configuration Issue

Your `config/ensemble/config.yaml` doesn't actually create an ensemble. It defines two independent strategies:

```yaml
strategy:
  - keltner_bands: {period: 26, multiplier: 3.0}
  - bollinger_bands: {period: 11, std_dev: 2.0}
```

This configuration:
- Runs each strategy separately
- Each generates (or doesn't generate) its own signals
- No combination or voting logic
- Results in separate signal traces for each strategy

## Why No Signals Were Generated

Looking at the metadata showing 0 signals over 16,614 bars:

1. **Keltner Bands (period=26, multiplier=3.0)**:
   - Very wide bands (3x ATR)
   - Longer period (26 bars)
   - Would only signal on extreme moves

2. **Bollinger Bands (period=11, std_dev=2.0)**:
   - Standard 2 std deviation bands
   - Shorter period but still conservative

Both strategies are quite conservative and may not have triggered any signals during the test period.

## How ADMF-PC Handles Multiple Strategies

### Option 1: Independent Strategies (Your Current Setup)
```yaml
strategy:
  - strategy_type_1: {params...}
  - strategy_type_2: {params...}
```
- Each runs independently
- Separate signal streams
- No interaction between strategies

### Option 2: True Ensemble Strategy
```yaml
strategies:
  - type: two_layer_ensemble
    name: my_ensemble
    params:
      baseline_strategies:
        - name: keltner_strategy
          type: keltner_bands
          params: {period: 26, multiplier: 3.0}
          weight: 0.5
        - name: bollinger_strategy  
          type: bollinger_bands
          params: {period: 11, std_dev: 2.0}
          weight: 0.5
      combination_method: weighted_vote
      threshold: 0.5
```

## Ensemble Types in ADMF-PC

1. **Two-Layer Ensemble** (`two_layer_ensemble`):
   - Baseline strategies (always active)
   - Regime-specific boosters (conditionally active)
   - Weighted voting or other combination methods

2. **DuckDB Ensemble** (`duckdb_ensemble`):
   - SQL-based strategy combination
   - Can query historical signals
   - More flexible but requires DuckDB

3. **Trend-Momentum Composite** (`trend_momentum_composite`):
   - Specific ensemble for trend/momentum strategies
   - Pre-configured combination logic

## Recommendations

1. **For Testing Basic Ensemble**:
   Use a proper ensemble configuration like `two_layer_actual.yaml`

2. **For Your Strategies**:
   Convert to ensemble format if you want combined signals

3. **For Debugging**:
   - Run strategies with more aggressive parameters first
   - Check individual strategy performance
   - Then combine into ensemble

## Next Steps

1. Copy and modify `two_layer_actual.yaml` for your needs
2. Or adjust parameters to be less conservative:
   - Keltner: Try multiplier=2.0 or 1.5
   - Bollinger: Already reasonable at 2.0 std dev
3. Test strategies individually first to ensure they generate signals