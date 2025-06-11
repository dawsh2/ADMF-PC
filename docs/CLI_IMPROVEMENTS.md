# CLI Improvements: Clean Topology Action Flags

## Summary

We've implemented cleaner CLI action flags to replace the chunky `--topology signal_generation` syntax with more intuitive action-specific flags.

## Before (Chunky)
```bash
python main.py --topology signal_generation --config config.yaml
python main.py --topology backtest --config config.yaml  
python main.py --topology signal_replay --config config.yaml
python main.py --topology optimization --config config.yaml
```

## After (Clean)
```bash
python main.py --signal-generation config.yaml
python main.py --backtest config.yaml
python main.py --signal-replay config.yaml
python main.py --optimize config.yaml
```

## Benefits

1. **Cleaner syntax** - No redundant "--topology" prefix
2. **Self-documenting** - Flag name tells you exactly what it does  
3. **Pure configs** - Config files now contain only business logic, no topology/workflow specs
4. **Unix-like** - Follows standard CLI patterns (like `git commit`, `docker run`, etc.)

## Implementation Details

### Updated Files

1. **`src/core/cli/parser.py`**:
   - Added new action flag fields to `CLIArgs` dataclass
   - Created mutually exclusive group for topology action flags
   - Marked old `--topology` and `--mode` flags as deprecated

2. **`main.py`**:
   - Updated to detect config path from action flags
   - Routes to appropriate topology based on action flag used
   - Maintains backward compatibility with old flags

### Example Config (Pure Business Logic)

```yaml
# config/clean_config.yaml - NO topology/workflow specs!
data:
  source: csv
  file_path: "data/SPY.csv"
  start_date: "2023-01-01"
  end_date: "2023-12-31"
  
strategies:
  - type: momentum
    params:
      fast_period: 10
      slow_period: 30
      
execution:
  initial_capital: 100000
  commission_rate: 0.001
```

## Migration Path

1. **Phase 1** ✓ Add new flags alongside old ones (COMPLETE)
2. **Phase 2**: Deprecate `--topology` and `--mode` flags with warnings
3. **Phase 3**: Update all configs to remove `topology:` and `workflow:` fields  
4. **Phase 4**: Remove deprecated flags in future version

## Future Enhancements

### Command Composition (Future)
```bash
# Generate signals and immediately replay them
python main.py --signal-generation config.yaml | python main.py --signal-replay -

# Run backtests on multiple configs
for config in configs/*.yaml; do
    python main.py --backtest $config
done
```

### Shell Aliases (User Convenience)
```bash
# In .bashrc or .zshrc
alias admf-backtest='python ~/ADMF-PC/main.py --backtest'
alias admf-signals='python ~/ADMF-PC/main.py --signal-generation'

# Usage becomes even cleaner:
admf-backtest my_strategy.yaml
admf-signals momentum_config.yaml
```

## Testing

Verified both new action flags work correctly:
```bash
# Signal generation
python main.py --signal-generation config/test_clean_cli.yaml
# ✅ Successfully executed signal_generation topology

# Backtesting  
python main.py --backtest config/test_clean_cli.yaml
# ✅ Successfully executed backtest topology
```

## Next Steps

1. Remove deprecated code (auto-wrapping logic, old mode handling)
2. Update all example configs to pure business logic format
3. Add shell completion support for the new flags
4. Document the new CLI interface in user guide