# Auto-Generated Analysis Notebooks

## Overview

The ADMF-PC system can automatically generate pre-configured Jupyter notebooks after each backtest run. These notebooks eliminate the friction of post-processing analysis by providing:

- **Pre-configured paths** pointing to your exact results
- **Strategy-specific analysis** tailored to what you tested
- **Automatic visualizations** with parameter heatmaps and equity curves
- **Performance calculations** with Sharpe ratio, drawdown, and win rate
- **Best parameter export** saved to `recommendations.json`

## Usage

### Basic Usage

Add `--notebook` to generate a notebook after your backtest:

```bash
python main.py --config config/bollinger/config.yaml --signal-generation --notebook
```

### Auto-Launch Jupyter

Add `--launch-notebook` to automatically open Jupyter after generation:

```bash
python main.py --config config/bollinger/config.yaml --signal-generation --notebook --launch-notebook
```

### Generated Notebook Structure

The auto-generated notebook includes these sections:

1. **Header & Navigation** - Quick links to all sections
2. **Setup & Configuration** - Pre-configured imports and paths
3. **Data Loading** - Load strategy index and trace files
4. **Strategy-Specific Analysis** - Custom analysis for your strategy type:
   - Bollinger Bands: Band width analysis, signal frequency heatmaps
   - Momentum: Intraday patterns, burst analysis
   - MA Crossover: Parameter space visualization
   - RSI: Threshold effectiveness analysis
   - Ensemble: Component correlation, voting patterns
5. **Performance Metrics** - Calculate returns, Sharpe, drawdown
6. **Visualizations** - Equity curves, parameter sensitivity
7. **Pattern Discovery** - Find successful parameter combinations
8. **Export Results** - Save best parameters to `recommendations.json`

## Example Workflow

1. **Run backtest with notebook generation:**
   ```bash
   python main.py --config config/bollinger/sweep.yaml --signal-generation --notebook
   ```

2. **Output shows notebook location:**
   ```
   ✅ Generated analysis notebook: config/bollinger/results/run_20250623_143030/analysis_bollinger_20250623_143035.ipynb
   ```

3. **Open the notebook:**
   ```bash
   jupyter lab config/bollinger/results/run_20250623_143030/analysis_bollinger_20250623_143035.ipynb
   ```

4. **Run all cells** (Cell → Run All)

5. **Find your results:**
   - Best parameters in `recommendations.json`
   - Full analysis in `performance_analysis.csv`
   - Visualizations inline in the notebook

## Strategy-Specific Features

### Bollinger Bands
- Parameter heatmaps (period vs std_dev)
- Signal frequency analysis
- Optimal trading frequency identification (1-3 signals/day)

### Momentum Strategies
- Intraday signal distribution
- Momentum burst patterns
- Time-of-day analysis

### Moving Average Crossovers
- Fast vs slow period scatter plots
- Crossover frequency analysis

### RSI Strategies
- Oversold/overbought threshold analysis
- Period effectiveness comparison

### Ensemble Strategies
- Component strategy breakdown
- Voting pattern analysis
- Strategy correlation matrices

## Configuration

### Using Custom Templates

Specify a custom notebook template:

```bash
python main.py --config config.yaml --signal-generation --notebook --notebook-template my_template.ipynb
```

### Directory Structure

Notebooks are saved alongside your results:

```
configs/
└── strategy_name/
    ├── config.yaml
    └── results/
        └── run_20250623_143025/
            ├── traces/              # Signal data
            ├── strategy_index.parquet    # Strategy catalog
            ├── metadata.json        # Run metadata
            └── analysis_bollinger_20250623_143030.ipynb  # AUTO-GENERATED!
```

## Benefits

1. **Zero Setup** - No manual path configuration
2. **Instant Analysis** - From backtest to insights in seconds
3. **Reproducible** - Every notebook is self-contained
4. **Strategy-Aware** - Analysis adapts to your strategy type
5. **Cumulative Learning** - Patterns discovered in one run inform future analysis

## Advanced Features

### Pattern Library Integration

The system maintains a pattern library of successful parameter combinations. When you discover a high-performing configuration, it's automatically saved and suggested in future notebooks.

### Cross-Run Analysis

Using strategy hashes, you can compare identical strategies across different runs:

```python
# In the notebook
SELECT * FROM '../*/strategy_index.parquet'
WHERE strategy_hash = 'a3f4b2c1d5e6'
```

### Regime Analysis

For strategies with regime classifiers, the notebook includes regime-specific performance breakdowns.

## Troubleshooting

### Notebook not generated?
- Ensure the backtest completed successfully
- Check that results were saved (look for the results directory)
- Verify you have the `--notebook` flag

### Can't launch Jupyter?
- Install Jupyter: `pip install jupyter jupyterlab`
- Try manual launch: `jupyter lab <notebook_path>`

### Missing data in notebook?
- Ensure market data is available (e.g., `data/SPY_5m.parquet`)
- Check that signal generation completed
- Verify the strategy_index.parquet file exists

## Next Steps

1. Run the test script to see it in action:
   ```bash
   python test_notebook_workflow.py
   ```

2. Try with your own strategies

3. Customize the notebook templates in `src/analytics/notebook_templates/`

The auto-generated notebooks transform post-processing from a chore into an exploration. No more copy-pasting code or adjusting paths - just immediate insights from your backtests.