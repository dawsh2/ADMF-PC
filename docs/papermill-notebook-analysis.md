# Papermill-Based Analysis Notebooks

## Overview

ADMF-PC now uses [papermill](https://papermill.readthedocs.io/) for notebook-based analysis. This approach provides:

- **Parameterized notebooks** - Pass run directories and config values without string manipulation
- **Headless execution** - Run analysis without opening Jupyter
- **Cross-strategy analysis** - One universal template analyzes all strategies in a sweep
- **HTML reports** - Generate static reports for sharing
- **Error handling** - Papermill tracks execution errors by cell

## Installation

```bash
pip install papermill
```

## Usage

### Basic: Generate Parameterized Notebook

Just create a notebook with parameters filled in (no execution):

```bash
python main.py --config config/my_sweep.yaml --signal-generation --notebook
```

### Execute Analysis

Run the analysis automatically (no Jupyter needed):

```bash
python main.py --config config/my_sweep.yaml --signal-generation --launch-notebook
```

This will:
1. Execute the universal analysis notebook
2. Calculate performance for all strategies
3. Find optimal ensemble combinations
4. Save `recommendations.json` and `performance_analysis.csv`
5. Launch Jupyter with the executed notebook

### Generate HTML Report

Add HTML generation in your config:

```yaml
analysis:
  generate_html: true
```

## Signal Analysis Template

The system uses one main template: `src/analytics/templates/signal_analysis.ipynb`

This template:
- Loads the `strategy_index.parquet` (catalog of ALL strategies)
- Calculates performance across all strategy types
- Performs correlation analysis for ensemble building
- Identifies best individual strategies and optimal combinations
- Exports actionable recommendations

### Key Sections

1. **Strategy Overview** - Distribution of strategies by type
2. **Cross-Strategy Performance** - Compare all strategies regardless of type
3. **Correlation Analysis** - Find uncorrelated strategies for ensembles
4. **Ensemble Recommendations** - Optimal strategy combinations
5. **Export Results** - Save recommendations.json

## Parameters

The notebook accepts these parameters:

```python
# parameters
run_dir = "/path/to/results/run_20250623_143030"
config_name = "my_sweep"
symbols = ["SPY"]
timeframe = "5m"
min_strategies_to_analyze = 20  # Performance optimization
sharpe_threshold = 1.0         # For filtering
correlation_threshold = 0.7    # For ensemble building
top_n_strategies = 10         # Top performers to show
ensemble_size = 5            # Target ensemble size
```

## Benefits Over Previous Approach

### Before (Complex Generation)
- 1000+ lines of code building notebook JSON
- String concatenation for code cells
- Strategy-specific templates needed
- Manual parameter injection with f-strings

### After (Papermill)
- ~100 lines of code
- Real .ipynb template files
- One universal template for all strategies
- Clean parameter passing
- Headless execution capability

## Example Workflow

1. **Run parameter sweep testing 100+ strategies:**
   ```bash
   python main.py --config config/big_sweep.yaml --signal-generation
   ```

2. **Execute analysis automatically:**
   ```bash
   python main.py --config config/big_sweep.yaml --signal-generation --launch-notebook
   ```

3. **Results appear in Jupyter showing:**
   - Top 10 strategies across ALL types
   - Performance by strategy type
   - Correlation heatmap
   - Recommended 5-strategy ensemble
   - Saved recommendations.json

4. **Use recommendations for production:**
   ```python
   with open('recommendations.json') as f:
       recs = json.load(f)
   
   # Best individual strategy
   best = recs['best_individual']
   print(f"Best: {best['strategy_type']} with Sharpe {best['sharpe_ratio']:.2f}")
   
   # Optimal ensemble
   ensemble = recs['ensemble']
   print(f"Ensemble of {len(ensemble)} uncorrelated strategies")
   ```

## Customization

### Adding Analysis Parameters

In your config.yaml:

```yaml
analysis:
  min_strategies: 50      # Analyze more strategies
  sharpe_threshold: 1.5   # Higher bar for "good"
  correlation_threshold: 0.5  # Stricter correlation limit
  ensemble_size: 7       # Larger ensemble
  generate_html: true    # Create HTML report
```

### Creating Additional Templates

For specialized analysis, create new templates:

```
src/analytics/templates/
├── signal_analysis.ipynb       # Main cross-strategy analysis
├── regime_analysis.ipynb       # Deep regime performance
├── correlation_matrix.ipynb    # Detailed correlation study
└── parameter_sensitivity.ipynb # Parameter optimization
```

Then run supplementary analysis:

```python
runner = PapermillNotebookRunner()
runner.run_supplementary_analysis(
    run_dir,
    template_name='regime_analysis',
    params={'min_sharpe': 0.5}
)
```

## Troubleshooting

### "Papermill not installed"
```bash
pip install papermill
```

### "No such kernel: python3"
```bash
python -m ipykernel install --user --name python3
```

### Notebook execution errors
- Check the output notebook - papermill marks which cell failed
- Look for missing dependencies or data files
- Ensure market data is available (e.g., `data/SPY_5m.parquet`)

## Migration from Legacy System

The system automatically falls back to the legacy notebook generator if papermill is not installed. However, we strongly recommend installing papermill for:

- Much simpler codebase
- Better error handling
- Headless execution
- Professional reports

## Summary

The papermill-based approach transforms our analysis workflow:

- **From**: Complex code generation → **To**: Simple template parameterization
- **From**: Strategy-specific templates → **To**: Universal cross-strategy analysis
- **From**: Manual Jupyter launch → **To**: Automated execution and reports
- **From**: 1000+ lines of code → **To**: ~100 lines + reusable templates

This makes the analysis phase as frictionless as the vision in WORKFLOW.md!