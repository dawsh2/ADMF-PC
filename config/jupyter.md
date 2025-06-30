# Auto-Generated Analysis Notebooks: The Holy Grail Workflow

## Overview

This system automatically generates pre-configured Jupyter notebooks after each backtest run, with all paths, queries, and visualizations ready to go. No more copy-pasting code or adjusting paths - just run your backtest and dive straight into analysis.

## Core Features

1. **Auto-configured paths** - Notebooks always point to the correct results directory
2. **Strategy-specific analysis** - Different templates for different strategy types
3. **Pre-loaded queries** - Common analyses ready to run
4. **Dynamic content** - Cells adapt based on what data is available
5. **Instant launch** - Option to auto-open notebook after generation

## Implementation

### 1. Notebook Generator Class

```python
# notebook_generator.py
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

class AnalysisNotebookGenerator:
    """Generate analysis notebooks tailored to specific runs and strategies"""
    
    def __init__(self, template_dir: Optional[Path] = None):
        self.template_dir = template_dir or Path(__file__).parent / "notebook_templates"
        self.template_dir.mkdir(exist_ok=True)
        
    def generate(self, 
                 run_dir: Path,
                 config: Dict[str, Any],
                 strategy_type: str,
                 launch: bool = False) -> Path:
        """Generate analysis notebook for a specific run"""
        
        # Determine paths
        results_path = run_dir
        config_dir = run_dir.parent.parent  # Assuming configs/strategy_name/results/run_id
        
        # Create notebook structure
        notebook = {
            "cells": [
                self._create_header_cell(config, run_dir),
                self._create_setup_cell(results_path),
                *self._create_strategy_specific_cells(strategy_type, config),
                *self._create_common_analysis_cells(),
                *self._create_performance_cells(),
                *self._create_pattern_discovery_cells(),
                *self._create_visualization_cells(),
                self._create_export_cell()
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.8.0"
                },
                "generated": {
                    "timestamp": datetime.now().isoformat(),
                    "run_id": run_dir.name,
                    "strategy_type": strategy_type,
                    "config_name": config.get('name', 'unnamed')
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        # Save notebook
        notebook_name = f"analysis_{strategy_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ipynb"
        notebook_path = run_dir / notebook_name
        
        with open(notebook_path, 'w') as f:
            json.dump(notebook, f, indent=2)
            
        print(f"✅ Generated analysis notebook: {notebook_path}")
        
        if launch:
            self._launch_notebook(notebook_path)
            
        return notebook_path
    
    def _create_header_cell(self, config: Dict, run_dir: Path) -> Dict:
        """Create header markdown cell"""
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# {config.get('name', 'Strategy')} Analysis\n",
                f"\n",
                f"**Run ID**: `{run_dir.name}`  \n",
                f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n",
                f"**Config**: `{config.get('name', 'unnamed')}`  \n",
                f"\n",
                f"## Quick Navigation\n",
                f"1. [Setup](#setup)\n",
                f"2. [Load Data](#load-data)\n",
                f"3. [Signal Analysis](#signal-analysis)\n",
                f"4. [Performance Metrics](#performance-metrics)\n",
                f"5. [Visualizations](#visualizations)\n",
                f"6. [Export Results](#export-results)"
            ]
        }
    
    def _create_setup_cell(self, results_path: Path) -> Dict:
        """Create setup and imports cell"""
        return {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Auto-generated setup\n",
                "import duckdb\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from pathlib import Path\n",
                "import json\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "# Configure plotting\n",
                "plt.style.use('seaborn-v0_8-darkgrid')\n",
                "sns.set_palette('husl')\n",
                "plt.rcParams['figure.figsize'] = (12, 6)\n",
                "\n",
                "# Set paths\n",
                f"results_path = Path('{results_path}')\n",
                "print(f'Analyzing results in: {results_path}')\n",
                "print(f'Run ID: {results_path.name}')\n",
                "\n",
                "# Initialize DuckDB\n",
                "con = duckdb.connect()"
            ]
        }
    
    def _create_strategy_specific_cells(self, strategy_type: str, config: Dict) -> List[Dict]:
        """Create cells specific to the strategy type"""
        cells = []
        
        if strategy_type == "bollinger_bands":
            cells.extend(self._bollinger_bands_cells(config))
        elif strategy_type == "momentum_burst":
            cells.extend(self._momentum_burst_cells(config))
        elif strategy_type == "ensemble":
            cells.extend(self._ensemble_cells(config))
        else:
            cells.extend(self._generic_strategy_cells(config))
            
        return cells
    
    def _bollinger_bands_cells(self, config: Dict) -> List[Dict]:
        """Bollinger Bands specific analysis cells"""
        return [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Load Data"]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Load strategy index\n",
                    "strategy_index = pd.read_parquet(results_path / 'strategy_index.parquet')\n",
                    "print(f'Total strategies tested: {len(strategy_index)}')\n",
                    "\n",
                    "# Query signal statistics\n",
                    "signal_stats = con.execute(f\"\"\"\n",
                    "    WITH trace_data AS (\n",
                    "        SELECT \n",
                    "            strategy_hash,\n",
                    "            json_extract_string(metadata, '$.parameters.period') as period,\n",
                    "            json_extract_string(metadata, '$.parameters.std_dev') as std_dev,\n",
                    "            COUNT(*) as num_signals,\n",
                    "            COUNT(DISTINCT DATE(ts)) as trading_days\n",
                    "        FROM read_parquet('{results_path}/traces/**/*.parquet')\n",
                    "        WHERE val != 0 AND metadata IS NOT NULL\n",
                    "        GROUP BY strategy_hash, period, std_dev\n",
                    "    )\n",
                    "    SELECT \n",
                    "        strategy_hash,\n",
                    "        CAST(period AS INT) as period,\n",
                    "        CAST(std_dev AS FLOAT) as std_dev,\n",
                    "        num_signals,\n",
                    "        trading_days,\n",
                    "        num_signals::FLOAT / trading_days as signals_per_day\n",
                    "    FROM trace_data\n",
                    "    WHERE period IS NOT NULL\n",
                    "    ORDER BY period, std_dev\n",
                    "\"\"\").df()\n",
                    "\n",
                    "print(f'Loaded {len(signal_stats)} strategies with signals')\n",
                    "signal_stats.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Signal Analysis"]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Create parameter heatmaps\n",
                    "fig, axes = plt.subplots(2, 2, figsize=(15, 12))\n",
                    "\n",
                    "# Signals per day heatmap\n",
                    "signals_pivot = signal_stats.pivot(index='period', columns='std_dev', values='signals_per_day')\n",
                    "sns.heatmap(signals_pivot, annot=True, fmt='.2f', cmap='viridis', ax=axes[0, 0])\n",
                    "axes[0, 0].set_title('Average Signals Per Day')\n",
                    "\n",
                    "# Total signals heatmap\n",
                    "total_pivot = signal_stats.pivot(index='period', columns='std_dev', values='num_signals')\n",
                    "sns.heatmap(total_pivot, annot=False, cmap='plasma', ax=axes[0, 1])\n",
                    "axes[0, 1].set_title('Total Number of Signals')\n",
                    "\n",
                    "# Optimal frequency region\n",
                    "optimal_mask = (signals_pivot >= 1) & (signals_pivot <= 3)\n",
                    "sns.heatmap(optimal_mask.astype(int), cmap='RdYlGn', ax=axes[1, 0])\n",
                    "axes[1, 0].set_title('Optimal Signal Frequency (1-3/day)')\n",
                    "\n",
                    "# Trading days active\n",
                    "days_pivot = signal_stats.pivot(index='period', columns='std_dev', values='trading_days')\n",
                    "sns.heatmap(days_pivot, annot=False, cmap='magma', ax=axes[1, 1])\n",
                    "axes[1, 1].set_title('Trading Days with Signals')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                    "\n",
                    "# Find optimal frequency strategies\n",
                    "optimal_freq = signal_stats[(signal_stats['signals_per_day'] >= 1) & \n",
                    "                           (signal_stats['signals_per_day'] <= 3)]\n",
                    "print(f'\\nStrategies with optimal frequency: {len(optimal_freq)}')"
                ]
            }
        ]
    
    def _common_analysis_cells(self) -> List[Dict]:
        """Cells common to all strategies"""
        return [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Performance Metrics"]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Load market data for performance calculation\n",
                    "market_data_path = Path('../../../../data/SPY_5m.parquet')\n",
                    "if not market_data_path.exists():\n",
                    "    # Try alternative paths\n",
                    "    market_data_path = list(Path('.').glob('**/SPY_5m.parquet'))[0]\n",
                    "    \n",
                    "market_data = pd.read_parquet(market_data_path)\n",
                    "print(f'Market data shape: {market_data.shape}')\n",
                    "\n",
                    "def calculate_performance(strategy_hash, signals_df):\n",
                    "    \"\"\"Calculate performance metrics for a strategy\"\"\"\n",
                    "    # Merge with market data\n",
                    "    df = market_data.merge(\n",
                    "        signals_df[['ts', 'val']], \n",
                    "        left_on='timestamp', \n",
                    "        right_on='ts', \n",
                    "        how='left'\n",
                    "    )\n",
                    "    \n",
                    "    # Forward fill signals\n",
                    "    df['signal'] = df['val'].fillna(method='ffill').fillna(0)\n",
                    "    \n",
                    "    # Calculate returns\n",
                    "    df['returns'] = df['close'].pct_change()\n",
                    "    df['strategy_returns'] = df['returns'] * df['signal'].shift(1)\n",
                    "    df['cum_returns'] = (1 + df['strategy_returns']).cumprod()\n",
                    "    \n",
                    "    # Metrics\n",
                    "    total_return = df['cum_returns'].iloc[-1] - 1\n",
                    "    sharpe = df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(252 * 78)\n",
                    "    max_dd = (df['cum_returns'] / df['cum_returns'].expanding().max() - 1).min()\n",
                    "    \n",
                    "    return {\n",
                    "        'total_return': total_return,\n",
                    "        'sharpe_ratio': sharpe,\n",
                    "        'max_drawdown': max_dd,\n",
                    "        'df': df  # Return full dataframe for plotting\n",
                    "    }"
                ]
            }
        ]
    
    def _create_performance_cells(self) -> List[Dict]:
        """Create performance analysis cells"""
        return [
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Calculate performance for top strategies\n",
                    "performance_results = []\n",
                    "\n",
                    "# Get top N strategies to analyze (adjust based on compute time)\n",
                    "strategies_to_analyze = signal_stats.head(20) if 'signal_stats' in locals() else strategy_index.head(20)\n",
                    "\n",
                    "for idx, row in strategies_to_analyze.iterrows():\n",
                    "    # Load signals\n",
                    "    signals = con.execute(f\"\"\"\n",
                    "        SELECT ts, val FROM read_parquet('{results_path}/traces/**/*.parquet')\n",
                    "        WHERE strategy_hash = '{row['strategy_hash']}'\n",
                    "        ORDER BY ts\n",
                    "    \"\"\").df()\n",
                    "    \n",
                    "    if len(signals) > 0:\n",
                    "        signals['ts'] = pd.to_datetime(signals['ts'])\n",
                    "        perf = calculate_performance(row['strategy_hash'], signals)\n",
                    "        perf.update(row.to_dict())\n",
                    "        performance_results.append(perf)\n",
                    "\n",
                    "performance_df = pd.DataFrame(performance_results)\n",
                    "print(f'Calculated performance for {len(performance_df)} strategies')\n",
                    "\n",
                    "# Show top performers\n",
                    "print('\\nTop 10 by Sharpe Ratio:')\n",
                    "cols_to_show = [col for col in ['period', 'std_dev', 'sharpe_ratio', 'total_return', 'max_drawdown'] \n",
                    "                if col in performance_df.columns]\n",
                    "print(performance_df.nlargest(10, 'sharpe_ratio')[cols_to_show].round(3))"
                ]
            }
        ]
    
    def _create_visualization_cells(self) -> List[Dict]:
        """Create visualization cells"""
        return [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Visualizations"]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Plot best strategy equity curve\n",
                    "if len(performance_df) > 0:\n",
                    "    best_idx = performance_df['sharpe_ratio'].idxmax()\n",
                    "    best_strategy = performance_df.loc[best_idx]\n",
                    "    \n",
                    "    plt.figure(figsize=(15, 8))\n",
                    "    \n",
                    "    # Equity curve\n",
                    "    plt.subplot(2, 1, 1)\n",
                    "    df = best_strategy['df']\n",
                    "    plt.plot(df.index, df['cum_returns'], label='Strategy', linewidth=2)\n",
                    "    plt.plot(df.index, (1 + df['returns']).cumprod(), label='Buy & Hold', alpha=0.7)\n",
                    "    plt.title(f'Best Strategy Performance (Sharpe: {best_strategy[\"sharpe_ratio\"]:.2f})')\n",
                    "    plt.ylabel('Cumulative Returns')\n",
                    "    plt.legend()\n",
                    "    plt.grid(True, alpha=0.3)\n",
                    "    \n",
                    "    # Drawdown\n",
                    "    plt.subplot(2, 1, 2)\n",
                    "    drawdown = (df['cum_returns'] / df['cum_returns'].expanding().max() - 1)\n",
                    "    plt.fill_between(df.index, drawdown, 0, alpha=0.3, color='red')\n",
                    "    plt.ylabel('Drawdown')\n",
                    "    plt.grid(True, alpha=0.3)\n",
                    "    \n",
                    "    plt.tight_layout()\n",
                    "    plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Parameter sensitivity analysis\n",
                    "if 'period' in performance_df.columns and 'std_dev' in performance_df.columns:\n",
                    "    fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n",
                    "    \n",
                    "    # Sharpe by period\n",
                    "    period_stats = performance_df.groupby('period')['sharpe_ratio'].agg(['mean', 'std'])\n",
                    "    period_stats['mean'].plot(ax=axes[0, 0], marker='o')\n",
                    "    axes[0, 0].set_title('Sharpe Ratio by Period')\n",
                    "    axes[0, 0].set_xlabel('Period')\n",
                    "    axes[0, 0].grid(True)\n",
                    "    \n",
                    "    # Sharpe by std_dev\n",
                    "    std_stats = performance_df.groupby('std_dev')['sharpe_ratio'].agg(['mean', 'std'])\n",
                    "    std_stats['mean'].plot(ax=axes[0, 1], marker='o')\n",
                    "    axes[0, 1].set_title('Sharpe Ratio by Std Dev')\n",
                    "    axes[0, 1].set_xlabel('Standard Deviation')\n",
                    "    axes[0, 1].grid(True)\n",
                    "    \n",
                    "    # Scatter plots\n",
                    "    axes[1, 0].scatter(performance_df['total_return'], performance_df['sharpe_ratio'], alpha=0.6)\n",
                    "    axes[1, 0].set_xlabel('Total Return')\n",
                    "    axes[1, 0].set_ylabel('Sharpe Ratio')\n",
                    "    axes[1, 0].set_title('Return vs Risk-Adjusted Return')\n",
                    "    axes[1, 0].grid(True)\n",
                    "    \n",
                    "    axes[1, 1].scatter(performance_df['max_drawdown'], performance_df['sharpe_ratio'], alpha=0.6)\n",
                    "    axes[1, 1].set_xlabel('Max Drawdown')\n",
                    "    axes[1, 1].set_ylabel('Sharpe Ratio')\n",
                    "    axes[1, 1].set_title('Drawdown vs Sharpe')\n",
                    "    axes[1, 1].grid(True)\n",
                    "    \n",
                    "    plt.tight_layout()\n",
                    "    plt.show()"
                ]
            }
        ]
    
    def _create_pattern_discovery_cells(self) -> List[Dict]:
        """Create pattern discovery and library cells"""
        return [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Pattern Discovery & Library"]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Initialize pattern discovery\n",
                    "from pattern_discovery import PatternDiscovery, TraceAnalysis\n",
                    "\n",
                    "# Load existing patterns\n",
                    "pattern_lib = PatternDiscovery()\n",
                    "existing_patterns = pattern_lib.load_patterns()\n",
                    "print(f'Loaded {len(existing_patterns)} existing patterns')\n",
                    "\n",
                    "# Initialize trace analysis for this run\n",
                    "trace_analysis = TraceAnalysis(results_path)\n",
                    "\n",
                    "# Test existing patterns on this run\n",
                    "print('\\nTesting existing patterns on current data:')\n",
                    "for pattern in existing_patterns[:5]:  # Test top 5\n",
                    "    result = pattern_lib.test_pattern(pattern['name'], trace_analysis)\n",
                    "    print(f\"  {pattern['name']}: {result['success_rate']:.1%} success rate\")"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Discover new patterns in this run\n",
                    "if 'performance_df' in locals() and len(performance_df) > 0:\n",
                    "    # Find common patterns in top performers\n",
                    "    top_performers = performance_df.nlargest(10, 'sharpe_ratio')\n",
                    "    \n",
                    "    # Example: Parameter sweet spots\n",
                    "    if 'period' in top_performers.columns:\n",
                    "        common_periods = top_performers['period'].mode().values\n",
                    "        if len(common_periods) > 0:\n",
                    "            pattern_query = f\"\"\"\n",
                    "            SELECT COUNT(*) as signals, AVG(val) as avg_strength\n",
                    "            FROM traces\n",
                    "            WHERE json_extract(metadata, '$.parameters.period') = '{common_periods[0]}'\n",
                    "            AND val != 0\n",
                    "            \"\"\"\n",
                    "            \n",
                    "            # Save as pattern if successful\n",
                    "            pattern_name = f'optimal_period_{strategy_type}_{common_periods[0]}'\n",
                    "            if trace_analysis.query(pattern_query)['signals'] > 100:\n",
                    "                trace_analysis.save_pattern(\n",
                    "                    name=pattern_name,\n",
                    "                    query=pattern_query,\n",
                    "                    description=f'Period {common_periods[0]} shows consistent performance'\n",
                    "                )\n",
                    "                print(f'✅ Saved new pattern: {pattern_name}')"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Interactive pattern exploration\n",
                    "# Find regime-specific patterns\n",
                    "regime_pattern = \"\"\"\n",
                    "WITH regime_performance AS (\n",
                    "    SELECT \n",
                    "        DATE(ts) as trading_day,\n",
                    "        AVG(CASE WHEN val > 0 THEN 1 ELSE 0 END) as long_ratio,\n",
                    "        COUNT(*) as daily_signals\n",
                    "    FROM traces\n",
                    "    WHERE val != 0\n",
                    "    GROUP BY DATE(ts)\n",
                    ")\n",
                    "SELECT \n",
                    "    AVG(long_ratio) as avg_long_bias,\n",
                    "    AVG(daily_signals) as avg_daily_signals,\n",
                    "    STDDEV(daily_signals) as signal_consistency\n",
                    "FROM regime_performance\n",
                    "\"\"\"\n",
                    "\n",
                    "regime_stats = trace_analysis.query(regime_pattern)\n",
                    "print('Regime analysis:', regime_stats)\n",
                    "\n",
                    "# Save if interesting\n",
                    "if regime_stats['avg_long_bias'] > 0.7:\n",
                    "    trace_analysis.save_pattern(\n",
                    "        name='bullish_bias_regime',\n",
                    "        query=regime_pattern,\n",
                    "        description='Strategy shows strong bullish bias in certain regimes'\n",
                    "    )"
                ]
            }
        ]
    
    def _create_export_cell(self) -> Dict:
        """Create results export cell"""
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Export Results"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Export results\n",
                "if len(performance_df) > 0:\n",
                "    # Best parameters\n",
                "    best = performance_df.loc[performance_df['sharpe_ratio'].idxmax()]\n",
                "    \n",
                "    recommendations = {\n",
                "        'best_overall': {\n",
                "            'strategy_hash': best.get('strategy_hash'),\n",
                "            'sharpe_ratio': float(best['sharpe_ratio']),\n",
                "            'total_return': float(best['total_return']),\n",
                "            'max_drawdown': float(best['max_drawdown']),\n",
                "        },\n",
                "        'run_info': {\n",
                "            'run_id': results_path.name,\n",
                "            'total_strategies': len(strategy_index),\n",
                "            'analyzed': len(performance_df)\n",
                "        }\n",
                "    }\n",
                "    \n",
                "    # Add strategy-specific params\n",
                "    for col in performance_df.columns:\n",
                "        if col in ['period', 'std_dev', 'fast_period', 'slow_period']:\n",
                "            recommendations['best_overall'][col] = best.get(col)\n",
                "    \n",
                "    # Save recommendations\n",
                "    with open(results_path / 'recommendations.json', 'w') as f:\n",
                "        json.dump(recommendations, f, indent=2)\n",
                "        \n",
                "    print('✅ Recommendations saved to recommendations.json')\n",
                "    print(f'\\nBest strategy: {best.get(\"strategy_hash\")}')\n",
                "    print(f'Sharpe Ratio: {best[\"sharpe_ratio\"]:.2f}')\n",
                "    print(f'Total Return: {best[\"total_return\"]:.1%}')\n",
                "    print(f'Max Drawdown: {best[\"max_drawdown\"]:.1%}')"
            ]
        }
    }
    
    def _launch_notebook(self, notebook_path: Path):
        """Launch Jupyter with the generated notebook"""
        try:
            subprocess.run(["jupyter", "lab", str(notebook_path)], check=False)
        except Exception as e:
            print(f"Could not auto-launch Jupyter: {e}")
            print(f"You can manually open: jupyter lab {notebook_path}")
    
    def _generic_strategy_cells(self, config: Dict) -> List[Dict]:
        """Generic strategy analysis for unknown types"""
        return [
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Generic strategy analysis\n",
                    "# Load available data\n",
                    "print('Available files:')\n",
                    "for f in results_path.rglob('*.parquet'):\n",
                    "    print(f'  {f.relative_to(results_path)}')\n",
                    "\n",
                    "# Load strategy index if available\n",
                    "if (results_path / 'strategy_index.parquet').exists():\n",
                    "    strategy_index = pd.read_parquet(results_path / 'strategy_index.parquet')\n",
                    "    print(f'\\nLoaded {len(strategy_index)} strategies')\n",
                    "    print('\\nColumns:', strategy_index.columns.tolist())"
                ]
            }
        ]
    
    def _momentum_burst_cells(self, config: Dict) -> List[Dict]:
        """Momentum burst specific analysis"""
        # Similar structure to bollinger_bands_cells but with momentum-specific queries
        return []
    
    def _ensemble_cells(self, config: Dict) -> List[Dict]:
        """Ensemble strategy specific analysis"""
        # Analysis for composite strategies
        return []
```

### 2. Integration with Main System

```python
# In your main run script or workflow manager
from notebook_generator import AnalysisNotebookGenerator

class WorkflowManager:
    def __init__(self):
        self.notebook_gen = AnalysisNotebookGenerator()
        
    def run_backtest(self, config_path: str, args):
        # Run your backtest
        results = run_strategy_backtest(config_path)
        
        # Generate analysis notebook if requested
        if args.generate_notebook:
            notebook_path = self.notebook_gen.generate(
                run_dir=results.output_dir,
                config=results.config,
                strategy_type=results.strategy_type,
                launch=args.launch_notebook
            )
            
            # Optional: Add to run metadata
            results.metadata['analysis_notebook'] = str(notebook_path)
            
        return results

# CLI Integration
parser.add_argument('--notebook', action='store_true', 
                   help='Generate analysis notebook after run')
parser.add_argument('--launch-notebook', action='store_true',
                   help='Auto-launch Jupyter after generating notebook')
```

### 3. Template System for Different Analysis Types

```python
# notebook_templates/parameter_sweep.py
def parameter_sweep_cells(param_space: Dict) -> List[Dict]:
    """Generate cells for parameter sweep analysis"""
    return [
        {
            "cell_type": "code",
            "source": [
                f"# Parameter space analyzed\n",
                f"param_space = {json.dumps(param_space, indent=2)}\n",
                "\n",
                "# Create parameter combination matrix\n",
                "param_combos = con.execute(\"\"\"\n",
                "    SELECT DISTINCT\n",
                "        json_extract(metadata, '$.parameters') as params,\n",
                "        strategy_hash\n",
                "    FROM read_parquet('traces/**/*.parquet')\n",
                "    WHERE metadata IS NOT NULL\n",
                "\"\"\").df()\n",
                "\n",
                "print(f'Total parameter combinations tested: {len(param_combos)}')"
            ]
        }
    ]

# notebook_templates/single_strategy.py
def single_strategy_cells(strategy_config: Dict) -> List[Dict]:
    """Generate cells for single strategy deep-dive"""
    return [
        {
            "cell_type": "code",
            "source": [
                "# Load detailed signal data\n",
                "signals = pd.read_parquet(list(results_path.glob('traces/**/*.parquet'))[0])\n",
                "print(f'Total signals generated: {(signals[\"val\"] != 0).sum()}')\n",
                "\n",
                "# Signal distribution\n",
                "plt.figure(figsize=(10, 6))\n",
                "signals['val'].value_counts().plot(kind='bar')\n",
                "plt.title('Signal Distribution')\n",
                "plt.show()"
            ]
        }
    ]
```

### 4. Advanced Features

```python
class AdvancedNotebookGenerator(AnalysisNotebookGenerator):
    """Extended generator with advanced features"""
    
    def generate_comparative(self, run_dirs: List[Path], comparison_name: str) -> Path:
        """Generate notebook comparing multiple runs"""
        # Compare different runs/strategies
        pass
    
    def generate_production_ready(self, run_dir: Path, best_strategy_hash: str) -> Path:
        """Generate notebook for production deployment preparation"""
        # Focus on single best strategy
        # Include stability analysis, edge cases, etc.
        pass
    
    def generate_from_template(self, template_name: str, **kwargs) -> Path:
        """Generate from user-defined templates"""
        template_path = self.template_dir / f"{template_name}.json"
        # Load and customize template
        pass
```

## Usage Examples

### Basic Usage
```bash
# Run backtest and generate notebook
python main.py --config configs/bollinger/config.yaml --notebook

# Run and immediately launch analysis
python main.py --config configs/bollinger/config.yaml --notebook --launch-notebook
```

### In Python Script
```python
# After backtest completes
from notebook_generator import AnalysisNotebookGenerator

gen = AnalysisNotebookGenerator()
notebook = gen.generate(
    run_dir=Path("configs/bollinger/results/run_20250623_120000"),
    config=config_dict,
    strategy_type="bollinger_bands",
    launch=True
)
```

### Custom Templates
```python
# Create custom template for specific analysis
gen.register_template("my_analysis", my_custom_cells_function)
notebook = gen.generate_from_template("my_analysis", 
                                     run_dir=run_dir,
                                     custom_param="value")
```

## Benefits

1. **Zero Friction Analysis**
   - No manual path configuration
   - No copy-pasting queries
   - Instant analysis readiness

2. **Consistency**
   - Same analysis structure across all runs
   - Standardized visualizations
   - Comparable results

3. **Adaptability**
   - Strategy-specific analysis
   - Dynamic content based on available data
   - Extensible template system

4. **Documentation**
   - Each run has its analysis notebook
   - Self-documenting results
   - Reproducible analysis

5. **Time Savings**
   - From run completion to analysis: seconds
   - No setup overhead
   - Focus on insights, not infrastructure

## Future Enhancements

1. **HTML Reports**
   - Export notebooks to static HTML reports
   - Email-ready summaries
   - Dashboard generation

2. **Analysis Caching**
   - Cache expensive calculations
   - Incremental updates
   - Fast re-runs

3. **Cloud Integration**
   - Upload to S3/GCS
   - Share via links
   - Collaborative analysis

4. **AI-Powered Insights**
   - Auto-generate observations
   - Anomaly detection
   - Parameter recommendations

## Pattern Library Implementation

### Core Pattern Classes

```python
# pattern_discovery.py
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import duckdb
import hashlib

class Pattern:
    """A discovered trading pattern"""
    def __init__(self, name: str, query: str, description: str, 
                 metadata: Optional[Dict] = None):
        self.name = name
        self.query = query
        self.description = description
        self.metadata = metadata or {}
        self.discovered_at = datetime.now()
        self.success_rate = None
        self.sample_size = None
        self.hash = self._compute_hash()
        
    def _compute_hash(self) -> str:
        """Unique identifier for pattern"""
        content = f"{self.name}{self.query}{self.description}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
        
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'query': self.query,
            'description': self.description,
            'metadata': self.metadata,
            'discovered_at': self.discovered_at.isoformat(),
            'success_rate': self.success_rate,
            'sample_size': self.sample_size,
            'hash': self.hash
        }

class PatternLibrary:
    """Persistent storage for discovered patterns"""
    
    def __init__(self, patterns_dir: Path = None):
        self.patterns_dir = patterns_dir or Path("saved_patterns")
        self.patterns_dir.mkdir(exist_ok=True)
        self.patterns_file = self.patterns_dir / "patterns.yaml"
        self.patterns = self.load_patterns()
        
    def load_patterns(self) -> List[Pattern]:
        """Load patterns from disk"""
        if not self.patterns_file.exists():
            return []
            
        with open(self.patterns_file, 'r') as f:
            data = yaml.safe_load(f) or []
            
        return [Pattern(**p) for p in data]
    
    def save_pattern(self, pattern: Pattern):
        """Add pattern to library"""
        # Check if pattern already exists
        existing_hashes = {p.hash for p in self.patterns}
        if pattern.hash in existing_hashes:
            print(f"Pattern '{pattern.name}' already exists")
            return
            
        self.patterns.append(pattern)
        self._persist()
        print(f"✅ Saved pattern: {pattern.name}")
        
    def _persist(self):
        """Save patterns to disk"""
        data = [p.to_dict() for p in self.patterns]
        with open(self.patterns_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
            
    def search(self, strategy_type: str = None, min_success_rate: float = None) -> List[Pattern]:
        """Search patterns by criteria"""
        results = self.patterns
        
        if strategy_type:
            results = [p for p in results 
                      if p.metadata.get('strategy_type') == strategy_type]
                      
        if min_success_rate:
            results = [p for p in results 
                      if p.success_rate and p.success_rate >= min_success_rate]
                      
        return results

class TraceAnalysis:
    """Interface for analyzing trace data"""
    
    def __init__(self, results_path: Path):
        self.results_path = results_path
        self.con = duckdb.connect()
        self.pattern_library = PatternLibrary()
        
        # Create view for traces
        trace_pattern = f"{results_path}/traces/**/*.parquet"
        self.con.execute(f"""
            CREATE VIEW traces AS 
            SELECT * FROM read_parquet('{trace_pattern}')
        """)
        
    def query(self, sql: str) -> pd.DataFrame:
        """Execute query on trace data"""
        return self.con.execute(sql).df()
        
    def save_pattern(self, name: str, query: str, description: str, 
                    metadata: Optional[Dict] = None):
        """Save a discovered pattern"""
        # Test the pattern first
        try:
            result = self.query(query)
            success_metrics = self._calculate_success_metrics(result)
        except Exception as e:
            print(f"Pattern query failed: {e}")
            return
            
        # Create pattern
        pattern = Pattern(
            name=name,
            query=query,
            description=description,
            metadata=metadata or {}
        )
        pattern.success_rate = success_metrics.get('success_rate')
        pattern.sample_size = success_metrics.get('sample_size')
        
        # Save to library
        self.pattern_library.save_pattern(pattern)
        
    def _calculate_success_metrics(self, result: pd.DataFrame) -> Dict:
        """Calculate pattern success metrics"""
        # Basic implementation - extend based on your needs
        return {
            'success_rate': 0.0,  # Implement based on result
            'sample_size': len(result)
        }

class PatternDiscovery:
    """Interactive pattern discovery and testing"""
    
    def __init__(self, patterns_dir: Path = None):
        self.library = PatternLibrary(patterns_dir)
        self.discovered_patterns = []
        
    def load_patterns(self) -> List[Dict]:
        """Load existing patterns"""
        return [p.to_dict() for p in self.library.patterns]
        
    def test_pattern(self, pattern_name: str, trace_analysis: TraceAnalysis) -> Dict:
        """Test a pattern on new data"""
        # Find pattern
        pattern = next((p for p in self.library.patterns if p.name == pattern_name), None)
        if not pattern:
            return {'error': f'Pattern {pattern_name} not found'}
            
        # Run query
        try:
            result = trace_analysis.query(pattern.query)
            metrics = trace_analysis._calculate_success_metrics(result)
            return {
                'success_rate': metrics['success_rate'],
                'sample_size': metrics['sample_size'],
                'data': result
            }
        except Exception as e:
            return {'error': str(e)}
            
    def discover_parameter_patterns(self, trace_analysis: TraceAnalysis, 
                                  top_n: int = 5) -> List[Pattern]:
        """Automatically discover parameter patterns"""
        patterns = []
        
        # Query for parameter performance
        param_query = """
        SELECT 
            json_extract(metadata, '$.parameters') as params,
            COUNT(*) as signal_count,
            AVG(CASE WHEN val > 0 THEN 1 ELSE -1 END) as avg_direction
        FROM traces
        WHERE metadata IS NOT NULL
        GROUP BY params
        ORDER BY signal_count DESC
        LIMIT 20
        """
        
        try:
            results = trace_analysis.query(param_query)
            
            # Find patterns in top performers
            for idx, row in results.head(top_n).iterrows():
                pattern_name = f"high_signal_params_{idx}"
                pattern = Pattern(
                    name=pattern_name,
                    query=f"SELECT * FROM traces WHERE json_extract(metadata, '$.parameters') = '{row['params']}'",
                    description=f"Parameters with high signal generation: {row['params']}",
                    metadata={'type': 'parameter_pattern', 'signal_count': row['signal_count']}
                )
                patterns.append(pattern)
                
        except Exception as e:
            print(f"Pattern discovery error: {e}")
            
        return patterns
```

### Integration with Notebook Generator

```python
# Enhanced notebook generator with pattern support
class PatternAwareNotebookGenerator(AnalysisNotebookGenerator):
    """Notebook generator that leverages pattern library"""
    
    def __init__(self, template_dir: Optional[Path] = None):
        super().__init__(template_dir)
        self.pattern_library = PatternLibrary()
        
    def _create_pattern_cells(self, strategy_type: str) -> List[Dict]:
        """Create cells that use existing patterns"""
        cells = []
        
        # Find relevant patterns
        relevant_patterns = self.pattern_library.search(strategy_type=strategy_type)
        
        if relevant_patterns:
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Apply Known Patterns"]
            })
            
            for pattern in relevant_patterns[:3]:  # Top 3 patterns
                cells.append({
                    "cell_type": "code",
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                    "source": [
                        f"# Pattern: {pattern.name}\n",
                        f"# Description: {pattern.description}\n",
                        f"# Historical success rate: {pattern.success_rate:.1%}\n",
                        f"\n",
                        f"pattern_result = trace_analysis.query(\"\"\"\n",
                        f"{pattern.query}\n",
                        f"\"\"\")\n",
                        f"\n",
                        f"print(f'Pattern {pattern.name} matches: {{len(pattern_result)}} instances')\n",
                        f"if len(pattern_result) > 0:\n",
                        f"    print(pattern_result.head())"
                    ]
                })
                
        return cells
```

### Example Pattern Library File

```yaml
# saved_patterns/patterns.yaml
- name: optimal_bollinger_period_20
  query: |
    SELECT strategy_hash, COUNT(*) as signals, AVG(val) as avg_strength
    FROM traces
    WHERE json_extract(metadata, '$.parameters.period') = '20'
    AND val != 0
    GROUP BY strategy_hash
    HAVING COUNT(*) > 50
  description: Period 20 shows optimal signal frequency for Bollinger Bands
  metadata:
    strategy_type: bollinger_bands
    discovered_in_run: run_20250623_120000
  discovered_at: '2025-06-23T12:30:00'
  success_rate: 0.68
  sample_size: 1250
  hash: a3f4b2c1d5e6

- name: high_volatility_entry_pattern
  query: |
    WITH volatility_entries AS (
      SELECT ts, val, 
             json_extract(metadata, '$.market_conditions.vix') as vix
      FROM traces
      WHERE val = 1  -- Long entries only
    )
    SELECT * FROM volatility_entries
    WHERE CAST(vix AS FLOAT) > 20
  description: Entries during high VIX (>20) show better risk/reward
  metadata:
    strategy_type: all
    pattern_type: entry_timing
  discovered_at: '2025-06-22T15:45:00'
  success_rate: 0.74
  sample_size: 89
  hash: b7e8f9a0c1d2

- name: morning_momentum_sweet_spot
  query: |
    SELECT 
      DATE(ts) as trading_day,
      COUNT(*) as morning_signals,
      AVG(val) as avg_direction
    FROM traces
    WHERE EXTRACT(HOUR FROM ts) BETWEEN 9 AND 11
    AND val != 0
    GROUP BY DATE(ts)
    HAVING COUNT(*) BETWEEN 2 AND 5
  description: 2-5 signals in first 2 hours indicates strong trend day
  metadata:
    strategy_type: momentum
    time_of_day: morning
    market_session: regular
  discovered_at: '2025-06-20T10:15:00'
  success_rate: 0.82
  sample_size: 45
  hash: c3d4e5f6a7b8
```

## Benefits of Pattern Library Integration

1. **Cumulative Learning**: Every backtest contributes to collective knowledge
2. **Rapid Validation**: Test if known patterns hold in new data
3. **Cross-Strategy Insights**: Patterns discovered in one strategy can benefit others
4. **Hypothesis Generation**: Browse patterns for new research ideas
5. **Risk Awareness**: Anti-patterns help avoid known pitfalls

This integration transforms each notebook from an isolated analysis into part of a growing knowledge system. The pattern library becomes your "trading research memory" that gets smarter with every run.
