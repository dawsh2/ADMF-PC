"""
Auto-Generated Analysis Notebooks for ADMF-PC

This module automatically generates pre-configured Jupyter notebooks after each backtest run,
with all paths, queries, and visualizations ready to go. No more copy-pasting code or 
adjusting paths - just run your backtest and dive straight into analysis.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


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
        """
        Generate analysis notebook for a specific run
        
        Args:
            run_dir: Path to the results directory
            config: The configuration dictionary used for the run
            strategy_type: Primary strategy type (bollinger_bands, momentum, etc)
            launch: Whether to auto-launch Jupyter after generation
            
        Returns:
            Path to the generated notebook
        """
        # Create notebook structure
        notebook = {
            "cells": [
                self._create_header_cell(config, run_dir),
                self._create_setup_cell(run_dir),
                self._create_load_data_cell(),
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
            
        logger.info(f"✅ Generated analysis notebook: {notebook_path}")
        
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
            "metadata": {"tags": ["setup"]},
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
                "con = duckdb.connect()\n",
                "\n",
                "# Helper functions\n",
                "def format_number(x):\n",
                "    if abs(x) >= 1e6:\n",
                "        return f'{x/1e6:.1f}M'\n",
                "    elif abs(x) >= 1e3:\n",
                "        return f'{x/1e3:.1f}K'\n",
                "    else:\n",
                "        return f'{x:.2f}'"
            ]
        }
    
    def _create_load_data_cell(self) -> Dict:
        """Create data loading cell"""
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Load Data <a name='load-data'></a>"]
        }, {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Load strategy index\n",
                "try:\n",
                "    strategy_index = pd.read_parquet(results_path / 'strategy_index.parquet')\n",
                "    print(f'Total strategies tested: {len(strategy_index)}')\n",
                "    print(f'Strategy types: {strategy_index[\"strategy_type\"].value_counts().to_dict()}')\n",
                "except FileNotFoundError:\n",
                "    print('No strategy index found - using legacy format')\n",
                "    strategy_index = None\n",
                "\n",
                "# Load metadata\n",
                "with open(results_path / 'metadata.json', 'r') as f:\n",
                "    metadata = json.load(f)\n",
                "    \n",
                "print(f'\\nTotal bars processed: {metadata.get(\"total_bars\", 0):,}')\n",
                "print(f'Total signals generated: {metadata.get(\"total_signals\", 0):,}')\n",
                "print(f'Compression ratio: {metadata.get(\"compression_ratio\", 0):.2f}%')"
            ]
        }
    
    def _create_strategy_specific_cells(self, strategy_type: str, config: Dict) -> List[Dict]:
        """Create cells specific to the strategy type"""
        cells = []
        
        # Map strategy types to their specific analysis methods
        strategy_cells = {
            "bollinger_bands": self._bollinger_bands_cells,
            "momentum": self._momentum_cells,
            "ma_crossover": self._ma_crossover_cells,
            "sma_crossover": self._ma_crossover_cells,
            "rsi": self._rsi_cells,
            "ensemble": self._ensemble_cells,
        }
        
        # Get the appropriate method or use generic
        cell_method = strategy_cells.get(strategy_type, self._generic_strategy_cells)
        cells.extend(cell_method(config))
        
        return cells
    
    def _bollinger_bands_cells(self, config: Dict) -> List[Dict]:
        """Bollinger Bands specific analysis cells"""
        return [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Signal Analysis <a name='signal-analysis'></a>"]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Query signal statistics using enhanced metadata\n",
                    "if strategy_index is not None:\n",
                    "    # Using new enhanced format with strategy index\n",
                    "    signal_stats = con.execute(f\"\"\"\n",
                    "        WITH trace_data AS (\n",
                    "            SELECT \n",
                    "                t.strategy_hash,\n",
                    "                si.param_period as period,\n",
                    "                si.param_std_dev as std_dev,\n",
                    "                COUNT(*) as num_signals,\n",
                    "                COUNT(DISTINCT DATE(t.ts)) as trading_days,\n",
                    "                SUM(CASE WHEN t.val > 0 THEN 1 ELSE 0 END) as long_signals,\n",
                    "                SUM(CASE WHEN t.val < 0 THEN 1 ELSE 0 END) as short_signals\n",
                    "            FROM read_parquet('{results_path}/traces/**/*.parquet') t\n",
                    "            JOIN strategy_index si ON t.strategy_hash = si.strategy_hash\n",
                    "            WHERE t.val != 0 AND si.strategy_type = 'bollinger_bands'\n",
                    "            GROUP BY t.strategy_hash, si.param_period, si.param_std_dev\n",
                    "        )\n",
                    "        SELECT \n",
                    "            strategy_hash,\n",
                    "            CAST(period AS INT) as period,\n",
                    "            CAST(std_dev AS FLOAT) as std_dev,\n",
                    "            num_signals,\n",
                    "            trading_days,\n",
                    "            long_signals,\n",
                    "            short_signals,\n",
                    "            num_signals::FLOAT / trading_days as signals_per_day,\n",
                    "            long_signals::FLOAT / NULLIF(num_signals, 0) as long_ratio\n",
                    "        FROM trace_data\n",
                    "        ORDER BY period, std_dev\n",
                    "    \"\"\").df()\n",
                    "else:\n",
                    "    # Fallback for legacy format\n",
                    "    signal_stats = con.execute(f\"\"\"\n",
                    "        SELECT \n",
                    "            strat as strategy_id,\n",
                    "            COUNT(*) as num_signals,\n",
                    "            COUNT(DISTINCT DATE(ts)) as trading_days\n",
                    "        FROM read_parquet('{results_path}/traces/**/*.parquet')\n",
                    "        WHERE val != 0\n",
                    "        GROUP BY strat\n",
                    "    \"\"\").df()\n",
                    "\n",
                    "print(f'Loaded {len(signal_stats)} strategies with signals')\n",
                    "signal_stats.head()"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Create parameter heatmaps\n",
                    "if 'period' in signal_stats.columns and 'std_dev' in signal_stats.columns:\n",
                    "    fig, axes = plt.subplots(2, 2, figsize=(15, 12))\n",
                    "    \n",
                    "    # Signals per day heatmap\n",
                    "    signals_pivot = signal_stats.pivot(index='period', columns='std_dev', values='signals_per_day')\n",
                    "    sns.heatmap(signals_pivot, annot=True, fmt='.2f', cmap='viridis', ax=axes[0, 0])\n",
                    "    axes[0, 0].set_title('Average Signals Per Day')\n",
                    "    \n",
                    "    # Total signals heatmap\n",
                    "    total_pivot = signal_stats.pivot(index='period', columns='std_dev', values='num_signals')\n",
                    "    sns.heatmap(total_pivot, annot=False, cmap='plasma', ax=axes[0, 1])\n",
                    "    axes[0, 1].set_title('Total Number of Signals')\n",
                    "    \n",
                    "    # Optimal frequency region (1-3 signals per day)\n",
                    "    optimal_mask = (signals_pivot >= 1) & (signals_pivot <= 3)\n",
                    "    sns.heatmap(optimal_mask.astype(int), cmap='RdYlGn', ax=axes[1, 0])\n",
                    "    axes[1, 0].set_title('Optimal Signal Frequency (1-3/day)')\n",
                    "    \n",
                    "    # Long/short ratio\n",
                    "    if 'long_ratio' in signal_stats.columns:\n",
                    "        ratio_pivot = signal_stats.pivot(index='period', columns='std_dev', values='long_ratio')\n",
                    "        sns.heatmap(ratio_pivot, annot=True, fmt='.2f', cmap='coolwarm', center=0.5, ax=axes[1, 1])\n",
                    "        axes[1, 1].set_title('Long Signal Ratio')\n",
                    "    \n",
                    "    plt.tight_layout()\n",
                    "    plt.show()\n",
                    "    \n",
                    "    # Find optimal frequency strategies\n",
                    "    optimal_freq = signal_stats[(signal_stats['signals_per_day'] >= 1) & \n",
                    "                               (signal_stats['signals_per_day'] <= 3)]\n",
                    "    print(f'\\nStrategies with optimal frequency (1-3 signals/day): {len(optimal_freq)}')\n",
                    "    if len(optimal_freq) > 0:\n",
                    "        print('\\nTop 5 by signal frequency:')\n",
                    "        print(optimal_freq.nlargest(5, 'signals_per_day')[['period', 'std_dev', 'signals_per_day', 'num_signals']])"
                ]
            }
        ]
    
    def _momentum_cells(self, config: Dict) -> List[Dict]:
        """Momentum strategy specific analysis"""
        return [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Momentum Strategy Analysis <a name='momentum-analysis'></a>"]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Analyze momentum burst patterns\n",
                    "if strategy_index is not None:\n",
                    "    momentum_strategies = strategy_index[\n",
                    "        strategy_index['strategy_type'].str.contains('momentum', case=False, na=False)\n",
                    "    ]\n",
                    "    print(f'Found {len(momentum_strategies)} momentum strategies')\n",
                    "else:\n",
                    "    momentum_strategies = pd.DataFrame()\n",
                    "\n",
                    "# Analyze signal timing patterns\n",
                    "momentum_analysis = con.execute(f\"\"\"\n",
                    "    WITH momentum_signals AS (\n",
                    "        SELECT \n",
                    "            ts,\n",
                    "            val as signal,\n",
                    "            strategy_hash,\n",
                    "            EXTRACT(HOUR FROM ts) as hour,\n",
                    "            EXTRACT(DOW FROM ts) as day_of_week,\n",
                    "            LAG(ts) OVER (PARTITION BY strategy_hash ORDER BY ts) as prev_ts\n",
                    "        FROM read_parquet('{results_path}/traces/**/*.parquet')\n",
                    "        WHERE val != 0\n",
                    "    ),\n",
                    "    burst_analysis AS (\n",
                    "        SELECT \n",
                    "            strategy_hash,\n",
                    "            EXTRACT(EPOCH FROM (ts - prev_ts))/3600 as hours_between_signals\n",
                    "        FROM momentum_signals\n",
                    "        WHERE prev_ts IS NOT NULL\n",
                    "    )\n",
                    "    SELECT \n",
                    "        hour,\n",
                    "        COUNT(*) as signal_count,\n",
                    "        SUM(CASE WHEN signal > 0 THEN 1 ELSE 0 END) as long_count,\n",
                    "        SUM(CASE WHEN signal < 0 THEN 1 ELSE 0 END) as short_count,\n",
                    "        AVG(ABS(signal)) as avg_strength\n",
                    "    FROM momentum_signals\n",
                    "    GROUP BY hour\n",
                    "    ORDER BY hour\n",
                    "\"\"\").df()\n",
                    "\n",
                    "# Plot intraday patterns\n",
                    "if len(momentum_analysis) > 0:\n",
                    "    fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n",
                    "    \n",
                    "    # Hourly signal distribution\n",
                    "    axes[0, 0].bar(momentum_analysis['hour'], momentum_analysis['signal_count'], alpha=0.7)\n",
                    "    axes[0, 0].set_xlabel('Hour of Day')\n",
                    "    axes[0, 0].set_ylabel('Number of Signals')\n",
                    "    axes[0, 0].set_title('Momentum Signals by Hour')\n",
                    "    axes[0, 0].grid(True, alpha=0.3)\n",
                    "    \n",
                    "    # Long vs Short distribution\n",
                    "    if 'long_count' in momentum_analysis.columns:\n",
                    "        x = momentum_analysis['hour']\n",
                    "        axes[0, 1].bar(x, momentum_analysis['long_count'], label='Long', alpha=0.7, color='green')\n",
                    "        axes[0, 1].bar(x, -momentum_analysis['short_count'], label='Short', alpha=0.7, color='red')\n",
                    "        axes[0, 1].set_xlabel('Hour of Day')\n",
                    "        axes[0, 1].set_ylabel('Signal Count')\n",
                    "        axes[0, 1].set_title('Long vs Short Signals by Hour')\n",
                    "        axes[0, 1].legend()\n",
                    "        axes[0, 1].grid(True, alpha=0.3)\n",
                    "    \n",
                    "    # Parameter sensitivity (if available)\n",
                    "    if len(momentum_strategies) > 0 and 'param_period' in momentum_strategies.columns:\n",
                    "        period_stats = con.execute(f\"\"\"\n",
                    "            SELECT \n",
                    "                si.param_period as period,\n",
                    "                COUNT(DISTINCT t.ts) as signal_count,\n",
                    "                COUNT(DISTINCT DATE(t.ts)) as active_days\n",
                    "            FROM read_parquet('{results_path}/traces/**/*.parquet') t\n",
                    "            JOIN strategy_index si ON t.strategy_hash = si.strategy_hash\n",
                    "            WHERE t.val != 0 AND si.strategy_type LIKE '%momentum%'\n",
                    "            GROUP BY si.param_period\n",
                    "            ORDER BY period\n",
                    "        \"\"\").df()\n",
                    "        \n",
                    "        if len(period_stats) > 0:\n",
                    "            axes[1, 0].plot(period_stats['period'], period_stats['signal_count'], 'o-')\n",
                    "            axes[1, 0].set_xlabel('Momentum Period')\n",
                    "            axes[1, 0].set_ylabel('Total Signals')\n",
                    "            axes[1, 0].set_title('Signal Frequency by Period')\n",
                    "            axes[1, 0].grid(True, alpha=0.3)\n",
                    "    \n",
                    "    plt.tight_layout()\n",
                    "    plt.show()\n",
                    "    \n",
                    "    # Summary statistics\n",
                    "    print(f'\\nMomentum Strategy Summary:')\n",
                    "    print(f\"Total momentum signals: {momentum_analysis['signal_count'].sum()}\")\n",
                    "    print(f\"Most active hour: {momentum_analysis.loc[momentum_analysis['signal_count'].idxmax(), 'hour']}:00\")\n",
                    "    if 'long_count' in momentum_analysis.columns:\n",
                    "        total_long = momentum_analysis['long_count'].sum()\n",
                    "        total_short = momentum_analysis['short_count'].sum()\n",
                    "        print(f'Long/Short ratio: {total_long/total_short:.2f}' if total_short > 0 else 'No short signals')"
                ]
            }
        ]
    
    def _ma_crossover_cells(self, config: Dict) -> List[Dict]:
        """Moving average crossover analysis"""
        return [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Moving Average Crossover Analysis"]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Analyze crossover patterns\n",
                    "if strategy_index is not None and 'param_fast_period' in strategy_index.columns:\n",
                    "    ma_stats = strategy_index[\n",
                    "        strategy_index['strategy_type'].str.contains('crossover', case=False, na=False)\n",
                    "    ].copy()\n",
                    "    \n",
                    "    if len(ma_stats) > 0:\n",
                    "        # Create scatter plot of fast vs slow periods\n",
                    "        plt.figure(figsize=(10, 8))\n",
                    "        plt.scatter(ma_stats['param_fast_period'], ma_stats['param_slow_period'], alpha=0.6)\n",
                    "        plt.xlabel('Fast Period')\n",
                    "        plt.ylabel('Slow Period')\n",
                    "        plt.title('MA Crossover Parameter Space Tested')\n",
                    "        \n",
                    "        # Add diagonal line (fast = slow)\n",
                    "        max_period = max(ma_stats['param_slow_period'].max(), ma_stats['param_fast_period'].max())\n",
                    "        plt.plot([0, max_period], [0, max_period], 'r--', alpha=0.3)\n",
                    "        plt.grid(True, alpha=0.3)\n",
                    "        plt.show()\n",
                    "        \n",
                    "        print(f'Total MA crossover combinations tested: {len(ma_stats)}')"
                ]
            }
        ]
    
    def _rsi_cells(self, config: Dict) -> List[Dict]:
        """RSI strategy analysis"""
        return [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## RSI Strategy Analysis <a name='rsi-analysis'></a>"]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Analyze RSI threshold effectiveness\n",
                    "if strategy_index is not None:\n",
                    "    rsi_strategies = strategy_index[\n",
                    "        strategy_index['strategy_type'].str.contains('rsi', case=False, na=False)\n",
                    "    ]\n",
                    "    \n",
                    "    if len(rsi_strategies) > 0:\n",
                    "        print(f'Found {len(rsi_strategies)} RSI strategies')\n",
                    "        \n",
                    "        # Analyze oversold/overbought threshold effectiveness\n",
                    "        if 'param_oversold' in rsi_strategies.columns:\n",
                    "            fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n",
                    "            \n",
                    "            # Scatter plot of thresholds\n",
                    "            axes[0, 0].scatter(rsi_strategies['param_oversold'], \n",
                    "                              rsi_strategies['param_overbought'], alpha=0.6)\n",
                    "            axes[0, 0].set_xlabel('Oversold Threshold')\n",
                    "            axes[0, 0].set_ylabel('Overbought Threshold')\n",
                    "            axes[0, 0].set_title('RSI Threshold Combinations Tested')\n",
                    "            axes[0, 0].grid(True, alpha=0.3)\n",
                    "            \n",
                    "            # Period distribution\n",
                    "            if 'param_period' in rsi_strategies.columns:\n",
                    "                period_counts = rsi_strategies['param_period'].value_counts().sort_index()\n",
                    "                axes[0, 1].bar(period_counts.index, period_counts.values)\n",
                    "                axes[0, 1].set_xlabel('RSI Period')\n",
                    "                axes[0, 1].set_ylabel('Count')\n",
                    "                axes[0, 1].set_title('RSI Period Distribution')\n",
                    "                axes[0, 1].grid(True, alpha=0.3)\n",
                    "            \n",
                    "            plt.tight_layout()\n",
                    "            plt.show()\n",
                    "        \n",
                    "        # Signal frequency analysis\n",
                    "        rsi_signals = con.execute(f\"\"\"\n",
                    "            SELECT \n",
                    "                si.param_period as period,\n",
                    "                si.param_oversold as oversold,\n",
                    "                si.param_overbought as overbought,\n",
                    "                COUNT(*) as signal_count,\n",
                    "                SUM(CASE WHEN t.val > 0 THEN 1 ELSE 0 END) as buy_signals,\n",
                    "                SUM(CASE WHEN t.val < 0 THEN 1 ELSE 0 END) as sell_signals\n",
                    "            FROM read_parquet('{results_path}/traces/**/*.parquet') t\n",
                    "            JOIN strategy_index si ON t.strategy_hash = si.strategy_hash\n",
                    "            WHERE t.val != 0 AND si.strategy_type LIKE '%rsi%'\n",
                    "            GROUP BY si.param_period, si.param_oversold, si.param_overbought\n",
                    "            ORDER BY signal_count DESC\n",
                    "            LIMIT 10\n",
                    "        \"\"\").df()\n",
                    "        \n",
                    "        if len(rsi_signals) > 0:\n",
                    "            print('\\nTop 10 RSI configurations by signal count:')\n",
                    "            print(rsi_signals)\n",
                    "    else:\n",
                    "        print('No RSI strategies found in this run')"
                ]
            }
        ]
    
    def _ensemble_cells(self, config: Dict) -> List[Dict]:
        """Ensemble strategy analysis"""
        return [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Ensemble Strategy Analysis <a name='ensemble-analysis'></a>"]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Analyze ensemble components and voting patterns\n",
                    "if strategy_index is not None:\n",
                    "    ensemble_strategies = strategy_index[\n",
                    "        strategy_index['strategy_type'].str.contains('ensemble', case=False, na=False)\n",
                    "    ]\n",
                    "    \n",
                    "    if len(ensemble_strategies) > 0:\n",
                    "        print(f'Found {len(ensemble_strategies)} ensemble strategies')\n",
                    "        \n",
                    "        # Try to extract component information from metadata\n",
                    "        for idx, row in ensemble_strategies.head(5).iterrows():\n",
                    "            print(f\"\\nEnsemble {idx}: {row['strategy_hash']}\")\n",
                    "            if 'metadata' in row and row['metadata']:\n",
                    "                try:\n",
                    "                    import json\n",
                    "                    meta = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']\n",
                    "                    if 'composite_strategies' in meta:\n",
                    "                        print('Components:')\n",
                    "                        for comp in meta['composite_strategies']:\n",
                    "                            print(f\"  - {comp.get('type', 'unknown')}: weight={comp.get('weight', 'N/A')}\")\n",
                    "                except:\n",
                    "                    pass\n",
                    "    else:\n",
                    "        print('No ensemble strategies found in this run')\n",
                    "\n",
                    "# Analyze voting patterns\n",
                    "voting_analysis = con.execute(f\"\"\"\n",
                    "    WITH ensemble_signals AS (\n",
                    "        SELECT \n",
                    "            ts,\n",
                    "            COUNT(DISTINCT strategy_hash) as strategies_voting,\n",
                    "            SUM(CASE WHEN val > 0 THEN 1 ELSE 0 END) as long_votes,\n",
                    "            SUM(CASE WHEN val < 0 THEN 1 ELSE 0 END) as short_votes,\n",
                    "            SUM(CASE WHEN val = 0 THEN 1 ELSE 0 END) as neutral_votes\n",
                    "        FROM read_parquet('{results_path}/traces/**/*.parquet')\n",
                    "        GROUP BY ts\n",
                    "        HAVING strategies_voting > 1\n",
                    "    )\n",
                    "    SELECT \n",
                    "        strategies_voting,\n",
                    "        COUNT(*) as occurrences,\n",
                    "        AVG(long_votes) as avg_long_votes,\n",
                    "        AVG(short_votes) as avg_short_votes\n",
                    "    FROM ensemble_signals\n",
                    "    GROUP BY strategies_voting\n",
                    "    ORDER BY strategies_voting\n",
                    "\"\"\").df()\n",
                    "\n",
                    "if len(voting_analysis) > 0:\n",
                    "    print('\\nVoting Pattern Analysis:')\n",
                    "    print(voting_analysis)"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Component correlation analysis (if multiple strategies)\n",
                    "if strategy_index is not None and len(strategy_index) > 1:\n",
                    "    print('\\nCalculating strategy correlations...')\n",
                    "    \n",
                    "    # This is a simplified correlation - in practice you'd want to\n",
                    "    # calculate based on actual signal alignment\n",
                    "    print('Note: Correlation analysis requires signal alignment across strategies')"
                ]
            }
        ]
    
    def _generic_strategy_cells(self, config: Dict) -> List[Dict]:
        """Generic strategy analysis for unknown types"""
        return [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Strategy Analysis"]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Generic strategy analysis\n",
                    "print('Available files:')\n",
                    "for f in results_path.rglob('*.parquet'):\n",
                    "    print(f'  {f.relative_to(results_path)}')\n",
                    "\n",
                    "# Basic signal statistics\n",
                    "signal_count = con.execute(f\"\"\"\n",
                    "    SELECT \n",
                    "        COUNT(*) as total_signals,\n",
                    "        COUNT(DISTINCT strat) as unique_strategies,\n",
                    "        COUNT(DISTINCT DATE(ts)) as trading_days\n",
                    "    FROM read_parquet('{results_path}/traces/**/*.parquet')\n",
                    "    WHERE val != 0\n",
                    "\"\"\").df()\n",
                    "\n",
                    "print(f'\\nSignal Statistics:')\n",
                    "print(signal_count)"
                ]
            }
        ]
    
    def _create_common_analysis_cells(self) -> List[Dict]:
        """Cells common to all strategies"""
        return [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Performance Metrics <a name='performance-metrics'></a>"]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Load market data for performance calculation\n",
                    "# Try to find market data in standard locations\n",
                    "market_data_paths = [\n",
                    "    Path('data/SPY_5m.parquet'),\n",
                    "    Path('../data/SPY_5m.parquet'),\n",
                    "    Path('../../data/SPY_5m.parquet'),\n",
                    "    Path('../../../data/SPY_5m.parquet'),\n",
                    "    Path('../../../../data/SPY_5m.parquet'),\n",
                    "]\n",
                    "\n",
                    "market_data = None\n",
                    "for path in market_data_paths:\n",
                    "    if path.exists():\n",
                    "        market_data = pd.read_parquet(path)\n",
                    "        print(f'Loaded market data from: {path}')\n",
                    "        print(f'Market data shape: {market_data.shape}')\n",
                    "        break\n",
                    "\n",
                    "if market_data is None:\n",
                    "    print('WARNING: Could not find market data file. Performance calculation will be skipped.')\n",
                    "    print('Searched in:', market_data_paths)"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "def calculate_performance(strategy_hash, signals_df, market_data):\n",
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
                    "    # Calculate metrics\n",
                    "    total_return = df['cum_returns'].iloc[-1] - 1\n",
                    "    \n",
                    "    # Annualized Sharpe (assuming 5-minute bars, 78 per day, 252 trading days)\n",
                    "    if df['strategy_returns'].std() > 0:\n",
                    "        sharpe = df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(252 * 78)\n",
                    "    else:\n",
                    "        sharpe = 0\n",
                    "    \n",
                    "    # Max drawdown\n",
                    "    cummax = df['cum_returns'].expanding().max()\n",
                    "    drawdown = (df['cum_returns'] / cummax - 1)\n",
                    "    max_dd = drawdown.min()\n",
                    "    \n",
                    "    # Win rate\n",
                    "    winning_trades = (df['strategy_returns'] > 0).sum()\n",
                    "    losing_trades = (df['strategy_returns'] < 0).sum()\n",
                    "    total_trades = winning_trades + losing_trades\n",
                    "    win_rate = winning_trades / total_trades if total_trades > 0 else 0\n",
                    "    \n",
                    "    return {\n",
                    "        'strategy_hash': strategy_hash,\n",
                    "        'total_return': total_return,\n",
                    "        'sharpe_ratio': sharpe,\n",
                    "        'max_drawdown': max_dd,\n",
                    "        'win_rate': win_rate,\n",
                    "        'total_trades': total_trades,\n",
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
                    "# Calculate performance for strategies\n",
                    "performance_results = []\n",
                    "\n",
                    "if market_data is not None and strategy_index is not None:\n",
                    "    # Limit to top N strategies to avoid long computation\n",
                    "    strategies_to_analyze = strategy_index.head(20)\n",
                    "    \n",
                    "    print(f'Calculating performance for {len(strategies_to_analyze)} strategies...')\n",
                    "    \n",
                    "    for idx, row in strategies_to_analyze.iterrows():\n",
                    "        # Load signals for this strategy\n",
                    "        trace_path = results_path / row['trace_path']\n",
                    "        \n",
                    "        if trace_path.exists():\n",
                    "            signals = pd.read_parquet(trace_path)\n",
                    "            signals['ts'] = pd.to_datetime(signals['ts'])\n",
                    "            \n",
                    "            perf = calculate_performance(row['strategy_hash'], signals, market_data)\n",
                    "            \n",
                    "            # Add strategy metadata\n",
                    "            perf['strategy_type'] = row.get('strategy_type', 'unknown')\n",
                    "            perf['period'] = row.get('param_period')\n",
                    "            perf['std_dev'] = row.get('param_std_dev')\n",
                    "            \n",
                    "            performance_results.append(perf)\n",
                    "    \n",
                    "    if performance_results:\n",
                    "        performance_df = pd.DataFrame(performance_results)\n",
                    "        print(f'\\nCalculated performance for {len(performance_df)} strategies')\n",
                    "        \n",
                    "        # Show top performers\n",
                    "        print('\\nTop 10 by Sharpe Ratio:')\n",
                    "        cols_to_show = ['strategy_type', 'period', 'std_dev', 'sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']\n",
                    "        cols_to_show = [col for col in cols_to_show if col in performance_df.columns]\n",
                    "        top_performers = performance_df.nlargest(10, 'sharpe_ratio')[cols_to_show].round(3)\n",
                    "        print(top_performers)\n",
                    "    else:\n",
                    "        print('No performance results calculated')\n",
                    "        performance_df = pd.DataFrame()\n",
                    "else:\n",
                    "    print('Skipping performance calculation - missing market data or strategy index')\n",
                    "    performance_df = pd.DataFrame()"
                ]
            }
        ]
    
    def _create_visualization_cells(self) -> List[Dict]:
        """Create visualization cells"""
        return [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Visualizations <a name='visualizations'></a>"]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Plot best strategy equity curve\n",
                    "if len(performance_df) > 0 and 'df' in performance_df.columns:\n",
                    "    best_idx = performance_df['sharpe_ratio'].idxmax()\n",
                    "    best_strategy = performance_df.loc[best_idx]\n",
                    "    \n",
                    "    plt.figure(figsize=(15, 8))\n",
                    "    \n",
                    "    # Equity curve\n",
                    "    plt.subplot(2, 1, 1)\n",
                    "    df = best_strategy['df']\n",
                    "    plt.plot(df.index, df['cum_returns'], label='Strategy', linewidth=2)\n",
                    "    \n",
                    "    # Add buy & hold for comparison\n",
                    "    df['bh_returns'] = (1 + df['returns']).cumprod()\n",
                    "    plt.plot(df.index, df['bh_returns'], label='Buy & Hold', alpha=0.7)\n",
                    "    \n",
                    "    plt.title(f'Best Strategy Performance (Sharpe: {best_strategy[\"sharpe_ratio\"]:.2f})')\n",
                    "    plt.ylabel('Cumulative Returns')\n",
                    "    plt.legend()\n",
                    "    plt.grid(True, alpha=0.3)\n",
                    "    \n",
                    "    # Drawdown\n",
                    "    plt.subplot(2, 1, 2)\n",
                    "    cummax = df['cum_returns'].expanding().max()\n",
                    "    drawdown = (df['cum_returns'] / cummax - 1)\n",
                    "    plt.fill_between(df.index, drawdown, 0, alpha=0.3, color='red')\n",
                    "    plt.ylabel('Drawdown')\n",
                    "    plt.xlabel('Time')\n",
                    "    plt.grid(True, alpha=0.3)\n",
                    "    \n",
                    "    plt.tight_layout()\n",
                    "    plt.show()\n",
                    "    \n",
                    "    # Print strategy details\n",
                    "    print(f'\\nBest Strategy Details:')\n",
                    "    print(f'Type: {best_strategy.get(\"strategy_type\", \"unknown\")}')\n",
                    "    if 'period' in best_strategy:\n",
                    "        print(f'Period: {best_strategy[\"period\"]}')\n",
                    "    if 'std_dev' in best_strategy:\n",
                    "        print(f'Std Dev: {best_strategy[\"std_dev\"]}')"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Parameter sensitivity analysis\n",
                    "if len(performance_df) > 0 and 'period' in performance_df.columns:\n",
                    "    fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n",
                    "    \n",
                    "    # Sharpe by period\n",
                    "    if 'period' in performance_df.columns:\n",
                    "        period_stats = performance_df.groupby('period')['sharpe_ratio'].agg(['mean', 'std'])\n",
                    "        period_stats['mean'].plot(ax=axes[0, 0], marker='o')\n",
                    "        axes[0, 0].fill_between(period_stats.index, \n",
                    "                               period_stats['mean'] - period_stats['std'],\n",
                    "                               period_stats['mean'] + period_stats['std'],\n",
                    "                               alpha=0.3)\n",
                    "        axes[0, 0].set_title('Sharpe Ratio by Period')\n",
                    "        axes[0, 0].set_xlabel('Period')\n",
                    "        axes[0, 0].grid(True)\n",
                    "    \n",
                    "    # Sharpe by std_dev\n",
                    "    if 'std_dev' in performance_df.columns:\n",
                    "        std_stats = performance_df.groupby('std_dev')['sharpe_ratio'].agg(['mean', 'std'])\n",
                    "        std_stats['mean'].plot(ax=axes[0, 1], marker='o')\n",
                    "        axes[0, 1].set_title('Sharpe Ratio by Std Dev')\n",
                    "        axes[0, 1].set_xlabel('Standard Deviation')\n",
                    "        axes[0, 1].grid(True)\n",
                    "    \n",
                    "    # Return vs Risk scatter\n",
                    "    axes[1, 0].scatter(performance_df['total_return'], performance_df['sharpe_ratio'], alpha=0.6)\n",
                    "    axes[1, 0].set_xlabel('Total Return')\n",
                    "    axes[1, 0].set_ylabel('Sharpe Ratio')\n",
                    "    axes[1, 0].set_title('Return vs Risk-Adjusted Return')\n",
                    "    axes[1, 0].grid(True)\n",
                    "    \n",
                    "    # Drawdown vs Sharpe\n",
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
        """Create pattern discovery cells"""
        return [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Pattern Discovery"]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Discover patterns in successful strategies\n",
                    "if len(performance_df) > 0:\n",
                    "    # Find common characteristics in top performers\n",
                    "    top_n = min(10, len(performance_df))\n",
                    "    top_performers = performance_df.nlargest(top_n, 'sharpe_ratio')\n",
                    "    \n",
                    "    print(f'Analyzing top {top_n} strategies...')\n",
                    "    \n",
                    "    # Parameter patterns\n",
                    "    if 'period' in top_performers.columns:\n",
                    "        print(f'\\nPeriod distribution in top performers:')\n",
                    "        print(top_performers['period'].value_counts().sort_index())\n",
                    "    \n",
                    "    if 'std_dev' in top_performers.columns:\n",
                    "        print(f'\\nStd Dev distribution in top performers:')\n",
                    "        print(top_performers['std_dev'].value_counts().sort_index())\n",
                    "    \n",
                    "    # Save pattern for future use\n",
                    "    if len(top_performers) > 0:\n",
                    "        best = top_performers.iloc[0]\n",
                    "        pattern = {\n",
                    "            'name': f'{best.get(\"strategy_type\", \"unknown\")}_high_sharpe',\n",
                    "            'discovered_at': datetime.now().isoformat(),\n",
                    "            'run_id': results_path.name,\n",
                    "            'performance': {\n",
                    "                'sharpe_ratio': float(best['sharpe_ratio']),\n",
                    "                'total_return': float(best['total_return']),\n",
                    "                'max_drawdown': float(best['max_drawdown'])\n",
                    "            },\n",
                    "            'parameters': {}\n",
                    "        }\n",
                    "        \n",
                    "        # Add available parameters\n",
                    "        for param in ['period', 'std_dev', 'fast_period', 'slow_period']:\n",
                    "            if param in best:\n",
                    "                pattern['parameters'][param] = best[param]\n",
                    "        \n",
                    "        print(f'\\nDiscovered pattern: {pattern[\"name\"]}')\n",
                    "        print(f'Parameters: {pattern[\"parameters\"]}')"
                ]
            }
        ]
    
    def _create_export_cell(self) -> Dict:
        """Create results export cell"""
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Export Results <a name='export-results'></a>"]
        }, {
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
                "            'strategy_hash': best.get('strategy_hash', ''),\n",
                "            'strategy_type': best.get('strategy_type', 'unknown'),\n",
                "            'sharpe_ratio': float(best['sharpe_ratio']),\n",
                "            'total_return': float(best['total_return']),\n",
                "            'max_drawdown': float(best['max_drawdown']),\n",
                "            'win_rate': float(best.get('win_rate', 0)),\n",
                "            'parameters': {}\n",
                "        },\n",
                "        'alternative_strategies': [],\n",
                "        'run_info': {\n",
                "            'run_id': results_path.name,\n",
                "            'generated_at': datetime.now().isoformat(),\n",
                "            'total_strategies': len(strategy_index) if strategy_index is not None else 0,\n",
                "            'analyzed': len(performance_df)\n",
                "        }\n",
                "    }\n",
                "    \n",
                "    # Add strategy-specific params\n",
                "    for col in performance_df.columns:\n",
                "        if col.startswith('param_') or col in ['period', 'std_dev', 'fast_period', 'slow_period']:\n",
                "            if col in best and pd.notna(best[col]):\n",
                "                param_name = col.replace('param_', '')\n",
                "                recommendations['best_overall']['parameters'][param_name] = best[col]\n",
                "    \n",
                "    # Add top 5 alternatives\n",
                "    if len(performance_df) > 5:\n",
                "        for idx, row in performance_df.nlargest(5, 'sharpe_ratio').iloc[1:].iterrows():\n",
                "            alt = {\n",
                "                'strategy_hash': row.get('strategy_hash', ''),\n",
                "                'sharpe_ratio': float(row['sharpe_ratio']),\n",
                "                'parameters': {}\n",
                "            }\n",
                "            \n",
                "            for col in ['period', 'std_dev', 'fast_period', 'slow_period']:\n",
                "                if col in row and pd.notna(row[col]):\n",
                "                    alt['parameters'][col] = row[col]\n",
                "            \n",
                "            recommendations['alternative_strategies'].append(alt)\n",
                "    \n",
                "    # Save recommendations\n",
                "    with open(results_path / 'recommendations.json', 'w') as f:\n",
                "        json.dump(recommendations, f, indent=2)\n",
                "        \n",
                "    print('✅ Recommendations saved to recommendations.json')\n",
                "    print(f'\\nBest strategy: {best.get(\"strategy_hash\", \"N/A\")}')\n",
                "    print(f'Strategy type: {best.get(\"strategy_type\", \"unknown\")}')\n",
                "    print(f'Sharpe Ratio: {best[\"sharpe_ratio\"]:.2f}')\n",
                "    print(f'Total Return: {best[\"total_return\"]:.1%}')\n",
                "    print(f'Max Drawdown: {best[\"max_drawdown\"]:.1%}')\n",
                "    print(f'Win Rate: {best.get(\"win_rate\", 0):.1%}')\n",
                "    \n",
                "    # Export performance DataFrame for further analysis\n",
                "    performance_df.to_csv(results_path / 'performance_analysis.csv', index=False)\n",
                "    print('\\n✅ Performance data exported to performance_analysis.csv')\n",
                "else:\n",
                "    print('No performance results to export')"
            ]
        }
    
    def _launch_notebook(self, notebook_path: Path):
        """Launch Jupyter with the generated notebook"""
        try:
            subprocess.run(["jupyter", "lab", str(notebook_path)], check=False)
        except FileNotFoundError:
            try:
                subprocess.run(["jupyter", "notebook", str(notebook_path)], check=False)
            except Exception as e:
                logger.error(f"Could not auto-launch Jupyter: {e}")
                logger.info(f"You can manually open: jupyter lab {notebook_path}")
        except Exception as e:
            logger.error(f"Could not auto-launch Jupyter: {e}")
            logger.info(f"You can manually open: jupyter lab {notebook_path}")