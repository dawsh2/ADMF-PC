#!/usr/bin/env python3
"""
Comprehensive analysis of two-layer ensemble strategy performance.
Reads actual signal and regime data from Parquet files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

class TwoLayerEnsembleAnalyzer:
    def __init__(self, workspace_path):
        self.workspace_path = Path(workspace_path)
        self.traces_path = self.workspace_path / "traces" / "SPY_1m"
        
        # Load metadata
        with open(self.workspace_path / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
            
        self.signal_data = {}
        self.regime_data = None
        
    def load_data(self):
        """Load all signal and regime data."""
        print("=== LOADING DATA ===\n")
        
        # Load regime data
        regime_file = self.traces_path / "classifiers" / "regime" / "SPY_market_regime_detector.parquet"
        if regime_file.exists():
            self.regime_data = pd.read_parquet(regime_file)
            print(f"Loaded regime data: {len(self.regime_data):,} rows")
            print(f"Regime columns: {list(self.regime_data.columns)}")
            print(f"Date range: {self.regime_data['timestamp'].min()} to {self.regime_data['timestamp'].max()}")
        else:
            print("No regime data found")
            
        # Load signal data for each strategy
        signal_dirs = [
            self.traces_path / "signals" / "ma_crossover",
            self.traces_path / "signals" / "regime"
        ]
        
        strategy_files = []
        for signal_dir in signal_dirs:
            if signal_dir.exists():
                strategy_files.extend(list(signal_dir.glob("*.parquet")))
                
        print(f"\nFound {len(strategy_files)} strategy files:")
        
        for file_path in strategy_files:
            try:
                strategy_name = file_path.stem.replace("SPY_", "")
                data = pd.read_parquet(file_path)
                self.signal_data[strategy_name] = data
                
                print(f"  - {strategy_name}: {len(data):,} rows")
                print(f"    Columns: {list(data.columns)}")
                print(f"    Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
                
                # Basic signal statistics
                if 'signal' in data.columns:
                    signal_counts = data['signal'].value_counts().sort_index()
                    print(f"    Signal distribution: {dict(signal_counts)}")
                
            except Exception as e:
                print(f"    Error loading {file_path}: {e}")
                
        print()
        
    def analyze_regime_performance(self):
        """Analyze regime detection performance."""
        print("=== REGIME ANALYSIS ===\n")
        
        if self.regime_data is None:
            print("No regime data available")
            return
            
        # Regime distribution
        if 'regime' in self.regime_data.columns:
            regime_dist = self.regime_data['regime'].value_counts()
            total_bars = len(self.regime_data)
            
            print(f"Total bars analyzed: {total_bars:,}")
            print("\nRegime Distribution:")
            for regime, count in regime_dist.items():
                print(f"  - {regime}: {count:,} bars ({count/total_bars*100:.2f}%)")
                
            # Regime transitions
            regime_changes = (self.regime_data['regime'] != self.regime_data['regime'].shift(1)).sum()
            print(f"\nRegime Changes: {regime_changes:,}")
            print(f"Average regime duration: {total_bars/regime_changes:.1f} bars")
            
            # Transition matrix
            print("\nRegime Transitions:")
            transitions = pd.crosstab(
                self.regime_data['regime'].shift(1), 
                self.regime_data['regime'], 
                dropna=False
            )
            print(transitions)
            
        # Additional regime statistics
        if 'confidence' in self.regime_data.columns:
            print(f"\nRegime Confidence Statistics:")
            print(f"  - Mean: {self.regime_data['confidence'].mean():.3f}")
            print(f"  - Std: {self.regime_data['confidence'].std():.3f}")
            print(f"  - Min: {self.regime_data['confidence'].min():.3f}")
            print(f"  - Max: {self.regime_data['confidence'].max():.3f}")
            
        print()
        
    def analyze_strategy_performance(self):
        """Analyze individual strategy performance."""
        print("=== STRATEGY PERFORMANCE ANALYSIS ===\n")
        
        baseline_strategies = [
            'dema_crossover',
            'elder_ray',
            'sma_crossover', 
            'stochastic_crossover',
            'pivot_channel_bounces'
        ]
        
        performance_summary = []
        
        for strategy_name, data in self.signal_data.items():
            print(f"Strategy: {strategy_name}")
            
            # Extract baseline strategy name
            baseline = None
            for base in baseline_strategies:
                if base in strategy_name:
                    baseline = base
                    break
                    
            if 'signal' not in data.columns:
                print("  - No signal column found")
                continue
                
            # Basic signal statistics
            total_bars = len(data)
            signal_counts = data['signal'].value_counts().sort_index()
            
            long_bars = signal_counts.get(1, 0)
            short_bars = signal_counts.get(-1, 0)
            flat_bars = signal_counts.get(0, 0)
            
            # Signal changes
            signal_changes = (data['signal'] != data['signal'].shift(1)).sum()
            
            print(f"  - Total bars: {total_bars:,}")
            print(f"  - Long signals: {long_bars:,} ({long_bars/total_bars*100:.1f}%)")
            print(f"  - Short signals: {short_bars:,} ({short_bars/total_bars*100:.1f}%)")
            print(f"  - Flat signals: {flat_bars:,} ({flat_bars/total_bars*100:.1f}%)")
            print(f"  - Signal changes: {signal_changes:,} ({signal_changes/total_bars*100:.2f}%)")
            
            # Average holding period
            avg_holding = total_bars / max(signal_changes, 1)
            print(f"  - Avg holding period: {avg_holding:.1f} bars")
            
            # Calculate returns if possible
            returns_analysis = self.calculate_returns(data)
            if returns_analysis:
                print(f"  - Total return: {returns_analysis['total_return']:.2%}")
                print(f"  - Annualized return: {returns_analysis['annualized_return']:.2%}")
                print(f"  - Win rate: {returns_analysis['win_rate']:.2%}")
                print(f"  - Sharpe ratio: {returns_analysis['sharpe_ratio']:.3f}")
                print(f"  - Max drawdown: {returns_analysis['max_drawdown']:.2%}")
                
            summary_entry = {
                'strategy': strategy_name,
                'baseline': baseline,
                'total_bars': total_bars,
                'long_pct': long_bars/total_bars*100,
                'short_pct': short_bars/total_bars*100,
                'flat_pct': flat_bars/total_bars*100,
                'signal_changes': signal_changes,
                'change_rate': signal_changes/total_bars*100,
                'avg_holding': avg_holding
            }
            
            if returns_analysis:
                summary_entry.update(returns_analysis)
                
            performance_summary.append(summary_entry)
            
            print()
            
        # Summary comparison
        if performance_summary:
            df = pd.DataFrame(performance_summary)
            print("\n=== STRATEGY COMPARISON SUMMARY ===")
            
            if 'baseline' in df.columns:
                baseline_summary = df.groupby('baseline').agg({
                    'long_pct': 'mean',
                    'short_pct': 'mean', 
                    'flat_pct': 'mean',
                    'change_rate': 'mean',
                    'avg_holding': 'mean'
                }).round(2)
                
                print("\nBy Baseline Strategy:")
                print(baseline_summary)
                
            if any(col in df.columns for col in ['total_return', 'win_rate', 'sharpe_ratio']):
                print("\nPerformance Metrics:")
                perf_cols = [col for col in ['total_return', 'annualized_return', 'win_rate', 'sharpe_ratio', 'max_drawdown'] if col in df.columns]
                print(df[['strategy'] + perf_cols].round(4))
        
        return performance_summary
        
    def calculate_returns(self, data):
        """Calculate returns and performance metrics."""
        if 'close' not in data.columns or 'signal' not in data.columns:
            return None
            
        try:
            # Calculate price returns
            data = data.copy()
            data['price_return'] = data['close'].pct_change()
            
            # Calculate strategy returns (signal * next period return)
            data['strategy_return'] = data['signal'].shift(1) * data['price_return']
            
            # Remove NaN values
            strategy_returns = data['strategy_return'].dropna()
            
            if len(strategy_returns) == 0:
                return None
                
            # Calculate metrics
            total_return = (1 + strategy_returns).prod() - 1
            
            # Annualized return (assuming 1-minute data, ~252*390 trading minutes per year)
            trading_minutes_per_year = 252 * 390
            periods_per_year = trading_minutes_per_year / len(strategy_returns)
            annualized_return = (1 + total_return) ** periods_per_year - 1
            
            # Win rate
            winning_trades = (strategy_returns > 0).sum()
            total_trades = len(strategy_returns)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Sharpe ratio
            sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(periods_per_year) if strategy_returns.std() > 0 else 0
            
            # Max drawdown
            cumulative_returns = (1 + strategy_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = cumulative_returns / rolling_max - 1
            max_drawdown = drawdown.min()
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }
            
        except Exception as e:
            print(f"    Error calculating returns: {e}")
            return None
            
    def analyze_regime_strategy_interaction(self):
        """Analyze how strategies perform in different regimes."""
        print("=== REGIME-STRATEGY INTERACTION ===\n")
        
        if self.regime_data is None:
            print("No regime data available for interaction analysis")
            return
            
        # Merge regime data with strategy signals
        for strategy_name, strategy_data in self.signal_data.items():
            print(f"\nStrategy: {strategy_name}")
            
            # Merge on timestamp
            merged = pd.merge(
                strategy_data[['timestamp', 'signal']],
                self.regime_data[['timestamp', 'regime']], 
                on='timestamp',
                how='inner'
            )
            
            if len(merged) == 0:
                print("  - No matching timestamps with regime data")
                continue
                
            # Analyze signal distribution by regime
            regime_signal_dist = pd.crosstab(merged['regime'], merged['signal'], normalize='index') * 100
            
            print("  Signal distribution by regime (%):")
            print(regime_signal_dist.round(1))
            
            # Signal changes by regime
            merged['signal_change'] = (merged['signal'] != merged['signal'].shift(1)).astype(int)
            regime_changes = merged.groupby('regime')['signal_change'].agg(['sum', 'count', 'mean'])
            regime_changes['change_rate'] = regime_changes['mean'] * 100
            
            print("\n  Signal change rates by regime:")
            for regime in regime_changes.index:
                rate = regime_changes.loc[regime, 'change_rate']
                total = regime_changes.loc[regime, 'count']
                changes = regime_changes.loc[regime, 'sum']
                print(f"    {regime}: {changes} changes in {total} bars ({rate:.2f}%)")
                
    def analyze_ensemble_effectiveness(self):
        """Analyze the effectiveness of the two-layer ensemble approach."""
        print("=== ENSEMBLE EFFECTIVENESS ===\n")
        
        # Extract configuration details from metadata
        print("Configuration Analysis:")
        if 'config' in self.metadata:
            config = self.metadata['config']
            
            print(f"  - Baseline allocation: {config.get('baseline_allocation', 'N/A')}")
            print(f"  - Baseline aggregation: {config.get('baseline_aggregation', 'N/A')}")
            print(f"  - Booster aggregation: {config.get('booster_aggregation', 'N/A')}")
            print(f"  - Min baseline agreement: {config.get('min_baseline_agreement', 'N/A')}")
            print(f"  - Min booster agreement: {config.get('min_booster_agreement', 'N/A')}")
            
            if 'regime_boosters' in config:
                boosters = config['regime_boosters']
                print(f"\n  Regime Boosters:")
                for regime, strategies in boosters.items():
                    print(f"    {regime}: {len(strategies)} strategies")
                    for strategy in strategies:
                        print(f"      - {strategy['name']}")
                        
        # Analyze signal diversity
        print(f"\nSignal Diversity Analysis:")
        if len(self.signal_data) > 1:
            # Calculate correlation matrix of signals
            signal_matrix = {}
            common_timestamps = None
            
            for strategy_name, data in self.signal_data.items():
                if 'signal' in data.columns:
                    signal_series = data.set_index('timestamp')['signal']
                    signal_matrix[strategy_name] = signal_series
                    
                    if common_timestamps is None:
                        common_timestamps = signal_series.index
                    else:
                        common_timestamps = common_timestamps.intersection(signal_series.index)
                        
            if len(signal_matrix) > 1 and len(common_timestamps) > 0:
                # Align all signals to common timestamps
                aligned_signals = pd.DataFrame({
                    name: series.reindex(common_timestamps) 
                    for name, series in signal_matrix.items()
                })
                
                # Calculate correlation matrix
                correlation_matrix = aligned_signals.corr()
                
                print(f"Signal Correlation Matrix ({len(common_timestamps):,} bars):")
                print(correlation_matrix.round(3))
                
                # Average correlation (excluding diagonal)
                mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
                avg_correlation = correlation_matrix.values[mask].mean()
                print(f"\nAverage inter-strategy correlation: {avg_correlation:.3f}")
                
                # Signal agreement analysis
                print(f"\nSignal Agreement Analysis:")
                for i, strategy1 in enumerate(aligned_signals.columns):
                    for j, strategy2 in enumerate(aligned_signals.columns):
                        if i < j:
                            agreement = (aligned_signals[strategy1] == aligned_signals[strategy2]).mean()
                            print(f"  {strategy1} vs {strategy2}: {agreement:.2%} agreement")
        
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print("\n" + "="*80)
        print("TWO-LAYER ENSEMBLE STRATEGY ANALYSIS SUMMARY")  
        print("="*80)
        
        print(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Workspace: {self.workspace_path.name}")
        
        # Run metadata
        print(f"\nRun Configuration:")
        if 'run_info' in self.metadata:
            run_info = self.metadata['run_info']
            print(f"  - Symbol: {run_info.get('symbol', 'N/A')}")
            print(f"  - Timeframe: {run_info.get('timeframe', 'N/A')}")
            print(f"  - Start Date: {run_info.get('start_date', 'N/A')}")
            print(f"  - End Date: {run_info.get('end_date', 'N/A')}")
            print(f"  - Duration: {run_info.get('duration_seconds', 'N/A')} seconds")
            
        # Data summary
        print(f"\nData Summary:")
        print(f"  - Total strategies: {len(self.signal_data)}")
        print(f"  - Regime classifier: {'Available' if self.regime_data is not None else 'Not found'}")
        
        if self.regime_data is not None:
            print(f"  - Total bars: {len(self.regime_data):,}")
            if 'regime' in self.regime_data.columns:
                unique_regimes = self.regime_data['regime'].nunique()
                print(f"  - Unique regimes detected: {unique_regimes}")
                
        # Key findings
        print(f"\nKey Findings:")
        
        if self.signal_data:
            avg_signal_changes = np.mean([
                (data['signal'] != data['signal'].shift(1)).sum() / len(data) * 100
                for data in self.signal_data.values() 
                if 'signal' in data.columns
            ])
            print(f"  - Average signal change rate: {avg_signal_changes:.2f}% of bars")
            
        if len(self.signal_data) > 1:
            print(f"  - Strategy diversity: {len(self.signal_data)} different ensemble instances")
            
        # Recommendations
        print(f"\nRecommendations:")
        if self.regime_data is not None and 'regime' in self.regime_data.columns:
            regime_changes = (self.regime_data['regime'] != self.regime_data['regime'].shift(1)).sum()
            avg_regime_duration = len(self.regime_data) / regime_changes
            if avg_regime_duration < 100:
                print(f"  - Consider smoothing regime detection (avg duration: {avg_regime_duration:.1f} bars)")
            else:
                print(f"  - Regime detection appears well-calibrated (avg duration: {avg_regime_duration:.1f} bars)")
        
        print(f"  - Monitor ensemble diversity to avoid over-correlation")
        print(f"  - Consider regime-specific performance evaluation")
        
        print("\n" + "="*80)
        
    def run_full_analysis(self):
        """Run the complete analysis."""
        print("Two-Layer Ensemble Strategy Analysis")
        print("=" * 50)
        
        self.load_data()
        self.analyze_regime_performance()
        self.analyze_strategy_performance()
        self.analyze_regime_strategy_interaction()
        self.analyze_ensemble_effectiveness()
        self.generate_summary_report()


def main():
    workspace_path = Path("workspaces/two_layer_regime_ensemble_v1_4f71d9e1")
    
    if not workspace_path.exists():
        print(f"Error: Workspace not found at {workspace_path}")
        return
        
    analyzer = TwoLayerEnsembleAnalyzer(workspace_path)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()