#!/usr/bin/env python3
"""
Comprehensive analysis of two-layer ensemble strategy performance.
Uses the correct column names from the parquet files.
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
            # Convert timestamp column
            self.regime_data['timestamp'] = pd.to_datetime(self.regime_data['ts'])
            self.regime_data['regime'] = self.regime_data['val']
            self.regime_data['price'] = self.regime_data['px']
            
            print(f"Loaded regime data: {len(self.regime_data):,} rows")
            print(f"Date range: {self.regime_data['timestamp'].min()} to {self.regime_data['timestamp'].max()}")
            
            # Show regime distribution
            regime_dist = self.regime_data['regime'].value_counts()
            print(f"Regime distribution:")
            for regime, count in regime_dist.items():
                print(f"  - {regime}: {count:,} ({count/len(self.regime_data)*100:.1f}%)")
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
                
                # Standardize column names
                data['timestamp'] = pd.to_datetime(data['ts'])
                data['signal'] = data['val']  # For strategies, val contains -1/0/1
                data['price'] = data['px']
                data['strategy_name'] = data['strat'].iloc[0] if len(data) > 0 else strategy_name
                
                self.signal_data[strategy_name] = data
                
                print(f"  - {strategy_name}: {len(data):,} rows")
                print(f"    Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
                
                # Basic signal statistics
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
            
        total_bars = len(self.regime_data)
        print(f"Total regime detection points: {total_bars:,}")
        
        # Regime distribution
        regime_dist = self.regime_data['regime'].value_counts()
        print(f"\nRegime Distribution:")
        for regime, count in regime_dist.items():
            print(f"  - {regime}: {count:,} bars ({count/total_bars*100:.2f}%)")
            
        # Regime stability analysis
        regime_changes = (self.regime_data['regime'] != self.regime_data['regime'].shift(1)).sum()
        print(f"\nRegime Changes: {regime_changes:,}")
        print(f"Average regime duration: {total_bars/max(regime_changes, 1):.1f} detection points")
        
        # Regime transitions
        print(f"\nRegime Transitions:")
        self.regime_data['prev_regime'] = self.regime_data['regime'].shift(1)
        transitions = self.regime_data[
            (self.regime_data['regime'] != self.regime_data['prev_regime']) & 
            (self.regime_data['prev_regime'].notna())
        ][['prev_regime', 'regime']].value_counts()
        
        for (prev_regime, current_regime), count in transitions.head(10).items():
            print(f"  - {prev_regime} → {current_regime}: {count} times")
            
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
                    
            # Basic signal statistics
            total_signals = len(data)
            signal_counts = data['signal'].value_counts().sort_index()
            
            long_signals = signal_counts.get(1, 0)
            short_signals = signal_counts.get(-1, 0)
            flat_signals = signal_counts.get(0, 0)
            
            # Signal changes (only count changes, not every signal)
            signal_changes = (data['signal'] != data['signal'].shift(1)).sum()
            
            print(f"  - Total signal points: {total_signals:,}")
            print(f"  - Long signals: {long_signals:,} ({long_signals/total_signals*100:.1f}%)")
            print(f"  - Short signals: {short_signals:,} ({short_signals/total_signals*100:.1f}%)")
            print(f"  - Flat signals: {flat_signals:,} ({flat_signals/total_signals*100:.1f}%)")
            print(f"  - Signal changes: {signal_changes:,} ({signal_changes/total_signals*100:.2f}%)")
            
            # Average holding period (in signal points, not necessarily time)
            avg_holding = total_signals / max(signal_changes, 1)
            print(f"  - Avg holding period: {avg_holding:.1f} signal points")
            
            # Time analysis
            time_span = data['timestamp'].max() - data['timestamp'].min()
            print(f"  - Time span: {time_span}")
            
            # Price analysis
            price_range = data['price'].max() - data['price'].min()
            price_start = data['price'].iloc[0]
            price_end = data['price'].iloc[-1]
            price_return = (price_end - price_start) / price_start
            
            print(f"  - Price range: ${data['price'].min():.2f} - ${data['price'].max():.2f}")
            print(f"  - Buy & hold return: {price_return:.2%}")
            
            # Calculate strategy returns
            returns_analysis = self.calculate_strategy_returns(data)
            if returns_analysis:
                print(f"  - Strategy return: {returns_analysis['total_return']:.2%}")
                print(f"  - Win rate: {returns_analysis['win_rate']:.2%}")
                print(f"  - Avg win: {returns_analysis['avg_win']:.2%}")
                print(f"  - Avg loss: {returns_analysis['avg_loss']:.2%}")
                print(f"  - Profit factor: {returns_analysis['profit_factor']:.2f}")
                
            performance_summary.append({
                'strategy': strategy_name,
                'baseline': baseline,
                'total_signals': total_signals,
                'long_pct': long_signals/total_signals*100,
                'short_pct': short_signals/total_signals*100,
                'flat_pct': flat_signals/total_signals*100,
                'signal_changes': signal_changes,
                'change_rate': signal_changes/total_signals*100,
                'avg_holding': avg_holding,
                'price_return': price_return * 100,
                **(returns_analysis if returns_analysis else {})
            })
            
            print()
            
        # Summary comparison
        if performance_summary:
            df = pd.DataFrame(performance_summary)
            print("\n=== STRATEGY COMPARISON SUMMARY ===")
            
            if 'baseline' in df.columns and df['baseline'].notna().any():
                baseline_summary = df.groupby('baseline').agg({
                    'long_pct': 'mean',
                    'short_pct': 'mean', 
                    'flat_pct': 'mean',
                    'change_rate': 'mean',
                    'avg_holding': 'mean',
                    'price_return': 'first'  # Should be same for all
                }).round(2)
                
                print("\nBy Baseline Strategy:")
                print(baseline_summary)
                
            # Performance metrics if available
            perf_cols = [col for col in ['total_return', 'win_rate', 'profit_factor'] if col in df.columns]
            if perf_cols:
                print(f"\nPerformance Metrics:")
                display_cols = ['strategy', 'baseline'] + perf_cols
                available_cols = [col for col in display_cols if col in df.columns]
                print(df[available_cols].round(3))
        
        return performance_summary
        
    def calculate_strategy_returns(self, data):
        """Calculate strategy returns and performance metrics."""
        try:
            # Sort by timestamp to ensure correct order
            data_sorted = data.sort_values('timestamp').copy()
            
            # Calculate price returns
            data_sorted['price_return'] = data_sorted['price'].pct_change()
            
            # Calculate strategy returns (signal * next period return)
            # Shift signal to avoid look-ahead bias
            data_sorted['strategy_return'] = data_sorted['signal'].shift(1) * data_sorted['price_return']
            
            # Remove NaN values
            strategy_returns = data_sorted['strategy_return'].dropna()
            
            if len(strategy_returns) == 0:
                return None
                
            # Calculate metrics
            total_return = (1 + strategy_returns).prod() - 1
            
            # Trade-based metrics
            trades = strategy_returns[strategy_returns != 0]  # Only non-zero returns are trades
            if len(trades) == 0:
                return {'total_return': total_return, 'win_rate': 0, 'avg_win': 0, 'avg_loss': 0, 'profit_factor': 0}
                
            wins = trades[trades > 0]
            losses = trades[trades < 0]
            
            win_rate = len(wins) / len(trades) if len(trades) > 0 else 0
            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = losses.mean() if len(losses) > 0 else 0
            
            profit_factor = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float('inf')
            
            return {
                'total_return': total_return * 100,  # Convert to percentage
                'win_rate': win_rate * 100,
                'avg_win': avg_win * 100,
                'avg_loss': avg_loss * 100,
                'profit_factor': profit_factor
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
            
        # Create a regime lookup by timestamp
        regime_lookup = self.regime_data.set_index('timestamp')['regime']
        
        for strategy_name, strategy_data in self.signal_data.items():
            print(f"\nStrategy: {strategy_name}")
            
            # Add regime information to strategy data
            strategy_with_regime = strategy_data.copy()
            strategy_with_regime['regime'] = strategy_with_regime['timestamp'].map(regime_lookup)
            
            # Remove rows where regime is not available
            strategy_with_regime = strategy_with_regime.dropna(subset=['regime'])
            
            if len(strategy_with_regime) == 0:
                print("  - No matching timestamps with regime data")
                continue
                
            print(f"  - Matched {len(strategy_with_regime):,} signals with regime data")
            
            # Signal distribution by regime
            regime_signal_dist = pd.crosstab(
                strategy_with_regime['regime'], 
                strategy_with_regime['signal'], 
                normalize='index'
            ) * 100
            
            print("  Signal distribution by regime (%):")
            for regime in regime_signal_dist.index:
                signal_dist = regime_signal_dist.loc[regime]
                print(f"    {regime}:")
                for signal, pct in signal_dist.items():
                    if pct > 0:
                        print(f"      Signal {signal}: {pct:.1f}%")
            
            # Signal change rates by regime
            strategy_with_regime['signal_change'] = (
                strategy_with_regime['signal'] != strategy_with_regime['signal'].shift(1)
            ).astype(int)
            
            regime_change_rates = strategy_with_regime.groupby('regime')['signal_change'].agg(['sum', 'count', 'mean'])
            regime_change_rates['change_rate'] = regime_change_rates['mean'] * 100
            
            print("\n  Signal change rates by regime:")
            for regime in regime_change_rates.index:
                rate = regime_change_rates.loc[regime, 'change_rate']
                total = regime_change_rates.loc[regime, 'count']
                changes = regime_change_rates.loc[regime, 'sum']
                print(f"    {regime}: {changes} changes in {total} points ({rate:.2f}%)")
                
    def analyze_ensemble_effectiveness(self):
        """Analyze the effectiveness of the two-layer ensemble approach."""
        print("=== ENSEMBLE EFFECTIVENESS ===\n")
        
        # Configuration analysis
        print("Configuration Details:")
        print(f"  - Total strategies analyzed: {len(self.signal_data)}")
        print(f"  - Regime classifier: {'Available' if self.regime_data is not None else 'Not available'}")
        
        if self.regime_data is not None:
            unique_regimes = self.regime_data['regime'].nunique()
            print(f"  - Regimes detected: {unique_regimes}")
            
        # Signal diversity analysis
        print(f"\nSignal Diversity Analysis:")
        if len(self.signal_data) > 1:
            # Find common timestamps across all strategies
            common_timestamps = None
            for strategy_name, data in self.signal_data.items():
                timestamps = set(data['timestamp'])
                if common_timestamps is None:
                    common_timestamps = timestamps
                else:
                    common_timestamps = common_timestamps.intersection(timestamps)
                    
            print(f"  - Common timestamps across strategies: {len(common_timestamps):,}")
            
            if len(common_timestamps) > 100:  # Only analyze if we have sufficient overlap
                # Create signal matrix for correlation analysis
                signal_matrix = {}
                
                for strategy_name, data in self.signal_data.items():
                    # Filter to common timestamps and create series
                    common_data = data[data['timestamp'].isin(common_timestamps)].copy()
                    common_data = common_data.sort_values('timestamp')
                    signal_series = common_data.set_index('timestamp')['signal']
                    signal_matrix[strategy_name] = signal_series
                    
                # Calculate correlation matrix
                signal_df = pd.DataFrame(signal_matrix)
                correlation_matrix = signal_df.corr()
                
                print(f"\nSignal Correlation Matrix:")
                print(correlation_matrix.round(3))
                
                # Average correlation (excluding diagonal)
                mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
                avg_correlation = correlation_matrix.values[mask].mean()
                print(f"\nAverage inter-strategy correlation: {avg_correlation:.3f}")
                
                # Signal agreement analysis
                print(f"\nSignal Agreement Analysis:")
                strategies = list(signal_df.columns)
                for i in range(len(strategies)):
                    for j in range(i+1, len(strategies)):
                        strategy1, strategy2 = strategies[i], strategies[j]
                        agreement = (signal_df[strategy1] == signal_df[strategy2]).mean()
                        print(f"  {strategy1.split('_')[-1] if '_' in strategy1 else strategy1} vs {strategy2.split('_')[-1] if '_' in strategy2 else strategy2}: {agreement:.2%} agreement")
        
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print("\n" + "="*80)
        print("TWO-LAYER ENSEMBLE STRATEGY ANALYSIS SUMMARY")  
        print("="*80)
        
        print(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Workspace: {self.workspace_path.name}")
        
        # Data summary
        print(f"\nData Summary:")
        print(f"  - Total ensemble strategies: {len(self.signal_data)}")
        print(f"  - Regime classifier: {'Available' if self.regime_data is not None else 'Not available'}")
        
        if self.regime_data is not None:
            print(f"  - Regime detection points: {len(self.regime_data):,}")
            regime_changes = (self.regime_data['regime'] != self.regime_data['regime'].shift(1)).sum()
            avg_regime_duration = len(self.regime_data) / max(regime_changes, 1)
            print(f"  - Average regime duration: {avg_regime_duration:.1f} detection points")
            unique_regimes = self.regime_data['regime'].nunique()
            print(f"  - Unique regimes: {unique_regimes}")
            
        # Strategy summary
        if self.signal_data:
            total_signals = sum(len(data) for data in self.signal_data.values())
            avg_signals_per_strategy = total_signals / len(self.signal_data)
            print(f"  - Total signal points across all strategies: {total_signals:,}")
            print(f"  - Average signals per strategy: {avg_signals_per_strategy:.0f}")
            
            # Average signal change rate
            change_rates = []
            for data in self.signal_data.values():
                changes = (data['signal'] != data['signal'].shift(1)).sum()
                rate = changes / len(data) * 100
                change_rates.append(rate)
            avg_change_rate = np.mean(change_rates)
            print(f"  - Average signal change rate: {avg_change_rate:.2f}% of signal points")
        
        # Key insights
        print(f"\nKey Insights:")
        
        if self.regime_data is not None:
            # Most common regime
            most_common_regime = self.regime_data['regime'].value_counts().index[0]
            regime_share = self.regime_data['regime'].value_counts().iloc[0] / len(self.regime_data) * 100
            print(f"  - Most common regime: {most_common_regime} ({regime_share:.1f}% of time)")
            
        if len(self.signal_data) > 1:
            print(f"  - Ensemble provides diversification across {len(self.signal_data)} different baseline strategies")
            
        # Configuration insights
        baseline_strategies = set()
        for strategy_name in self.signal_data.keys():
            for base in ['dema_crossover', 'elder_ray', 'sma_crossover', 'stochastic_crossover', 'pivot_channel_bounces']:
                if base in strategy_name:
                    baseline_strategies.add(base)
                    break
        
        if baseline_strategies:
            print(f"  - Baseline strategies used: {', '.join(sorted(baseline_strategies))}")
            
        print(f"\nAnalysis Complete!")
        print("="*80)
        
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