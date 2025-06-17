#!/usr/bin/env python3
"""
Ensemble Strategy Performance Analysis - Last 12k Bars
Using signal-storage-replay methodology for efficient analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Constants
WORKSPACE_PATH = "/Users/daws/ADMF-PC/workspaces/duckdb_ensemble_cost_optimized_v1_014a539f"
DATA_PATH = "/Users/daws/ADMF-PC/data/SPY_1m.parquet"
ANALYSIS_BARS = 22000  # Out-of-sample period only
TRANSACTION_COST = 0.0001  # 0.01% per trade

class EnsembleAnalyzer:
    def __init__(self):
        self.data = None
        self.signals_default = None
        self.signals_custom = None
        self.classifier_signals = None
        self.analysis_start_idx = None
        
    def load_source_data(self):
        """Load SPY data and identify last 12k bars"""
        print("📊 Loading SPY_1m.parquet...")
        self.data = pd.read_parquet(DATA_PATH)
        
        # Ensure datetime index
        if 'timestamp' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            self.data = self.data.set_index('timestamp')
        elif 'datetime' in self.data.columns:
            self.data['datetime'] = pd.to_datetime(self.data['datetime'])
            self.data = self.data.set_index('datetime')
        elif not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)
            
        # Sort by datetime to ensure proper order
        self.data = self.data.sort_index()
        
        # Calculate analysis window
        total_bars = len(self.data)
        self.analysis_start_idx = max(0, total_bars - ANALYSIS_BARS)
        
        # Get analysis window
        self.analysis_data = self.data.iloc[self.analysis_start_idx:].copy()
        
        print(f"✅ Loaded {total_bars:,} total bars")
        print(f"📈 Analysis window: {len(self.analysis_data):,} bars")
        print(f"🗓️  Period: {self.analysis_data.index[0]} to {self.analysis_data.index[-1]}")
        
        # Calculate trading days
        trading_days = len(self.analysis_data.index.normalize().unique())
        print(f"📅 Trading days: {trading_days}")
        
        return self.analysis_data
    
    def load_ensemble_signals(self):
        """Load and filter ensemble signals using sparse storage"""
        print("\n🔄 Loading ensemble signals...")
        
        # Load signal files
        default_path = f"{WORKSPACE_PATH}/traces/SPY_1m/signals/unknown/SPY_adaptive_ensemble_default.parquet"
        custom_path = f"{WORKSPACE_PATH}/traces/SPY_1m/signals/unknown/SPY_adaptive_ensemble_custom.parquet"
        classifier_path = f"{WORKSPACE_PATH}/traces/SPY_1m/classifiers/unknown/SPY_vol_mom_classifier.parquet"
        
        self.signals_default = pd.read_parquet(default_path)
        self.signals_custom = pd.read_parquet(custom_path)
        self.classifier_signals = pd.read_parquet(classifier_path)
        
        print(f"✅ Default ensemble: {len(self.signals_default):,} signal changes")
        print(f"✅ Custom ensemble: {len(self.signals_custom):,} signal changes")
        print(f"✅ Classifier: {len(self.classifier_signals):,} signal changes")
        
        # Filter signals to analysis window using indexed approach
        self.filter_signals_to_window()
        
    def filter_signals_to_window(self):
        """Filter signals to last 12k bars using efficient indexing"""
        print("\n🔍 Filtering signals to analysis window...")
        
        # Filter each signal dataset to the analysis window
        # Note: The signal files use 'idx' column for bar index
        analysis_mask_default = self.signals_default['idx'] >= self.analysis_start_idx
        analysis_mask_custom = self.signals_custom['idx'] >= self.analysis_start_idx
        analysis_mask_classifier = self.classifier_signals['idx'] >= self.analysis_start_idx
        
        self.signals_default_filtered = self.signals_default[analysis_mask_default].copy()
        self.signals_custom_filtered = self.signals_custom[analysis_mask_custom].copy()
        self.classifier_filtered = self.classifier_signals[analysis_mask_classifier].copy()
        
        print(f"📊 Filtered default signals: {len(self.signals_default_filtered):,}")
        print(f"📊 Filtered custom signals: {len(self.signals_custom_filtered):,}")
        print(f"📊 Filtered classifier signals: {len(self.classifier_filtered):,}")
        
    def replay_signals_to_timeline(self, signals, signal_name):
        """Reconstruct full signal timeline from sparse storage"""
        print(f"\n🎬 Replaying {signal_name} signals...")
        
        # Create timeline for analysis period
        timeline = pd.Series(index=self.analysis_data.index, dtype=float, name=signal_name)
        timeline.fillna(0.0, inplace=True)
        
        # Replay sparse signals
        for _, row in signals.iterrows():
            bar_idx = row['idx']  # Use 'idx' column from signal files
            signal_value = row['val']  # Use 'val' column for signal value
            
            # Convert global bar index to analysis window index
            analysis_bar_idx = bar_idx - self.analysis_start_idx
            
            if 0 <= analysis_bar_idx < len(timeline):
                # Set signal and forward fill until next change
                timeline.iloc[analysis_bar_idx:] = signal_value
                
        return timeline
    
    def calculate_trade_returns(self, signals, prices, signal_name):
        """Calculate returns from signal timeline"""
        print(f"\n💰 Calculating {signal_name} trade returns...")
        
        # Align signals and prices
        aligned_signals = signals.reindex(prices.index, method='ffill')
        aligned_signals.fillna(0.0, inplace=True)
        
        # Calculate position changes (trades)
        position_changes = aligned_signals.diff()
        trades = position_changes[position_changes != 0].dropna()
        
        print(f"📈 Total trades: {len(trades)}")
        
        # Calculate returns
        price_changes = prices.pct_change()
        strategy_returns = aligned_signals.shift(1) * price_changes
        
        # Remove NaN values
        strategy_returns = strategy_returns.dropna()
        
        # Calculate trade-level returns
        trade_returns = []
        cumulative_return = 0.0
        current_position = 0.0
        
        for timestamp, position in aligned_signals.items():
            if timestamp in prices.index:
                if current_position != position:  # Position change = trade
                    # Apply transaction cost
                    cost = abs(position - current_position) * TRANSACTION_COST
                    current_position = position
                    
                if current_position != 0:
                    # Calculate return for this bar
                    if timestamp in price_changes.index and not pd.isna(price_changes[timestamp]):
                        bar_return = current_position * price_changes[timestamp] - (cost if position != current_position else 0)
                        trade_returns.append(bar_return)
                        cumulative_return += bar_return
        
        return strategy_returns, trades, trade_returns
    
    def calculate_performance_metrics(self, returns, trades, signal_name):
        """Calculate comprehensive performance metrics"""
        print(f"\n📊 Calculating {signal_name} performance metrics...")
        
        metrics = {}
        
        # Basic returns
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns.mean()) ** (252 * 24 * 60) - 1  # 1-minute data
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252 * 24 * 60)  # Annualized
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
        profit_factor = abs(positive_returns.sum() / negative_returns.sum()) if negative_returns.sum() != 0 else np.inf
        
        metrics.update({
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': len(trades),
            'total_bars': len(returns)
        })
        
        return metrics
    
    def analyze_market_conditions(self):
        """Analyze market conditions and regime classifier performance"""
        print("\n🌍 Analyzing market conditions...")
        
        # Market statistics
        price_data = self.analysis_data['close']
        returns = price_data.pct_change().dropna()
        
        market_metrics = {
            'total_return': (price_data.iloc[-1] / price_data.iloc[0]) - 1,
            'volatility': returns.std() * np.sqrt(252 * 24 * 60),
            'max_price': price_data.max(),
            'min_price': price_data.min(),
            'avg_daily_range': ((self.analysis_data['high'] - self.analysis_data['low']) / self.analysis_data['close']).mean()
        }
        
        print(f"📈 Market total return: {market_metrics['total_return']:.2%}")
        print(f"📊 Market volatility: {market_metrics['volatility']:.2%}")
        print(f"💹 Price range: ${market_metrics['min_price']:.2f} - ${market_metrics['max_price']:.2f}")
        
        # Regime analysis
        if len(self.classifier_filtered) > 0:
            regime_timeline = self.replay_signals_to_timeline(self.classifier_filtered, 'regime')
            regime_changes = len(self.classifier_filtered)
            unique_regimes = regime_timeline.nunique()
            
            print(f"🔄 Regime changes: {regime_changes}")
            print(f"🎯 Unique regimes: {unique_regimes}")
            
            # Regime distribution
            regime_dist = regime_timeline.value_counts(normalize=True)
            print("📊 Regime distribution:")
            for regime, pct in regime_dist.items():
                print(f"   Regime {regime}: {pct:.1%}")
                
        return market_metrics
        
    def generate_report(self, metrics_default, metrics_custom, market_metrics):
        """Generate comprehensive performance report"""
        print("\n" + "="*80)
        print("📋 ENSEMBLE STRATEGY PERFORMANCE REPORT - LAST 22K BARS (OUT-OF-SAMPLE)")
        print("="*80)
        
        print(f"\n🗓️  ANALYSIS PERIOD")
        print(f"   Start: {self.analysis_data.index[0]}")
        print(f"   End: {self.analysis_data.index[-1]}")
        print(f"   Bars: {len(self.analysis_data):,}")
        trading_days = len(self.analysis_data.index.normalize().unique())
        print(f"   Trading Days: {trading_days}")
        
        print(f"\n🌍 MARKET CONDITIONS")
        print(f"   Market Return: {market_metrics['total_return']:>10.2%}")
        print(f"   Market Volatility: {market_metrics['volatility']:>8.2%}")
        print(f"   Price Range: ${market_metrics['min_price']:.2f} - ${market_metrics['max_price']:.2f}")
        
        print(f"\n📊 STRATEGY PERFORMANCE COMPARISON")
        print(f"{'Metric':<25} {'Default':<15} {'Custom':<15} {'Difference':<15}")
        print("-" * 70)
        
        comparisons = [
            ('Total Return', 'total_return', '.2%'),
            ('Annualized Return', 'annualized_return', '.2%'),
            ('Sharpe Ratio', 'sharpe_ratio', '.2f'),
            ('Max Drawdown', 'max_drawdown', '.2%'),
            ('Win Rate', 'win_rate', '.2%'),
            ('Profit Factor', 'profit_factor', '.2f'),
            ('Total Trades', 'total_trades', ',d'),
        ]
        
        for label, key, fmt in comparisons:
            default_val = metrics_default[key]
            custom_val = metrics_custom[key]
            
            # Format values
            if fmt == '.2%':
                default_str = f"{default_val:.2%}"
                custom_str = f"{custom_val:.2%}"
                diff = custom_val - default_val
                diff_str = f"{diff:+.2%}"
            elif fmt == '.2f':
                default_str = f"{default_val:.2f}"
                custom_str = f"{custom_val:.2f}"
                diff = custom_val - default_val
                diff_str = f"{diff:+.2f}"
            elif fmt == ',d':
                default_str = f"{default_val:,d}"
                custom_str = f"{custom_val:,d}"
                diff = custom_val - default_val
                diff_str = f"{diff:+,d}"
            else:
                default_str = str(default_val)
                custom_str = str(custom_val)
                diff_str = str(custom_val - default_val)
            
            print(f"{label:<25} {default_str:<15} {custom_str:<15} {diff_str:<15}")
        
        print(f"\n💸 TRANSACTION COST IMPACT")
        print(f"   Cost Rate: {TRANSACTION_COST:.2%} per trade")
        
        # Estimate gross vs net performance
        default_gross_return = metrics_default['total_return'] / (1 - metrics_default['total_trades'] * TRANSACTION_COST)
        custom_gross_return = metrics_custom['total_return'] / (1 - metrics_custom['total_trades'] * TRANSACTION_COST)
        
        print(f"   Default - Gross: {default_gross_return:>6.2%}, Net: {metrics_default['total_return']:>6.2%}")
        print(f"   Custom  - Gross: {custom_gross_return:>6.2%}, Net: {metrics_custom['total_return']:>6.2%}")
        
        print(f"\n🏆 PERFORMANCE SUMMARY")
        if metrics_custom['total_return'] > metrics_default['total_return']:
            winner = "Custom Ensemble"
            advantage = (metrics_custom['total_return'] - metrics_default['total_return']) * 100
        else:
            winner = "Default Ensemble"
            advantage = (metrics_default['total_return'] - metrics_custom['total_return']) * 100
            
        print(f"   Winner: {winner}")
        print(f"   Advantage: {advantage:.2f} percentage points")
        
        if metrics_custom['sharpe_ratio'] > metrics_default['sharpe_ratio']:
            print(f"   Risk-Adjusted Winner: Custom (Sharpe: {metrics_custom['sharpe_ratio']:.2f})")
        else:
            print(f"   Risk-Adjusted Winner: Default (Sharpe: {metrics_default['sharpe_ratio']:.2f})")
            
        print("="*80)
        
    def run_analysis(self):
        """Execute complete analysis workflow"""
        print("🚀 Starting Ensemble Performance Analysis...")
        
        # Step 1: Load source data
        self.load_source_data()
        
        # Step 2: Load and filter signals
        self.load_ensemble_signals()
        
        # Step 3: Replay signals and calculate returns
        default_timeline = self.replay_signals_to_timeline(self.signals_default_filtered, 'default_ensemble')
        custom_timeline = self.replay_signals_to_timeline(self.signals_custom_filtered, 'custom_ensemble')
        
        # Step 4: Calculate performance metrics
        prices = self.analysis_data['close']
        
        default_returns, default_trades, default_trade_returns = self.calculate_trade_returns(
            default_timeline, prices, 'Default Ensemble'
        )
        custom_returns, custom_trades, custom_trade_returns = self.calculate_trade_returns(
            custom_timeline, prices, 'Custom Ensemble'
        )
        
        metrics_default = self.calculate_performance_metrics(default_returns, default_trades, 'Default')
        metrics_custom = self.calculate_performance_metrics(custom_returns, custom_trades, 'Custom')
        
        # Step 5: Market analysis
        market_metrics = self.analyze_market_conditions()
        
        # Step 6: Generate report
        self.generate_report(metrics_default, metrics_custom, market_metrics)
        
        return {
            'metrics_default': metrics_default,
            'metrics_custom': metrics_custom,
            'market_metrics': market_metrics,
            'default_returns': default_returns,
            'custom_returns': custom_returns,
            'default_timeline': default_timeline,
            'custom_timeline': custom_timeline
        }

if __name__ == "__main__":
    analyzer = EnsembleAnalyzer()
    results = analyzer.run_analysis()
    
    print("\n✅ Analysis complete!")
    print("📁 Results available in returned dictionary")