#!/usr/bin/env python3
"""
Comprehensive analysis of ensemble strategies using SQL data mining.
Focuses on finding effective combinations and validating signal timing.
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class EnsembleAnalyzer:
    def __init__(self, db_path: str = "analytics.duckdb"):
        """Initialize analyzer with database connection."""
        self.conn = duckdb.connect(db_path, read_only=True)
        self.conn.execute("SET memory_limit='2GB'")
        
    def analyze_signal_timing(self):
        """Validate that signal timing is correct in event-driven system."""
        print("=" * 80)
        print("📊 SIGNAL TIMING VALIDATION")
        print("=" * 80)
        
        # Check if we have signal_performance table
        tables = self.conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]
        
        if 'signal_performance' in table_names:
            # Analyze signal-to-trade timing
            timing_analysis = self.conn.execute("""
                SELECT 
                    strategy_type,
                    strategy_name,
                    AVG(entry_bar - signal_bar) as avg_bars_to_entry,
                    MIN(entry_bar - signal_bar) as min_bars_to_entry,
                    MAX(entry_bar - signal_bar) as max_bars_to_entry,
                    COUNT(*) as total_trades,
                    AVG(return_pct) as avg_return_pct
                FROM signal_performance
                WHERE entry_bar IS NOT NULL
                GROUP BY strategy_type, strategy_name
                ORDER BY avg_return_pct DESC
            """).df()
            
            print("\n🎯 Signal-to-Trade Timing Analysis:")
            print(timing_analysis.to_string(index=False))
            
            # Check for look-ahead bias
            lookahead = self.conn.execute("""
                SELECT 
                    strategy_name,
                    COUNT(*) as negative_delays
                FROM signal_performance
                WHERE entry_bar < signal_bar
                GROUP BY strategy_name
            """).df()
            
            if not lookahead.empty:
                print("\n⚠️  WARNING: Potential look-ahead bias detected!")
                print(lookahead.to_string(index=False))
            else:
                print("\n✅ No look-ahead bias detected (all entries after signals)")
                
        else:
            print("\n⚠️  No signal_performance table found. Running basic analysis...")
            self._analyze_basic_signal_timing()
    
    def _analyze_basic_signal_timing(self):
        """Basic signal timing analysis without performance table."""
        # Check signal_changes table
        if self._table_exists('signal_changes'):
            signal_distribution = self.conn.execute("""
                SELECT 
                    component_type,
                    COUNT(DISTINCT component_id) as unique_components,
                    COUNT(*) as total_signals,
                    MIN(bar_index) as first_signal_bar,
                    MAX(bar_index) as last_signal_bar,
                    COUNT(DISTINCT bar_index) as unique_bars_with_signals
                FROM signal_changes
                GROUP BY component_type
            """).df()
            
            print("\n📈 Signal Distribution by Component Type:")
            print(signal_distribution.to_string(index=False))
            
            # Analyze signal density
            density = self.conn.execute("""
                SELECT 
                    bar_index,
                    COUNT(*) as signals_at_bar,
                    COUNT(DISTINCT component_id) as components_signaling
                FROM signal_changes
                WHERE component_type = 'strategy'
                GROUP BY bar_index
                ORDER BY signals_at_bar DESC
                LIMIT 20
            """).df()
            
            print("\n📊 Bars with Most Signal Activity:")
            print(density.to_string(index=False))
    
    def find_ensemble_combinations(self):
        """Find effective ensemble strategy combinations."""
        print("\n" + "=" * 80)
        print("🎭 ENSEMBLE STRATEGY DISCOVERY")
        print("=" * 80)
        
        # First, analyze correlations between strategies
        if self._table_exists('strategy_correlations'):
            correlations = self.conn.execute("""
                SELECT 
                    strategy_a,
                    strategy_b,
                    correlation,
                    a_signal_freq,
                    b_signal_freq,
                    ABS(a_signal_freq - b_signal_freq) as freq_diff
                FROM strategy_correlations
                WHERE ABS(correlation) < 0.3  -- Low correlation for diversity
                AND a_signal_freq > 0.01      -- Active strategies
                AND b_signal_freq > 0.01
                ORDER BY freq_diff DESC
                LIMIT 20
            """).df()
            
            print("\n🔀 Low-Correlation Strategy Pairs (Good for Ensembles):")
            print(correlations.to_string(index=False))
        
        # Analyze complementary signal patterns
        if self._table_exists('signal_changes'):
            # Find strategies that signal at different times
            complementary = self.conn.execute("""
                WITH strategy_signals AS (
                    SELECT 
                        component_id as strategy_id,
                        bar_index,
                        signal_value
                    FROM signal_changes
                    WHERE component_type = 'strategy'
                ),
                signal_overlaps AS (
                    SELECT 
                        s1.strategy_id as strategy_a,
                        s2.strategy_id as strategy_b,
                        COUNT(DISTINCT s1.bar_index) as a_signals,
                        COUNT(DISTINCT s2.bar_index) as b_signals,
                        COUNT(DISTINCT CASE 
                            WHEN s1.bar_index = s2.bar_index 
                            THEN s1.bar_index 
                        END) as overlapping_bars
                    FROM strategy_signals s1
                    CROSS JOIN strategy_signals s2
                    WHERE s1.strategy_id < s2.strategy_id
                    GROUP BY s1.strategy_id, s2.strategy_id
                )
                SELECT 
                    strategy_a,
                    strategy_b,
                    a_signals,
                    b_signals,
                    overlapping_bars,
                    ROUND(100.0 * overlapping_bars / LEAST(a_signals, b_signals), 2) as overlap_pct,
                    a_signals + b_signals - overlapping_bars as combined_coverage
                FROM signal_overlaps
                WHERE overlap_pct < 50  -- Low overlap
                AND a_signals > 10      -- Meaningful signal count
                AND b_signals > 10
                ORDER BY combined_coverage DESC
                LIMIT 20
            """).df()
            
            print("\n🎯 Complementary Strategy Pairs (Low Signal Overlap):")
            print(complementary.to_string(index=False))
    
    def analyze_regime_based_ensembles(self):
        """Analyze how strategies perform under different market regimes."""
        print("\n" + "=" * 80)
        print("🌍 REGIME-BASED ENSEMBLE ANALYSIS")
        print("=" * 80)
        
        if not self._table_exists('component_metrics'):
            print("⚠️  component_metrics table not found")
            return
            
        # Find classifiers
        classifiers = self.conn.execute("""
            SELECT 
                component_id,
                strategy_type as classifier_type,
                signal_frequency as regime_change_freq,
                regime_classifications
            FROM component_metrics
            WHERE component_type = 'classifier'
            ORDER BY signal_frequency DESC
        """).df()
        
        print(f"\n🔍 Found {len(classifiers)} classifiers")
        print("\nTop Classifiers by Regime Change Frequency:")
        print(classifiers[['component_id', 'classifier_type', 'regime_change_freq']].head(10).to_string(index=False))
        
        # Analyze strategy performance by regime if we have the data
        if self._table_exists('signal_changes'):
            # Find strategies that work well in specific regimes
            regime_strategy_analysis = self.conn.execute("""
                WITH strategy_signals AS (
                    SELECT 
                        component_id,
                        bar_index,
                        signal_value
                    FROM signal_changes
                    WHERE component_type = 'strategy'
                ),
                classifier_states AS (
                    SELECT 
                        component_id as classifier_id,
                        bar_index,
                        signal_value as regime
                    FROM signal_changes
                    WHERE component_type = 'classifier'
                ),
                aligned_signals AS (
                    SELECT 
                        s.component_id as strategy_id,
                        c.classifier_id,
                        c.regime,
                        COUNT(*) as signals_in_regime
                    FROM strategy_signals s
                    JOIN classifier_states c ON s.bar_index = c.bar_index
                    GROUP BY s.component_id, c.classifier_id, c.regime
                )
                SELECT 
                    strategy_id,
                    classifier_id,
                    regime,
                    signals_in_regime,
                    RANK() OVER (PARTITION BY classifier_id, regime ORDER BY signals_in_regime DESC) as rank_in_regime
                FROM aligned_signals
                WHERE signals_in_regime > 5
                ORDER BY classifier_id, regime, signals_in_regime DESC
            """).df()
            
            if not regime_strategy_analysis.empty:
                print("\n📊 Top Strategies by Regime:")
                # Show top 3 strategies for each regime
                top_by_regime = regime_strategy_analysis[regime_strategy_analysis['rank_in_regime'] <= 3]
                for classifier in top_by_regime['classifier_id'].unique()[:3]:  # Show first 3 classifiers
                    print(f"\n🎯 Classifier: {classifier}")
                    classifier_data = top_by_regime[top_by_regime['classifier_id'] == classifier]
                    print(classifier_data[['regime', 'strategy_id', 'signals_in_regime']].to_string(index=False))
    
    def suggest_ensemble_portfolios(self):
        """Suggest specific ensemble portfolio configurations."""
        print("\n" + "=" * 80)
        print("💼 ENSEMBLE PORTFOLIO SUGGESTIONS")
        print("=" * 80)
        
        suggestions = []
        
        # 1. Momentum + Mean Reversion Ensemble
        print("\n1️⃣ Momentum + Mean Reversion Ensemble")
        print("   - Combines trend-following with counter-trend")
        print("   - Works well in both trending and ranging markets")
        
        if self._table_exists('component_metrics'):
            momentum_mr = self.conn.execute("""
                SELECT 
                    strategy_type,
                    COUNT(*) as strategy_count,
                    AVG(signal_frequency) as avg_signal_freq,
                    AVG(compression_ratio) as avg_compression
                FROM component_metrics
                WHERE component_type = 'strategy'
                AND strategy_type IN ('momentum', 'mean_reversion', 'ma_crossover', 'rsi')
                GROUP BY strategy_type
            """).df()
            
            if not momentum_mr.empty:
                print(momentum_mr.to_string(index=False))
                
                # Suggest specific combination
                best_momentum = self.conn.execute("""
                    SELECT component_id, signal_frequency
                    FROM component_metrics
                    WHERE component_type = 'strategy' 
                    AND strategy_type IN ('momentum', 'ma_crossover')
                    AND signal_frequency BETWEEN 0.02 AND 0.10
                    ORDER BY signal_frequency DESC
                    LIMIT 1
                """).fetchone()
                
                best_mr = self.conn.execute("""
                    SELECT component_id, signal_frequency
                    FROM component_metrics
                    WHERE component_type = 'strategy'
                    AND strategy_type IN ('mean_reversion', 'rsi')
                    AND signal_frequency BETWEEN 0.02 AND 0.10
                    ORDER BY signal_frequency DESC
                    LIMIT 1
                """).fetchone()
                
                if best_momentum and best_mr:
                    print(f"\n   Suggested combination:")
                    print(f"   - Momentum: {best_momentum[0]} (freq: {best_momentum[1]:.3f})")
                    print(f"   - Mean Rev: {best_mr[0]} (freq: {best_mr[1]:.3f})")
        
        # 2. Multi-Timeframe Ensemble
        print("\n2️⃣ Multi-Timeframe Ensemble")
        print("   - Same strategy logic on different timeframes")
        print("   - Captures both short-term and long-term moves")
        
        # 3. Regime-Adaptive Ensemble
        print("\n3️⃣ Regime-Adaptive Ensemble")
        print("   - Different strategies for different market conditions")
        print("   - Requires regime classifier for switching")
        
        if self._table_exists('component_metrics'):
            # Find a good regime classifier
            best_classifier = self.conn.execute("""
                SELECT 
                    component_id,
                    strategy_type,
                    signal_frequency,
                    regime_classifications
                FROM component_metrics
                WHERE component_type = 'classifier'
                AND signal_frequency BETWEEN 0.001 AND 0.05
                ORDER BY signal_frequency DESC
                LIMIT 1
            """).fetchone()
            
            if best_classifier:
                print(f"\n   Suggested classifier: {best_classifier[0]}")
                print(f"   Type: {best_classifier[1]}, Freq: {best_classifier[2]:.3f}")
                
                # Parse regime classifications if available
                if best_classifier[3]:
                    try:
                        regimes = json.loads(best_classifier[3]) if isinstance(best_classifier[3], str) else best_classifier[3]
                        print(f"   Regimes: {list(regimes.keys())}")
                    except:
                        pass
        
        # 4. Signal Strength Voting Ensemble
        print("\n4️⃣ Signal Strength Voting Ensemble")
        print("   - Multiple strategies vote on direction")
        print("   - Position size based on agreement strength")
        print("   - Reduces false signals through consensus")
        
        return suggestions
    
    def analyze_optimal_parameters(self):
        """Analyze optimal parameters for ensemble strategies."""
        print("\n" + "=" * 80)
        print("⚙️  OPTIMAL PARAMETER ANALYSIS")
        print("=" * 80)
        
        if not self._table_exists('component_metrics'):
            print("⚠️  component_metrics table not found")
            return
            
        # Analyze RSI parameters
        print("\n📊 RSI Strategy Parameter Analysis:")
        rsi_params = self.conn.execute("""
            SELECT 
                CASE 
                    WHEN component_id LIKE '%rsi_7_%' THEN 7
                    WHEN component_id LIKE '%rsi_14_%' THEN 14
                    WHEN component_id LIKE '%rsi_21_%' THEN 21
                    WHEN component_id LIKE '%rsi_30_%' THEN 30
                    ELSE NULL
                END as rsi_period,
                COUNT(*) as count,
                AVG(signal_frequency) as avg_signal_freq,
                AVG(total_positions) as avg_positions,
                AVG(avg_position_duration) as avg_duration
            FROM component_metrics
            WHERE component_type = 'strategy'
            AND strategy_type = 'rsi'
            AND component_id LIKE '%rsi_%'
            GROUP BY 1
            HAVING rsi_period IS NOT NULL
            ORDER BY avg_signal_freq DESC
        """).df()
        
        if not rsi_params.empty:
            print(rsi_params.to_string(index=False))
            print(f"\n✨ Recommended RSI period: {rsi_params.iloc[0]['rsi_period']} (highest signal frequency)")
        
        # Analyze MA crossover parameters
        print("\n📊 MA Crossover Parameter Analysis:")
        ma_params = self.conn.execute("""
            SELECT 
                strategy_type,
                COUNT(*) as variations,
                MIN(signal_frequency) as min_freq,
                AVG(signal_frequency) as avg_freq,
                MAX(signal_frequency) as max_freq,
                AVG(compression_ratio) as avg_compression
            FROM component_metrics
            WHERE component_type = 'strategy'
            AND strategy_type IN ('ma_crossover', 'momentum')
            GROUP BY strategy_type
        """).df()
        
        if not ma_params.empty:
            print(ma_params.to_string(index=False))
    
    def export_ensemble_configs(self):
        """Export recommended ensemble configurations as YAML configs."""
        print("\n" + "=" * 80)
        print("📝 EXPORTING ENSEMBLE CONFIGURATIONS")
        print("=" * 80)
        
        # Create configs directory if it doesn't exist
        configs_dir = Path("config/ensembles")
        configs_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Momentum + Mean Reversion Config
        momentum_mr_config = """# Momentum + Mean Reversion Ensemble Strategy
# Combines trend-following with counter-trend strategies

workflow_type: ensemble_backtest

data:
  type: file
  file_path: data/synthetic_data.parquet
  symbols: ["AAPL"]
  
ensemble:
  name: momentum_mean_reversion
  voting_method: weighted_average
  min_agreement: 0.6
  
  strategies:
    - name: trend_follower
      type: ma_crossover
      weight: 0.5
      params:
        fast_period: 10
        slow_period: 30
        signal_threshold: 0.02
        
    - name: mean_reverter  
      type: rsi
      weight: 0.5
      params:
        period: 14
        oversold: 30
        overbought: 70
        
portfolio:
  initial_capital: 100000
  position_size: 0.1
  max_positions: 5
  
risk:
  stop_loss: 0.02
  take_profit: 0.05
  max_drawdown: 0.10
"""
        
        config_path = configs_dir / "momentum_mean_reversion.yaml"
        with open(config_path, 'w') as f:
            f.write(momentum_mr_config)
        print(f"✅ Created: {config_path}")
        
        # 2. Multi-Strategy Voting Ensemble
        voting_config = """# Multi-Strategy Voting Ensemble
# Multiple strategies vote on trade direction

workflow_type: ensemble_backtest

data:
  type: file
  file_path: data/synthetic_data.parquet
  symbols: ["AAPL"]
  
ensemble:
  name: multi_strategy_voting
  voting_method: majority_vote
  min_voters: 2
  
  strategies:
    - name: momentum_fast
      type: momentum
      params:
        lookback: 20
        threshold: 0.02
        
    - name: rsi_oversold
      type: rsi
      params:
        period: 14
        oversold: 30
        overbought: 70
        
    - name: ma_cross
      type: ma_crossover
      params:
        fast_period: 5
        slow_period: 20
        
    - name: mean_rev
      type: mean_reversion
      params:
        lookback: 30
        num_std: 2.0
        
portfolio:
  initial_capital: 100000
  position_size_method: kelly
  max_positions: 3
  
risk:
  stop_loss: 0.03
  trailing_stop: true
  max_correlation: 0.7
"""
        
        config_path = configs_dir / "multi_strategy_voting.yaml"
        with open(config_path, 'w') as f:
            f.write(voting_config)
        print(f"✅ Created: {config_path}")
        
        # 3. Regime-Adaptive Ensemble
        regime_config = """# Regime-Adaptive Ensemble Strategy
# Switches strategies based on market regime

workflow_type: regime_adaptive_backtest

data:
  type: file
  file_path: data/synthetic_data.parquet
  symbols: ["AAPL"]
  
classifier:
  type: volatility_regime
  params:
    lookback: 50
    thresholds: [0.01, 0.02]  # Low, Medium, High volatility
    
regime_strategies:
  low_volatility:
    - name: trend_follower
      type: ma_crossover
      weight: 1.0
      params:
        fast_period: 20
        slow_period: 50
        
  medium_volatility:
    - name: momentum
      type: momentum
      weight: 0.6
      params:
        lookback: 20
        
    - name: mean_reversion
      type: mean_reversion  
      weight: 0.4
      params:
        lookback: 30
        num_std: 1.5
        
  high_volatility:
    - name: rsi_extreme
      type: rsi
      weight: 1.0
      params:
        period: 7
        oversold: 20
        overbought: 80
        
portfolio:
  initial_capital: 100000
  position_size: 0.05  # Smaller size in high volatility
  max_positions: 3
  
risk:
  dynamic_stops: true  # Adjust stops based on volatility
  max_var: 0.02       # Value at Risk limit
"""
        
        config_path = configs_dir / "regime_adaptive.yaml"
        with open(config_path, 'w') as f:
            f.write(regime_config)
        print(f"✅ Created: {config_path}")
        
        print(f"\n📁 All ensemble configs saved to: {configs_dir}")
        
    def _table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        result = self.conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name]
        ).fetchone()
        return result[0] > 0
    
    def close(self):
        """Close database connection."""
        self.conn.close()

def main():
    """Run comprehensive ensemble analysis."""
    analyzer = EnsembleAnalyzer()
    
    try:
        # 1. Validate signal timing
        analyzer.analyze_signal_timing()
        
        # 2. Find ensemble combinations
        analyzer.find_ensemble_combinations()
        
        # 3. Analyze regime-based ensembles
        analyzer.analyze_regime_based_ensembles()
        
        # 4. Suggest ensemble portfolios
        analyzer.suggest_ensemble_portfolios()
        
        # 5. Analyze optimal parameters
        analyzer.analyze_optimal_parameters()
        
        # 6. Export ensemble configurations
        analyzer.export_ensemble_configs()
        
        print("\n" + "=" * 80)
        print("✅ ENSEMBLE ANALYSIS COMPLETE")
        print("=" * 80)
        
    finally:
        analyzer.close()

if __name__ == "__main__":
    main()