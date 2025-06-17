#\!/usr/bin/env python3
"""
Framework for selecting optimal strategy ensembles per regime
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import duckdb

class EnsembleSelector:
    """
    Select optimal strategy ensembles for each regime based on multiple criteria
    """
    
    def __init__(self, results_file: str, analytics_db: str):
        self.results_file = results_file
        self.analytics_db = analytics_db
        self.conn = duckdb.connect(analytics_db)
        
    def load_results(self):
        """Load strategy performance results"""
        query = f"""
        CREATE OR REPLACE TABLE ensemble_analysis AS 
        SELECT 
            strategy_id,
            strategy_name,
            strategy_type,
            current_regime,
            trading_days,
            avg_daily_return_pct,
            daily_volatility_pct,
            TRY_CAST(annualized_sharpe_ratio AS DOUBLE) as sharpe_ratio,
            total_return_pct,
            win_days_pct,
            -- Add computed metrics
            total_return_pct / 100.0 as total_return,
            avg_daily_return_pct / 100.0 as avg_daily_return,
            daily_volatility_pct / 100.0 as daily_volatility
        FROM read_csv_auto('{self.results_file}')
        WHERE TRY_CAST(annualized_sharpe_ratio AS DOUBLE) IS NOT NULL
          AND trading_days >= 20;
        """
        self.conn.execute(query)
        
    def calculate_selection_metrics(self):
        """Calculate comprehensive metrics for ensemble selection"""
        
        # 1. Risk-adjusted performance metrics
        query = """
        CREATE OR REPLACE TABLE strategy_scores AS
        WITH base_metrics AS (
            SELECT 
                strategy_name,
                strategy_type,
                current_regime,
                sharpe_ratio,
                total_return,
                daily_volatility,
                win_days_pct / 100.0 as win_rate,
                trading_days,
                
                -- Sortino ratio (downside deviation)
                (SELECT STDDEV(CASE WHEN r2.avg_daily_return < 0 
                                   THEN r2.avg_daily_return ELSE 0 END) * SQRT(252)
                 FROM ensemble_analysis r2 
                 WHERE r2.strategy_name = ensemble_analysis.strategy_name
                   AND r2.current_regime = ensemble_analysis.current_regime) as downside_vol,
                
                -- Calmar ratio (return / max drawdown proxy)
                total_return / GREATEST(ABS(
                    (SELECT MIN(SUM(r3.avg_daily_return) OVER (ORDER BY r3.strategy_name))
                     FROM ensemble_analysis r3
                     WHERE r3.strategy_name = ensemble_analysis.strategy_name
                       AND r3.current_regime = ensemble_analysis.current_regime)
                ), 0.01) as calmar_proxy
                
            FROM ensemble_analysis
        ),
        scored AS (
            SELECT 
                *,
                -- Sortino ratio
                COALESCE(avg_daily_return * 252 / NULLIF(downside_vol, 0), 0) as sortino_ratio,
                
                -- Consistency score (combines Sharpe and win rate)
                sharpe_ratio * SQRT(win_rate) as consistency_score,
                
                -- Risk-adjusted return score
                sharpe_ratio * (1 - daily_volatility) as risk_adj_score,
                
                -- Regime stability (how well it performs in this specific regime)
                sharpe_ratio / GREATEST(
                    (SELECT STDDEV(sharpe_ratio) 
                     FROM ensemble_analysis e2 
                     WHERE e2.strategy_name = base_metrics.strategy_name), 0.1
                ) as regime_stability
                
            FROM base_metrics
        )
        SELECT 
            *,
            -- Composite selection score
            (0.30 * (sharpe_ratio / 5.0) +           -- Sharpe (normalized to 0-1 range)
             0.20 * (consistency_score / 3.0) +       -- Consistency 
             0.20 * (regime_stability / 10.0) +       -- Regime stability
             0.15 * (win_rate) +                      -- Win rate
             0.15 * (1 - daily_volatility / 0.10)    -- Low volatility bonus
            ) as selection_score
        FROM scored;
        """
        self.conn.execute(query)
        
    def analyze_correlations(self):
        """Analyze strategy correlations within each regime"""
        # This would require access to the actual return series
        # For now, we'll use strategy type as a proxy for correlation
        query = """
        CREATE OR REPLACE TABLE strategy_correlations AS
        WITH type_groups AS (
            SELECT 
                current_regime,
                strategy_type,
                COUNT(*) as strategy_count,
                AVG(sharpe_ratio) as avg_sharpe,
                AVG(daily_volatility) as avg_volatility
            FROM strategy_scores
            GROUP BY current_regime, strategy_type
        )
        SELECT 
            s.*,
            -- Diversification score (penalty for same strategy type)
            CASE 
                WHEN tg.strategy_count > 5 THEN 0.7  -- Many similar strategies
                WHEN tg.strategy_count > 3 THEN 0.85 -- Some similar strategies
                ELSE 1.0                              -- Few similar strategies
            END as diversification_multiplier
        FROM strategy_scores s
        JOIN type_groups tg 
          ON s.current_regime = tg.current_regime 
         AND s.strategy_type = tg.strategy_type;
        """
        self.conn.execute(query)
        
    def select_ensemble_candidates(self, 
                                 min_sharpe: float = 0.5,
                                 max_strategies_per_type: int = 2,
                                 target_ensemble_size: int = 10):
        """Select ensemble candidates based on multiple criteria"""
        
        query = f"""
        CREATE OR REPLACE TABLE ensemble_candidates AS
        WITH ranked_strategies AS (
            SELECT 
                *,
                selection_score * diversification_multiplier as final_score,
                ROW_NUMBER() OVER (
                    PARTITION BY current_regime, strategy_type 
                    ORDER BY selection_score DESC
                ) as type_rank,
                ROW_NUMBER() OVER (
                    PARTITION BY current_regime 
                    ORDER BY selection_score * diversification_multiplier DESC
                ) as overall_rank
            FROM strategy_correlations
            WHERE sharpe_ratio >= {min_sharpe}
        ),
        type_limited AS (
            -- Limit strategies per type to ensure diversification
            SELECT *
            FROM ranked_strategies
            WHERE type_rank <= {max_strategies_per_type}
        )
        SELECT 
            current_regime,
            strategy_name,
            strategy_type,
            sharpe_ratio,
            total_return,
            daily_volatility,
            win_rate,
            consistency_score,
            regime_stability,
            selection_score,
            final_score,
            overall_rank
        FROM type_limited
        WHERE overall_rank <= {target_ensemble_size * 2}  -- Keep more for final selection
        ORDER BY current_regime, final_score DESC;
        """
        self.conn.execute(query)
        
    def generate_ensemble_recommendations(self):
        """Generate final ensemble recommendations per regime"""
        
        query = """
        -- Final ensemble selection
        WITH regime_ensembles AS (
            SELECT 
                current_regime,
                strategy_name,
                strategy_type,
                sharpe_ratio,
                final_score,
                ROW_NUMBER() OVER (PARTITION BY current_regime ORDER BY final_score DESC) as rank
            FROM ensemble_candidates
        ),
        ensemble_stats AS (
            SELECT 
                current_regime,
                COUNT(*) as ensemble_size,
                COUNT(DISTINCT strategy_type) as strategy_types,
                AVG(sharpe_ratio) as avg_sharpe,
                MIN(sharpe_ratio) as min_sharpe,
                MAX(sharpe_ratio) as max_sharpe,
                STRING_AGG(strategy_name || ' (' || ROUND(sharpe_ratio, 2) || ')', ', ') as strategies
            FROM regime_ensembles
            WHERE rank <= 10  -- Top 10 per regime
            GROUP BY current_regime
        )
        SELECT * FROM ensemble_stats
        ORDER BY current_regime;
        """
        
        return self.conn.execute(query).df()
    
    def analyze_ensemble_characteristics(self):
        """Analyze characteristics of selected ensembles"""
        
        query = """
        -- Ensemble characteristics
        WITH selected_strategies AS (
            SELECT * 
            FROM ensemble_candidates 
            WHERE overall_rank <= 10
        )
        SELECT 
            current_regime,
            COUNT(DISTINCT strategy_type) as unique_strategy_types,
            COUNT(*) as total_strategies,
            ROUND(AVG(sharpe_ratio), 3) as avg_sharpe,
            ROUND(AVG(total_return * 100), 2) as avg_return_pct,
            ROUND(AVG(daily_volatility * 100), 2) as avg_volatility_pct,
            ROUND(AVG(win_rate * 100), 1) as avg_win_rate_pct,
            ROUND(MIN(sharpe_ratio), 3) as min_sharpe,
            ROUND(MAX(sharpe_ratio), 3) as max_sharpe
        FROM selected_strategies
        GROUP BY current_regime
        ORDER BY current_regime;
        """
        
        return self.conn.execute(query).df()

def main():
    # Configuration
    results_file = "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/results/all_strategies_analysis_20250616_175832.csv"
    analytics_db = "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/analytics.duckdb"
    
    # Initialize selector
    selector = EnsembleSelector(results_file, analytics_db)
    
    print("=== ENSEMBLE SELECTION FRAMEWORK ===\n")
    
    # Step 1: Load and prepare data
    print("1. Loading strategy results...")
    selector.load_results()
    
    # Step 2: Calculate selection metrics
    print("2. Calculating selection metrics...")
    selector.calculate_selection_metrics()
    
    # Step 3: Analyze correlations/diversification
    print("3. Analyzing strategy diversification...")
    selector.analyze_correlations()
    
    # Step 4: Select ensemble candidates
    print("4. Selecting ensemble candidates...")
    selector.select_ensemble_candidates(
        min_sharpe=0.5,
        max_strategies_per_type=2,
        target_ensemble_size=10
    )
    
    # Step 5: Generate recommendations
    print("\n=== ENSEMBLE RECOMMENDATIONS ===")
    recommendations = selector.generate_ensemble_recommendations()
    print(recommendations.to_string(index=False))
    
    # Step 6: Analyze characteristics
    print("\n=== ENSEMBLE CHARACTERISTICS ===")
    characteristics = selector.analyze_ensemble_characteristics()
    print(characteristics.to_string(index=False))
    
    # Export detailed candidates
    print("\n5. Exporting detailed ensemble candidates...")
    detailed_query = """
    SELECT * FROM ensemble_candidates 
    WHERE overall_rank <= 10
    ORDER BY current_regime, final_score DESC
    """
    detailed_df = selector.conn.execute(detailed_query).df()
    output_file = "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/results/ensemble_candidates.csv"
    detailed_df.to_csv(output_file, index=False)
    print(f"Detailed candidates saved to: {output_file}")

if __name__ == "__main__":
    main()