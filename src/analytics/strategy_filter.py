"""
Strategy filtering and selection toolkit for production-scale analysis.
"""
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


class StrategyFilter:
    """Advanced filtering for viable trading strategies."""
    
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.db_path = self.workspace_path / "analytics.duckdb"
        self.con = duckdb.connect(str(self.db_path))
        
    def filter_by_trade_count(self, min_trades: int = 10) -> pd.DataFrame:
        """
        Filter strategies by minimum trade count.
        Very efficient - no joins required.
        """
        query = f"""
        SELECT 
            strat,
            COUNT(*) as trade_count,
            CASE 
                WHEN strat LIKE '%breakout%' THEN 'breakout'
                WHEN strat LIKE '%ma_crossover%' THEN 'ma_crossover'
                WHEN strat LIKE '%mean_reversion%' THEN 'mean_reversion'
                WHEN strat LIKE '%momentum%' THEN 'momentum'
                WHEN strat LIKE '%rsi%' THEN 'rsi'
            END as strategy_type
        FROM read_parquet('{self.workspace_path}/traces/*/signals/*/*.parquet')
        WHERE val != 0
        GROUP BY strat
        HAVING COUNT(*) >= {min_trades}
        ORDER BY trade_count DESC
        """
        return self.con.execute(query).df()
    
    def sensitivity_analysis(self, strategy_type: str, data_path: str) -> pd.DataFrame:
        """
        Analyze parameter sensitivity by examining neighboring parameter values.
        Returns strategies that exist in stable positive return neighborhoods.
        """
        # First get all strategies of this type with their parameters
        query = f"""
        WITH strategy_params AS (
            SELECT DISTINCT
                strat,
                -- Extract parameters from strategy name
                REGEXP_EXTRACT(strat, '_{strategy_type}_grid_([0-9]+)_', 1) as param1,
                REGEXP_EXTRACT(strat, '_{strategy_type}_grid_[0-9]+_([0-9]+)_', 1) as param2
            FROM read_parquet('{self.workspace_path}/traces/*/signals/{strategy_type}_grid/*.parquet')
        ),
        strategy_performance AS (
            SELECT 
                s.strat,
                COUNT(*) as trades,
                AVG(CASE 
                    WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
                    WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
                END) as avg_return
            FROM read_parquet('{self.workspace_path}/traces/*/signals/{strategy_type}_grid/*.parquet') s
            JOIN read_parquet('{data_path}') m1 ON s.idx = m1.bar_index
            JOIN read_parquet('{data_path}') m2 ON s.idx + 1 = m2.bar_index
            WHERE s.val != 0
            GROUP BY s.strat
            HAVING COUNT(*) >= 5
        )
        SELECT 
            sp.strat,
            p.param1::INTEGER as param1,
            p.param2::INTEGER as param2,
            sp.trades,
            ROUND(sp.avg_return, 4) as avg_return
        FROM strategy_performance sp
        JOIN strategy_params p ON sp.strat = p.strat
        WHERE sp.avg_return > 0
        ORDER BY param1, param2
        """
        
        df = self.con.execute(query).df()
        
        # Analyze neighborhoods
        stable_strategies = []
        for _, row in df.iterrows():
            p1, p2 = row['param1'], row['param2']
            # Check if neighbors also have positive returns
            neighbors = df[
                (df['param1'].between(p1-5, p1+5)) & 
                (df['param2'].between(p2-5, p2+5))
            ]
            if len(neighbors) >= 3 and neighbors['avg_return'].mean() > 0:
                stable_strategies.append(row)
        
        return pd.DataFrame(stable_strategies)
    
    def regime_analysis(self, classifier_path: str, data_path: str, 
                       min_sharpe: float = 0.5) -> pd.DataFrame:
        """
        Analyze strategy performance by market regime.
        Filter strategies by Sharpe ratio within each regime.
        """
        query = f"""
        WITH regimes AS (
            -- Load classifier signals
            SELECT 
                idx as bar_index,
                val as regime
            FROM read_parquet('{classifier_path}')
            WHERE val != 0
        ),
        signal_returns AS (
            SELECT 
                s.strat,
                s.idx,
                r.regime,
                CASE 
                    WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
                    WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
                END as return_pct
            FROM read_parquet('{self.workspace_path}/traces/*/signals/*/*.parquet') s
            JOIN read_parquet('{data_path}') m1 ON s.idx = m1.bar_index
            JOIN read_parquet('{data_path}') m2 ON s.idx + 1 = m2.bar_index
            LEFT JOIN regimes r ON s.idx = r.bar_index
            WHERE s.val != 0
        )
        SELECT 
            strat,
            regime,
            COUNT(*) as trades,
            ROUND(AVG(return_pct), 4) as avg_return,
            ROUND(STDDEV(return_pct), 4) as volatility,
            -- Simple Sharpe for regime (not annualized due to short periods)
            ROUND(AVG(return_pct) / NULLIF(STDDEV(return_pct), 0), 3) as sharpe_ratio
        FROM signal_returns
        WHERE regime IS NOT NULL
        GROUP BY strat, regime
        HAVING COUNT(*) >= 5 AND sharpe_ratio >= {min_sharpe}
        ORDER BY regime, sharpe_ratio DESC
        """
        
        return self.con.execute(query).df()
    
    def correlation_filter(self, data_path: str, max_correlation: float = 0.7) -> pd.DataFrame:
        """
        Filter out highly correlated strategies using fast signal overlap method.
        """
        # Get all strategies in workspace
        all_strategies_query = f"""
        SELECT DISTINCT strat 
        FROM read_parquet('{self.workspace_path}/traces/*/signals/*/*.parquet')
        WHERE val != 0
        """
        all_strategies = self.con.execute(all_strategies_query).df()['strat'].tolist()
        
        # Use fast correlation filter
        from .fast_correlation_filter import ultra_fast_correlation_filter
        
        # Convert correlation threshold to overlap percentage
        # 0.7 correlation ~ 30% signal overlap
        max_overlap_pct = (1 - max_correlation) * 100
        
        selected = ultra_fast_correlation_filter(
            str(self.workspace_path), 
            all_strategies,
            max_overlap_pct
        )
        
        # Get returns for selected strategies only
        if selected:
            selected_list = "'" + "','".join(selected) + "'"
            returns_query = f"""
            SELECT 
                strat as strategy,
                AVG(CASE 
                    WHEN val = 1 THEN (m2.close - m1.close) / m1.close * 100
                    WHEN val = -1 THEN (m1.close - m2.close) / m1.close * 100
                END) as avg_return
            FROM read_parquet('{self.workspace_path}/traces/*/signals/*/*.parquet') s
            JOIN read_parquet('{data_path}') m1 ON s.idx = m1.bar_index
            JOIN read_parquet('{data_path}') m2 ON s.idx + 1 = m2.bar_index
            WHERE s.val != 0 AND s.strat IN ({selected_list})
            GROUP BY s.strat
            """
            return self.con.execute(returns_query).df()
        else:
            return pd.DataFrame({'strategy': [], 'avg_return': []})
    
    def apply_execution_costs(self, data_path: str, 
                            commission_bps: float = 1.0,  # basis points
                            slippage_bps: float = 1.0) -> pd.DataFrame:
        """
        Apply realistic execution costs and filter strategies that remain profitable.
        Commission and slippage in basis points (1 bp = 0.01%).
        """
        # Convert basis points to percentage
        one_way_cost = (commission_bps + slippage_bps) / 100.0
        round_trip_cost = one_way_cost * 2
        
        query = f"""
        WITH strategy_performance AS (
            SELECT 
                s.strat,
                COUNT(*) as trades,
                AVG(CASE 
                    WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
                    WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
                END) as gross_return,
                -- Apply round-trip costs
                AVG(CASE 
                    WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100 - {round_trip_cost}
                    WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100 - {round_trip_cost}
                END) as net_return
            FROM read_parquet('{self.workspace_path}/traces/*/signals/*/*.parquet') s
            JOIN read_parquet('{data_path}') m1 ON s.idx = m1.bar_index
            JOIN read_parquet('{data_path}') m2 ON s.idx + 1 = m2.bar_index
            WHERE s.val != 0
            GROUP BY s.strat
        )
        SELECT 
            strat,
            trades,
            ROUND(gross_return, 4) as gross_return_pct,
            ROUND(net_return, 4) as net_return_pct,
            ROUND(gross_return - net_return, 4) as cost_impact_pct
        FROM strategy_performance
        WHERE net_return > 0  -- Only profitable after costs
        ORDER BY net_return DESC
        """
        
        return self.con.execute(query).df()
    
    def comprehensive_filter(self, data_path: str, 
                           classifier_path: Optional[str] = None,
                           min_trades: int = 10,
                           min_sharpe: float = 0.5,
                           max_correlation: float = 0.7,
                           commission_pct: float = 0.01,
                           slippage_pct: float = 0.01) -> Dict[str, pd.DataFrame]:
        """
        Apply all filters and return comprehensive analysis.
        """
        results = {}
        
        # 1. Trade count filter
        print("1. Filtering by trade count...")
        viable = self.filter_by_trade_count(min_trades)
        results['trade_count_filter'] = viable
        print(f"   {len(viable)} strategies with >= {min_trades} trades")
        
        # 2. Execution costs
        print("\n2. Applying execution costs...")
        profitable = self.apply_execution_costs(data_path, commission_pct, slippage_pct)
        results['after_costs'] = profitable
        print(f"   {len(profitable)} strategies profitable after costs")
        
        # 3. Correlation filter - only among profitable strategies
        print("\n3. Filtering correlated strategies...")
        if len(profitable) > 0:
            from .correlation_filter import filter_correlated_strategies
            
            # Prepare data for correlation filter
            strategies_to_filter = profitable[['strat', 'net_return_pct']].copy()
            strategies_to_filter.rename(columns={'net_return_pct': 'avg_return'}, inplace=True)
            
            # Convert correlation threshold to overlap percentage
            max_overlap_pct = (1 - max_correlation) * 100
            
            # Filter correlated strategies, keeping best performers
            uncorrelated_df = filter_correlated_strategies(
                workspace_path=str(self.workspace_path),
                strategies_with_scores=strategies_to_filter,
                max_overlap_pct=max_overlap_pct,
                score_column='avg_return'
            )
            
            # Convert to expected format
            uncorrelated = uncorrelated_df[['strat', 'avg_return']].copy()
            uncorrelated.columns = ['strategy', 'avg_return']
        else:
            uncorrelated = pd.DataFrame({'strategy': [], 'avg_return': []})
            
        results['uncorrelated'] = uncorrelated
        print(f"   {len(uncorrelated)} uncorrelated strategies selected")
        
        # 4. Regime analysis (if classifier provided)
        if classifier_path:
            print("\n4. Analyzing by regime...")
            regime_performers = self.regime_analysis(classifier_path, data_path, min_sharpe)
            results['regime_analysis'] = regime_performers
            print(f"   {len(regime_performers)} strategy-regime combinations with Sharpe >= {min_sharpe}")
        
        # 5. Final selection
        print("\n5. Final selection...")
        # The uncorrelated strategies ARE the final selection
        # They were already filtered from profitable strategies
        results['final_selection'] = uncorrelated.copy()
        print(f"   {len(uncorrelated)} strategies selected")
        
        return results
    
    def close(self):
        """Close database connection."""
        self.con.close()


def analyze_grid_search(workspace_path: str, data_path: str, 
                       classifier_path: Optional[str] = None):
    """
    Run comprehensive analysis on grid search results.
    """
    filter = StrategyFilter(workspace_path)
    
    try:
        results = filter.comprehensive_filter(
            data_path=data_path,
            classifier_path=classifier_path,
            min_trades=10,
            min_sharpe=0.5,
            max_correlation=0.7,
            commission_pct=0.01,
            slippage_pct=0.01
        )
        
        # Save results
        output_dir = Path(workspace_path) / "analysis_results"
        output_dir.mkdir(exist_ok=True)
        
        for name, df in results.items():
            df.to_csv(output_dir / f"{name}.csv", index=False)
        
        # Create summary report
        summary = {
            'total_strategies_tested': len(filter.filter_by_trade_count(1)),
            'viable_by_trade_count': len(results['trade_count_filter']),
            'profitable_after_costs': len(results['after_costs']),
            'uncorrelated_selected': len(results['uncorrelated']),
            'final_strategies': len(results['final_selection'])
        }
        
        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to {output_dir}")
        
        return results
        
    finally:
        filter.close()


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python strategy_filter.py <workspace_path> <data_path> [classifier_path]")
        sys.exit(1)
    
    workspace = sys.argv[1]
    data = sys.argv[2]
    classifier = sys.argv[3] if len(sys.argv) > 3 else None
    
    analyze_grid_search(workspace, data, classifier)