"""
Optimized strategy filter that properly chains filters.
"""
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Optional
import json


class OptimizedStrategyFilter:
    """Strategy filtering with proper filter chaining."""
    
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.db_path = self.workspace_path / "analytics.duckdb"
        self.con = duckdb.connect(str(self.db_path))
    
    def correlation_filter_subset(self, data_path: str, strategies: List[str], 
                                 max_correlation: float = 0.7) -> pd.DataFrame:
        """
        Filter correlations only among specified strategies.
        Much faster than correlating all strategies.
        """
        if not strategies:
            return pd.DataFrame({'strategy': [], 'avg_return': []})
        
        # Create strategy list for SQL IN clause
        strategy_list = "'" + "','".join(strategies) + "'"
        
        query = f"""
        WITH strategy_returns AS (
            SELECT 
                s.strat,
                s.idx,
                CASE 
                    WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close
                    WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close
                    ELSE 0
                END as return_pct
            FROM read_parquet('{self.workspace_path}/traces/*/signals/*/*.parquet') s
            JOIN read_parquet('{data_path}') m1 ON s.idx = m1.bar_index
            JOIN read_parquet('{data_path}') m2 ON s.idx + 1 = m2.bar_index
            WHERE s.strat IN ({strategy_list})  -- Only specified strategies
        )
        SELECT * FROM strategy_returns
        ORDER BY strat, idx
        """
        
        print(f"  Calculating correlations for {len(strategies)} strategies...")
        returns_df = self.con.execute(query).df()
        
        if returns_df.empty:
            return pd.DataFrame({'strategy': [], 'avg_return': []})
        
        # Pivot returns
        returns_pivot = returns_df.pivot_table(
            index='idx', 
            columns='strat', 
            values='return_pct',
            fill_value=0
        )
        
        # Calculate correlations
        corr_matrix = returns_pivot.corr()
        
        # Find uncorrelated strategies
        selected_strategies = []
        remaining = set(corr_matrix.columns)
        
        while remaining:
            # Get strategy with best average return
            avg_returns = returns_pivot[list(remaining)].mean()
            best_strat = avg_returns.idxmax()
            selected_strategies.append(best_strat)
            
            # Remove correlated strategies
            correlated = corr_matrix[best_strat][corr_matrix[best_strat] > max_correlation].index
            remaining -= set(correlated)
        
        return pd.DataFrame({
            'strategy': selected_strategies,
            'avg_return': returns_pivot[selected_strategies].mean()
        })
    
    def comprehensive_filter_optimized(self, data_path: str, 
                                     min_trades: int = 10,
                                     min_sharpe: float = 0.5,
                                     max_correlation: float = 0.7,
                                     commission_bps: float = 1.0,
                                     slippage_bps: float = 1.0) -> Dict[str, pd.DataFrame]:
        """
        Optimized filtering that properly chains filters.
        """
        from .strategy_filter import StrategyFilter
        
        # Use existing filter for first steps
        base_filter = StrategyFilter(str(self.workspace_path))
        results = {}
        
        try:
            # 1. Trade count filter
            print("1. Filtering by trade count...")
            viable = base_filter.filter_by_trade_count(min_trades)
            results['trade_count_filter'] = viable
            print(f"   {len(viable)} strategies with >= {min_trades} trades")
            
            if viable.empty:
                return results
            
            # 2. Execution costs - only on viable strategies
            print("\n2. Applying execution costs...")
            viable_list = viable['strat'].tolist()
            profitable = base_filter.apply_execution_costs(data_path, commission_bps, slippage_bps)
            # Filter to only include viable strategies
            profitable = profitable[profitable['strat'].isin(viable_list)]
            results['after_costs'] = profitable
            print(f"   {len(profitable)} strategies profitable after costs")
            
            if profitable.empty:
                return results
            
            # 3. Correlation filter - ONLY on profitable strategies
            print("\n3. Filtering correlated strategies...")
            profitable_list = profitable['strat'].tolist()
            uncorrelated = self.correlation_filter_subset(data_path, profitable_list, max_correlation)
            results['uncorrelated'] = uncorrelated
            print(f"   {len(uncorrelated)} uncorrelated strategies selected")
            
            # 4. Final selection
            print("\n4. Final selection...")
            final_strategies = set(uncorrelated['strategy'])
            
            # Get full details for final strategies
            if final_strategies:
                final_details_query = f"""
                WITH performance AS (
                    SELECT 
                        s.strat,
                        COUNT(*) as trades,
                        AVG(CASE 
                            WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
                            WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
                        END) as avg_return,
                        STDDEV(CASE 
                            WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
                            WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
                        END) as volatility
                    FROM read_parquet('{self.workspace_path}/traces/*/signals/*/*.parquet') s
                    JOIN read_parquet('{data_path}') m1 ON s.idx = m1.bar_index
                    JOIN read_parquet('{data_path}') m2 ON s.idx + 1 = m2.bar_index
                    WHERE s.val != 0 AND s.strat IN ('{"','".join(final_strategies)}')
                    GROUP BY s.strat
                )
                SELECT 
                    strat as strategy,
                    trades,
                    ROUND(avg_return, 4) as avg_return_pct,
                    ROUND(volatility, 4) as volatility,
                    ROUND(avg_return / NULLIF(volatility, 0), 3) as sharpe_ratio
                FROM performance
                ORDER BY sharpe_ratio DESC
                """
                
                results['final_selection'] = self.con.execute(final_details_query).df()
            else:
                results['final_selection'] = pd.DataFrame()
            
            print(f"   {len(final_strategies)} strategies selected")
            
            return results
            
        finally:
            base_filter.close()
    
    def close(self):
        """Close database connection."""
        self.con.close()


def run_optimized_filter(workspace_path: str, data_path: str, **kwargs):
    """Run optimized filtering."""
    filter = OptimizedStrategyFilter(workspace_path)
    
    try:
        results = filter.comprehensive_filter_optimized(data_path, **kwargs)
        
        # Save results
        output_dir = Path(workspace_path) / "analysis_results"
        output_dir.mkdir(exist_ok=True)
        
        for name, df in results.items():
            if not df.empty:
                df.to_csv(output_dir / f"{name}.csv", index=False)
        
        # Create summary
        summary = {
            'total_strategies_tested': len(results.get('trade_count_filter', [])),
            'profitable_after_costs': len(results.get('after_costs', [])),
            'uncorrelated_selected': len(results.get('uncorrelated', [])),
            'final_strategies': len(results.get('final_selection', []))
        }
        
        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to {output_dir}")
        
        # Show final selection
        if not results['final_selection'].empty:
            print("\n=== Final Selected Strategies ===")
            print(results['final_selection'].to_string(index=False))
        
        return results
        
    finally:
        filter.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python strategy_filter_optimized.py <workspace_path> <data_path>")
        sys.exit(1)
    
    run_optimized_filter(
        workspace_path=sys.argv[1],
        data_path=sys.argv[2],
        min_trades=50,  # Higher for 10k bars
        commission_bps=1.0,
        slippage_bps=1.0
    )