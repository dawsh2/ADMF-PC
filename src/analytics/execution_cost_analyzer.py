"""
Execution cost analysis using proper execution models from src/execution.
"""
import duckdb
import pandas as pd
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

# Import execution models
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.execution.synchronous.models.commission import (
    ZeroCommissionModel,
    PerShareCommissionModel,
    PercentageCommissionModel,
    TieredCommissionModel
)
from src.execution.synchronous.models.slippage import (
    PercentageSlippageModel,
    FixedSlippageModel,
    ZeroSlippageModel
)


class ExecutionCostAnalyzer:
    """Analyze strategy performance with realistic execution costs."""
    
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.con = duckdb.connect(str(self.workspace_path / "analytics.duckdb"))
        
    def analyze_with_broker_costs(self, data_path: str, broker: str = "interactive_brokers",
                                 shares_per_trade: int = 1) -> pd.DataFrame:
        """
        Analyze strategies with specific broker cost models.
        
        Args:
            data_path: Path to market data
            broker: Broker name for cost model
            shares_per_trade: Number of shares per trade (default 1 for simple math)
        """
        # Get broker-specific models
        commission_model, slippage_model = self._get_broker_models(broker)
        
        # Query strategy signals with prices
        query = f"""
        WITH signals_with_prices AS (
            SELECT 
                s.strat,
                s.val as signal,
                m1.close as entry_price,
                m2.close as exit_price,
                -- Fixed shares per trade
                {shares_per_trade} as shares,
                -- Raw return
                CASE 
                    WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close
                    WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close
                END as return_pct
            FROM read_parquet('{self.workspace_path}/traces/*/signals/*/*.parquet') s
            JOIN read_parquet('{data_path}') m1 ON s.idx = m1.bar_index
            JOIN read_parquet('{data_path}') m2 ON s.idx + 1 = m2.bar_index
            WHERE s.val != 0
        ),
        strategy_summary AS (
            SELECT 
                strat,
                COUNT(*) as trades,
                AVG(entry_price) as avg_price,
                AVG(shares) as avg_shares,
                AVG(return_pct) as avg_return_pct,
                SUM(return_pct) as total_return_pct
            FROM signals_with_prices
            GROUP BY strat
        )
        SELECT * FROM strategy_summary
        ORDER BY trades DESC
        """
        
        df = self.con.execute(query).df()
        
        # Calculate costs for each strategy
        results = []
        for _, row in df.iterrows():
            # Commission cost (entry + exit)
            avg_trade_value = row['avg_price'] * row['avg_shares']
            
            # Entry commission
            entry_comm = self._calculate_commission(
                commission_model, row['avg_shares'], row['avg_price']
            )
            
            # Exit commission (assume same shares)
            exit_price = row['avg_price'] * (1 + row['avg_return_pct'])
            exit_comm = self._calculate_commission(
                commission_model, row['avg_shares'], exit_price
            )
            
            # Slippage cost (entry + exit)
            entry_slip = self._calculate_slippage(
                slippage_model, row['avg_price'], row['avg_shares']
            )
            exit_slip = self._calculate_slippage(
                slippage_model, exit_price, row['avg_shares']
            )
            
            # Total costs
            total_commission = (entry_comm + exit_comm) * row['trades']
            total_slippage = (entry_slip + exit_slip) * row['trades']
            total_costs = total_commission + total_slippage
            
            # Cost per trade as percentage
            cost_per_trade_pct = (entry_comm + exit_comm + entry_slip + exit_slip) / avg_trade_value * 100
            
            # Net returns (dollar basis)
            gross_pnl = row['avg_return_pct'] * avg_trade_value * row['trades']
            net_pnl = gross_pnl - total_costs
            net_return_per_trade = (gross_pnl - total_costs) / row['trades']
            
            results.append({
                'strategy': row['strat'],
                'trades': row['trades'],
                'avg_shares': row['avg_shares'],
                'avg_price': row['avg_price'],
                'gross_return_pct': row['avg_return_pct'] * 100,
                'commission_per_trade': entry_comm + exit_comm,
                'slippage_per_trade': entry_slip + exit_slip,
                'total_cost_per_trade': entry_comm + exit_comm + entry_slip + exit_slip,
                'cost_per_trade_pct': cost_per_trade_pct,
                'net_return_per_trade': net_return_per_trade,
                'net_return_pct': net_return_per_trade / avg_trade_value * 100,
                'total_gross_pnl': gross_pnl,
                'total_costs': total_costs,
                'total_net_pnl': net_pnl
            })
        
        return pd.DataFrame(results)
    
    def _get_broker_models(self, broker: str):
        """Get commission and slippage models for specific broker."""
        
        if broker == "interactive_brokers":
            # IB Pro pricing: $0.005/share, $1 min
            commission = PerShareCommissionModel(
                rate_per_share=0.005,
                minimum_commission=1.0,
                maximum_commission=10.0
            )
            # Assume 1bp slippage for liquid stocks
            slippage = PercentageSlippageModel(
                base_slippage_pct=0.0001,  # 1bp
                volatility_multiplier=0,    # Ignore volatility for now
                volume_impact_factor=0      # Ignore volume impact
            )
            
        elif broker == "alpaca":
            # Alpaca: Zero commission
            commission = ZeroCommissionModel()
            # Still have slippage
            slippage = PercentageSlippageModel(base_slippage_pct=0.0001)
            
        elif broker == "robinhood":
            # Robinhood: Zero commission
            commission = ZeroCommissionModel()
            # Higher slippage due to payment for order flow
            slippage = PercentageSlippageModel(base_slippage_pct=0.0002)  # 2bp
            
        elif broker == "td_ameritrade":
            # TD Ameritrade: Zero commission
            commission = ZeroCommissionModel()
            slippage = PercentageSlippageModel(base_slippage_pct=0.00015)
            
        else:
            # Default: percentage model
            commission = PercentageCommissionModel(
                commission_percent=0.001,  # 0.1%
                minimum_commission=1.0
            )
            slippage = PercentageSlippageModel(base_slippage_pct=0.0001)
            
        return commission, slippage
    
    def _calculate_commission(self, model, shares: float, price: float) -> float:
        """Calculate commission for a trade."""
        if isinstance(model, PerShareCommissionModel):
            commission = shares * model.rate_per_share
            commission = max(commission, model.minimum_commission)
            commission = min(commission, model.maximum_commission)
            return commission
        elif isinstance(model, PercentageCommissionModel):
            trade_value = shares * price
            commission = trade_value * model.commission_percent
            return max(commission, model.minimum_commission)
        elif isinstance(model, ZeroCommissionModel):
            return 0.0
        else:
            return 0.0
    
    def _calculate_slippage(self, model, price: float, shares: float) -> float:
        """Calculate slippage cost in dollars."""
        if isinstance(model, PercentageSlippageModel):
            # Slippage as price impact
            slippage_pct = float(model.base_slippage_pct)
            return price * shares * slippage_pct
        elif isinstance(model, FixedSlippageModel):
            return model.slippage_amount * shares
        else:
            return 0.0
    
    def compare_brokers(self, data_path: str, strategies: List[str],
                       shares_per_trade: int = 1) -> pd.DataFrame:
        """Compare costs across different brokers for specific strategies."""
        brokers = ["interactive_brokers", "alpaca", "robinhood", "td_ameritrade"]
        
        results = []
        for broker in brokers:
            broker_df = self.analyze_with_broker_costs(data_path, broker, shares_per_trade)
            broker_df = broker_df[broker_df['strategy'].isin(strategies)]
            broker_df['broker'] = broker
            results.append(broker_df)
        
        comparison = pd.concat(results)
        
        # Pivot for easy comparison
        pivot = comparison.pivot_table(
            index='strategy',
            columns='broker',
            values=['commission_per_trade', 'total_cost_per_trade', 'net_return_pct']
        )
        
        return pivot
    
    def find_profitable_with_costs(self, data_path: str, broker: str,
                                  shares_per_trade: int = 1,
                                  min_trades: int = 50) -> pd.DataFrame:
        """Find strategies that remain profitable after realistic costs."""
        
        df = self.analyze_with_broker_costs(data_path, broker, shares_per_trade)
        
        # Filter profitable strategies
        profitable = df[
            (df['trades'] >= min_trades) & 
            (df['net_return_pct'] > 0)
        ].sort_values('total_net_pnl', ascending=False)
        
        return profitable
    
    def close(self):
        """Close database connection."""
        self.con.close()


def analyze_with_real_costs(workspace_path: str, data_path: str):
    """Run analysis with real broker costs."""
    
    analyzer = ExecutionCostAnalyzer(workspace_path)
    
    try:
        print("=== Strategy Analysis with Real Execution Costs ===\n")
        
        # Analyze with 1 share per trade for simple math
        print("\n--- Trading 1 Share Per Trade ---")
        print("(SPY at ~$520 means ~$520 per trade)\n")
        
        profitable = analyzer.find_profitable_with_costs(
            data_path, 
            broker="interactive_brokers",
            shares_per_trade=1,
            min_trades=100
        )
            
        if len(profitable) > 0:
            print(f"Profitable strategies: {len(profitable)}")
            print("\nTop 5:")
            cols = ['strategy', 'trades', 'gross_return_pct', 'commission_per_trade', 
                   'cost_per_trade_pct', 'net_return_pct', 'total_net_pnl']
            print(profitable[cols].head().to_string(index=False))
        else:
            print("No profitable strategies with these costs!")
        
        # Compare brokers for top strategies
        if len(profitable) > 0:
            print("\n\n=== Broker Comparison for Top Strategies ===")
            top_strategies = profitable.head(5)['strategy'].tolist()
            comparison = analyzer.compare_brokers(data_path, top_strategies)
            print(comparison)
            
    finally:
        analyzer.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python execution_cost_analyzer.py <workspace_path> <data_path>")
        sys.exit(1)
    
    analyze_with_real_costs(sys.argv[1], sys.argv[2])