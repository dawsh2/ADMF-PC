# Step 10.2: Multi-Asset Support

**Status**: Intermediate Complexity Step
**Complexity**: High
**Prerequisites**: [Step 10.1: Advanced Analytics](step-10.1-advanced-analytics.md) completed
**Architecture Ref**: [Portfolio Architecture](../architecture/portfolio-architecture.md)

## 🎯 Objective

Implement comprehensive multi-asset portfolio management:
- Cross-asset correlation and cointegration analysis
- Portfolio optimization (Mean-Variance, Black-Litterman, Risk Parity)
- Dynamic asset allocation and rebalancing
- Currency exposure management
- Sector and factor exposure tracking

## 📋 Required Reading

Before starting:
1. [Modern Portfolio Theory](../references/modern-portfolio-theory.md)
2. [Multi-Asset Strategies](../references/multi-asset-strategies.md)
3. [Currency Risk Management](../references/currency-risk.md)

## 🏗️ Implementation Tasks

### 1. Multi-Asset Data Management

```python
# src/portfolio/multi_asset_data.py
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

@dataclass
class AssetUniverse:
    """Defines the investable universe"""
    assets: List[str]
    asset_classes: Dict[str, str]  # asset -> class mapping
    currencies: Dict[str, str]  # asset -> currency mapping
    sectors: Dict[str, str]  # asset -> sector mapping
    
    # Constraints
    min_weights: Dict[str, float]
    max_weights: Dict[str, float]
    
    # Reference data
    benchmarks: Dict[str, str]  # asset class -> benchmark
    risk_free_rates: Dict[str, float]  # currency -> rate
    
    def get_assets_by_class(self, asset_class: str) -> List[str]:
        """Get all assets in a given class"""
        return [asset for asset, cls in self.asset_classes.items() 
                if cls == asset_class]
    
    def get_currency_groups(self) -> Dict[str, List[str]]:
        """Group assets by currency"""
        groups = {}
        for asset, currency in self.currencies.items():
            if currency not in groups:
                groups[currency] = []
            groups[currency].append(asset)
        return groups

class MultiAssetDataManager:
    """
    Manages data for multiple assets with different frequencies.
    Handles alignment, missing data, and corporate actions.
    """
    
    def __init__(self, universe: AssetUniverse):
        self.universe = universe
        self.data_cache = {}
        self.aligned_data = None
        self.logger = ComponentLogger("MultiAssetDataManager", "portfolio")
        
        # Data quality tracking
        self.data_quality_scores = {}
        self.missing_data_stats = {}
    
    async def load_asset_data(self, assets: List[str], 
                            start_date: datetime,
                            end_date: datetime,
                            frequency: str = 'D') -> Dict[str, pd.DataFrame]:
        """Load data for multiple assets asynchronously"""
        tasks = []
        
        async with asyncio.TaskGroup() as tg:
            for asset in assets:
                task = tg.create_task(
                    self._load_single_asset(asset, start_date, end_date, frequency)
                )
                tasks.append((asset, task))
        
        # Collect results
        asset_data = {}
        for asset, task in tasks:
            try:
                data = await task
                asset_data[asset] = data
                self.data_cache[asset] = data
            except Exception as e:
                self.logger.error(f"Failed to load data for {asset}: {e}")
        
        return asset_data
    
    async def _load_single_asset(self, asset: str,
                               start_date: datetime,
                               end_date: datetime,
                               frequency: str) -> pd.DataFrame:
        """Load data for a single asset"""
        # This would connect to your data source
        # For now, simulate with generated data
        dates = pd.date_range(start_date, end_date, freq=frequency)
        
        # Simulate OHLCV data
        np.random.seed(hash(asset) % 2**32)
        returns = np.random.normal(0.0001, 0.02, len(dates))
        prices = 100 * np.cumprod(1 + returns)
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
            'high': prices * (1 + np.random.uniform(0, 0.02, len(dates))),
            'low': prices * (1 + np.random.uniform(-0.02, 0, len(dates))),
            'close': prices,
            'volume': np.random.uniform(1e6, 1e7, len(dates)),
            'returns': returns
        }, index=dates)
        
        # Add asset-specific fields
        data['asset'] = asset
        data['currency'] = self.universe.currencies.get(asset, 'USD')
        data['asset_class'] = self.universe.asset_classes.get(asset, 'equity')
        
        return data
    
    def align_data(self, asset_data: Dict[str, pd.DataFrame],
                  method: str = 'intersection') -> pd.DataFrame:
        """Align multi-asset data to common timestamps"""
        if not asset_data:
            return pd.DataFrame()
        
        # Get all timestamps
        all_timestamps = set()
        for data in asset_data.values():
            all_timestamps.update(data.index)
        
        if method == 'intersection':
            # Use only common timestamps
            common_timestamps = set(asset_data[list(asset_data.keys())[0]].index)
            for data in asset_data.values():
                common_timestamps = common_timestamps.intersection(data.index)
            aligned_index = sorted(common_timestamps)
        
        elif method == 'union':
            # Use all timestamps
            aligned_index = sorted(all_timestamps)
        
        else:
            raise ValueError(f"Unknown alignment method: {method}")
        
        # Create aligned dataframe
        aligned_data = {}
        
        for asset, data in asset_data.items():
            # Reindex to aligned timestamps
            asset_aligned = data.reindex(aligned_index)
            
            # Handle missing data
            asset_aligned = self._handle_missing_data(asset_aligned, asset)
            
            # Store in multi-level structure
            for col in ['open', 'high', 'low', 'close', 'volume', 'returns']:
                if col in asset_aligned.columns:
                    aligned_data[(col, asset)] = asset_aligned[col]
        
        # Create multi-level column dataframe
        self.aligned_data = pd.DataFrame(aligned_data)
        self.aligned_data.columns = pd.MultiIndex.from_tuples(
            self.aligned_data.columns
        )
        
        # Calculate data quality
        self._calculate_data_quality()
        
        return self.aligned_data
    
    def _handle_missing_data(self, data: pd.DataFrame, 
                           asset: str) -> pd.DataFrame:
        """Handle missing data with appropriate methods"""
        # Track missing data
        missing_count = data['close'].isna().sum()
        total_count = len(data)
        
        self.missing_data_stats[asset] = {
            'missing_count': missing_count,
            'missing_pct': missing_count / total_count * 100,
            'longest_gap': self._find_longest_gap(data['close'])
        }
        
        # Forward fill for prices (last known price)
        for col in ['open', 'high', 'low', 'close']:
            if col in data.columns:
                data[col] = data[col].fillna(method='ffill')
        
        # Zero fill for volume and returns
        data['volume'] = data['volume'].fillna(0)
        data['returns'] = data['returns'].fillna(0)
        
        return data
    
    def calculate_correlation_matrix(self, 
                                   window: Optional[int] = None,
                                   method: str = 'pearson') -> pd.DataFrame:
        """Calculate correlation matrix for all assets"""
        if self.aligned_data is None:
            raise ValueError("No aligned data available")
        
        returns = self.aligned_data['returns']
        
        if window:
            # Rolling correlation
            return returns.rolling(window).corr()
        else:
            # Full period correlation
            if method == 'pearson':
                return returns.corr()
            elif method == 'spearman':
                return returns.corr(method='spearman')
            elif method == 'kendall':
                return returns.corr(method='kendall')
            else:
                raise ValueError(f"Unknown correlation method: {method}")
```

### 2. Portfolio Optimization

```python
# src/portfolio/optimization.py
import cvxpy as cp
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

class PortfolioOptimizer:
    """
    Multi-asset portfolio optimization with various methods.
    Handles constraints and transaction costs.
    """
    
    def __init__(self, universe: AssetUniverse):
        self.universe = universe
        self.optimization_history = []
        self.logger = ComponentLogger("PortfolioOptimizer", "portfolio")
    
    def optimize(self, expected_returns: pd.Series,
                covariance_matrix: pd.DataFrame,
                method: str = 'mean_variance',
                constraints: Optional[Dict] = None,
                **kwargs) -> Dict[str, float]:
        """Main optimization interface"""
        
        # Validate inputs
        self._validate_inputs(expected_returns, covariance_matrix)
        
        # Select optimization method
        if method == 'mean_variance':
            weights = self._mean_variance_optimize(
                expected_returns, covariance_matrix, constraints, **kwargs
            )
        elif method == 'min_variance':
            weights = self._min_variance_optimize(
                covariance_matrix, constraints
            )
        elif method == 'risk_parity':
            weights = self._risk_parity_optimize(
                covariance_matrix, constraints
            )
        elif method == 'black_litterman':
            weights = self._black_litterman_optimize(
                expected_returns, covariance_matrix, constraints, **kwargs
            )
        elif method == 'max_diversification':
            weights = self._max_diversification_optimize(
                covariance_matrix, constraints
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Post-process weights
        weights = self._post_process_weights(weights, constraints)
        
        # Store optimization result
        self._store_optimization_result(
            weights, expected_returns, covariance_matrix, method
        )
        
        return weights
    
    def _mean_variance_optimize(self, expected_returns: pd.Series,
                               covariance_matrix: pd.DataFrame,
                               constraints: Optional[Dict],
                               target_return: Optional[float] = None,
                               risk_aversion: float = 1.0) -> Dict[str, float]:
        """Classic Markowitz mean-variance optimization"""
        n_assets = len(expected_returns)
        
        # Setup optimization variables
        weights = cp.Variable(n_assets)
        returns = expected_returns.values
        cov_matrix = covariance_matrix.values
        
        # Portfolio return and risk
        portfolio_return = returns.T @ weights
        portfolio_risk = cp.quad_form(weights, cov_matrix)
        
        # Objective function
        if target_return is not None:
            # Minimize risk for target return
            objective = cp.Minimize(portfolio_risk)
            constraints_list = [
                cp.sum(weights) == 1,
                portfolio_return >= target_return
            ]
        else:
            # Maximize utility (return - risk_aversion * risk)
            objective = cp.Maximize(
                portfolio_return - risk_aversion * portfolio_risk
            )
            constraints_list = [cp.sum(weights) == 1]
        
        # Add user constraints
        constraints_list.extend(
            self._convert_constraints_to_cvxpy(weights, constraints, expected_returns.index)
        )
        
        # Solve
        problem = cp.Problem(objective, constraints_list)
        problem.solve()
        
        if problem.status != 'optimal':
            self.logger.warning(f"Optimization status: {problem.status}")
        
        # Convert to dictionary
        weight_dict = dict(zip(expected_returns.index, weights.value))
        
        return weight_dict
    
    def _risk_parity_optimize(self, covariance_matrix: pd.DataFrame,
                            constraints: Optional[Dict]) -> Dict[str, float]:
        """Risk parity optimization - equal risk contribution"""
        n_assets = len(covariance_matrix)
        
        # Initial guess - equal weights
        x0 = np.ones(n_assets) / n_assets
        
        # Objective: minimize difference from equal risk contribution
        def objective(weights):
            portfolio_vol = np.sqrt(weights @ covariance_matrix @ weights)
            marginal_contrib = covariance_matrix @ weights
            contrib = weights * marginal_contrib / portfolio_vol
            
            # Target equal contribution
            target_contrib = 1.0 / n_assets
            
            # Sum of squared differences from target
            return np.sum((contrib - target_contrib) ** 2)
        
        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Sum to 1
        ]
        
        # Bounds (all weights positive)
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        # Convert to dictionary
        weight_dict = dict(zip(covariance_matrix.index, result.x))
        
        return weight_dict
    
    def _black_litterman_optimize(self, market_returns: pd.Series,
                                covariance_matrix: pd.DataFrame,
                                constraints: Optional[Dict],
                                views: Optional[Dict] = None,
                                tau: float = 0.05) -> Dict[str, float]:
        """Black-Litterman optimization with views"""
        
        # Market capitalization weights (could be passed in)
        # For now, use equal weights as market weights
        market_weights = pd.Series(
            1.0 / len(market_returns), 
            index=market_returns.index
        )
        
        # Equilibrium returns
        lam = self._calculate_risk_aversion(market_returns, covariance_matrix)
        equilibrium_returns = lam * covariance_matrix @ market_weights
        
        if views is None:
            # No views - return market weights
            return market_weights.to_dict()
        
        # Process views
        P, Q, omega = self._process_views(views, covariance_matrix)
        
        # Black-Litterman formula
        tau_sigma = tau * covariance_matrix
        
        # Posterior covariance
        posterior_cov = np.linalg.inv(
            np.linalg.inv(tau_sigma) + P.T @ np.linalg.inv(omega) @ P
        )
        
        # Posterior returns
        posterior_returns = posterior_cov @ (
            np.linalg.inv(tau_sigma) @ equilibrium_returns + 
            P.T @ np.linalg.inv(omega) @ Q
        )
        
        # Convert to Series
        posterior_returns = pd.Series(posterior_returns, index=market_returns.index)
        
        # Optimize with posterior estimates
        return self._mean_variance_optimize(
            posterior_returns, covariance_matrix, constraints
        )
    
    def _calculate_efficient_frontier(self, expected_returns: pd.Series,
                                    covariance_matrix: pd.DataFrame,
                                    n_portfolios: int = 100) -> pd.DataFrame:
        """Calculate the efficient frontier"""
        min_ret = expected_returns.min()
        max_ret = expected_returns.max()
        
        target_returns = np.linspace(min_ret, max_ret, n_portfolios)
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            try:
                weights = self._mean_variance_optimize(
                    expected_returns, 
                    covariance_matrix,
                    constraints=None,
                    target_return=target_return
                )
                
                # Calculate portfolio metrics
                portfolio_return = sum(
                    weights[asset] * expected_returns[asset] 
                    for asset in weights
                )
                
                portfolio_risk = np.sqrt(
                    sum(
                        weights[asset1] * weights[asset2] * 
                        covariance_matrix.loc[asset1, asset2]
                        for asset1 in weights
                        for asset2 in weights
                    )
                )
                
                efficient_portfolios.append({
                    'return': portfolio_return,
                    'risk': portfolio_risk,
                    'sharpe': portfolio_return / portfolio_risk,
                    'weights': weights
                })
                
            except Exception as e:
                self.logger.debug(f"Failed for target return {target_return}: {e}")
        
        return pd.DataFrame(efficient_portfolios)
```

### 3. Dynamic Rebalancing

```python
# src/portfolio/rebalancing.py
class DynamicRebalancer:
    """
    Handles portfolio rebalancing with transaction costs.
    Implements various rebalancing strategies.
    """
    
    def __init__(self, universe: AssetUniverse,
                 transaction_costs: Dict[str, float]):
        self.universe = universe
        self.transaction_costs = transaction_costs
        self.rebalancing_history = []
        self.logger = ComponentLogger("DynamicRebalancer", "portfolio")
    
    def should_rebalance(self, current_weights: Dict[str, float],
                        target_weights: Dict[str, float],
                        method: str = 'threshold',
                        **kwargs) -> bool:
        """Determine if rebalancing is needed"""
        
        if method == 'threshold':
            return self._threshold_rebalancing(
                current_weights, target_weights, **kwargs
            )
        elif method == 'periodic':
            return self._periodic_rebalancing(**kwargs)
        elif method == 'range':
            return self._range_rebalancing(
                current_weights, target_weights, **kwargs
            )
        elif method == 'cost_benefit':
            return self._cost_benefit_rebalancing(
                current_weights, target_weights, **kwargs
            )
        else:
            raise ValueError(f"Unknown rebalancing method: {method}")
    
    def _threshold_rebalancing(self, current_weights: Dict[str, float],
                             target_weights: Dict[str, float],
                             threshold: float = 0.05) -> bool:
        """Rebalance if any weight deviates by more than threshold"""
        for asset in target_weights:
            current = current_weights.get(asset, 0)
            target = target_weights[asset]
            
            if abs(current - target) > threshold:
                return True
        
        return False
    
    def _cost_benefit_rebalancing(self, current_weights: Dict[str, float],
                                target_weights: Dict[str, float],
                                expected_returns: pd.Series,
                                horizon_days: int = 30) -> bool:
        """Rebalance if expected benefit exceeds costs"""
        # Calculate rebalancing trades
        trades = self.calculate_rebalancing_trades(
            current_weights, target_weights, portfolio_value=1.0
        )
        
        # Calculate transaction costs
        total_cost = 0
        for asset, trade in trades.items():
            cost_rate = self.transaction_costs.get(asset, 0.001)
            total_cost += abs(trade) * cost_rate
        
        # Calculate expected benefit
        expected_benefit = 0
        for asset in target_weights:
            weight_diff = target_weights[asset] - current_weights.get(asset, 0)
            expected_return = expected_returns.get(asset, 0)
            
            # Benefit from moving to target weight
            expected_benefit += weight_diff * expected_return * horizon_days / 252
        
        return expected_benefit > total_cost * 2  # Require 2x benefit
    
    def calculate_rebalancing_trades(self, 
                                   current_weights: Dict[str, float],
                                   target_weights: Dict[str, float],
                                   portfolio_value: float,
                                   constraints: Optional[Dict] = None) -> Dict[str, float]:
        """Calculate optimal rebalancing trades"""
        
        # Basic rebalancing
        trades = {}
        
        for asset in set(current_weights.keys()) | set(target_weights.keys()):
            current = current_weights.get(asset, 0)
            target = target_weights.get(asset, 0)
            
            # Trade amount in portfolio value terms
            trade = (target - current) * portfolio_value
            
            if abs(trade) > 0.0001:  # Minimum trade size
                trades[asset] = trade
        
        # Apply constraints
        if constraints:
            trades = self._apply_trade_constraints(trades, constraints)
        
        # Optimize for transaction costs
        trades = self._optimize_trades_for_costs(trades, current_weights, target_weights)
        
        return trades
    
    def _optimize_trades_for_costs(self, trades: Dict[str, float],
                                 current_weights: Dict[str, float],
                                 target_weights: Dict[str, float]) -> Dict[str, float]:
        """Optimize trades to minimize transaction costs while reaching targets"""
        
        # For small deviations, consider not trading
        optimized_trades = {}
        
        for asset, trade in trades.items():
            cost_rate = self.transaction_costs.get(asset, 0.001)
            trade_cost = abs(trade) * cost_rate
            
            # Weight deviation
            current = current_weights.get(asset, 0)
            target = target_weights.get(asset, 0)
            deviation = abs(current - target)
            
            # Don't trade if cost exceeds benefit threshold
            if trade_cost > deviation * 0.1:  # 10% of deviation
                continue
            
            optimized_trades[asset] = trade
        
        return optimized_trades
```

### 4. Currency Management

```python
# src/portfolio/currency_management.py
class CurrencyManager:
    """
    Manages currency exposure in multi-asset portfolios.
    Handles hedging and currency overlay strategies.
    """
    
    def __init__(self, base_currency: str = 'USD'):
        self.base_currency = base_currency
        self.fx_rates = {}
        self.hedge_ratios = {}
        self.logger = ComponentLogger("CurrencyManager", "portfolio")
    
    def calculate_currency_exposure(self, 
                                  portfolio_weights: Dict[str, float],
                                  asset_currencies: Dict[str, str]) -> Dict[str, float]:
        """Calculate portfolio exposure by currency"""
        currency_exposure = {}
        
        for asset, weight in portfolio_weights.items():
            currency = asset_currencies.get(asset, self.base_currency)
            
            if currency not in currency_exposure:
                currency_exposure[currency] = 0
            
            currency_exposure[currency] += weight
        
        return currency_exposure
    
    def calculate_hedged_returns(self, 
                               asset_returns: pd.DataFrame,
                               fx_returns: pd.DataFrame,
                               hedge_ratios: Dict[str, float]) -> pd.DataFrame:
        """Calculate returns with currency hedging"""
        hedged_returns = asset_returns.copy()
        
        for asset in asset_returns.columns:
            if asset in self.asset_currencies:
                currency = self.asset_currencies[asset]
                
                if currency != self.base_currency and currency in fx_returns.columns:
                    # Get hedge ratio
                    hedge_ratio = hedge_ratios.get(currency, 0)
                    
                    # Hedged return = asset return - hedge_ratio * fx return
                    hedged_returns[asset] = (
                        asset_returns[asset] - 
                        hedge_ratio * fx_returns[currency]
                    )
        
        return hedged_returns
    
    def optimize_currency_hedge(self, 
                              portfolio_weights: Dict[str, float],
                              asset_currencies: Dict[str, str],
                              fx_volatilities: Dict[str, float],
                              fx_correlations: pd.DataFrame,
                              method: str = 'min_variance') -> Dict[str, float]:
        """Optimize currency hedge ratios"""
        
        # Get currency exposures
        currency_exposure = self.calculate_currency_exposure(
            portfolio_weights, asset_currencies
        )
        
        # Remove base currency
        foreign_currencies = {
            curr: exp for curr, exp in currency_exposure.items() 
            if curr != self.base_currency
        }
        
        if method == 'min_variance':
            # Minimize portfolio variance from currency
            hedge_ratios = self._min_variance_hedge(
                foreign_currencies, fx_volatilities, fx_correlations
            )
        elif method == 'black':
            # Black's universal hedging constant
            hedge_ratios = {curr: 0.77 for curr in foreign_currencies}
        elif method == 'full':
            # Full hedge
            hedge_ratios = {curr: 1.0 for curr in foreign_currencies}
        elif method == 'none':
            # No hedge
            hedge_ratios = {curr: 0.0 for curr in foreign_currencies}
        else:
            raise ValueError(f"Unknown hedge method: {method}")
        
        self.hedge_ratios = hedge_ratios
        return hedge_ratios
```

### 5. Multi-Asset Strategy Coordination

```python
# src/portfolio/multi_asset_strategy.py
class MultiAssetStrategy:
    """
    Coordinates strategies across multiple assets.
    Handles portfolio-level decisions and risk management.
    """
    
    def __init__(self, universe: AssetUniverse,
                 asset_strategies: Dict[str, TradingStrategy]):
        self.universe = universe
        self.asset_strategies = asset_strategies
        self.portfolio_optimizer = PortfolioOptimizer(universe)
        self.rebalancer = DynamicRebalancer(universe, transaction_costs={})
        self.currency_manager = CurrencyManager()
        self.logger = ComponentLogger("MultiAssetStrategy", "portfolio")
        
        # Portfolio state
        self.current_weights = {}
        self.target_weights = {}
        self.positions = {}
    
    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """Generate signals for all assets"""
        signals = {}
        
        # Generate individual asset signals
        for asset, strategy in self.asset_strategies.items():
            if asset in market_data:
                try:
                    signal = strategy.generate_signal(market_data[asset])
                    signals[asset] = signal
                except Exception as e:
                    self.logger.error(f"Signal generation failed for {asset}: {e}")
        
        # Apply portfolio-level filters
        signals = self._apply_portfolio_constraints(signals, market_data)
        
        # Correlation-based signal adjustment
        signals = self._adjust_signals_for_correlation(signals, market_data)
        
        return signals
    
    def _apply_portfolio_constraints(self, signals: Dict[str, Signal],
                                   market_data: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """Apply portfolio-level constraints to signals"""
        
        # Calculate current exposure
        total_long_exposure = sum(
            1 for signal in signals.values() 
            if signal.direction == SignalDirection.LONG
        )
        total_short_exposure = sum(
            1 for signal in signals.values() 
            if signal.direction == SignalDirection.SHORT
        )
        
        # Apply exposure limits
        max_long_positions = 10
        max_short_positions = 5
        
        if total_long_exposure > max_long_positions:
            # Keep only strongest long signals
            long_signals = [
                (asset, signal) for asset, signal in signals.items()
                if signal.direction == SignalDirection.LONG
            ]
            long_signals.sort(key=lambda x: x[1].strength, reverse=True)
            
            # Cancel weaker signals
            for asset, signal in long_signals[max_long_positions:]:
                signals[asset] = Signal(
                    direction=SignalDirection.FLAT,
                    strength=0,
                    metadata={'reason': 'portfolio_exposure_limit'}
                )
        
        return signals
    
    def _adjust_signals_for_correlation(self, signals: Dict[str, Signal],
                                      market_data: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """Adjust signals based on correlation"""
        
        # Calculate recent correlations
        returns_data = {}
        for asset, data in market_data.items():
            if 'returns' in data.columns:
                returns_data[asset] = data['returns'].tail(60)
        
        if len(returns_data) < 2:
            return signals
        
        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr()
        
        # Reduce signals for highly correlated assets
        adjusted_signals = signals.copy()
        
        for asset1 in signals:
            for asset2 in signals:
                if asset1 != asset2 and asset1 in correlation_matrix and asset2 in correlation_matrix:
                    correlation = correlation_matrix.loc[asset1, asset2]
                    
                    # High correlation threshold
                    if abs(correlation) > 0.8:
                        # Same direction signals
                        if signals[asset1].direction == signals[asset2].direction:
                            # Reduce strength of weaker signal
                            if signals[asset1].strength < signals[asset2].strength:
                                adjusted_signals[asset1].strength *= (1 - abs(correlation))
        
        return adjusted_signals
    
    def optimize_portfolio(self, signals: Dict[str, Signal],
                         market_data: Dict[str, pd.DataFrame],
                         risk_budget: float = 0.02) -> Dict[str, float]:
        """Optimize portfolio weights based on signals"""
        
        # Convert signals to expected returns
        expected_returns = self._signals_to_expected_returns(signals, market_data)
        
        # Calculate covariance matrix
        returns_data = self._extract_returns(market_data)
        
        # Use shrinkage estimator for more stable covariance
        lw = LedoitWolf()
        covariance_matrix = pd.DataFrame(
            lw.fit(returns_data.T).covariance_,
            index=returns_data.index,
            columns=returns_data.index
        )
        
        # Add constraints
        constraints = {
            'long_only': True,
            'max_weight': 0.2,
            'min_weight': 0,
            'sector_limits': {'technology': 0.3, 'finance': 0.25},
            'risk_budget': risk_budget
        }
        
        # Optimize
        if len(expected_returns) > 0:
            target_weights = self.portfolio_optimizer.optimize(
                expected_returns,
                covariance_matrix,
                method='mean_variance',
                constraints=constraints,
                risk_aversion=2.0
            )
        else:
            target_weights = {}
        
        self.target_weights = target_weights
        return target_weights
    
    def execute_portfolio_trades(self, target_weights: Dict[str, float],
                               current_prices: Dict[str, float],
                               portfolio_value: float) -> List[Order]:
        """Generate orders to reach target weights"""
        
        # Check if rebalancing needed
        if not self.rebalancer.should_rebalance(
            self.current_weights, 
            target_weights,
            method='cost_benefit'
        ):
            return []
        
        # Calculate required trades
        trades = self.rebalancer.calculate_rebalancing_trades(
            self.current_weights,
            target_weights,
            portfolio_value
        )
        
        # Generate orders
        orders = []
        
        for asset, trade_value in trades.items():
            if asset in current_prices:
                # Calculate shares
                shares = int(trade_value / current_prices[asset])
                
                if shares != 0:
                    order = Order(
                        asset=asset,
                        quantity=shares,
                        order_type=OrderType.MARKET,
                        side=OrderSide.BUY if shares > 0 else OrderSide.SELL,
                        metadata={
                            'strategy': 'multi_asset_rebalance',
                            'target_weight': target_weights.get(asset, 0),
                            'current_weight': self.current_weights.get(asset, 0)
                        }
                    )
                    orders.append(order)
        
        return orders
```

## 🧪 Testing Requirements

### Unit Tests

Create `tests/unit/test_step10_2_multi_asset.py`:

```python
class TestMultiAssetDataManager:
    """Test multi-asset data management"""
    
    async def test_async_data_loading(self):
        """Test asynchronous multi-asset data loading"""
        universe = create_test_universe(['SPY', 'TLT', 'GLD'])
        manager = MultiAssetDataManager(universe)
        
        # Load data
        asset_data = await manager.load_asset_data(
            ['SPY', 'TLT', 'GLD'],
            datetime(2023, 1, 1),
            datetime(2023, 12, 31)
        )
        
        assert len(asset_data) == 3
        assert all(isinstance(df, pd.DataFrame) for df in asset_data.values())
    
    def test_data_alignment(self):
        """Test multi-asset data alignment"""
        # Create test data with different timestamps
        data1 = pd.DataFrame(
            {'close': [100, 101, 102], 'returns': [0, 0.01, 0.0099]},
            index=pd.date_range('2023-01-01', periods=3)
        )
        
        data2 = pd.DataFrame(
            {'close': [50, 51], 'returns': [0, 0.02]},
            index=pd.date_range('2023-01-02', periods=2)
        )
        
        manager = MultiAssetDataManager(create_test_universe(['A', 'B']))
        aligned = manager.align_data({'A': data1, 'B': data2}, method='intersection')
        
        # Should only have common dates
        assert len(aligned) == 2
        assert ('close', 'A') in aligned.columns
        assert ('close', 'B') in aligned.columns

class TestPortfolioOptimizer:
    """Test portfolio optimization methods"""
    
    def test_mean_variance_optimization(self):
        """Test Markowitz optimization"""
        universe = create_test_universe(['A', 'B', 'C'])
        optimizer = PortfolioOptimizer(universe)
        
        expected_returns = pd.Series([0.10, 0.15, 0.12], index=['A', 'B', 'C'])
        cov_matrix = pd.DataFrame([
            [0.01, 0.002, 0.001],
            [0.002, 0.015, 0.002],
            [0.001, 0.002, 0.01]
        ], index=['A', 'B', 'C'], columns=['A', 'B', 'C'])
        
        weights = optimizer.optimize(
            expected_returns, 
            cov_matrix,
            method='mean_variance',
            constraints={'long_only': True}
        )
        
        # Weights should sum to 1
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        # All weights non-negative
        assert all(w >= 0 for w in weights.values())
    
    def test_risk_parity_optimization(self):
        """Test risk parity allocation"""
        optimizer = PortfolioOptimizer(create_test_universe(['A', 'B']))
        
        # Equal volatility assets should get equal weight
        cov_matrix = pd.DataFrame([
            [0.01, 0],
            [0, 0.01]
        ], index=['A', 'B'], columns=['A', 'B'])
        
        weights = optimizer.optimize(
            expected_returns=None,
            covariance_matrix=cov_matrix,
            method='risk_parity'
        )
        
        assert abs(weights['A'] - 0.5) < 0.01
        assert abs(weights['B'] - 0.5) < 0.01
```

### Integration Tests

Create `tests/integration/test_step10_2_multi_asset_integration.py`:

```python
def test_complete_multi_asset_workflow():
    """Test full multi-asset portfolio workflow"""
    # Setup universe
    universe = AssetUniverse(
        assets=['SPY', 'TLT', 'GLD', 'VNQ', 'EFA'],
        asset_classes={
            'SPY': 'equity', 'TLT': 'bond', 'GLD': 'commodity',
            'VNQ': 'reit', 'EFA': 'intl_equity'
        },
        currencies={
            'SPY': 'USD', 'TLT': 'USD', 'GLD': 'USD',
            'VNQ': 'USD', 'EFA': 'USD'
        },
        sectors={},
        min_weights={asset: 0.05 for asset in ['SPY', 'TLT', 'GLD', 'VNQ', 'EFA']},
        max_weights={asset: 0.40 for asset in ['SPY', 'TLT', 'GLD', 'VNQ', 'EFA']},
        benchmarks={'equity': 'SPX', 'bond': 'AGG'},
        risk_free_rates={'USD': 0.05}
    )
    
    # Load data
    data_manager = MultiAssetDataManager(universe)
    asset_data = asyncio.run(data_manager.load_asset_data(
        universe.assets,
        datetime(2023, 1, 1),
        datetime(2023, 12, 31)
    ))
    
    # Align data
    aligned_data = data_manager.align_data(asset_data)
    
    # Calculate correlations
    correlation_matrix = data_manager.calculate_correlation_matrix()
    
    # Create strategies
    asset_strategies = {
        asset: create_momentum_strategy(asset) 
        for asset in universe.assets
    }
    
    # Multi-asset strategy
    multi_asset = MultiAssetStrategy(universe, asset_strategies)
    
    # Generate signals
    signals = multi_asset.generate_signals(asset_data)
    
    # Optimize portfolio
    target_weights = multi_asset.optimize_portfolio(signals, asset_data)
    
    # Verify results
    assert len(target_weights) > 0
    assert abs(sum(target_weights.values()) - 1.0) < 1e-6
    assert all(0 <= w <= 0.4 for w in target_weights.values())

def test_currency_hedging():
    """Test currency exposure management"""
    # Create multi-currency universe
    universe = AssetUniverse(
        assets=['SPY', 'EFA', 'EWJ', 'FXE'],
        asset_classes={'SPY': 'equity', 'EFA': 'equity', 'EWJ': 'equity', 'FXE': 'currency'},
        currencies={'SPY': 'USD', 'EFA': 'EUR', 'EWJ': 'JPY', 'FXE': 'EUR'},
        sectors={},
        min_weights={},
        max_weights={},
        benchmarks={},
        risk_free_rates={'USD': 0.05, 'EUR': 0.02, 'JPY': 0.0}
    )
    
    # Portfolio weights
    portfolio_weights = {
        'SPY': 0.4,
        'EFA': 0.3,
        'EWJ': 0.2,
        'FXE': 0.1
    }
    
    # Currency manager
    currency_mgr = CurrencyManager(base_currency='USD')
    
    # Calculate exposure
    currency_exposure = currency_mgr.calculate_currency_exposure(
        portfolio_weights, universe.currencies
    )
    
    assert currency_exposure['USD'] == 0.4
    assert currency_exposure['EUR'] == 0.4  # EFA + FXE
    assert currency_exposure['JPY'] == 0.2
    
    # Optimize hedge
    hedge_ratios = currency_mgr.optimize_currency_hedge(
        portfolio_weights,
        universe.currencies,
        fx_volatilities={'EUR': 0.08, 'JPY': 0.10},
        fx_correlations=pd.DataFrame([
            [1.0, 0.3],
            [0.3, 1.0]
        ], index=['EUR', 'JPY'], columns=['EUR', 'JPY']),
        method='min_variance'
    )
    
    assert 'EUR' in hedge_ratios
    assert 'JPY' in hedge_ratios
    assert all(0 <= h <= 1 for h in hedge_ratios.values())
```

### System Tests

Create `tests/system/test_step10_2_production_portfolio.py`:

```python
def test_large_universe_optimization():
    """Test optimization with large asset universe"""
    # Create universe with 100 assets
    assets = [f'ASSET_{i}' for i in range(100)]
    
    universe = AssetUniverse(
        assets=assets,
        asset_classes={asset: 'equity' for asset in assets},
        currencies={asset: 'USD' for asset in assets},
        sectors={asset: f'sector_{i%10}' for i, asset in enumerate(assets)},
        min_weights={},
        max_weights={asset: 0.05 for asset in assets},  # Max 5% per asset
        benchmarks={},
        risk_free_rates={'USD': 0.05}
    )
    
    # Generate random returns and covariance
    np.random.seed(42)
    expected_returns = pd.Series(
        np.random.normal(0.08, 0.02, 100), 
        index=assets
    )
    
    # Generate positive definite covariance matrix
    A = np.random.randn(100, 50)
    cov_matrix = pd.DataFrame(
        A @ A.T / 50 + np.eye(100) * 0.01,
        index=assets,
        columns=assets
    )
    
    # Optimize
    optimizer = PortfolioOptimizer(universe)
    
    start_time = time.time()
    weights = optimizer.optimize(
        expected_returns,
        cov_matrix,
        method='mean_variance',
        constraints={
            'long_only': True,
            'max_weight': 0.05,
            'sector_limits': {f'sector_{i}': 0.15 for i in range(10)}
        }
    )
    optimization_time = time.time() - start_time
    
    # Performance requirement
    assert optimization_time < 5.0  # Should optimize in under 5 seconds
    
    # Validate constraints
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert all(w <= 0.05 for w in weights.values())
    
    # Check sector constraints
    sector_weights = {}
    for asset, weight in weights.items():
        sector = universe.sectors[asset]
        sector_weights[sector] = sector_weights.get(sector, 0) + weight
    
    assert all(w <= 0.15 for w in sector_weights.values())

def test_real_time_rebalancing():
    """Test real-time portfolio rebalancing"""
    # Setup
    universe = create_production_universe()
    multi_asset = MultiAssetStrategy(
        universe,
        asset_strategies=create_asset_strategies(universe)
    )
    
    # Initial portfolio
    multi_asset.current_weights = {
        'SPY': 0.35,
        'TLT': 0.25,
        'GLD': 0.15,
        'VNQ': 0.15,
        'EFA': 0.10
    }
    
    # Simulate market data updates
    for i in range(100):
        # Generate new market data
        market_data = generate_market_update(universe.assets)
        
        # Time the full cycle
        start_time = time.time()
        
        # Generate signals
        signals = multi_asset.generate_signals(market_data)
        
        # Optimize portfolio
        target_weights = multi_asset.optimize_portfolio(signals, market_data)
        
        # Generate rebalancing trades
        orders = multi_asset.execute_portfolio_trades(
            target_weights,
            current_prices={asset: data['close'].iloc[-1] 
                          for asset, data in market_data.items()},
            portfolio_value=1000000
        )
        
        cycle_time = time.time() - start_time
        
        # Performance requirement
        assert cycle_time < 0.1  # 100ms max per update
        
        # Update current weights (simulate fills)
        if orders:
            multi_asset.current_weights = target_weights
```

## ✅ Validation Checklist

### Data Management
- [ ] Multi-asset data loading works
- [ ] Data alignment handles missing data
- [ ] Correlation calculations accurate
- [ ] Data quality tracked

### Portfolio Optimization
- [ ] Mean-variance optimization working
- [ ] Risk parity implemented correctly
- [ ] Black-Litterman with views
- [ ] Constraints properly enforced
- [ ] Efficient frontier calculation

### Rebalancing
- [ ] Threshold rebalancing logic
- [ ] Cost-benefit analysis working
- [ ] Trade optimization functional
- [ ] Transaction costs considered

### Currency Management
- [ ] Exposure calculation accurate
- [ ] Hedge optimization working
- [ ] Multiple hedge methods available
- [ ] Currency returns adjusted

### Integration
- [ ] All components work together
- [ ] Real-time performance adequate
- [ ] Large universe handled
- [ ] Constraints respected

## 📊 Performance Benchmarks

### Optimization Performance
- 10 assets: < 50ms
- 50 assets: < 200ms
- 100 assets: < 1s
- 500 assets: < 5s

### Rebalancing Performance
- Trade calculation: < 10ms
- Cost analysis: < 20ms
- Order generation: < 5ms

### Data Processing
- Alignment (1 year, 10 assets): < 100ms
- Correlation matrix: < 50ms
- Data quality check: < 20ms

## 🐛 Common Issues

1. **Data Alignment**
   - Handle holidays differently across markets
   - Consider timezone differences
   - Manage corporate actions

2. **Optimization Stability**
   - Use covariance shrinkage
   - Add regularization
   - Handle singular matrices

3. **Currency Complexity**
   - Track all currency pairs needed
   - Handle triangular arbitrage
   - Consider forward points

## 🎯 Success Criteria

Step 10.2 is complete when:
1. ✅ Multi-asset data management operational
2. ✅ All optimization methods working
3. ✅ Rebalancing logic implemented
4. ✅ Currency management functional
5. ✅ Performance targets met

## 🚀 Next Steps

Once all validations pass, proceed to:
[Step 10.3: Execution Algorithms](step-10.3-execution-algos.md)

## 📚 Additional Resources

- [Portfolio Theory Advanced](../references/portfolio-theory-advanced.md)
- [Currency Risk Handbook](../references/currency-risk-handbook.md)
- [Multi-Asset Class Investing](../references/multi-asset-investing.md)
- [Rebalancing Best Practices](../references/rebalancing-strategies.md)