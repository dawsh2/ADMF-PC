"""
Parameter selector components for train/test splits and optimization.

These components analyze training results and select optimal parameters
for the test phase based on various criteria.
"""

from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ParameterSelector(ABC):
    """Base protocol for parameter selectors."""
    
    @abstractmethod
    def select(self, results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Select optimal parameters from training results.
        
        Args:
            results: List of result dictionaries from training phase
            
        Returns:
            Selected parameters dict or None if no valid selection
        """
        pass


class BestMetricSelector(ParameterSelector):
    """Selects parameters based on best value of a single metric."""
    
    def __init__(self, 
                 metric_name: str = 'sharpe_ratio',
                 minimize: bool = False,
                 min_trades: int = 0,
                 constraints: Optional[Dict[str, Any]] = None):
        """
        Initialize selector.
        
        Args:
            metric_name: Name of metric to optimize
            minimize: Whether to minimize (True) or maximize (False)
            min_trades: Minimum number of trades required
            constraints: Additional constraints on metrics
        """
        self.metric_name = metric_name
        self.minimize = minimize
        self.min_trades = min_trades
        self.constraints = constraints or {}
        
    def select(self, results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select parameters with best metric value."""
        if not results:
            logger.warning("No results provided for parameter selection")
            return None
            
        # Filter valid results
        valid_results = []
        for result in results:
            if not self._is_valid_result(result):
                continue
            valid_results.append(result)
                
        if not valid_results:
            logger.warning("No valid results found after filtering")
            return None
            
        # Find best
        best_result = None
        best_value = float('inf') if self.minimize else float('-inf')
        
        for result in valid_results:
            value = self._extract_metric(result, self.metric_name)
            if value is None:
                continue
                
            if self.minimize:
                if value < best_value:
                    best_value = value
                    best_result = result
            else:
                if value > best_value:
                    best_value = value
                    best_result = result
                    
        if best_result:
            logger.info(f"Selected parameters with {self.metric_name}={best_value}")
            return self._extract_parameters(best_result)
        else:
            logger.warning("Could not find valid parameters")
            return None
            
    def _is_valid_result(self, result: Dict[str, Any]) -> bool:
        """Check if result meets all constraints."""
        metrics = result.get('metrics', result.get('aggregate_metrics', {}))
        
        # Check min trades
        num_trades = metrics.get('num_trades', metrics.get('total_trades', 0))
        if num_trades < self.min_trades:
            return False
            
        # Check additional constraints
        for metric, constraint in self.constraints.items():
            value = metrics.get(metric)
            if value is None:
                continue
                
            if isinstance(constraint, dict):
                if 'min' in constraint and value < constraint['min']:
                    return False
                if 'max' in constraint and value > constraint['max']:
                    return False
            else:
                # Simple equality or threshold
                if value != constraint:
                    return False
                    
        return True
        
    def _extract_metric(self, result: Dict[str, Any], metric_name: str) -> Optional[float]:
        """Extract metric value from result."""
        # Try different locations
        locations = [
            result.get('metrics', {}),
            result.get('aggregate_metrics', {}),
            result.get('phase_results', {}).get('aggregate_metrics', {}),
            result
        ]
        
        for location in locations:
            if metric_name in location:
                return location[metric_name]
                
        return None
        
    def _extract_parameters(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters from result."""
        # Try different locations
        if 'parameters' in result:
            return result['parameters']
        elif 'config' in result and 'parameters' in result['config']:
            return result['config']['parameters']
        elif 'strategy_params' in result:
            return result['strategy_params']
        elif 'config' in result:
            # Extract strategy and risk parameters
            config = result['config']
            params = {}
            if 'strategy_type' in config:
                params['strategy_type'] = config['strategy_type']
            if 'strategy_params' in config:
                params['strategy_params'] = config['strategy_params']
            if 'risk_type' in config:
                params['risk_type'] = config['risk_type']
            if 'risk_params' in config:
                params['risk_params'] = config['risk_params']
            return params
        else:
            # Return full result as fallback
            return result


class BestSharpeSelector(BestMetricSelector):
    """Convenience selector for Sharpe ratio optimization."""
    
    def __init__(self, min_trades: int = 30, max_drawdown: Optional[float] = None):
        constraints = {}
        if max_drawdown is not None:
            constraints['max_drawdown'] = {'max': max_drawdown}
            
        super().__init__(
            metric_name='sharpe_ratio',
            minimize=False,
            min_trades=min_trades,
            constraints=constraints
        )


class RobustSelector(ParameterSelector):
    """
    Selects parameters that perform well across multiple metrics.
    
    Uses a scoring system that considers:
    - Sharpe ratio
    - Maximum drawdown
    - Win rate
    - Profit factor
    """
    
    def __init__(self,
                 weights: Optional[Dict[str, float]] = None,
                 min_trades: int = 30,
                 require_positive_sharpe: bool = True):
        """
        Initialize robust selector.
        
        Args:
            weights: Metric weights for scoring (default: balanced)
            min_trades: Minimum number of trades
            require_positive_sharpe: Only consider positive Sharpe strategies
        """
        self.weights = weights or {
            'sharpe_ratio': 0.4,
            'profit_factor': 0.2,
            'win_rate': 0.2,
            'max_drawdown': 0.2  # Lower is better
        }
        self.min_trades = min_trades
        self.require_positive_sharpe = require_positive_sharpe
        
    def select(self, results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select parameters with best composite score."""
        if not results:
            return None
            
        # Score all results
        scored_results = []
        for result in results:
            score = self._score_result(result)
            if score is not None:
                scored_results.append((score, result))
                
        if not scored_results:
            logger.warning("No valid results to score")
            return None
            
        # Sort by score (descending)
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        # Return parameters from best result
        best_score, best_result = scored_results[0]
        logger.info(f"Selected parameters with robust score={best_score:.3f}")
        
        # Extract parameters similar to BestMetricSelector
        return self._extract_parameters(best_result)
        
    def _score_result(self, result: Dict[str, Any]) -> Optional[float]:
        """Calculate composite score for result."""
        metrics = result.get('metrics', result.get('aggregate_metrics', {}))
        
        # Check constraints
        num_trades = metrics.get('num_trades', metrics.get('total_trades', 0))
        if num_trades < self.min_trades:
            return None
            
        sharpe = metrics.get('sharpe_ratio', 0)
        if self.require_positive_sharpe and sharpe <= 0:
            return None
            
        # Calculate normalized scores for each metric
        scores = {}
        
        # Sharpe ratio (normalized to 0-1 range, capped at 3)
        scores['sharpe_ratio'] = min(sharpe / 3.0, 1.0) if sharpe > 0 else 0
        
        # Profit factor (normalized, capped at 3)
        pf = metrics.get('profit_factor', 1.0)
        scores['profit_factor'] = min((pf - 1.0) / 2.0, 1.0) if pf > 1 else 0
        
        # Win rate (already 0-1)
        scores['win_rate'] = metrics.get('win_rate', 0.5)
        
        # Max drawdown (inverted - lower is better)
        dd = abs(metrics.get('max_drawdown', 0))
        scores['max_drawdown'] = 1.0 - min(dd, 1.0)  # 0% dd = score 1, 100% dd = score 0
        
        # Calculate weighted score
        total_score = 0
        total_weight = 0
        for metric, weight in self.weights.items():
            if metric in scores:
                total_score += scores[metric] * weight
                total_weight += weight
                
        return total_score / total_weight if total_weight > 0 else None
        
    def _extract_parameters(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters from result (same as BestMetricSelector)."""
        if 'parameters' in result:
            return result['parameters']
        elif 'config' in result and 'parameters' in result['config']:
            return result['config']['parameters']
        elif 'strategy_params' in result:
            return result['strategy_params']
        elif 'config' in result:
            config = result['config']
            params = {}
            if 'strategy_type' in config:
                params['strategy_type'] = config['strategy_type']
            if 'strategy_params' in config:
                params['strategy_params'] = config['strategy_params']
            if 'risk_type' in config:
                params['risk_type'] = config['risk_type']
            if 'risk_params' in config:
                params['risk_params'] = config['risk_params']
            return params
        else:
            return result


class StabilitySelector(ParameterSelector):
    """
    Selects parameters based on stability across different market conditions.
    
    Expects results to contain sub-period performance (e.g., monthly returns).
    """
    
    def __init__(self,
                 min_sharpe: float = 0.5,
                 max_return_std: float = 0.3,
                 min_positive_periods: float = 0.6):
        """
        Initialize stability selector.
        
        Args:
            min_sharpe: Minimum acceptable Sharpe ratio
            max_return_std: Maximum standard deviation of period returns
            min_positive_periods: Minimum fraction of positive periods
        """
        self.min_sharpe = min_sharpe
        self.max_return_std = max_return_std
        self.min_positive_periods = min_positive_periods
        
    def select(self, results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select most stable parameters."""
        if not results:
            return None
            
        best_stability = float('-inf')
        best_result = None
        
        for result in results:
            stability = self._calculate_stability(result)
            if stability is not None and stability > best_stability:
                best_stability = stability
                best_result = result
                
        if best_result:
            logger.info(f"Selected parameters with stability score={best_stability:.3f}")
            return self._extract_parameters(best_result)
        else:
            return None
            
    def _calculate_stability(self, result: Dict[str, Any]) -> Optional[float]:
        """Calculate stability score."""
        metrics = result.get('metrics', result.get('aggregate_metrics', {}))
        
        # Check minimum Sharpe
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe < self.min_sharpe:
            return None
            
        # Get period returns if available
        period_returns = metrics.get('period_returns', [])
        if not period_returns:
            # Fallback to using overall metrics
            return sharpe  # Just use Sharpe as stability measure
            
        # Calculate stability metrics
        returns_array = np.array(period_returns)
        
        # Standard deviation of returns
        return_std = np.std(returns_array)
        if return_std > self.max_return_std:
            return None
            
        # Fraction of positive periods
        positive_periods = np.sum(returns_array > 0) / len(returns_array)
        if positive_periods < self.min_positive_periods:
            return None
            
        # Stability score combines multiple factors
        stability_score = (
            sharpe * 0.4 +  # Sharpe ratio
            (1 - return_std) * 0.3 +  # Low volatility of returns
            positive_periods * 0.3  # Consistency
        )
        
        return stability_score
        
    def _extract_parameters(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters from result."""
        if 'parameters' in result:
            return result['parameters']
        elif 'config' in result:
            return result['config']
        else:
            return result


# Factory function
def create_parameter_selector(selector_type: str, 
                            config: Optional[Dict[str, Any]] = None) -> ParameterSelector:
    """
    Factory function to create parameter selectors.
    
    Args:
        selector_type: Type of selector ('best_sharpe', 'robust', 'stability', etc.)
        config: Configuration for the selector
        
    Returns:
        ParameterSelector instance
    """
    config = config or {}
    
    if selector_type == 'best_sharpe':
        return BestSharpeSelector(**config)
    elif selector_type == 'robust':
        return RobustSelector(**config)
    elif selector_type == 'stability':
        return StabilitySelector(**config)
    elif selector_type == 'best_metric':
        return BestMetricSelector(**config)
    else:
        # Default to best Sharpe
        logger.warning(f"Unknown selector type '{selector_type}', using best_sharpe")
        return BestSharpeSelector(**config)