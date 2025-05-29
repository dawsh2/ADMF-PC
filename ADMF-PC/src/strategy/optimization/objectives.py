"""
Optimization objective implementations.
"""

from typing import Dict, Any, List, Optional
import math
import logging

logger = logging.getLogger(__name__)


class SharpeObjective:
    """Maximize Sharpe ratio"""
    
    def __init__(self, risk_free_rate: float = 0.0, periods_per_year: int = 252):
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
    
    def calculate(self, results: Dict[str, Any]) -> float:
        """
        Calculate Sharpe ratio from results.
        
        Args:
            results: Must contain 'returns' list or 'sharpe_ratio'
            
        Returns:
            Sharpe ratio (higher is better)
        """
        # If Sharpe already calculated, use it
        if 'sharpe_ratio' in results:
            return results['sharpe_ratio']
        
        # Calculate from returns
        if 'returns' not in results:
            raise ValueError("Results must contain 'returns' or 'sharpe_ratio'")
        
        returns = results['returns']
        if not returns or len(returns) < 2:
            return 0.0
        
        # Calculate Sharpe
        avg_return = sum(returns) / len(returns)
        variance = sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1)
        std_dev = math.sqrt(variance)
        
        if std_dev == 0:
            return 0.0
        
        # Annualize
        sharpe = (avg_return - self.risk_free_rate) / std_dev
        sharpe *= math.sqrt(self.periods_per_year)
        
        return sharpe
    
    def get_direction(self) -> str:
        """Maximize Sharpe ratio"""
        return "maximize"
    
    def get_requirements(self) -> List[str]:
        """Either returns or sharpe_ratio required"""
        return ["returns|sharpe_ratio"]


class MaxReturnObjective:
    """Maximize total return"""
    
    def calculate(self, results: Dict[str, Any]) -> float:
        """
        Calculate total return from results.
        
        Args:
            results: Must contain 'total_return' or 'equity_curve'
            
        Returns:
            Total return (higher is better)
        """
        # Direct total return
        if 'total_return' in results:
            return results['total_return']
        
        # Calculate from equity curve
        if 'equity_curve' in results:
            equity_curve = results['equity_curve']
            if len(equity_curve) >= 2:
                initial = equity_curve[0]
                final = equity_curve[-1]
                return (final - initial) / initial if initial != 0 else 0.0
        
        # Calculate from returns
        if 'returns' in results:
            returns = results['returns']
            total_return = 1.0
            for r in returns:
                total_return *= (1 + r)
            return total_return - 1
        
        raise ValueError("Results must contain 'total_return', 'equity_curve', or 'returns'")
    
    def get_direction(self) -> str:
        """Maximize return"""
        return "maximize"
    
    def get_requirements(self) -> List[str]:
        """Need return data"""
        return ["total_return|equity_curve|returns"]


class MinDrawdownObjective:
    """Minimize maximum drawdown"""
    
    def calculate(self, results: Dict[str, Any]) -> float:
        """
        Calculate maximum drawdown from results.
        
        Args:
            results: Must contain 'max_drawdown' or 'equity_curve'
            
        Returns:
            Negative of max drawdown (higher is better, closer to 0)
        """
        # Direct max drawdown
        if 'max_drawdown' in results:
            # Return negative so higher is better
            return -abs(results['max_drawdown'])
        
        # Calculate from equity curve
        if 'equity_curve' in results:
            equity_curve = results['equity_curve']
            if len(equity_curve) < 2:
                return 0.0
            
            max_drawdown = 0.0
            peak = equity_curve[0]
            
            for value in equity_curve:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak if peak != 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
            
            # Return negative so higher is better
            return -max_drawdown
        
        raise ValueError("Results must contain 'max_drawdown' or 'equity_curve'")
    
    def get_direction(self) -> str:
        """Maximize (which minimizes drawdown due to negative return)"""
        return "maximize"
    
    def get_requirements(self) -> List[str]:
        """Need drawdown or equity data"""
        return ["max_drawdown|equity_curve"]


class CompositeObjective:
    """Combine multiple objectives with weights"""
    
    def __init__(self, objectives: List[tuple[Any, float]]):
        """
        Initialize composite objective.
        
        Args:
            objectives: List of (objective, weight) tuples
        """
        self.objectives = objectives
        
        # Normalize weights
        total_weight = sum(weight for _, weight in objectives)
        if total_weight > 0:
            self.objectives = [(obj, weight / total_weight) 
                              for obj, weight in objectives]
    
    def calculate(self, results: Dict[str, Any]) -> float:
        """
        Calculate weighted combination of objectives.
        
        Args:
            results: Backtest results
            
        Returns:
            Weighted objective value
        """
        total_score = 0.0
        
        for objective, weight in self.objectives:
            try:
                score = objective.calculate(results)
                
                # Handle minimize objectives
                if hasattr(objective, 'get_direction') and objective.get_direction() == 'minimize':
                    score = -score
                
                total_score += weight * score
                
            except Exception as e:
                logger.warning(f"Error calculating objective {objective.__class__.__name__}: {e}")
                # Skip failed objectives
                continue
        
        return total_score
    
    def get_direction(self) -> str:
        """Always maximize composite score"""
        return "maximize"
    
    def get_requirements(self) -> List[str]:
        """Combine all sub-objective requirements"""
        all_requirements = []
        for objective, _ in self.objectives:
            if hasattr(objective, 'get_requirements'):
                all_requirements.extend(objective.get_requirements())
        return list(set(all_requirements))


class CalmarObjective:
    """Maximize Calmar ratio (return / max drawdown)"""
    
    def __init__(self, periods_per_year: int = 252):
        self.periods_per_year = periods_per_year
    
    def calculate(self, results: Dict[str, Any]) -> float:
        """
        Calculate Calmar ratio from results.
        
        Args:
            results: Must contain return and drawdown data
            
        Returns:
            Calmar ratio
        """
        # Get annualized return
        if 'annual_return' in results:
            annual_return = results['annual_return']
        elif 'total_return' in results and 'num_periods' in results:
            total_return = results['total_return']
            num_periods = results['num_periods']
            annual_return = ((1 + total_return) ** (self.periods_per_year / num_periods)) - 1
        else:
            raise ValueError("Cannot calculate annual return from results")
        
        # Get max drawdown
        if 'max_drawdown' in results:
            max_drawdown = abs(results['max_drawdown'])
        else:
            raise ValueError("Results must contain 'max_drawdown'")
        
        # Calculate Calmar
        if max_drawdown == 0:
            return float('inf') if annual_return > 0 else 0.0
        
        return annual_return / max_drawdown
    
    def get_direction(self) -> str:
        """Maximize Calmar ratio"""
        return "maximize"
    
    def get_requirements(self) -> List[str]:
        """Need return and drawdown data"""
        return ["annual_return|total_return", "max_drawdown"]


class SortinoObjective:
    """Maximize Sortino ratio (uses downside deviation)"""
    
    def __init__(self, risk_free_rate: float = 0.0, periods_per_year: int = 252):
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
    
    def calculate(self, results: Dict[str, Any]) -> float:
        """
        Calculate Sortino ratio from results.
        
        Args:
            results: Must contain 'returns' list
            
        Returns:
            Sortino ratio
        """
        if 'sortino_ratio' in results:
            return results['sortino_ratio']
        
        if 'returns' not in results:
            raise ValueError("Results must contain 'returns' or 'sortino_ratio'")
        
        returns = results['returns']
        if not returns or len(returns) < 2:
            return 0.0
        
        # Calculate downside deviation
        avg_return = sum(returns) / len(returns)
        downside_returns = [min(0, r - self.risk_free_rate) for r in returns]
        downside_variance = sum(dr ** 2 for dr in downside_returns) / len(downside_returns)
        downside_dev = math.sqrt(downside_variance)
        
        if downside_dev == 0:
            return float('inf') if avg_return > self.risk_free_rate else 0.0
        
        # Calculate Sortino
        sortino = (avg_return - self.risk_free_rate) / downside_dev
        sortino *= math.sqrt(self.periods_per_year)
        
        return sortino
    
    def get_direction(self) -> str:
        """Maximize Sortino ratio"""
        return "maximize"
    
    def get_requirements(self) -> List[str]:
        """Need returns data"""
        return ["returns|sortino_ratio"]