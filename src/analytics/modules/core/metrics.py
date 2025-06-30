"""Performance metrics calculations."""

import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple


def calculate_sharpe(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio.
    
    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns series
    risk_free_rate : float
        Annual risk-free rate
    periods_per_year : int
        Number of periods per year (252 for daily, 252*78 for 5-min bars)
        
    Returns
    -------
    float
        Sharpe ratio
        
    Examples
    --------
    >>> returns = pd.Series([0.01, -0.005, 0.008, 0.003, -0.002])
    >>> calculate_sharpe(returns)
    1.523
    """
    if len(returns) < 2:
        return np.nan
    
    excess_returns = returns - risk_free_rate / periods_per_year
    
    mean_return = excess_returns.mean()
    std_return = excess_returns.std()
    
    if std_return == 0:
        return np.nan
    
    return mean_return / std_return * np.sqrt(periods_per_year)


def calculate_compound_sharpe(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio using compound (geometric) returns.
    
    This is more accurate for strategies with high volatility or 
    when compounding effects are significant.
    
    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Simple returns series
    risk_free_rate : float
        Annual risk-free rate
    periods_per_year : int
        Number of periods per year
        
    Returns
    -------
    float
        Compound Sharpe ratio
    """
    if len(returns) < 2:
        return np.nan
    
    # Convert to growth factors
    growth_factors = 1 + returns
    
    # Calculate geometric mean return
    geo_mean = np.prod(growth_factors) ** (1 / len(returns)) - 1
    
    # Subtract risk-free rate
    excess_geo_return = geo_mean - risk_free_rate / periods_per_year
    
    # Use standard deviation of simple returns
    std_return = returns.std()
    
    if std_return == 0:
        return np.nan
    
    return excess_geo_return / std_return * np.sqrt(periods_per_year)


def calculate_max_drawdown(
    returns: Union[pd.Series, np.ndarray],
    return_type: str = 'simple'
) -> Tuple[float, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Calculate maximum drawdown and dates.
    
    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns series
    return_type : str
        'simple' or 'log' returns
        
    Returns
    -------
    tuple
        (max_drawdown, peak_date, trough_date)
    """
    if len(returns) == 0:
        return (0.0, None, None)
    
    # Calculate cumulative returns
    if return_type == 'log':
        cum_returns = np.exp(np.cumsum(returns))
    else:
        cum_returns = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = cum_returns.expanding().max()
    
    # Calculate drawdown
    drawdown = (cum_returns - running_max) / running_max
    
    # Find maximum drawdown
    max_dd = drawdown.min()
    
    # Find dates if Series with datetime index
    if isinstance(drawdown, pd.Series) and isinstance(drawdown.index, pd.DatetimeIndex):
        max_dd_idx = drawdown.idxmin()
        # Find the peak before the trough
        peak_idx = cum_returns[:max_dd_idx].idxmax()
        return (max_dd, peak_idx, max_dd_idx)
    
    return (max_dd, None, None)


def calculate_win_rate(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate win rate (percentage of positive returns).
    
    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns series
        
    Returns
    -------
    float
        Win rate between 0 and 1
    """
    if len(returns) == 0:
        return np.nan
    
    return (returns > 0).sum() / len(returns)


def calculate_profit_factor(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate profit factor (gross profits / gross losses).
    
    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns series
        
    Returns
    -------
    float
        Profit factor (> 1 is profitable)
    """
    if len(returns) == 0:
        return np.nan
    
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    
    return gains / losses


def calculate_calmar_ratio(
    returns: Union[pd.Series, np.ndarray],
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown).
    
    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns series
    periods_per_year : int
        Number of periods per year
        
    Returns
    -------
    float
        Calmar ratio
    """
    if len(returns) < periods_per_year:
        return np.nan
    
    # Annual return
    total_return = (1 + returns).prod() - 1
    years = len(returns) / periods_per_year
    annual_return = (1 + total_return) ** (1 / years) - 1
    
    # Max drawdown
    max_dd, _, _ = calculate_max_drawdown(returns)
    
    if max_dd == 0:
        return np.nan
    
    return annual_return / abs(max_dd)


def calculate_sortino_ratio(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio (uses downside deviation).
    
    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns series
    risk_free_rate : float
        Annual risk-free rate
    periods_per_year : int
        Number of periods per year
        
    Returns
    -------
    float
        Sortino ratio
    """
    if len(returns) < 2:
        return np.nan
    
    excess_returns = returns - risk_free_rate / periods_per_year
    
    mean_return = excess_returns.mean()
    
    # Downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.nan
    
    downside_std = np.sqrt((downside_returns ** 2).mean())
    
    if downside_std == 0:
        return np.nan
    
    return mean_return / downside_std * np.sqrt(periods_per_year)


def calculate_information_ratio(
    returns: Union[pd.Series, np.ndarray],
    benchmark_returns: Union[pd.Series, np.ndarray],
    periods_per_year: int = 252
) -> float:
    """
    Calculate Information ratio (active return / tracking error).
    
    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Strategy returns
    benchmark_returns : pd.Series or np.ndarray
        Benchmark returns
    periods_per_year : int
        Number of periods per year
        
    Returns
    -------
    float
        Information ratio
    """
    if len(returns) != len(benchmark_returns):
        raise ValueError("Returns and benchmark must have same length")
    
    if len(returns) < 2:
        return np.nan
    
    active_returns = returns - benchmark_returns
    
    mean_active = active_returns.mean()
    std_active = active_returns.std()
    
    if std_active == 0:
        return np.nan
    
    return mean_active / std_active * np.sqrt(periods_per_year)


def calculate_omega_ratio(
    returns: Union[pd.Series, np.ndarray],
    threshold: float = 0.0
) -> float:
    """
    Calculate Omega ratio.
    
    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns series
    threshold : float
        Threshold return (usually 0)
        
    Returns
    -------
    float
        Omega ratio
    """
    if len(returns) == 0:
        return np.nan
    
    # Returns above threshold
    gains = returns[returns > threshold] - threshold
    
    # Returns below threshold  
    losses = threshold - returns[returns <= threshold]
    
    sum_gains = gains.sum() if len(gains) > 0 else 0
    sum_losses = losses.sum() if len(losses) > 0 else 0
    
    if sum_losses == 0:
        return np.inf if sum_gains > 0 else np.nan
    
    return sum_gains / sum_losses