"""Performance analysis modules."""

from .sharpe import calculate_sharpe, calculate_compound_sharpe
from .equity_curves import calculate_equity_curve, plot_equity_curve
from .comparison import compare_strategies

__all__ = [
    'calculate_sharpe',
    'calculate_compound_sharpe',
    'calculate_equity_curve',
    'plot_equity_curve',
    'compare_strategies'
]