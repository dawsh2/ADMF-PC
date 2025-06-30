"""
Constrained Strategy Wrapper

Wraps any strategy to apply config-based constraints to its signals.
This ensures constraints work consistently across all execution modes.
"""

import logging
from typing import Dict, Any, Optional, Callable

from .config_filter import ConfigSignalFilter

logger = logging.getLogger(__name__)


class ConstrainedStrategyWrapper:
    """
    Wraps a strategy function to apply constraints to its output signals.
    
    This decouples constraint application from execution mode, ensuring
    constraints work consistently whether in signal generation, backtesting,
    or any other mode.
    """
    
    def __init__(self, strategy_func: Callable, filter_expr: Optional[str] = None,
                 filter_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the constrained strategy wrapper.
        
        Args:
            strategy_func: The original strategy function
            filter_expr: Constraint expression string (e.g., "intraday and volume > sma(volume, 20)")
            filter_params: Parameters for the constraint expression
        """
        self.strategy_func = strategy_func
        self.filter_expr = filter_expr
        self.filter_params = filter_params or {}
        
        # Initialize filter if expression provided
        self.filter = None
        if filter_expr:
            # Handle list-style filters from YAML config
            if isinstance(filter_expr, list):
                # Import parser to convert list to expression
                from ...core.coordinator.config.clean_syntax_parser import CleanSyntaxParser
                parser = CleanSyntaxParser()
                expr, params = parser._parse_combined_filter(filter_expr)
                # Store the parsed expression (without signal check - that's added later)
                self.filter_expr = expr
                # Merge params if provided
                if params:
                    self.filter_params.update(params)
                logger.info(f"Converted list filter to expression: {expr}")
            else:
                # Store as is for string expressions
                self.filter_expr = filter_expr
            
            # Replace any placeholders in the expression with filter params
            final_expr = self.filter_expr
            
            # Replace ${param} placeholders with actual values from filter_params
            if self.filter_params:
                import re
                placeholder_pattern = r'\$\{(\w+)\}'
                
                def replace_placeholder(match):
                    param_name = match.group(1)
                    if param_name in self.filter_params:
                        return str(self.filter_params[param_name])
                    else:
                        logger.warning(f"Filter parameter '{param_name}' not found in filter_params")
                        return match.group(0)  # Keep placeholder if not found
                
                final_expr = re.sub(placeholder_pattern, replace_placeholder, final_expr)
            
            # Add signal check if not present
            # Special handling for "intraday" constraint
            if final_expr == 'intraday':
                # For intraday, we want to allow signals only during market hours
                # This means signal != 0 AND intraday, OR signal == 0 (to allow flattening)
                final_expr = "(signal != 0 and intraday) or signal == 0"
            elif not final_expr.startswith('signal'):
                final_expr = f"signal != 0 and ({final_expr})"
                
            self.filter = ConfigSignalFilter()
            try:
                self.filter.compile_filter(final_expr)
                logger.info(f"Compiled filter: {final_expr}")
            except Exception as e:
                logger.error(f"Failed to compile filter '{final_expr}': {e}")
                self.filter = None
    
    def __call__(self, features: Dict[str, Any], bar: Dict[str, Any], 
                 params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute the wrapped strategy and apply filter.
        
        Args:
            features: Feature values from FeatureHub
            bar: Current bar data
            params: Strategy parameters
            
        Returns:
            Signal dictionary if filter passes, None otherwise
        """
        # Execute the original strategy
        signal = self.strategy_func(features, bar, params)
        
        # If no signal or no filter, return as-is
        if signal is None or self.filter is None:
            return signal
        
        # Apply the filter
        try:
            # Ensure signal has required structure
            if isinstance(signal, dict) and 'signal_value' in signal:
                # Evaluate filter
                if self.filter.evaluate_filter(signal, features, bar, filter_params=self.filter_params):
                    # Filter passed
                    logger.debug(f"Signal passed filter: {signal.get('signal_value')} at {bar.get('timestamp')}")
                    return signal
                else:
                    # Filter rejected - return flat signal instead of None
                    logger.debug(f"Signal rejected by filter: {signal.get('signal_value')} at {bar.get('timestamp')}")
                    # Return a flat signal (0) to force position closure
                    flat_signal = {
                        'signal_value': 0,
                        'timestamp': bar.get('timestamp'),
                        'metadata': {
                            'threshold_rejected': True,
                            'original_signal': signal.get('signal_value', 0),
                            'threshold_expr': self.filter_expr
                        }
                    }
                    return flat_signal
            else:
                # Invalid signal format, log warning but pass through
                logger.warning(f"Invalid signal format from strategy: {signal}")
                return signal
                
        except Exception as e:
            logger.error(f"Error applying filter: {e}")
            # On error, return the original signal (fail open)
            return signal
    
    @property
    def __name__(self):
        """Preserve the original function name for logging."""
        return getattr(self.strategy_func, '__name__', 'filtered_strategy')
    
    def __repr__(self):
        """String representation."""
        return f"FilteredStrategyWrapper({self.strategy_func}, filter='{self.filter_expr}')"


def wrap_strategy_with_filter(strategy_func: Callable, config: Dict[str, Any]) -> Callable:
    """
    Convenience function to wrap a strategy with filter from config.
    
    Args:
        strategy_func: The strategy function to wrap
        config: Strategy configuration containing 'constraints' (or 'threshold' or deprecated 'filter') and optional 'filter_params'
        
    Returns:
        Wrapped strategy function that applies the filter
    """
    # Support 'constraints' (new), 'threshold' (transitional) and 'filter' (deprecated)
    filter_expr = config.get('constraints') or config.get('threshold') or config.get('filter')
    filter_params = config.get('filter_params', {})
    
    if not filter_expr:
        # No filter, return original function
        return strategy_func
    
    # Create and return wrapped function
    return FilteredStrategyWrapper(strategy_func, filter_expr, filter_params)

# Keep old name for backward compatibility
FilteredStrategyWrapper = ConstrainedStrategyWrapper
