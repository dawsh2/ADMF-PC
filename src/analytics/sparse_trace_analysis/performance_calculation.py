"""
Performance calculation module for sparse trace analysis.

Handles log returns per trade with proper execution cost modeling.
Supports both multiplicative and additive execution costs.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union


@dataclass
class ExecutionCostConfig:
    """Configuration for execution cost modeling."""
    
    # Multiplicative costs (preferred for percentage-based costs)
    cost_multiplier: Optional[float] = None  # e.g., 0.97 for 3% total cost
    
    # Additive costs (for fixed dollar amounts)  
    commission_per_trade: Optional[float] = None  # Fixed commission per trade
    slippage_bps: Optional[float] = None  # Slippage in basis points
    
    # Combined additive cost per trade
    fixed_cost_per_trade: Optional[float] = None  # Total fixed cost per trade


def apply_execution_costs(
    gross_log_return: float,
    entry_price: float,
    exit_price: float,
    position_size: float,
    cost_config: ExecutionCostConfig
) -> float:
    """
    Apply execution costs to a gross log return.
    
    Args:
        gross_log_return: Log return before costs
        entry_price: Position entry price
        exit_price: Position exit price  
        position_size: Position size (signal value, typically ±1)
        cost_config: Execution cost configuration
        
    Returns:
        Net log return after execution costs
    """
    if cost_config.cost_multiplier is not None:
        # Multiplicative costs (simple and preferred)
        return gross_log_return * cost_config.cost_multiplier
    
    # Additive costs (more complex but sometimes necessary)
    # Convert to linear return, apply costs, convert back to log
    gross_linear_return = np.exp(abs(gross_log_return)) - 1
    
    # Calculate total additive costs
    total_cost = 0.0
    
    if cost_config.commission_per_trade is not None:
        total_cost += cost_config.commission_per_trade * 2  # Entry + exit
        
    if cost_config.slippage_bps is not None:
        avg_price = (entry_price + exit_price) / 2
        slippage_cost = avg_price * (cost_config.slippage_bps / 10000) * abs(position_size)
        total_cost += slippage_cost
        
    if cost_config.fixed_cost_per_trade is not None:
        total_cost += cost_config.fixed_cost_per_trade
    
    # Apply cost as percentage of notional
    notional_value = entry_price * abs(position_size)
    cost_pct = total_cost / notional_value if notional_value > 0 else 0
    
    net_linear_return = gross_linear_return - cost_pct
    
    # Convert back to log return with proper sign
    if net_linear_return > -1:  # Avoid log of negative numbers
        net_log_return = np.log(1 + net_linear_return) * np.sign(gross_log_return)
    else:
        net_log_return = -10  # Cap extreme losses
    
    return net_log_return


def calculate_trade_log_return(
    entry_price: float,
    exit_price: float,
    signal_value: float,
    cost_config: Optional[ExecutionCostConfig] = None
) -> Dict[str, float]:
    """
    Calculate log return for a single trade.
    
    Args:
        entry_price: Price when position opened
        exit_price: Price when position closed
        signal_value: Position direction and size (typically ±1)
        cost_config: Optional execution cost configuration
        
    Returns:
        Dictionary with gross and net log returns
    """
    if entry_price <= 0 or exit_price <= 0:
        return {'gross_log_return': 0.0, 'net_log_return': 0.0}
    
    # Calculate gross log return
    gross_log_return = float(np.log(exit_price / entry_price) * signal_value)
    
    # Apply execution costs if specified
    if cost_config is not None:
        net_log_return = apply_execution_costs(
            gross_log_return, entry_price, exit_price, signal_value, cost_config
        )
    else:
        net_log_return = gross_log_return
    
    return {
        'gross_log_return': gross_log_return,
        'net_log_return': net_log_return
    }


def calculate_log_returns_with_costs(
    signals_df: pd.DataFrame,
    cost_config: Optional[ExecutionCostConfig] = None,
    initial_capital: float = 10000.0
) -> Dict[str, Any]:
    """
    Calculate strategy performance using log returns with execution costs.
    
    Handles sparse signal format:
    - First non-zero signal opens position
    - Signal changes represent position changes (close previous, optionally open new)
    - Signal value determines position direction and size
    
    Args:
        signals_df: DataFrame with columns ['bar_idx', 'signal_value', 'price']
        cost_config: Optional execution cost configuration
        initial_capital: Starting capital for percentage calculations
        
    Returns:
        Dictionary with performance metrics and trade details
    """
    if signals_df.empty:
        return {
            'total_log_return': 0.0,
            'percentage_return': 0.0,
            'trades': [],
            'num_trades': 0,
            'win_rate': 0.0,
            'avg_trade_log_return': 0.0,
            'max_drawdown_pct': 0.0,
            'gross_log_return': 0.0,
            'net_log_return': 0.0
        }
    
    trades = []
    current_position = 0.0
    entry_price = None
    entry_bar_idx = None
    total_gross_log_return = 0.0
    total_net_log_return = 0.0
    log_return_curve = []
    
    # Sort by bar index to ensure proper chronological order
    signals_df = signals_df.sort_values('bar_idx').reset_index(drop=True)
    
    for _, row in signals_df.iterrows():
        bar_idx = row['bar_idx']
        signal = row.get('signal_value', 0)
        price = float(row['price'])
        
        # Track cumulative return for drawdown calculation
        log_return_curve.append(total_net_log_return)
        
        if current_position == 0:
            # No position, check if opening one
            if signal != 0:
                current_position = signal
                entry_price = price
                entry_bar_idx = bar_idx
                
        else:
            # Have position, check for close or flip
            if signal == 0 or signal != current_position:
                # Close current position
                if entry_price is not None and price > 0:
                    
                    # Calculate trade returns
                    trade_returns = calculate_trade_log_return(
                        entry_price, price, current_position, cost_config
                    )
                    
                    trade_data = {
                        'entry_bar': entry_bar_idx,
                        'exit_bar': bar_idx,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'signal': current_position,
                        'bars_held': bar_idx - entry_bar_idx,
                        'gross_log_return': trade_returns['gross_log_return'],
                        'net_log_return': trade_returns['net_log_return']
                    }
                    
                    trades.append(trade_data)
                    total_gross_log_return += trade_returns['gross_log_return']
                    total_net_log_return += trade_returns['net_log_return']
                
                # Reset position
                current_position = 0
                entry_price = None
                entry_bar_idx = None
                
                # Check if opening new position (signal flip)
                if signal != 0:
                    current_position = signal
                    entry_price = price
                    entry_bar_idx = bar_idx
    
    # Calculate final metrics
    if not trades:
        return {
            'total_log_return': 0.0,
            'percentage_return': 0.0,
            'trades': [],
            'num_trades': 0,
            'win_rate': 0.0,
            'avg_trade_log_return': 0.0,
            'max_drawdown_pct': 0.0,
            'gross_log_return': 0.0,
            'net_log_return': 0.0
        }
    
    # Convert to percentage returns
    gross_percentage_return = np.exp(total_gross_log_return) - 1
    net_percentage_return = np.exp(total_net_log_return) - 1
    
    # Calculate win rate and average trade return
    winning_trades = [t for t in trades if t['net_log_return'] > 0]
    win_rate = len(winning_trades) / len(trades)
    avg_trade_log_return = total_net_log_return / len(trades)
    
    # Calculate maximum drawdown
    log_return_curve = np.array(log_return_curve)
    percentage_curve = np.exp(log_return_curve) - 1
    running_max = np.maximum.accumulate(1 + percentage_curve)
    drawdown = (1 + percentage_curve) / running_max - 1
    max_drawdown_pct = np.min(drawdown)
    
    return {
        'total_log_return': total_net_log_return,
        'gross_log_return': total_gross_log_return,
        'percentage_return': net_percentage_return,
        'gross_percentage_return': gross_percentage_return,
        'trades': trades,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'avg_trade_log_return': avg_trade_log_return,
        'max_drawdown_pct': max_drawdown_pct,
        'initial_capital': initial_capital
    }


def summarize_performance(performance_results: Dict[str, Any]) -> str:
    """
    Generate a formatted summary of performance results.
    
    Args:
        performance_results: Output from calculate_log_returns_with_costs
        
    Returns:
        Formatted performance summary string
    """
    if performance_results['num_trades'] == 0:
        return "No trades executed"
    
    summary = f"""
Performance Summary:
├─ Total Log Return: {performance_results['total_log_return']:.4f}
├─ Percentage Return: {performance_results['percentage_return']:.2%}
├─ Number of Trades: {performance_results['num_trades']}
├─ Win Rate: {performance_results['win_rate']:.2%}
├─ Avg Trade Return: {performance_results['avg_trade_log_return']:.4f}
└─ Max Drawdown: {performance_results['max_drawdown_pct']:.2%}
"""
    
    if 'gross_percentage_return' in performance_results:
        cost_impact = performance_results['gross_percentage_return'] - performance_results['percentage_return']
        summary += f"└─ Execution Cost Impact: {cost_impact:.2%}"
    
    return summary.strip()


# Example cost configurations for common scenarios
ZERO_COST = ExecutionCostConfig()

TYPICAL_RETAIL = ExecutionCostConfig(
    commission_per_trade=1.0,  # $1 per trade
    slippage_bps=2.0  # 2 bps slippage
)

INSTITUTIONAL = ExecutionCostConfig(
    cost_multiplier=0.999  # 0.1% total cost multiplier
)

HIGH_FREQUENCY = ExecutionCostConfig(
    commission_per_trade=0.005,  # $0.005 per share
    slippage_bps=0.5  # 0.5 bps slippage
)