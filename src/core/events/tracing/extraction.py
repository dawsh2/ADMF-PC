"""Result extraction from event streams."""

from typing import Protocol, Dict, Any, List, Optional
from abc import abstractmethod

from ..types import Event, EventType

# The ResultExtractor protocol is already defined in protocols.py
# Here we implement concrete extractors

class MetricsExtractor:
    """Extract metrics directly from event stream."""
    
    def __init__(self):
        self.metrics = {
            'total_return': 0.0,
            'portfolio_values': [],
            'trades': [],
            'positions': {}
        }
        
    def extract(self, event: Event) -> Optional[Dict[str, Any]]:
        """Update metrics from relevant events."""
        if event.event_type == EventType.POSITION_CLOSE.value:
            # Extract trade data and update metrics
            trade_data = {
                'entry_price': event.payload.get('entry_price', 0),
                'exit_price': event.payload.get('exit_price', 0),
                'quantity': event.payload.get('quantity', 0),
                'direction': event.payload.get('direction', 'long'),
                'pnl': event.payload.get('pnl', 0)
            }
            self.metrics['trades'].append(trade_data)
            self.metrics['total_return'] += trade_data['pnl']
            
        elif event.event_type == EventType.PORTFOLIO_UPDATE.value:
            self.metrics['portfolio_values'].append({
                'timestamp': event.timestamp,
                'value': event.payload.get('portfolio_value', 0)
            })
            
        elif event.event_type == EventType.POSITION_OPEN.value:
            position_id = event.correlation_id
            if position_id:
                self.metrics['positions'][position_id] = {
                    'symbol': event.payload.get('symbol'),
                    'entry_price': event.payload.get('price'),
                    'quantity': event.payload.get('quantity'),
                    'direction': event.payload.get('direction'),
                    'timestamp': event.timestamp
                }
                
        return None
        
    def get_results(self) -> Dict[str, Any]:
        """Get final metrics."""
        # Calculate additional metrics
        if self.metrics['portfolio_values']:
            initial_value = self.metrics['portfolio_values'][0]['value']
            final_value = self.metrics['portfolio_values'][-1]['value']
            self.metrics['total_return_pct'] = (final_value - initial_value) / initial_value
            
        self.metrics['total_trades'] = len(self.metrics['trades'])
        self.metrics['winning_trades'] = sum(1 for t in self.metrics['trades'] if t['pnl'] > 0)
        
        if self.metrics['total_trades'] > 0:
            self.metrics['win_rate'] = self.metrics['winning_trades'] / self.metrics['total_trades']
        else:
            self.metrics['win_rate'] = 0.0
            
        return self.metrics


class ObjectiveFunctionExtractor:
    """Extract objective function value from event stream."""
    
    def __init__(self, objective_function: str = 'sharpe_ratio'):
        self.objective_function = objective_function
        self.portfolio_values = []
        self.trades = []
        
    def extract(self, event: Event) -> Optional[Dict[str, Any]]:
        """Extract data needed for objective function."""
        if event.event_type == EventType.PORTFOLIO_UPDATE.value:
            self.portfolio_values.append(event.payload.get('portfolio_value', 0))
            
        elif event.event_type == EventType.POSITION_CLOSE.value:
            self.trades.append({
                'pnl': event.payload.get('pnl', 0),
                'duration': event.payload.get('duration', 0)
            })
            
        return None
        
    def get_results(self) -> Dict[str, Any]:
        """Calculate objective function value."""
        if self.objective_function == 'sharpe_ratio':
            if len(self.portfolio_values) < 2:
                return {'objective_value': 0.0}
                
            # Calculate returns
            returns = []
            for i in range(1, len(self.portfolio_values)):
                ret = (self.portfolio_values[i] - self.portfolio_values[i-1]) / self.portfolio_values[i-1]
                returns.append(ret)
                
            # Calculate Sharpe ratio
            if returns:
                import numpy as np
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                if std_return > 0:
                    sharpe = np.sqrt(252) * mean_return / std_return
                else:
                    sharpe = 0.0
            else:
                sharpe = 0.0
                
            return {'objective_value': sharpe, 'objective_type': 'sharpe_ratio'}
            
        elif self.objective_function == 'total_return':
            total_pnl = sum(t['pnl'] for t in self.trades)
            return {'objective_value': total_pnl, 'objective_type': 'total_return'}
            
        else:
            raise ValueError(f"Unknown objective function: {self.objective_function}")