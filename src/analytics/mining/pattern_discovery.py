"""Pattern discovery for trading strategies."""

from typing import Dict, Any, List
import pandas as pd
import logging

from .query import EventQueryInterface

logger = logging.getLogger(__name__)


class TradingPatternMiner:
    """
    Specialized pattern discovery for trading strategies.
    
    Focuses on finding actionable patterns in event sequences.
    """
    
    def __init__(self, query_interface: EventQueryInterface):
        self.query = query_interface
        
    def find_regime_patterns(self, classifier_id: str) -> pd.DataFrame:
        """Find patterns related to regime changes."""
        # Query classifier state changes
        events = self.query.storage.query({
            'event_type': 'CLASSIFIER_STATE_CHANGE',
            'source_id': classifier_id
        })
        
        # Analyze performance around regime changes
        results = []
        for event in events:
            # Get events in window around regime change
            window_start = event.timestamp - pd.Timedelta(hours=1)
            window_end = event.timestamp + pd.Timedelta(hours=1)
            
            window_events = self.query.query_by_time_range(
                window_start, window_end, 
                event_types=['SIGNAL', 'POSITION_OPEN', 'POSITION_CLOSE']
            )
            
            # Calculate metrics
            signals_before = len(window_events[
                (window_events['timestamp'] < event.timestamp) & 
                (window_events['event_type'] == 'SIGNAL')
            ])
            signals_after = len(window_events[
                (window_events['timestamp'] >= event.timestamp) & 
                (window_events['event_type'] == 'SIGNAL')
            ])
            
            results.append({
                'regime_change_time': event.timestamp,
                'old_regime': event.payload.get('old_state'),
                'new_regime': event.payload.get('new_state'),
                'signals_before': signals_before,
                'signals_after': signals_after,
                'signal_change_ratio': signals_after / max(signals_before, 1)
            })
            
        return pd.DataFrame(results)
        
    def find_execution_patterns(self) -> Dict[str, Any]:
        """Find patterns in order execution."""
        # Analyze fill rates, slippage patterns, etc.
        order_events = self.query.storage.query({'event_type': 'ORDER'})
        fill_events = self.query.storage.query({'event_type': 'FILL'})
        
        # Match orders to fills
        order_to_fill = {}
        for fill in fill_events:
            order_id = fill.payload.get('order_id')
            if order_id:
                order_to_fill[order_id] = fill
                
        # Calculate execution metrics
        execution_times = []
        slippage_values = []
        
        for order in order_events:
            order_id = order.metadata.get('order_id')
            if order_id in order_to_fill:
                fill = order_to_fill[order_id]
                exec_time = (fill.timestamp - order.timestamp).total_seconds()
                execution_times.append(exec_time)
                
                # Calculate slippage
                expected_price = order.payload.get('limit_price', order.payload.get('price'))
                actual_price = fill.payload.get('price')
                if expected_price and actual_price:
                    slippage = abs(actual_price - expected_price) / expected_price
                    slippage_values.append(slippage)
                    
        return {
            'avg_execution_time': sum(execution_times) / len(execution_times) if execution_times else 0,
            'max_execution_time': max(execution_times) if execution_times else 0,
            'avg_slippage_pct': sum(slippage_values) / len(slippage_values) * 100 if slippage_values else 0,
            'fill_rate': len(order_to_fill) / len(order_events) if order_events else 0
        }
        
    def find_signal_quality_patterns(self) -> pd.DataFrame:
        """Find patterns in signal quality and performance."""
        # Get all signals and their outcomes
        signal_events = self.query.storage.query({'event_type': 'SIGNAL'})
        position_close_events = self.query.storage.query({'event_type': 'POSITION_CLOSE'})
        
        # Match signals to their outcomes
        signal_outcomes = []
        
        for signal in signal_events:
            correlation_id = signal.correlation_id
            if not correlation_id:
                continue
                
            # Find corresponding position close
            outcome = next(
                (pos for pos in position_close_events 
                 if pos.correlation_id == correlation_id), 
                None
            )
            
            if outcome:
                signal_outcomes.append({
                    'signal_strength': signal.payload.get('strength', 0),
                    'signal_confidence': signal.payload.get('confidence', 0),
                    'strategy_id': signal.payload.get('strategy_id'),
                    'symbol': signal.payload.get('symbol'),
                    'direction': signal.payload.get('direction'),
                    'pnl': outcome.payload.get('pnl', 0),
                    'pnl_pct': outcome.payload.get('pnl_pct', 0),
                    'duration_hours': (outcome.timestamp - signal.timestamp).total_seconds() / 3600
                })
        
        return pd.DataFrame(signal_outcomes)
        
    def analyze_strategy_regime_performance(self) -> Dict[str, Dict[str, Any]]:
        """Analyze how different strategies perform in different regimes."""
        # Get signals with regime information
        signal_events = self.query.storage.query({'event_type': 'SIGNAL'})
        position_close_events = self.query.storage.query({'event_type': 'POSITION_CLOSE'})
        
        strategy_regime_performance = {}
        
        for signal in signal_events:
            strategy_id = signal.payload.get('strategy_id')
            regime = signal.payload.get('classifier_states', {}).get('trend', 'unknown')
            
            if not strategy_id or regime == 'unknown':
                continue
                
            # Find outcome
            outcome = next(
                (pos for pos in position_close_events 
                 if pos.correlation_id == signal.correlation_id), 
                None
            )
            
            if outcome:
                key = f"{strategy_id}_{regime}"
                if key not in strategy_regime_performance:
                    strategy_regime_performance[key] = {
                        'strategy_id': strategy_id,
                        'regime': regime,
                        'total_trades': 0,
                        'winning_trades': 0,
                        'total_pnl': 0,
                        'pnl_list': []
                    }
                
                perf = strategy_regime_performance[key]
                pnl = outcome.payload.get('pnl', 0)
                
                perf['total_trades'] += 1
                perf['total_pnl'] += pnl
                perf['pnl_list'].append(pnl)
                
                if pnl > 0:
                    perf['winning_trades'] += 1
        
        # Calculate final metrics
        for key, perf in strategy_regime_performance.items():
            if perf['total_trades'] > 0:
                perf['win_rate'] = perf['winning_trades'] / perf['total_trades']
                perf['avg_pnl'] = perf['total_pnl'] / perf['total_trades']
                perf['sharpe_approx'] = (
                    sum(perf['pnl_list']) / (len(perf['pnl_list']) * max(1, pd.Series(perf['pnl_list']).std()))
                    if len(perf['pnl_list']) > 1 else 0
                )
            
            # Remove raw PnL list for cleaner output
            del perf['pnl_list']
        
        return strategy_regime_performance