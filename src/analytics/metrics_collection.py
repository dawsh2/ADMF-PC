"""
Simple Performance Metrics Collection

Provides a clean interface for collecting performance metrics from containers
without needing full event trace analysis.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Simple metrics collector that gets performance data from containers.
    
    No event tracing needed - just asks containers for their metrics.
    """
    
    @staticmethod
    def collect_from_containers(containers: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect metrics from all containers.
        
        Args:
            containers: Dict of container instances
            
        Returns:
            Aggregated metrics from all containers
        """
        metrics = {
            'by_container': {},
            'aggregate': {}
        }
        
        portfolio_metrics = []
        
        for container_name, container in containers.items():
            # Get metrics based on container type
            if 'portfolio' in container_name.lower():
                # Portfolio containers have performance metrics
                container_metrics = MetricsCollector._get_portfolio_metrics(container)
                if container_metrics:
                    metrics['by_container'][container_name] = container_metrics
                    portfolio_metrics.append(container_metrics)
            
            elif 'execution' in container_name.lower():
                # Execution containers have trade statistics
                exec_metrics = MetricsCollector._get_execution_metrics(container)
                if exec_metrics:
                    metrics['by_container'][container_name] = exec_metrics
        
        # Aggregate portfolio metrics
        if portfolio_metrics:
            metrics['aggregate'] = MetricsCollector._aggregate_portfolio_metrics(portfolio_metrics)
        
        return metrics
    
    @staticmethod
    def _get_portfolio_metrics(container) -> Optional[Dict[str, Any]]:
        """Get metrics from a portfolio container."""
        # Try common methods containers might have
        if hasattr(container, 'get_metrics'):
            return container.get_metrics()
        
        if hasattr(container, 'get_performance'):
            return container.get_performance()
        
        # Try to get from portfolio state component
        if hasattr(container, 'get_component'):
            portfolio_state = container.get_component('portfolio_state')
            if portfolio_state and hasattr(portfolio_state, 'get_metrics'):
                return portfolio_state.get_metrics()
        
        # Fallback: look for common attributes
        metrics = {}
        for attr in ['sharpe_ratio', 'total_return', 'max_drawdown', 'total_value']:
            if hasattr(container, attr):
                metrics[attr] = getattr(container, attr)
        
        return metrics if metrics else None
    
    @staticmethod
    def _get_execution_metrics(container) -> Optional[Dict[str, Any]]:
        """Get metrics from an execution container."""
        if hasattr(container, 'get_statistics'):
            return container.get_statistics()
        
        # Try execution engine component
        if hasattr(container, 'get_component'):
            exec_engine = container.get_component('execution_engine')
            if exec_engine and hasattr(exec_engine, 'get_statistics'):
                return exec_engine.get_statistics()
        
        return None
    
    @staticmethod
    def _aggregate_portfolio_metrics(portfolio_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics from multiple portfolios."""
        if not portfolio_metrics:
            return {}
        
        # Find best performing portfolio
        best_sharpe = -float('inf')
        best_portfolio = None
        
        for metrics in portfolio_metrics:
            sharpe = metrics.get('sharpe_ratio', -float('inf'))
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_portfolio = metrics
        
        # Calculate averages
        aggregate = {
            'best_sharpe_ratio': best_sharpe,
            'best_metrics': best_portfolio,
            'portfolio_count': len(portfolio_metrics)
        }
        
        # Average metrics
        for metric_name in ['sharpe_ratio', 'total_return', 'max_drawdown']:
            values = [m.get(metric_name) for m in portfolio_metrics if metric_name in m]
            if values:
                aggregate[f'avg_{metric_name}'] = sum(values) / len(values)
        
        return aggregate


def get_phase_metrics(topology: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple function to get metrics from a topology's containers.
    
    This is what the Sequencer would call instead of doing trace analysis.
    
    Args:
        topology: Topology dict with containers
        
    Returns:
        Performance metrics
    """
    containers = topology.get('containers', {})
    
    if not containers:
        return {
            'success': True,
            'metrics': {},
            'message': 'No containers to collect metrics from'
        }
    
    collector = MetricsCollector()
    metrics = collector.collect_from_containers(containers)
    
    # Return in a simple format
    result = {
        'success': True,
        'metrics': metrics.get('aggregate', {}),
        'container_metrics': metrics.get('by_container', {})
    }
    
    # Add the most important metrics at the top level for easy access
    if 'best_sharpe_ratio' in metrics.get('aggregate', {}):
        result['sharpe_ratio'] = metrics['aggregate']['best_sharpe_ratio']
    
    if 'avg_total_return' in metrics.get('aggregate', {}):
        result['total_return'] = metrics['aggregate']['avg_total_return']
    
    return result
