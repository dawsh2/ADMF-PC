"""
Adapter to run EVENT_FLOW_ARCHITECTURE backtests through the Coordinator.

This module adapts the new simplified configuration format to work with
the existing Coordinator infrastructure.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .workflows import WorkflowManager
from ..types.workflow import WorkflowConfig, WorkflowType, ExecutionContext, WorkflowResult
from ..containers.symbol_timeframe_container import SymbolTimeframeContainer
from ..containers.portfolio_container import PortfolioContainer
from ..containers.execution_container import ExecutionContainer
from ..communication.factory import AdapterFactory

logger = logging.getLogger(__name__)


async def run_event_flow_backtest(config: Dict[str, Any]) -> WorkflowResult:
    """
    Run a backtest using the EVENT_FLOW_ARCHITECTURE.
    
    This bypasses the complex Coordinator workflow patterns and directly
    implements the simplified event flow.
    
    Args:
        config: Backtest configuration dictionary
        
    Returns:
        WorkflowResult with backtest results
    """
    # Extract configuration sections
    backtest_config = config.get('backtest', {})
    data_config = backtest_config.get('data', {})
    features_config = backtest_config.get('features', {})
    strategies = backtest_config.get('strategies', [])
    risk_profiles = backtest_config.get('risk_profiles', [])
    execution_config = backtest_config.get('execution', {})
    portfolio_config = backtest_config.get('portfolio', {})
    output_config = backtest_config.get('output', {})
    
    # Create result object
    result = WorkflowResult(
        workflow_id=f"event_flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        workflow_type=WorkflowType.BACKTEST,
        success=True,
        metadata={'architecture': 'EVENT_FLOW'}
    )
    
    try:
        # 1. Create containers
        containers = {}
        
        # Create symbol-timeframe containers
        symbols = data_config.get('symbols', ['SPY'])
        for symbol in symbols:
            container_id = f"{symbol}_1d"
            symbol_container = SymbolTimeframeContainer(
                symbol=symbol,
                timeframe='1d',
                data_config={
                    'source': data_config.get('source', 'csv'),
                    'file': f"data/{symbol}.csv",
                    'start_date': data_config.get('start_date'),
                    'end_date': data_config.get('end_date')
                },
                feature_config=features_config,
                container_id=container_id
            )
            containers[container_id] = symbol_container
        
        # Create portfolio containers (one per strategy/risk combination)
        portfolio_containers = []
        combo_id = 0
        for strategy_config in strategies:
            for risk_config in risk_profiles:
                portfolio_id = f"portfolio_c{combo_id:04d}"
                portfolio = PortfolioContainer(
                    combo_id=f"c{combo_id:04d}",
                    strategy_params=strategy_config,
                    risk_params=risk_config,
                    initial_capital=portfolio_config.get('initial_capital', 100000),
                    container_id=portfolio_id
                )
                
                # Set up stateless services
                strategy_type = strategy_config.get('type', 'momentum')
                if strategy_type == 'momentum':
                    from ...strategy.strategies.momentum import momentum_strategy
                    portfolio.set_strategy_service(momentum_strategy)
                
                risk_type = risk_config.get('type', 'conservative')
                # Risk validation happens through RiskServiceAdapter, not directly in portfolio
                
                containers[portfolio_id] = portfolio
                portfolio_containers.append(portfolio)
                combo_id += 1
        
        # Create execution container
        execution = ExecutionContainer(
            execution_config=execution_config,
            container_id='execution'
        )
        containers['execution'] = execution
        
        # 2. Set up communication adapters
        adapter_factory = AdapterFactory()
        adapters = []
        
        # Symbol → Portfolio broadcasts (FEATURES)
        for symbol in symbols:
            symbol_id = f"{symbol}_1d"
            portfolio_ids = [f"portfolio_c{i:04d}" for i in range(len(portfolio_containers))]
            
            feature_broadcast = adapter_factory.create_adapter(
                name=f'feature_broadcast_{symbol}',
                config={
                    'type': 'broadcast',
                    'source': symbol_id,
                    'targets': portfolio_ids
                }
            )
            adapters.append(feature_broadcast)
        
        # Portfolio → Execution routing (ORDERS)
        from ..types.events import EventType
        for i, portfolio in enumerate(portfolio_containers):
            portfolio_id = f"portfolio_c{i:04d}"
            order_routing = adapter_factory.create_adapter(
                name=f'order_routing_{i}',
                config={
                    'type': 'selective',
                    'source': portfolio_id,
                    'route_by_type': {
                        EventType.ORDER: 'execution'
                    },
                    'routing_rules': [],
                    'default_target': None
                }
            )
            adapters.append(order_routing)
        
        # Execution → Portfolio broadcast (FILLS)
        portfolio_ids = [f"portfolio_c{i:04d}" for i in range(len(portfolio_containers))]
        fill_broadcast = adapter_factory.create_adapter(
            name='fill_broadcast',
            config={
                'type': 'broadcast',
                'source': 'execution',
                'targets': portfolio_ids
            }
        )
        adapters.append(fill_broadcast)
        
        # Wire up adapters
        for adapter in adapters:
            adapter.setup(containers)
            adapter.start()
        
        # 3. Initialize and start containers
        logger.info("Initializing containers...")
        for container in containers.values():
            await container.initialize()
        
        logger.info("Starting containers...")
        for container in containers.values():
            await container.start()
        
        # 4. Wait for processing to complete
        # In a real implementation, we'd wait for data exhaustion signal
        await asyncio.sleep(2)
        
        # 5. Collect results
        logger.info("Collecting results...")
        
        portfolio_results = {}
        for portfolio in portfolio_containers:
            portfolio_state = {
                'combo_id': portfolio.combo_id,
                'cash': portfolio.portfolio_state.cash,
                'total_value': portfolio.portfolio_state.total_value,
                'positions': len(portfolio.portfolio_state.positions),
                'metrics': portfolio.metrics,
                'orders_created': portfolio._orders_created,
                'signals_generated': portfolio._signals_generated
            }
            portfolio_results[portfolio.combo_id] = portfolio_state
        
        # Get execution stats
        exec_stats = execution.get_execution_stats()
        
        # Find best performing portfolio
        best_portfolio = max(
            portfolio_results.values(),
            key=lambda p: p['total_value'],
            default=None
        )
        
        # Prepare final results
        result.final_results = {
            'architecture': 'EVENT_FLOW',
            'portfolios': portfolio_results,
            'execution_stats': exec_stats,
            'best_portfolio': best_portfolio,
            'summary': {
                'total_portfolios': len(portfolio_containers),
                'total_orders': sum(p['orders_created'] for p in portfolio_results.values()),
                'total_fills': exec_stats['orders_filled'],
                'total_commission': exec_stats['total_commission']
            }
        }
        
        # 6. Save results if configured
        if output_config.get('results_dir'):
            results_dir = Path(output_config['results_dir'])
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save summary
            import json
            with open(results_dir / 'summary.json', 'w') as f:
                json.dump(result.final_results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {results_dir}")
        
        # 7. Stop containers and adapters
        logger.info("Stopping containers...")
        for container in containers.values():
            await container.stop()
        
        for adapter in adapters:
            adapter.stop()
        
        logger.info("EVENT_FLOW backtest completed successfully")
        
    except Exception as e:
        logger.error(f"EVENT_FLOW backtest failed: {e}")
        result.success = False
        result.add_error(str(e))
    
    return result


def adapt_yaml_config_for_event_flow(yaml_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapt a YAML configuration to the EVENT_FLOW format if needed.
    
    This allows backward compatibility with existing configs.
    """
    # If it's already in EVENT_FLOW format, return as-is
    if 'backtest' in yaml_config and yaml_config.get('backtest', {}).get('mode') == 'backtest':
        return yaml_config
    
    # Otherwise, try to adapt old format
    adapted = {'backtest': {}}
    
    # Map data config
    if 'data' in yaml_config:
        adapted['backtest']['data'] = yaml_config['data']
    
    # Map strategies
    if 'strategies' in yaml_config:
        adapted['backtest']['strategies'] = yaml_config['strategies']
    
    # Map risk
    if 'risk' in yaml_config:
        adapted['backtest']['risk_profiles'] = [yaml_config['risk']]
    
    # Map portfolio
    if 'portfolio' in yaml_config:
        adapted['backtest']['portfolio'] = yaml_config['portfolio']
    
    return adapted