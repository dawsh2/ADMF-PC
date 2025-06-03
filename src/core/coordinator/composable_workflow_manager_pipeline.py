"""
Composable workflow manager that uses pipeline communication adapters.

This version sets up pipeline adapters for linear event flow between containers,
eliminating circular dependencies.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .simple_types import WorkflowConfig, WorkflowType
from ..containers.composable import ComposableContainerProtocol, ContainerState
from ..containers.composition_engine import get_global_composition_engine
from ..events.event_bus import EventBus
from ..events.types import Event, EventType

logger = logging.getLogger(__name__)


class ComposableWorkflowManagerPipeline:
    """Manages workflow execution using composable containers with pipeline communication."""
    
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.container = None
        self.event_bus = EventBus()
        self.completion_event = asyncio.Event()
        
    async def execute_workflow(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Execute workflow using composable containers with pipeline communication."""
        logger.info("Executing backtest workflow with composable containers and pipeline communication")
        
        # Extract container pattern
        container_pattern = config.parameters.get('container_pattern', 'simple_backtest')
        
        # Build configuration for composition
        composition_config = self._build_composition_config(config)
        
        # Get composition engine
        engine = get_global_composition_engine()
        
        # Add strategy config
        strategies = config.backtest_config.get('strategies', [])
        if strategies:
            logger.info(f"Found {len(strategies)} strategies from config")
            for i, strategy in enumerate(strategies):
                logger.info(f"  Strategy {i+1}: {strategy}")
            
            # Configure strategy container with correct key
            if len(strategies) == 1:
                # Single strategy - use the strategy directly
                strategy_config = strategies[0]
                composition_config['strategy'] = {
                    'type': strategy_config.get('type', 'momentum'),
                    'parameters': strategy_config.get('parameters', {}),
                    'name': strategy_config.get('name', 'primary_strategy')
                }
                logger.info(f"Configured single strategy: {composition_config['strategy']}")
            else:
                # Multiple strategies - pass all strategies
                composition_config['strategy'] = {
                    'strategies': strategies,
                    'aggregation': {
                        'method': 'weighted_voting',
                        'min_confidence': 0.6
                    }
                }
                logger.info(f"Configured multi-strategy with {len(strategies)} strategies")
            
            # Infer required indicators from strategies
            required_indicators = self._infer_indicators_from_strategies(strategies)
            logger.info(f"Inferred {len(required_indicators)} indicators from strategies: {required_indicators}")
            
            # Add to indicator config
            composition_config['indicator'] = {
                'indicators': list(required_indicators)
            }
        
        # Add risk and portfolio config 
        if 'risk' in config.parameters:
            composition_config['risk'] = config.parameters['risk']
            logger.info(f"Added risk config to {container_pattern}: {config.parameters['risk']}")
            
        if 'risk' in composition_config and 'portfolio' not in composition_config:
            composition_config['portfolio'] = {
                'initial_capital': composition_config['risk'].get('initial_capital', 100000)
            }
            logger.info(f"Added portfolio config to {container_pattern}: {composition_config['portfolio']}")
        
        try:
            # For simple_backtest, create a custom structure without classifier
            if container_pattern == 'simple_backtest':
                logger.info("Creating custom simple_backtest structure without classifier")
                custom_structure = {
                    "root": {
                        "role": "backtest",
                        "children": {
                            "data": {"role": "data"},
                            "indicators": {"role": "indicator"},
                            "risk": {
                                "role": "risk",
                                "children": {
                                    "portfolio": {
                                        "role": "portfolio",
                                        "children": {
                                            "strategy": {"role": "strategy"}
                                        }
                                    }
                                }
                            },
                            "execution": {"role": "execution"}
                        }
                    }
                }
                self.container = engine.compose_custom_pattern(custom_structure, composition_config)
            else:
                # Use standard pattern
                logger.info(f"Executing standard pattern: {container_pattern}")
                self.container = engine.compose_pattern(container_pattern, composition_config)
            
            # Get all containers for pipeline setup
            all_containers = self._collect_all_containers(self.container)
            container_map = {c.metadata.container_id: c for c in all_containers}
            
            # Setup pipeline communication if coordinator has communication layer
            if hasattr(self.coordinator, 'communication_layer') and self.coordinator.communication_layer:
                # Configure pipeline order based on container pattern
                pipeline_order = self._determine_pipeline_order(container_pattern, all_containers)
                
                # Find the pipeline adapter and setup the complete pipeline
                for adapter_name, adapter in self.coordinator.communication_layer.adapters.items():
                    if hasattr(adapter, 'setup_pipeline'):
                        # Use proper setup_pipeline method which includes reverse routing
                        adapter.setup_pipeline(pipeline_order)
                        logger.info(f"Updated pipeline adapter with {len(pipeline_order)} containers")
                        break
            
            # Setup monitoring
            self._setup_completion_monitoring()
            
            # Initialize containers
            await self.container.initialize()
            
            # Start execution
            await self.container.start()
            
            # Wait for completion
            await self._wait_for_completion()
            
            # Extract results
            result = await self._extract_results()
            
            return result
            
        finally:
            # Cleanup
            if self.container:
                await self.container.stop()
    
    def _determine_pipeline_order(self, pattern_name: str, containers: List[ComposableContainerProtocol]) -> List[ComposableContainerProtocol]:
        """Determine the order of containers in the pipeline based on pattern."""
        # Create a map by role
        role_map = {}
        for container in containers:
            role = container.metadata.role.value
            if role not in role_map:
                role_map[role] = []
            role_map[role].append(container)
        
        # Define pipeline order for simple_backtest pattern
        if pattern_name == 'simple_backtest':
            pipeline_roles = ['data', 'indicator', 'strategy', 'portfolio', 'risk', 'execution']
        else:
            # Default order
            pipeline_roles = ['data', 'indicator', 'classifier', 'strategy', 'portfolio', 'risk', 'execution']
        
        # Build ordered list
        pipeline_order = []
        for role in pipeline_roles:
            if role in role_map:
                pipeline_order.extend(role_map[role])
        
        logger.info(f"Pipeline order: {[c.metadata.name for c in pipeline_order]}")
        return pipeline_order
    
    def _collect_all_containers(self, root: ComposableContainerProtocol) -> List[ComposableContainerProtocol]:
        """Collect all containers in the hierarchy."""
        containers = [root]
        for child in root.child_containers:
            containers.extend(self._collect_all_containers(child))
        return containers
    
    def _build_composition_config(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Build configuration for container composition."""
        composition_config = {
            'data': config.data_config,
            'backtest': config.backtest_config,
        }
        
        # Add any additional configuration from parameters
        if 'risk' in config.parameters:
            composition_config['risk'] = config.parameters['risk']
            
        if 'indicator' in config.parameters:
            composition_config['indicator'] = config.parameters['indicator']
            
        if 'execution' in config.parameters:
            composition_config['execution'] = config.parameters['execution']
        
        # Extract strategies if present
        if 'strategies' in config.parameters:
            composition_config['strategies'] = config.parameters['strategies']
        
        return composition_config
    
    def _infer_indicators_from_strategies(self, strategies: List[Dict[str, Any]]) -> set:
        """Infer required indicators from strategy configurations."""
        required_indicators = set()
        
        for strategy in strategies:
            strategy_type = strategy.get('type')
            
            if strategy_type == 'momentum':
                # Momentum strategy typically needs SMA and RSI
                params = strategy.get('parameters', {})
                lookback = params.get('lookback_period', 20)
                required_indicators.add(f'SMA_{lookback}')
                required_indicators.add('RSI')
                
            elif strategy_type == 'mean_reversion':
                # Mean reversion typically needs Bollinger Bands and RSI
                params = strategy.get('parameters', {})
                lookback = params.get('lookback_period', 20)
                required_indicators.add(f'BB_{lookback}')
                required_indicators.add('RSI')
                
            elif strategy_type == 'trend_following':
                # Trend following might need multiple SMAs
                required_indicators.add('SMA_10')
                required_indicators.add('SMA_20')
                required_indicators.add('SMA_50')
                
        return required_indicators
    
    def _setup_completion_monitoring(self):
        """Setup monitoring for workflow completion."""
        # Subscribe to system events
        if self.container:
            # Find data container for END_OF_DATA events
            data_containers = [c for c in self._collect_all_containers(self.container) 
                             if c.metadata.role.value == 'data']
            
            for data_container in data_containers:
                data_container.event_bus.subscribe(
                    EventType.SYSTEM,
                    self._handle_system_event
                )
    
    def _handle_system_event(self, event: Event):
        """Handle system events for completion detection."""
        if event.payload.get('message') == 'END_OF_DATA':
            logger.info("Received END_OF_DATA signal")
            self.completion_event.set()
    
    async def _wait_for_completion(self):
        """Wait for workflow completion."""
        # Use a timeout to prevent hanging
        timeout = 300  # 5 minutes
        
        try:
            await asyncio.wait_for(self.completion_event.wait(), timeout)
            logger.info("Workflow completed normally")
        except asyncio.TimeoutError:
            logger.warning(f"Workflow timeout after {timeout} seconds")
            
        # Give containers time to process final events
        await asyncio.sleep(2)
    
    async def _extract_results(self) -> Dict[str, Any]:
        """Extract results from completed workflow."""
        result = {
            'success': True,
            'container_id': self.container.metadata.container_id if self.container else None,
            'completion_time': datetime.now(),
        }
        
        # Extract portfolio performance
        try:
            portfolio_value = self._extract_portfolio_value()
            if portfolio_value:
                result['portfolio'] = portfolio_value
        except Exception as e:
            logger.error(f"Error extracting portfolio: {e}")
        
        # Extract execution data
        try:
            execution_data = self._extract_execution_data()
            if execution_data:
                result.update(execution_data)
        except Exception as e:
            logger.error(f"Error extracting execution data: {e}")
        
        return result
    
    def _extract_portfolio_value(self) -> Optional[Dict[str, Any]]:
        """Extract portfolio value from portfolio containers."""
        if not self.container:
            return None
            
        # Find portfolio container first (new architecture)
        all_containers = self._collect_all_containers(self.container)
        portfolio_containers = [c for c in all_containers if c.metadata.role.value == 'portfolio']
        
        for portfolio_container in portfolio_containers:
            if hasattr(portfolio_container, 'portfolio_state') and portfolio_container.portfolio_state:
                portfolio_state = portfolio_container.portfolio_state
                
                try:
                    # Get portfolio metrics
                    cash_balance = portfolio_state.get_cash_balance()
                    positions = portfolio_state.get_all_positions()
                    
                    # Calculate total position value
                    position_value = 0
                    for symbol, position in positions.items():
                        if hasattr(position, 'market_value'):
                            position_value += float(position.market_value)
                        elif hasattr(position, 'quantity') and hasattr(position, 'current_price'):
                            position_value += float(position.quantity) * float(position.current_price)
                    
                    total_equity = float(cash_balance) + position_value
                    
                    logger.info(f"📊 Final Portfolio Summary:")
                    logger.info(f"   💰 Cash Balance: ${cash_balance:.2f}")
                    logger.info(f"   📈 Position Value: ${position_value:.2f}")
                    logger.info(f"   🏆 Total Equity: ${total_equity:.2f}")
                    logger.info(f"   📍 Number of Positions: {len(positions)}")
                    
                    return {
                        'cash_balance': float(cash_balance),
                        'position_value': position_value,
                        'total_equity': total_equity,
                        'positions': len(positions),
                        'position_details': {
                            symbol: {
                                'quantity': float(position.quantity) if hasattr(position, 'quantity') else 0,
                                'market_value': float(position.market_value) if hasattr(position, 'market_value') else 0
                            }
                            for symbol, position in positions.items()
                        }
                    }
                except Exception as e:
                    logger.error(f"Error calculating portfolio value from PortfolioContainer: {e}")
        
        # Fallback to risk container (legacy architecture)
        risk_containers = [c for c in all_containers if c.metadata.role.value == 'risk']
        
        for risk_container in risk_containers:
            if hasattr(risk_container, 'risk_manager') and risk_container.risk_manager:
                risk_manager = risk_container.risk_manager
                portfolio_state = risk_manager.get_portfolio_state()
                
                if portfolio_state:
                    try:
                        # Get portfolio metrics
                        cash_balance = portfolio_state.get_cash_balance()
                        positions = portfolio_state.get_all_positions()
                        
                        # Calculate total position value
                        position_value = 0
                        for symbol, position in positions.items():
                            if hasattr(position, 'market_value'):
                                position_value += float(position.market_value)
                            elif hasattr(position, 'quantity') and hasattr(position, 'current_price'):
                                position_value += float(position.quantity) * float(position.current_price)
                        
                        total_equity = float(cash_balance) + position_value
                        
                        logger.info(f"📊 Final Portfolio Summary (from RiskContainer):")
                        logger.info(f"   💰 Cash Balance: ${cash_balance:.2f}")
                        logger.info(f"   📈 Position Value: ${position_value:.2f}")
                        logger.info(f"   🏆 Total Equity: ${total_equity:.2f}")
                        logger.info(f"   📍 Number of Positions: {len(positions)}")
                        
                        return {
                            'cash_balance': float(cash_balance),
                            'position_value': position_value,
                            'total_equity': total_equity,
                            'positions': len(positions),
                            'position_details': {
                                symbol: {
                                    'quantity': float(position.quantity) if hasattr(position, 'quantity') else 0,
                                    'market_value': float(position.market_value) if hasattr(position, 'market_value') else 0
                                }
                                for symbol, position in positions.items()
                            }
                        }
                    except Exception as e:
                        logger.error(f"Error calculating portfolio value from RiskContainer: {e}")
                        
        logger.warning("No portfolio or risk container found with portfolio state")
        return None
    
    def _extract_execution_data(self) -> Dict[str, Any]:
        """Extract execution data from execution container."""
        if not self.container:
            return {}
            
        # Find execution container
        all_containers = self._collect_all_containers(self.container)
        exec_containers = [c for c in all_containers if c.metadata.role.value == 'execution']
        
        data = {}
        for exec_container in exec_containers:
            if hasattr(exec_container, 'broker'):
                broker = exec_container.broker
                if hasattr(broker, 'get_fills'):
                    fills = broker.get_fills()
                    data['total_trades'] = len(fills)
                    data['fills'] = fills
                    
                if hasattr(broker, 'get_performance'):
                    data['performance'] = broker.get_performance()
        
        return data