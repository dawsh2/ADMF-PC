"""
Example of how the Coordinator uses the composable container system.

This demonstrates the flexibility of the container composition engine
and how different patterns can be easily configured and executed.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
import logging

from ..core.containers.composition_engine import (
    get_global_composition_engine, 
    ContainerCompositionEngine,
    PatternManager
)
from ..core.containers.composable import ContainerRole


logger = logging.getLogger(__name__)


class BacktestCoordinator:
    """
    Example coordinator showing how to use composable containers.
    
    This demonstrates how the coordinator can easily switch between
    different container arrangements without changing the core logic.
    """
    
    def __init__(self):
        self.composition_engine = get_global_composition_engine()
        self.pattern_manager = PatternManager()
        self.active_containers = {}
    
    async def run_simple_backtest(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a simple backtest using minimal container structure."""
        
        # Use simple_backtest pattern: Data → Strategy → Execution
        container_config = {
            'data': {
                'source': 'historical',
                'symbols': config.get('symbols', ['SPY']),
                'start_date': config.get('start_date'),
                'end_date': config.get('end_date')
            },
            'strategy': {
                'type': config.get('strategy_type', 'momentum'),
                'parameters': config.get('strategy_params', {})
            },
            'execution': {
                'mode': 'backtest',
                'initial_capital': config.get('initial_capital', 100000)
            }
        }
        
        # Compose container pattern
        root_container = self.composition_engine.compose_pattern(
            pattern_name="simple_backtest",
            config_overrides=container_config
        )
        
        return await self._execute_backtest(root_container, "simple_backtest")
    
    async def run_full_backtest(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run full backtest with complete container hierarchy."""
        
        # Use full_backtest pattern: Data → Indicator → Classifier → Risk → Portfolio → Strategy → Execution
        container_config = {
            'data': {
                'source': 'historical',
                'symbols': config.get('symbols', ['SPY']),
                'start_date': config.get('start_date'),
                'end_date': config.get('end_date')
            },
            'indicator': {
                'max_indicators': 50
            },
            'classifier': {
                'type': config.get('classifier_type', 'hmm'),
                'parameters': config.get('classifier_params', {})
            },
            'risk': {
                'profile': config.get('risk_profile', 'conservative'),
                'max_position_size': config.get('max_position_size', 0.02),
                'max_total_exposure': config.get('max_total_exposure', 0.10)
            },
            'portfolio': {
                'allocation': config.get('portfolio_allocation', 100000),
                'rebalance_frequency': config.get('rebalance_frequency', 'daily')
            },
            'strategy': {
                'type': config.get('strategy_type', 'momentum'),
                'parameters': config.get('strategy_params', {})
            },
            'execution': {
                'mode': 'backtest',
                'initial_capital': config.get('initial_capital', 100000)
            }
        }
        
        # Compose container pattern
        root_container = self.composition_engine.compose_pattern(
            pattern_name="full_backtest",
            config_overrides=container_config
        )
        
        return await self._execute_backtest(root_container, "full_backtest")
    
    async def run_signal_generation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run signal generation only - no execution."""
        
        container_config = {
            'data': {
                'source': 'historical',
                'symbols': config.get('symbols', ['SPY']),
                'start_date': config.get('start_date'),
                'end_date': config.get('end_date')
            },
            'indicator': {
                'max_indicators': 50
            },
            'classifier': {
                'type': config.get('classifier_type', 'hmm'),
                'parameters': config.get('classifier_params', {})
            },
            'strategy': {
                'type': config.get('strategy_type', 'momentum'),
                'parameters': config.get('strategy_params', {})
            },
            'analysis': {
                'mode': 'signal_generation',
                'output_path': config.get('signal_output_path', './signals/')
            }
        }
        
        # Compose container pattern
        root_container = self.composition_engine.compose_pattern(
            pattern_name="signal_generation",
            config_overrides=container_config
        )
        
        return await self._execute_backtest(root_container, "signal_generation")
    
    async def run_custom_pattern(self, pattern_structure: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Run backtest with custom container arrangement."""
        
        # Example custom pattern: Data → [Parallel Strategies] → Execution
        custom_structure = {
            "root": {
                "role": "data",
                "children": {
                    "strategy1": {
                        "role": "strategy",
                        "config": {"type": "momentum"}
                    },
                    "strategy2": {
                        "role": "strategy", 
                        "config": {"type": "mean_reversion"}
                    },
                    "execution": {
                        "role": "execution"
                    }
                }
            }
        }
        
        # Use provided structure or default
        structure = pattern_structure or custom_structure
        
        # Compose custom pattern
        root_container = self.composition_engine.compose_custom_pattern(
            structure=structure,
            config=config
        )
        
        return await self._execute_backtest(root_container, "custom_pattern")
    
    async def run_parallel_optimization(self, optimization_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run multiple backtests in parallel with different configurations."""
        
        parameter_grid = optimization_config.get('parameter_grid', [])
        pattern_name = optimization_config.get('pattern', 'full_backtest')
        
        # Create tasks for parallel execution
        tasks = []
        for i, params in enumerate(parameter_grid):
            task = asyncio.create_task(
                self._run_single_optimization_instance(pattern_name, params, f"opt_{i}")
            )
            tasks.append(task)
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]
        
        return {
            'total_runs': len(parameter_grid),
            'successful': len(successful_results),
            'failed': len(failed_results),
            'results': successful_results,
            'errors': failed_results
        }
    
    async def _run_single_optimization_instance(
        self, 
        pattern_name: str, 
        config: Dict[str, Any], 
        instance_id: str
    ) -> Dict[str, Any]:
        """Run a single optimization instance."""
        
        try:
            # Compose container for this instance
            root_container = self.composition_engine.compose_pattern(
                pattern_name=pattern_name,
                config_overrides=config
            )
            
            # Add instance identifier
            root_container.metadata.tags.add(f"optimization_instance_{instance_id}")
            
            # Execute
            result = await self._execute_backtest(root_container, f"{pattern_name}_{instance_id}")
            result['instance_id'] = instance_id
            result['config'] = config
            
            return result
            
        except Exception as e:
            logger.error(f"Optimization instance {instance_id} failed: {e}")
            raise e
    
    async def _execute_backtest(self, root_container, backtest_id: str) -> Dict[str, Any]:
        """Execute a backtest with the given container."""
        
        try:
            # Store active container
            self.active_containers[backtest_id] = root_container
            
            # Initialize and start container
            await root_container.initialize()
            await root_container.start()
            
            logger.info(f"Started backtest: {backtest_id}")
            
            # Wait for completion (this would be more sophisticated in real implementation)
            # For now, simulate some execution time
            await asyncio.sleep(0.1)  # Placeholder
            
            # Get results
            results = {
                'backtest_id': backtest_id,
                'container_id': root_container.metadata.container_id,
                'status': root_container.state.value,
                'metrics': root_container.get_status(),
                'start_time': datetime.now().isoformat(),
                'container_structure': self._get_container_structure(root_container)
            }
            
            # Stop and dispose container
            await root_container.stop()
            await root_container.dispose()
            
            # Remove from active containers
            del self.active_containers[backtest_id]
            
            logger.info(f"Completed backtest: {backtest_id}")
            return results
            
        except Exception as e:
            logger.error(f"Backtest {backtest_id} failed: {e}")
            
            # Clean up on error
            if backtest_id in self.active_containers:
                try:
                    await root_container.dispose()
                    del self.active_containers[backtest_id]
                except:
                    pass
            
            raise e
    
    def _get_container_structure(self, container) -> Dict[str, Any]:
        """Get hierarchical structure of container for debugging."""
        structure = {
            'id': container.metadata.container_id,
            'role': container.metadata.role.value,
            'name': container.metadata.name,
            'state': container.state.value,
            'children': []
        }
        
        for child in container.child_containers:
            structure['children'].append(self._get_container_structure(child))
        
        return structure
    
    def get_available_patterns(self) -> Dict[str, Any]:
        """Get all available container patterns."""
        patterns = self.composition_engine.registry.list_available_patterns()
        
        pattern_info = {}
        for pattern_name in patterns:
            pattern = self.composition_engine.registry.get_pattern(pattern_name)
            if pattern:
                pattern_info[pattern_name] = {
                    'description': pattern.description,
                    'required_capabilities': list(pattern.required_capabilities),
                    'default_config': pattern.default_config
                }
        
        return pattern_info
    
    async def shutdown(self) -> None:
        """Shutdown coordinator and clean up active containers."""
        logger.info("Shutting down coordinator...")
        
        # Stop all active containers
        for backtest_id, container in self.active_containers.items():
            try:
                await container.stop()
                await container.dispose()
                logger.info(f"Cleaned up container: {backtest_id}")
            except Exception as e:
                logger.error(f"Error cleaning up container {backtest_id}: {e}")
        
        self.active_containers.clear()
        logger.info("Coordinator shutdown complete")


# Example usage
async def main():
    """Example of how to use the coordinator."""
    
    coordinator = BacktestCoordinator()
    
    # Example 1: Simple backtest
    simple_config = {
        'symbols': ['SPY'],
        'start_date': datetime.now() - timedelta(days=365),
        'end_date': datetime.now(),
        'strategy_type': 'momentum',
        'initial_capital': 100000
    }
    
    print("Running simple backtest...")
    result = await coordinator.run_simple_backtest(simple_config)
    print(f"Simple backtest result: {result['backtest_id']}")
    
    # Example 2: Full backtest with all containers
    full_config = {
        'symbols': ['SPY', 'QQQ'],
        'start_date': datetime.now() - timedelta(days=365),
        'end_date': datetime.now(),
        'classifier_type': 'hmm',
        'risk_profile': 'conservative',
        'strategy_type': 'momentum',
        'initial_capital': 100000
    }
    
    print("Running full backtest...")
    result = await coordinator.run_full_backtest(full_config)
    print(f"Full backtest result: {result['backtest_id']}")
    
    # Example 3: Show available patterns
    patterns = coordinator.get_available_patterns()
    print(f"Available patterns: {list(patterns.keys())}")
    
    # Clean up
    await coordinator.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())