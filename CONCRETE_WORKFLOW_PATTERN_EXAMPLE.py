"""
Concrete example of smart multi-parameter workflow pattern implementation.

This shows exactly how containers are created and wired up with proper sharing.
"""

import asyncio
import uuid
from typing import Dict, Any, List
from dataclasses import dataclass

# Mock imports - in real code these would be actual imports
from core.containers.factory import get_global_factory
from core.communication.factory import AdapterFactory
from core.containers.protocols import ContainerRole, ComposableContainer


@dataclass 
class ContainerAnalysis:
    """Analysis of which containers can be shared vs. need separation."""
    shared_risk: Dict[str, Dict[str, Any]]
    shared_execution: Dict[str, Dict[str, Any]]  
    separate_portfolios: List[Dict[str, Any]]
    separate_strategies: List[Dict[str, Any]]


class MultiParameterWorkflowExample:
    """Concrete example of multi-parameter workflow with smart container sharing."""
    
    def __init__(self):
        self.factory = get_global_factory()
        self.adapter_factory = AdapterFactory()
        self.active_containers = {}
        self.active_adapters = []
    
    async def execute_example(self):
        """Execute concrete example with momentum strategy parameter grid."""
        
        # Example: 2 lookback periods × 2 thresholds × 2 capitals = 8 combinations
        param_combinations = [
            {'strategy': {'type': 'momentum', 'parameters': {'lookback_period': 10, 'signal_threshold': 0.01}}, 'initial_capital': 10000},
            {'strategy': {'type': 'momentum', 'parameters': {'lookback_period': 10, 'signal_threshold': 0.02}}, 'initial_capital': 10000},
            {'strategy': {'type': 'momentum', 'parameters': {'lookback_period': 20, 'signal_threshold': 0.01}}, 'initial_capital': 10000},
            {'strategy': {'type': 'momentum', 'parameters': {'lookback_period': 20, 'signal_threshold': 0.02}}, 'initial_capital': 10000},
            {'strategy': {'type': 'momentum', 'parameters': {'lookback_period': 10, 'signal_threshold': 0.01}}, 'initial_capital': 50000},
            {'strategy': {'type': 'momentum', 'parameters': {'lookback_period': 10, 'signal_threshold': 0.02}}, 'initial_capital': 50000},
            {'strategy': {'type': 'momentum', 'parameters': {'lookback_period': 20, 'signal_threshold': 0.01}}, 'initial_capital': 50000},
            {'strategy': {'type': 'momentum', 'parameters': {'lookback_period': 20, 'signal_threshold': 0.02}}, 'initial_capital': 50000},
        ]
        
        base_config = {
            'risk': {
                'max_position_size': 1000,
                'commission': 0.001,
                'position_size_method': 'percentage',
                'position_size_pct': 2.0
            },
            'execution': {
                'commission': 0.001,
                'slippage': 0.0005,
                'mode': 'backtest'
            }
        }
        
        print("🔍 Analyzing container sharing opportunities...")
        analysis = self._analyze_container_sharing(param_combinations, base_config)
        
        print(f"📊 Analysis Results:")
        print(f"   Shared Risk Containers: {len(analysis.shared_risk)}")
        print(f"   Shared Execution Containers: {len(analysis.shared_execution)}")
        print(f"   Separate Portfolio Containers: {len(analysis.separate_portfolios)}")
        print(f"   Separate Strategy Containers: {len(analysis.separate_strategies)}")
        
        print("\n🏗️  Creating containers...")
        containers = await self._create_all_containers(analysis, base_config)
        
        print("\n🔌 Setting up communication...")
        await self._setup_communication(containers, analysis)
        
        print("\n🚀 Starting workflow...")
        await self._start_all_containers(containers)
        
        print("\n⏱️  Simulating execution...")
        await asyncio.sleep(2.0)  # Simulate execution time
        
        print("\n📈 Collecting results...")
        results = self._collect_results(containers, param_combinations)
        
        print("\n🛑 Stopping workflow...")
        await self._stop_all_containers(containers)
        
        return results
    
    def _analyze_container_sharing(self, param_combinations: List[Dict[str, Any]], base_config: Dict[str, Any]) -> ContainerAnalysis:
        """Analyze which containers can be shared vs need separation."""
        
        # Risk analysis - check if all combinations have same risk config
        shared_risk = {}
        risk_config = base_config.get('risk', {})
        risk_hash = self._config_hash(risk_config)
        
        shared_risk[risk_hash] = {
            'container_id': 'shared_risk_0',
            'config': risk_config,
            'serves_combinations': list(range(len(param_combinations)))  # All combinations
        }
        
        # Execution analysis - check if all combinations have same execution config
        shared_execution = {}
        execution_config = base_config.get('execution', {})
        execution_hash = self._config_hash(execution_config)
        
        shared_execution[execution_hash] = {
            'container_id': 'shared_execution_0', 
            'config': execution_config,
            'serves_combinations': list(range(len(param_combinations)))  # All combinations
        }
        
        # Portfolio analysis - always separate (different capital/P&L tracking)
        separate_portfolios = []
        for i, combo in enumerate(param_combinations):
            separate_portfolios.append({
                'combination_index': i,
                'container_id': f'portfolio_combo_{i}',
                'config': {
                    'initial_capital': combo['initial_capital'],
                    'combination_id': f'combo_{i}',
                    'parameter_combination': combo
                }
            })
        
        # Strategy analysis - always separate (different parameters)
        separate_strategies = []
        for i, combo in enumerate(param_combinations):
            separate_strategies.append({
                'combination_index': i,
                'container_id': f'strategy_combo_{i}',
                'config': {
                    **combo['strategy'],
                    'combination_id': f'combo_{i}',
                    'target_portfolio': f'portfolio_combo_{i}'
                }
            })
        
        return ContainerAnalysis(
            shared_risk=shared_risk,
            shared_execution=shared_execution,
            separate_portfolios=separate_portfolios,
            separate_strategies=separate_strategies
        )
    
    async def _create_all_containers(self, analysis: ContainerAnalysis, base_config: Dict[str, Any]) -> Dict[str, ComposableContainer]:
        """Create all containers with proper sharing."""
        containers = {}
        
        # 1. Create hub container for coordination
        print("   Creating hub container...")
        hub_container = self.factory.create_container(
            role=ContainerRole.DATA,
            config={'role': 'coordination_hub', 'coordination_mode': 'multi_portfolio'},
            container_id='multi_param_hub'
        )
        containers['hub'] = hub_container
        
        # 2. Create shared risk containers
        print(f"   Creating {len(analysis.shared_risk)} shared risk containers...")
        for risk_hash, risk_info in analysis.shared_risk.items():
            risk_container = self.factory.create_container(
                role=ContainerRole.RISK,
                config=risk_info['config'],
                container_id=risk_info['container_id']
            )
            containers[risk_info['container_id']] = risk_container
            
            # Add as child of hub
            hub_container.add_child_container(risk_container)
        
        # 3. Create shared execution containers
        print(f"   Creating {len(analysis.shared_execution)} shared execution containers...")
        for exec_hash, exec_info in analysis.shared_execution.items():
            execution_container = self.factory.create_container(
                role=ContainerRole.EXECUTION,
                config=exec_info['config'],
                container_id=exec_info['container_id']
            )
            containers[exec_info['container_id']] = execution_container
            
            # Add as child of hub
            hub_container.add_child_container(execution_container)
        
        # 4. Create separate portfolio containers
        print(f"   Creating {len(analysis.separate_portfolios)} separate portfolio containers...")
        for portfolio_info in analysis.separate_portfolios:
            portfolio_container = self.factory.create_container(
                role=ContainerRole.PORTFOLIO,
                config=portfolio_info['config'],
                container_id=portfolio_info['container_id']
            )
            containers[portfolio_info['container_id']] = portfolio_container
            
            # Add as child of hub
            hub_container.add_child_container(portfolio_container)
        
        # 5. Create separate strategy containers
        print(f"   Creating {len(analysis.separate_strategies)} separate strategy containers...")
        for strategy_info in analysis.separate_strategies:
            strategy_container = self.factory.create_container(
                role=ContainerRole.STRATEGY,
                config=strategy_info['config'],
                container_id=strategy_info['container_id']
            )
            containers[strategy_info['container_id']] = strategy_container
            
            # Connect strategy to its portfolio
            portfolio_id = strategy_info['config']['target_portfolio']
            if portfolio_id in containers:
                containers[portfolio_id].add_child_container(strategy_container)
        
        print(f"   ✅ Created {len(containers)} total containers")
        return containers
    
    async def _setup_communication(self, containers: Dict[str, ComposableContainer], analysis: ContainerAnalysis):
        """Setup optimized communication between containers."""
        
        comm_configs = []
        
        # 1. Hub broadcasts market data to all strategies
        strategy_ids = [s['container_id'] for s in analysis.separate_strategies]
        comm_configs.append({
            'name': 'hub_to_strategies_broadcast',
            'type': 'broadcast',
            'source': 'multi_param_hub',
            'targets': strategy_ids,
            'event_types': ['BAR', 'TICK', 'FEATURE'],
            'log_level': 'INFO'
        })
        
        # 2. All strategies send signals to shared risk container
        # Since we have one shared risk container, use selective routing
        risk_container_id = list(analysis.shared_risk.values())[0]['container_id']
        comm_configs.append({
            'name': 'strategies_to_shared_risk',
            'type': 'selective',
            'sources': strategy_ids,
            'target': risk_container_id,
            'event_types': ['SIGNAL'],
            'routing_method': 'round_robin',  # Distribute signals
            'log_level': 'INFO'
        })
        
        # 3. Shared risk sends orders to shared execution
        execution_container_id = list(analysis.shared_execution.values())[0]['container_id']
        comm_configs.append({
            'name': 'shared_risk_to_shared_execution',
            'type': 'pipeline',
            'source': risk_container_id,
            'target': execution_container_id,
            'event_types': ['ORDER'],
            'log_level': 'INFO'
        })
        
        # 4. Shared execution sends fills to appropriate portfolios
        portfolio_ids = [p['container_id'] for p in analysis.separate_portfolios]
        comm_configs.append({
            'name': 'shared_execution_to_portfolios',
            'type': 'selective',
            'source': execution_container_id,
            'targets': portfolio_ids,
            'event_types': ['FILL'],
            'routing_key': 'target_portfolio',  # Route based on order's target portfolio
            'log_level': 'INFO'
        })
        
        # 5. Portfolios send performance updates back to hub
        comm_configs.append({
            'name': 'portfolios_to_hub_aggregation',
            'type': 'selective',
            'sources': portfolio_ids,
            'target': 'multi_param_hub',
            'event_types': ['PORTFOLIO', 'PERFORMANCE'],
            'aggregation_method': 'collect_all',
            'log_level': 'INFO'
        })
        
        # Create all adapters
        print(f"   Creating {len(comm_configs)} communication adapters...")
        adapters = self.adapter_factory.create_adapters_from_config(comm_configs, containers)
        self.active_adapters.extend(adapters)
        
        # Start adapters
        self.adapter_factory.start_all()
        print(f"   ✅ Communication setup complete")
    
    async def _start_all_containers(self, containers: Dict[str, ComposableContainer]):
        """Initialize and start all containers."""
        for container_id, container in containers.items():
            await container.initialize()
            await container.start()
            self.active_containers[container_id] = container
            print(f"   ✅ Started {container_id}")
    
    async def _stop_all_containers(self, containers: Dict[str, ComposableContainer]):
        """Stop all containers."""
        # Stop in reverse order
        for container_id, container in reversed(list(containers.items())):
            await container.stop()
            print(f"   🛑 Stopped {container_id}")
        
        # Stop adapters
        self.adapter_factory.stop_all()
        print(f"   🛑 Stopped all adapters")
    
    def _collect_results(self, containers: Dict[str, ComposableContainer], param_combinations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collect results from all containers."""
        results = {
            'summary': {
                'total_combinations': len(param_combinations),
                'containers_created': len(containers),
                'shared_containers': 3,  # hub + 1 risk + 1 execution
                'separate_containers': len(containers) - 3
            },
            'portfolios': {},
            'containers': {}
        }
        
        # Collect portfolio results
        for i, combo in enumerate(param_combinations):
            portfolio_id = f'portfolio_combo_{i}'
            if portfolio_id in containers:
                portfolio_container = containers[portfolio_id]
                
                # Get portfolio component if it exists
                portfolio_state = portfolio_container.get_component('portfolio_state')
                if portfolio_state:
                    results['portfolios'][f'combo_{i}'] = {
                        'parameters': combo,
                        'portfolio_pnl': float(getattr(portfolio_state, 'realized_pnl', 0)),
                        'final_value': float(getattr(portfolio_state, 'total_value', combo['initial_capital'])),
                        'trades': getattr(portfolio_state, 'trade_count', 0)
                    }
        
        # Collect container metrics
        for container_id, container in containers.items():
            status = container.get_status()
            results['containers'][container_id] = {
                'state': container.state.value,
                'metrics': status.get('metrics', {}),
                'type': container.metadata.role.value
            }
        
        return results
    
    def _config_hash(self, config: Dict[str, Any]) -> str:
        """Create hash of configuration for sharing detection."""
        import json
        import hashlib
        
        # Remove fields that shouldn't affect sharing
        clean_config = {k: v for k, v in config.items() 
                       if k not in ['combination_id', 'combination_index', 'target_portfolio']}
        
        config_str = json.dumps(clean_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


# Example usage
async def main():
    """Run the concrete multi-parameter workflow example."""
    print("🚀 Starting Multi-Parameter Workflow Example")
    print("=" * 60)
    
    workflow = MultiParameterWorkflowExample()
    results = await workflow.execute_example()
    
    print("\n📊 Final Results:")
    print(f"Total Combinations: {results['summary']['total_combinations']}")
    print(f"Containers Created: {results['summary']['containers_created']}")
    print(f"Shared Containers: {results['summary']['shared_containers']}")
    print(f"Separate Containers: {results['summary']['separate_containers']}")
    
    print(f"\nContainer Efficiency: {results['summary']['shared_containers']}/{results['summary']['containers_created']} containers shared")
    print(f"Memory Savings: ~{(1 - results['summary']['containers_created']/(len(results['portfolios'])*4 + 1))*100:.1f}% vs naive approach")
    
    print("\nPortfolio Results:")
    for combo_id, portfolio_data in results['portfolios'].items():
        params = portfolio_data['parameters']['strategy']['parameters']
        capital = portfolio_data['parameters']['initial_capital']
        pnl = portfolio_data['portfolio_pnl']
        print(f"  {combo_id}: lookback={params['lookback_period']}, threshold={params['signal_threshold']}, capital=${capital} → PnL: ${pnl:.2f}")
    
    print("\n✅ Multi-Parameter Workflow Example Complete!")


if __name__ == "__main__":
    asyncio.run(main())