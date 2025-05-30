"""
Minimal coordinator with clean architecture and no deep imports.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class WorkflowType(str, Enum):
    """Workflow types."""
    BACKTEST = "backtest"
    OPTIMIZATION = "optimization"
    LIVE = "live"


@dataclass
class MinimalWorkflowConfig:
    """Minimal workflow configuration."""
    workflow_type: WorkflowType
    data_config: Dict[str, Any]
    backtest_config: Dict[str, Any]
    parameters: Dict[str, Any]
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create from dictionary."""
        return cls(
            workflow_type=WorkflowType(config_dict.get('workflow_type', 'backtest')),
            data_config=config_dict.get('data_config', {}),
            backtest_config=config_dict.get('backtest_config', {}),
            parameters=config_dict.get('parameters', {})
        )


@dataclass
class WorkflowResult:
    """Result from workflow execution."""
    workflow_id: str
    success: bool
    results: Dict[str, Any]
    errors: list
    start_time: datetime
    end_time: Optional[datetime] = None
    
    def finalize(self):
        """Mark workflow as complete."""
        self.end_time = datetime.now()
        if self.end_time and self.start_time:
            duration = (self.end_time - self.start_time).total_seconds()
            self.results['duration_seconds'] = duration


class MinimalCoordinator:
    """
    Minimal coordinator that avoids deep import chains.
    
    Key principles:
    1. Lazy loading - only import what's needed when it's needed
    2. Plugin architecture - managers register themselves
    3. Simple interfaces - avoid complex hierarchies
    """
    
    def __init__(self):
        self.logger = logger
        self._managers = {}
        self._register_default_managers()
        
    def _register_default_managers(self):
        """Register default workflow managers."""
        # Register managers without importing them
        self._managers[WorkflowType.BACKTEST] = 'backtest'
        self._managers[WorkflowType.OPTIMIZATION] = 'optimization'
        
    async def execute_workflow(
        self, 
        config: Dict[str, Any],
        workflow_id: Optional[str] = None
    ) -> WorkflowResult:
        """Execute a workflow with minimal dependencies."""
        
        # Create workflow ID
        if not workflow_id:
            import uuid
            workflow_id = str(uuid.uuid4())
            
        # Parse config
        workflow_config = MinimalWorkflowConfig.from_dict(config)
        
        # Create result
        result = WorkflowResult(
            workflow_id=workflow_id,
            success=False,
            results={},
            errors=[],
            start_time=datetime.now()
        )
        
        try:
            self.logger.info(f"Executing {workflow_config.workflow_type} workflow {workflow_id}")
            
            # Route to appropriate handler
            if workflow_config.workflow_type == WorkflowType.BACKTEST:
                await self._execute_backtest(workflow_config, result)
            elif workflow_config.workflow_type == WorkflowType.OPTIMIZATION:
                await self._execute_optimization(workflow_config, result)
            else:
                result.errors.append(f"Unsupported workflow type: {workflow_config.workflow_type}")
                
        except Exception as e:
            self.logger.error(f"Workflow {workflow_id} failed: {e}")
            result.errors.append(str(e))
            
        finally:
            result.finalize()
            result.success = len(result.errors) == 0
            
        return result
        
    async def _execute_backtest(self, config: MinimalWorkflowConfig, result: WorkflowResult):
        """Execute backtest workflow with lazy imports."""
        
        # Only import when needed
        try:
            from ...execution.simple_backtest_engine import SimpleBacktestEngine
            from ...data.simple_loader import SimpleDataLoader
        except ImportError:
            # Fallback to inline implementation
            result.errors.append("Backtest engine not available")
            return
            
        # Check if we need train/test split
        dataset = config.data_config.get('dataset', 'full')
        
        if dataset in ['train', 'test']:
            # Coordinator handles data splitting
            self.logger.info(f"Preparing {dataset} dataset")
            
            # Create data loader
            data_loader = SimpleDataLoader(config.data_config)
            
            # Load and get split info
            data_loader.load_full_data(max_bars=config.data_config.get('max_bars'))
            split_info = data_loader.get_train_test_info()
            
            self.logger.info(f"Data split info: {split_info}")
            
            # Store split info in results
            result.results['data_split_info'] = split_info
            
        # Merge configuration
        engine_config = {
            'data': config.data_config,
            'strategies': config.backtest_config.get('strategies', []),
            'portfolio': config.backtest_config.get('portfolio', {'initial_capital': 10000}),
            'backtest': config.backtest_config
        }
        
        self.logger.debug(f"Engine config: {engine_config}")
        
        # Run backtest
        self.logger.info(f"Creating backtest engine for {dataset} dataset")
        engine = SimpleBacktestEngine(engine_config)
        
        # Load data - engine will use the dataset parameter
        max_bars = config.data_config.get('max_bars', config.parameters.get('max_bars'))
        self.logger.info(f"Loading data (dataset={dataset}, max_bars={max_bars})")
        engine.load_data(max_bars=max_bars, dataset=dataset)
        
        # Execute
        self.logger.info(f"Running backtest on {dataset} data")
        backtest_result = engine.run_backtest()
        
        # Store results
        result.results = {
            'dataset': dataset,
            'final_equity': backtest_result.final_equity,
            'total_return': backtest_result.total_return,
            'num_trades': backtest_result.num_trades,
            'win_rate': backtest_result.win_rate,
            'sharpe_ratio': backtest_result.sharpe_ratio,
            'max_drawdown': backtest_result.max_drawdown
        }
        
        self.logger.info(f"Backtest complete on {dataset}: {backtest_result.num_trades} trades, {backtest_result.total_return:.2%} return")
        
    async def _execute_optimization(self, config: MinimalWorkflowConfig, result: WorkflowResult):
        """Execute optimization workflow with full features."""
        
        # Check if we should use advanced optimization
        opt_config = config.parameters.get('optimization', {})
        method = opt_config.get('method', 'grid_search')
        
        if method == 'bayesian':
            # Lazy load advanced optimizer only if needed
            try:
                from ...strategy.optimization.optimizers import BayesianOptimizer
                optimizer = BayesianOptimizer(opt_config)
                # Full Bayesian optimization available!
                await self._run_advanced_optimization(optimizer, config, result)
                return
            except ImportError:
                self.logger.warning("Bayesian optimizer not available, falling back to grid search")
        
        # Default grid search
        param_space = opt_config.get('parameter_space', {
            'buy_threshold': [88, 90, 92],
            'sell_threshold': [98, 100, 102]
        })
        
        # Generate all combinations
        param_sets = self._generate_parameter_grid(param_space)
        
        best_return = -float('inf')
        best_params = None
        
        for params in param_sets:
            # Create test config
            test_config = config.backtest_config.copy()
            test_config['strategies'][0]['parameters'] = params
            
            # Run backtest
            test_workflow = MinimalWorkflowConfig(
                workflow_type=WorkflowType.BACKTEST,
                data_config=config.data_config,
                backtest_config=test_config,
                parameters=config.parameters
            )
            
            test_result = WorkflowResult(
                workflow_id=f"{result.workflow_id}_test",
                success=False,
                results={},
                errors=[],
                start_time=datetime.now()
            )
            
            await self._execute_backtest(test_workflow, test_result)
            
            if test_result.success:
                total_return = test_result.results.get('total_return', 0)
                if total_return > best_return:
                    best_return = total_return
                    best_params = params
                    
        result.results = {
            'best_parameters': best_params,
            'best_return': best_return,
            'tested_combinations': len(param_sets)
        }