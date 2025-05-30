"""
Simple backtest manager that avoids complex dependencies.
"""

from typing import Dict, Any
import logging

from .protocols import WorkflowManager
from .simple_types import WorkflowConfig, WorkflowResult, ExecutionContext

logger = logging.getLogger(__name__)


class SimpleBacktestWorkflowManager(WorkflowManager):
    """Simple backtest manager for testing."""
    
    def __init__(self, container_id: str, shared_services: Dict[str, Any] = None):
        self.container_id = container_id
        self.shared_services = shared_services or {}
        
    async def execute(
        self,
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> WorkflowResult:
        """Execute backtest workflow."""
        logger.info(f"Executing simple backtest workflow {context.workflow_id}")
        
        # For now, create a simple result
        result = WorkflowResult(
            workflow_id=context.workflow_id,
            workflow_type=config.workflow_type,
            success=True,
            results={
                'message': 'Simple backtest completed',
                'config': config.dict() if hasattr(config, 'dict') else config.__dict__
            }
        )
        
        # If we can import the simple backtest engine, use it
        try:
            from ...execution.simple_backtest_engine import SimpleBacktestEngine
            
            # Create merged config
            full_config = {
                'data': config.data_config,
                'backtest': config.backtest_config,
                'strategies': config.backtest_config.get('strategies', []),
                'portfolio': config.backtest_config.get('portfolio', {
                    'initial_capital': 10000
                })
            }
            
            # Create and run engine
            engine = SimpleBacktestEngine(full_config)
            engine.load_data(max_bars=config.data_config.get('max_bars', 100))
            backtest_result = engine.run_backtest()
            
            # Update result
            result.results = {
                'final_equity': backtest_result.final_equity,
                'total_return': backtest_result.total_return,
                'num_trades': backtest_result.num_trades,
                'win_rate': backtest_result.win_rate,
                'sharpe_ratio': backtest_result.sharpe_ratio,
                'max_drawdown': backtest_result.max_drawdown
            }
            
        except Exception as e:
            logger.warning(f"Could not run full backtest: {e}")
            result.results['warning'] = str(e)
        
        result.finalize()
        return result
        
    async def validate_config(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Validate configuration."""
        errors = []
        warnings = []
        
        # Check required fields
        if not config.data_config:
            errors.append("Missing data configuration")
            
        if not config.backtest_config:
            errors.append("Missing backtest configuration")
            
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
        
    def get_required_capabilities(self) -> Dict[str, Any]:
        """Get required capabilities."""
        return {
            'data_loading': True,
            'backtesting': True,
            'reporting': True
        }