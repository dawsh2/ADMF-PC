"""
Workflow-specific managers for the Coordinator.
"""
import asyncio
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import logging
from datetime import datetime

from .types import (
    WorkflowConfig, WorkflowResult, ExecutionContext,
    WorkflowPhase, PhaseResult, WorkflowType
)
from .protocols import WorkflowManager, PhaseExecutor


logger = logging.getLogger(__name__)


class BaseWorkflowManager(ABC):
    """Base class for workflow managers."""
    
    def __init__(self, container_manager: Any, container_id: str):
        self.container_manager = container_manager
        self.container_id = container_id
        
        # Only access container if it exists (not during validation)
        if container_id in container_manager.active_containers:
            self.container = container_manager.active_containers[container_id]
            self.event_bus = self.container.event_bus
        else:
            # For validation, we don't need a real container
            self.container = None
            self.event_bus = None
            
        self._phase_executors: Dict[WorkflowPhase, PhaseExecutor] = {}
        
    async def execute(
        self,
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> WorkflowResult:
        """Execute the workflow."""
        result = WorkflowResult(
            workflow_id=context.workflow_id,
            workflow_type=context.workflow_type,
            success=True
        )
        
        try:
            # Execute phases in order
            for phase in self.get_execution_phases():
                context.update_phase(phase)
                
                # Emit phase start event
                if self.event_bus:
                    from ..events import Event, EventType
                    start_event = Event(
                        event_type=EventType.INFO,
                        payload={
                            'type': f'workflow.phase.{phase.value}.start',
                            'workflow_id': context.workflow_id,
                            'phase': phase.value
                        },
                        source_id="workflow_manager",
                        container_id=self.container_id
                    )
                    self.event_bus.publish(start_event)
                
                # Execute phase
                phase_result = await self.execute_phase(
                    phase, config, context
                )
                result.add_phase_result(phase_result)
                
                # Emit phase complete event
                if self.event_bus:
                    complete_event = Event(
                        event_type=EventType.INFO,
                        payload={
                            'type': f'workflow.phase.{phase.value}.complete',
                            'workflow_id': context.workflow_id,
                            'phase': phase.value,
                            'success': phase_result.success
                        },
                        source_id="workflow_manager",
                        container_id=self.container_id
                    )
                    self.event_bus.publish(complete_event)
                
                # Stop on failure if critical phase
                if not phase_result.success and self.is_critical_phase(phase):
                    result.success = False
                    break
                    
            # Finalize results
            result.finalize()
            
            # Aggregate final results
            if result.success:
                result.final_results = await self.aggregate_results(
                    result.phase_results
                )
                
        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            result.success = False
            result.errors.append(str(e))
            result.finalize()
            
        return result
        
    @abstractmethod
    def get_execution_phases(self) -> List[WorkflowPhase]:
        """Get ordered list of phases to execute."""
        ...
        
    @abstractmethod
    async def execute_phase(
        self,
        phase: WorkflowPhase,
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> PhaseResult:
        """Execute a specific phase."""
        ...
        
    @abstractmethod
    def is_critical_phase(self, phase: WorkflowPhase) -> bool:
        """Check if phase failure should stop execution."""
        ...
        
    @abstractmethod
    async def aggregate_results(
        self,
        phase_results: Dict[WorkflowPhase, PhaseResult]
    ) -> Dict[str, Any]:
        """Aggregate phase results into final output."""
        ...
        
    async def validate_config(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Validate workflow configuration."""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Common validation
        if not config.data_config:
            validation_results['errors'].append("Data configuration is required")
            validation_results['valid'] = False
            
        # Type-specific validation
        specific_results = await self.validate_specific_config(config)
        validation_results['errors'].extend(specific_results.get('errors', []))
        validation_results['warnings'].extend(specific_results.get('warnings', []))
        
        if specific_results.get('errors'):
            validation_results['valid'] = False
            
        return validation_results
        
    @abstractmethod
    async def validate_specific_config(
        self,
        config: WorkflowConfig
    ) -> Dict[str, Any]:
        """Validate workflow-specific configuration."""
        ...


class OptimizationManager(BaseWorkflowManager):
    """Manager for optimization workflows."""
    
    def get_execution_phases(self) -> List[WorkflowPhase]:
        """Get optimization phases."""
        return [
            WorkflowPhase.INITIALIZATION,
            WorkflowPhase.DATA_PREPARATION,
            WorkflowPhase.COMPUTATION,
            WorkflowPhase.VALIDATION,
            WorkflowPhase.AGGREGATION,
            WorkflowPhase.FINALIZATION
        ]
        
    async def execute_phase(
        self,
        phase: WorkflowPhase,
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> PhaseResult:
        """Execute optimization phase."""
        result = PhaseResult(phase=phase, success=True)
        
        try:
            if phase == WorkflowPhase.INITIALIZATION:
                # Initialize optimization environment
                result.data['initialized'] = True
                
            elif phase == WorkflowPhase.DATA_PREPARATION:
                # Prepare data for optimization
                result.data['data_prepared'] = True
                
            elif phase == WorkflowPhase.COMPUTATION:
                # Run optimization algorithm
                optimization_results = await self._run_optimization(
                    config.optimization_config or {},
                    context
                )
                result.data['optimization_results'] = optimization_results
                # Store in context for validation phase
                context.shared_resources['optimization_results'] = optimization_results
                
            elif phase == WorkflowPhase.VALIDATION:
                # Validate optimization results
                # Get optimization results from context or previous phase
                optimization_results = {}
                if 'optimization_results' in context.shared_resources:
                    optimization_results = context.shared_resources['optimization_results']
                
                validation = await self._validate_results(optimization_results)
                result.data['validation'] = validation
                result.success = validation.get('valid', False)
                
            elif phase == WorkflowPhase.AGGREGATION:
                # Aggregate results
                result.data['aggregated'] = True
                
            elif phase == WorkflowPhase.FINALIZATION:
                # Finalize and cleanup
                result.data['finalized'] = True
                
        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            logger.error(f"Phase {phase} error: {e}")
            
        return result
        
    def is_critical_phase(self, phase: WorkflowPhase) -> bool:
        """All phases except finalization are critical."""
        return phase != WorkflowPhase.FINALIZATION
        
    async def aggregate_results(
        self,
        phase_results: Dict[WorkflowPhase, PhaseResult]
    ) -> Dict[str, Any]:
        """Aggregate optimization results."""
        computation_result = phase_results.get(WorkflowPhase.COMPUTATION)
        
        if computation_result and computation_result.success:
            return computation_result.data.get('optimization_results', {})
            
        return {}
        
    async def validate_specific_config(
        self,
        config: WorkflowConfig
    ) -> Dict[str, Any]:
        """Validate optimization configuration."""
        errors = []
        warnings = []
        
        if not config.optimization_config:
            errors.append("Optimization configuration is required")
        else:
            opt_config = config.optimization_config
            
            if 'algorithm' not in opt_config:
                errors.append("Optimization algorithm must be specified")
                
            if 'objective' not in opt_config:
                errors.append("Optimization objective must be specified")
                
        return {'errors': errors, 'warnings': warnings}
        
    async def _run_optimization(
        self,
        opt_config: Dict[str, Any],
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """Run the optimization algorithm."""
        # This would integrate with your optimization system
        logger.info(f"Running optimization: {opt_config.get('algorithm')}")
        
        # Placeholder implementation
        return {
            'algorithm': opt_config.get('algorithm'),
            'objective': opt_config.get('objective'),
            'results': {'optimal_value': 0.95}
        }
        
    async def _validate_results(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate optimization results."""
        return {
            'valid': bool(results),
            'checks': ['convergence', 'constraints', 'bounds']
        }
        
    def get_required_capabilities(self) -> Dict[str, Any]:
        """Get required capabilities for optimization."""
        return {
            'computation': ['optimization_engine'],
            'data': ['time_series', 'indicators'],
            'memory': '4GB'
        }


class BacktestManager(BaseWorkflowManager):
    """Manager for backtest workflows."""
    
    def get_execution_phases(self) -> List[WorkflowPhase]:
        """Get backtest phases."""
        return [
            WorkflowPhase.INITIALIZATION,
            WorkflowPhase.DATA_PREPARATION,
            WorkflowPhase.COMPUTATION,
            WorkflowPhase.VALIDATION,
            WorkflowPhase.AGGREGATION
        ]
        
    async def execute_phase(
        self,
        phase: WorkflowPhase,
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> PhaseResult:
        """Execute backtest phase."""
        result = PhaseResult(phase=phase, success=True)
        
        try:
            if phase == WorkflowPhase.INITIALIZATION:
                # Initialize backtest environment
                result.data['environment'] = await self._setup_backtest_env(
                    config.backtest_config or {}
                )
                
            elif phase == WorkflowPhase.DATA_PREPARATION:
                # Load and prepare historical data
                result.data['data'] = await self._prepare_backtest_data(
                    config.data_config,
                    context
                )
                
            elif phase == WorkflowPhase.COMPUTATION:
                # Run backtest simulation
                result.data['backtest_results'] = await self._run_backtest(
                    config.backtest_config or {},
                    context
                )
                
            elif phase == WorkflowPhase.VALIDATION:
                # Validate backtest results
                validation = await self._validate_backtest(
                    result.data.get('backtest_results', {})
                )
                result.data['validation'] = validation
                
            elif phase == WorkflowPhase.AGGREGATION:
                # Calculate performance metrics
                result.data['metrics'] = await self._calculate_metrics(
                    result.data.get('backtest_results', {})
                )
                
        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            logger.error(f"Backtest phase {phase} error: {e}")
            
        return result
        
    def is_critical_phase(self, phase: WorkflowPhase) -> bool:
        """Data preparation and computation are critical."""
        return phase in [
            WorkflowPhase.DATA_PREPARATION,
            WorkflowPhase.COMPUTATION
        ]
        
    async def aggregate_results(
        self,
        phase_results: Dict[WorkflowPhase, PhaseResult]
    ) -> Dict[str, Any]:
        """Aggregate backtest results."""
        results = {}
        
        # Get backtest results
        computation = phase_results.get(WorkflowPhase.COMPUTATION)
        if computation and computation.success:
            results['backtest'] = computation.data.get('backtest_results', {})
            
        # Get metrics
        aggregation = phase_results.get(WorkflowPhase.AGGREGATION)
        if aggregation and aggregation.success:
            results['metrics'] = aggregation.data.get('metrics', {})
            
        return results
        
    async def validate_specific_config(
        self,
        config: WorkflowConfig
    ) -> Dict[str, Any]:
        """Validate backtest configuration."""
        errors = []
        warnings = []
        
        if not config.backtest_config:
            errors.append("Backtest configuration is required")
        else:
            bt_config = config.backtest_config
            
            if 'start_date' not in bt_config:
                errors.append("Start date is required for backtest")
                
            if 'end_date' not in bt_config:
                errors.append("End date is required for backtest")
                
            if 'strategy' not in bt_config:
                errors.append("Strategy configuration is required")
                
        return {'errors': errors, 'warnings': warnings}
        
    async def _setup_backtest_env(
        self,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Set up backtest environment."""
        return {
            'mode': 'historical',
            'start_date': config.get('start_date'),
            'end_date': config.get('end_date')
        }
        
    async def _prepare_backtest_data(
        self,
        data_config: Dict[str, Any],
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """Prepare data for backtest."""
        # Use shared data feeds from context
        data_feeds = context.shared_resources.get('data_feeds', {})
        
        return {
            'feeds': list(data_feeds.keys()),
            'ready': True
        }
        
    async def _run_backtest(
        self,
        config: Dict[str, Any],
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """Run backtest simulation."""
        logger.info(f"Running backtest: {config.get('strategy')}")
        
        # Placeholder implementation
        return {
            'strategy': config.get('strategy'),
            'trades': 150,
            'final_equity': 125000
        }
        
    async def _validate_backtest(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate backtest results."""
        return {
            'data_integrity': True,
            'trade_validation': True,
            'position_consistency': True
        }
        
    async def _calculate_metrics(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate performance metrics."""
        return {
            'total_return': 0.25,
            'sharpe_ratio': 1.5,
            'max_drawdown': 0.12,
            'win_rate': 0.55
        }
        
    def get_required_capabilities(self) -> Dict[str, Any]:
        """Get required capabilities for backtest."""
        return {
            'data': ['historical_data', 'market_data'],
            'computation': ['backtest_engine'],
            'memory': '2GB'
        }


class LiveTradingManager(BaseWorkflowManager):
    """Manager for live trading workflows."""
    
    def get_execution_phases(self) -> List[WorkflowPhase]:
        """Get live trading phases."""
        return [
            WorkflowPhase.INITIALIZATION,
            WorkflowPhase.VALIDATION,
            WorkflowPhase.COMPUTATION  # Continuous trading
        ]
    
    async def execute_phase(
        self,
        phase: WorkflowPhase,
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> PhaseResult:
        """Execute live trading phase."""
        result = PhaseResult(phase=phase, success=True)
        
        try:
            if phase == WorkflowPhase.INITIALIZATION:
                # Initialize trading environment
                result.data['broker_connection'] = await self._connect_broker(
                    config.live_config or {}
                )
                result.data['risk_manager'] = await self._setup_risk_management(
                    config.live_config or {}
                )
                
            elif phase == WorkflowPhase.VALIDATION:
                # Validate trading setup
                validation = await self._validate_trading_setup(
                    config.live_config or {},
                    context
                )
                result.data['validation'] = validation
                result.success = validation.get('ready', False)
                
            elif phase == WorkflowPhase.COMPUTATION:
                # Start live trading
                result.data['trading_session'] = await self._start_trading(
                    config.live_config or {},
                    context
                )
                
        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            logger.error(f"Live trading phase {phase} error: {e}")
            
        return result
    
    def is_critical_phase(self, phase: WorkflowPhase) -> bool:
        """All phases are critical for live trading."""
        return True
    
    async def aggregate_results(
        self,
        phase_results: Dict[WorkflowPhase, PhaseResult]
    ) -> Dict[str, Any]:
        """Aggregate live trading results."""
        results = {}
        
        init_result = phase_results.get(WorkflowPhase.INITIALIZATION)
        if init_result and init_result.success:
            results['broker_status'] = 'connected'
            results['risk_status'] = 'active'
        
        computation = phase_results.get(WorkflowPhase.COMPUTATION)
        if computation and computation.success:
            results['trading_session'] = computation.data.get('trading_session', {})
        
        return results
    
    async def validate_specific_config(
        self,
        config: WorkflowConfig
    ) -> Dict[str, Any]:
        """Validate live trading configuration."""
        errors = []
        warnings = []
        
        if not config.live_config:
            errors.append("Live trading configuration is required")
        else:
            live_config = config.live_config
            
            if 'broker' not in live_config:
                errors.append("Broker configuration is required")
            
            if 'strategy' not in live_config:
                errors.append("Strategy configuration is required")
            
            if 'risk_limits' not in live_config:
                warnings.append("No risk limits configured - using defaults")
        
        return {'errors': errors, 'warnings': warnings}
    
    async def _connect_broker(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Connect to broker."""
        logger.info(f"Connecting to broker: {config.get('broker', {}).get('name')}")
        return {'connected': True, 'broker': config.get('broker')}
    
    async def _setup_risk_management(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Set up risk management."""
        return {
            'limits': config.get('risk_limits', {}),
            'active': True
        }
    
    async def _validate_trading_setup(self, config: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Validate trading setup."""
        return {
            'ready': True,
            'broker_connected': True,
            'data_feeds': True,
            'risk_manager': True
        }
    
    async def _start_trading(self, config: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Start live trading session."""
        return {
            'session_id': context.workflow_id,
            'status': 'active',
            'start_time': datetime.now().isoformat()
        }


class AnalysisManager(BaseWorkflowManager):
    """Manager for analysis workflows."""
    
    def get_execution_phases(self) -> List[WorkflowPhase]:
        """Get analysis phases."""
        return [
            WorkflowPhase.INITIALIZATION,
            WorkflowPhase.DATA_PREPARATION,
            WorkflowPhase.COMPUTATION,
            WorkflowPhase.AGGREGATION
        ]
    
    async def execute_phase(
        self,
        phase: WorkflowPhase,
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> PhaseResult:
        """Execute analysis phase."""
        result = PhaseResult(phase=phase, success=True)
        
        try:
            if phase == WorkflowPhase.INITIALIZATION:
                # Initialize analysis environment
                result.data['analyzers'] = await self._setup_analyzers(
                    config.analysis_config or {}
                )
                
            elif phase == WorkflowPhase.DATA_PREPARATION:
                # Load data for analysis
                result.data['data'] = await self._load_analysis_data(
                    config.data_config,
                    context
                )
                
            elif phase == WorkflowPhase.COMPUTATION:
                # Run analysis
                result.data['analysis_results'] = await self._run_analysis(
                    config.analysis_config or {},
                    context
                )
                
            elif phase == WorkflowPhase.AGGREGATION:
                # Aggregate analysis results
                result.data['summary'] = await self._summarize_analysis(
                    result.data.get('analysis_results', {})
                )
                
        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            logger.error(f"Analysis phase {phase} error: {e}")
            
        return result
    
    def is_critical_phase(self, phase: WorkflowPhase) -> bool:
        """Data and computation are critical."""
        return phase in [WorkflowPhase.DATA_PREPARATION, WorkflowPhase.COMPUTATION]
    
    async def aggregate_results(
        self,
        phase_results: Dict[WorkflowPhase, PhaseResult]
    ) -> Dict[str, Any]:
        """Aggregate analysis results."""
        results = {}
        
        computation = phase_results.get(WorkflowPhase.COMPUTATION)
        if computation and computation.success:
            results['analysis'] = computation.data.get('analysis_results', {})
        
        aggregation = phase_results.get(WorkflowPhase.AGGREGATION)
        if aggregation and aggregation.success:
            results['summary'] = aggregation.data.get('summary', {})
        
        return results
    
    async def validate_specific_config(
        self,
        config: WorkflowConfig
    ) -> Dict[str, Any]:
        """Validate analysis configuration."""
        errors = []
        warnings = []
        
        if not config.analysis_config:
            errors.append("Analysis configuration is required")
        else:
            analysis_config = config.analysis_config
            
            if 'analysis_type' not in analysis_config:
                errors.append("Analysis type must be specified")
        
        return {'errors': errors, 'warnings': warnings}
    
    async def _setup_analyzers(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Set up analysis components."""
        return {
            'type': config.get('analysis_type'),
            'configured': True
        }
    
    async def _load_analysis_data(self, data_config: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Load data for analysis."""
        return {
            'loaded': True,
            'sources': list(data_config.get('sources', {}).keys())
        }
    
    async def _run_analysis(self, config: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Run the analysis."""
        return {
            'analysis_type': config.get('analysis_type'),
            'results': {'sample_metric': 0.85}
        }
    
    async def _summarize_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize analysis results."""
        return {
            'key_findings': ['Finding 1', 'Finding 2'],
            'metrics': results
        }


class ValidationManager(BaseWorkflowManager):
    """Manager for validation workflows."""
    
    def get_execution_phases(self) -> List[WorkflowPhase]:
        """Get validation phases."""
        return [
            WorkflowPhase.INITIALIZATION,
            WorkflowPhase.DATA_PREPARATION,
            WorkflowPhase.VALIDATION,
            WorkflowPhase.AGGREGATION
        ]
    
    async def execute_phase(
        self,
        phase: WorkflowPhase,
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> PhaseResult:
        """Execute validation phase."""
        result = PhaseResult(phase=phase, success=True)
        
        try:
            if phase == WorkflowPhase.INITIALIZATION:
                # Initialize validation environment
                result.data['validators'] = await self._setup_validators(
                    config.validation_config or {}
                )
                
            elif phase == WorkflowPhase.DATA_PREPARATION:
                # Prepare validation data
                result.data['test_data'] = await self._prepare_test_data(
                    config.data_config,
                    context
                )
                
            elif phase == WorkflowPhase.VALIDATION:
                # Run validation tests
                result.data['validation_results'] = await self._run_validation(
                    config.validation_config or {},
                    context
                )
                
            elif phase == WorkflowPhase.AGGREGATION:
                # Aggregate validation results
                result.data['report'] = await self._generate_validation_report(
                    result.data.get('validation_results', {})
                )
                
        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            logger.error(f"Validation phase {phase} error: {e}")
            
        return result
    
    def is_critical_phase(self, phase: WorkflowPhase) -> bool:
        """Validation phase is critical."""
        return phase == WorkflowPhase.VALIDATION
    
    async def aggregate_results(
        self,
        phase_results: Dict[WorkflowPhase, PhaseResult]
    ) -> Dict[str, Any]:
        """Aggregate validation results."""
        results = {}
        
        validation = phase_results.get(WorkflowPhase.VALIDATION)
        if validation and validation.success:
            results['validation'] = validation.data.get('validation_results', {})
        
        aggregation = phase_results.get(WorkflowPhase.AGGREGATION)
        if aggregation and aggregation.success:
            results['report'] = aggregation.data.get('report', {})
        
        return results
    
    async def validate_specific_config(
        self,
        config: WorkflowConfig
    ) -> Dict[str, Any]:
        """Validate validation configuration."""
        errors = []
        warnings = []
        
        if not config.validation_config:
            errors.append("Validation configuration is required")
        else:
            val_config = config.validation_config
            
            if 'validation_type' not in val_config:
                errors.append("Validation type must be specified")
            
            if 'test_cases' not in val_config:
                warnings.append("No test cases specified")
        
        return {'errors': errors, 'warnings': warnings}
    
    async def _setup_validators(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Set up validation components."""
        return {
            'type': config.get('validation_type'),
            'validators': config.get('validators', [])
        }
    
    async def _prepare_test_data(self, data_config: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Prepare test data."""
        return {
            'test_set': 'prepared',
            'size': 1000
        }
    
    async def _run_validation(self, config: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Run validation tests."""
        return {
            'validation_type': config.get('validation_type'),
            'tests_passed': 95,
            'tests_failed': 5,
            'total_tests': 100
        }
    
    async def _generate_validation_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation report."""
        return {
            'summary': 'Validation completed',
            'pass_rate': 0.95,
            'details': results
        }


# Factory for creating managers
class WorkflowManagerFactory:
    """Factory for creating workflow managers."""
    
    def __init__(self, container_manager: Any, shared_services: Dict[str, Any]):
        self.container_manager = container_manager
        self.shared_services = shared_services
        
        # Use simple backtest manager to avoid circular imports
        try:
            from .backtest_manager import BacktestWorkflowManager
            backtest_manager_class = BacktestWorkflowManager
        except ImportError:
            from .simple_backtest_manager import SimpleBacktestWorkflowManager
            backtest_manager_class = SimpleBacktestWorkflowManager
        
        self._managers = {
            WorkflowType.OPTIMIZATION: OptimizationManager,
            WorkflowType.BACKTEST: backtest_manager_class,
            WorkflowType.LIVE_TRADING: LiveTradingManager,
            WorkflowType.ANALYSIS: AnalysisManager,
            WorkflowType.VALIDATION: ValidationManager
        }
        
    def create_manager(
        self, 
        workflow_type: WorkflowType, 
        container_id: str,
        use_composable: bool = False
    ) -> WorkflowManager:
        """Create a manager for the workflow type."""
        
        # Check if composable containers are requested
        if use_composable:
            return self._create_composable_manager(workflow_type, container_id)
        
        # Traditional manager creation
        manager_class = self._managers.get(workflow_type)
        
        if not manager_class:
            raise ValueError(f"No manager for workflow type: {workflow_type}")
            
        # Special handling for BacktestWorkflowManager
        if workflow_type == WorkflowType.BACKTEST:
            from .backtest_manager import BacktestWorkflowManager
            return BacktestWorkflowManager(container_id, self.shared_services)
            
        return manager_class(self.container_manager, container_id)
    
    def _create_composable_manager(
        self, 
        workflow_type: WorkflowType, 
        container_id: str
    ) -> WorkflowManager:
        """Create composable workflow manager."""
        
        try:
            from .composable_workflow_manager import ComposableWorkflowManager
            return ComposableWorkflowManager(container_id, self.shared_services)
        except ImportError as e:
            logger.error(f"Cannot create composable manager: {e}")
            # Fall back to traditional manager
            return self.create_manager(workflow_type, container_id, use_composable=False)