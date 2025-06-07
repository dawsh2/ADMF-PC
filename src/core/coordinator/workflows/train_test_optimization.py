"""
Train/Test optimization workflow implementation.

This workflow runs parameter optimization on training data,
then validates the best parameters on test data.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from ..protocols import WorkflowProtocol, PhaseConfig, PhaseEnhancerProtocol


class TrainTestOptimizationWorkflow:
    """
    Train/Test optimization workflow using Protocol + Composition.
    
    This workflow:
    1. Splits data into train/test periods
    2. Runs parameter optimization on training data
    3. Selects best parameters based on objective function
    4. Validates selected parameters on test data
    
    Generic - works with any strategy/parameter configuration.
    """
    
    def __init__(self, enhancers: Optional[List[PhaseEnhancerProtocol]] = None):
        """
        Initialize with optional phase enhancers.
        
        Args:
            enhancers: List of components that can enhance phase configs
        """
        self.enhancers = enhancers or []
        
        # Workflow defaults - user can override
        self.defaults = {
            'train_test_split': {
                'train_ratio': 0.8  # 80/20 split by default
            },
            'parameter_selection': {
                'method': 'top_n',
                'n': 5
            },
            'objective_function': {
                'name': 'sharpe_ratio'
            },
            'trace_level': 'minimal',  # Memory efficient for optimization
            'results': {
                'retention_policy': 'trade_complete',
                'streaming_metrics': True,
                'store_trades': False,  # Don't store trades during optimization
                'store_equity_curve': False
            }
        }
    
    def get_phases(self, config: Dict[str, Any]) -> Dict[str, PhaseConfig]:
        """
        Convert user config to phase definitions with memory management support.
        
        Creates two phases:
        1. Training phase with parameter optimization (disk storage by default)
        2. Testing phase with selected parameters (memory storage by default)
        
        Supports user overrides at both global and phase levels.
        
        Args:
            config: User configuration with parameter_space
            
        Returns:
            Dict with training and testing phases
        """
        # Calculate train/test periods
        train_period, test_period = self._calculate_periods(config)
        
        # Get any phase overrides from user config
        phase_overrides = config.get('phase_overrides', {})
        
        # Apply global override if present
        global_storage = config.get('results_storage')
        
        # Training phase configuration - default to disk for large optimization
        training_config = {
            **config,  # Include all user config
            'data': {
                **config.get('data', {}),
                'start': train_period[0],
                'end': train_period[1]
            },
            'parameter_space': config.get('parameter_space', {}),
            'objective_function': config.get('objective_function', self.defaults['objective_function']),
            'optimization_phase': True,  # Flag for sequence
            # Workflow defaults for training phase
            'results_storage': 'disk',  # Large optimization results
            'event_tracing': ['POSITION_OPEN', 'POSITION_CLOSE', 'FILL'],
            'retention_policy': 'trade_complete'
        }
        
        # Apply overrides for training phase
        if global_storage:
            training_config['results_storage'] = global_storage
        if 'training' in phase_overrides:
            training_config.update(phase_overrides['training'])
        
        # Testing phase configuration - default to memory with equity tracking
        testing_config = {
            **config,
            'data': {
                **config.get('data', {}),
                'start': test_period[0],
                'end': test_period[1]
            },
            'parameter_selection': config.get('parameter_selection', self.defaults['parameter_selection']),
            'validation_phase': True,  # Flag for sequence
            # Workflow defaults for testing phase
            'results_storage': 'memory',  # Smaller dataset, keep in memory
            'event_tracing': ['POSITION_OPEN', 'POSITION_CLOSE', 'FILL', 'PORTFOLIO_UPDATE'],
            'retention_policy': 'sliding_window',
            'sliding_window_size': 1000
        }
        
        # Apply overrides for testing phase
        if global_storage:
            testing_config['results_storage'] = global_storage
        if 'testing' in phase_overrides:
            testing_config.update(phase_overrides['testing'])
        
        # Create phases
        phases = {
            "training": PhaseConfig(
                name="training",
                sequence="train_test",  # Train/test sequence handles optimization
                topology="backtest",    # Full backtest topology
                description="Optimize parameters on training data",
                config=training_config,
                output={
                    'optimal_parameters': True,
                    'parameter_performance': True,
                    'optimization_metrics': True
                }
            ),
            "testing": PhaseConfig(
                name="testing",
                sequence="train_test",  # Same sequence, different mode
                topology="backtest",
                description="Validate optimal parameters on test data",
                config=testing_config,
                input={
                    'optimal_parameters': '{training.output.optimal_parameters}',
                    'parameter_performance': '{training.output.parameter_performance}'
                },
                output={
                    'validation_metrics': True,
                    'validation_trades': True,
                    'performance_comparison': True
                },
                depends_on=['training']  # Must run after training
            )
        }
        
        # Apply enhancers
        for enhancer in self.enhancers:
            phases = enhancer.enhance(phases)
        
        return phases
    
    def _calculate_periods(self, config: Dict[str, Any]) -> tuple:
        """
        Calculate train/test periods based on config.
        
        Returns:
            Tuple of (train_period, test_period) where each is (start, end)
        """
        data_config = config.get('data', {})
        start_date = data_config.get('start', data_config.get('start_date'))
        end_date = data_config.get('end', data_config.get('end_date'))
        
        if not start_date or not end_date:
            raise ValueError("Data start and end dates required")
        
        # Check if explicit periods provided
        split_config = config.get('train_test_split', self.defaults['train_test_split'])
        
        if 'train_period' in split_config and 'test_period' in split_config:
            # Use explicit periods
            train_period = split_config['train_period']
            test_period = split_config['test_period']
        else:
            # Calculate based on ratio
            train_ratio = split_config.get('train_ratio', 0.8)
            
            # Parse dates
            if isinstance(start_date, str):
                start = datetime.strptime(start_date, '%Y-%m-%d')
            else:
                start = start_date
                
            if isinstance(end_date, str):
                end = datetime.strptime(end_date, '%Y-%m-%d')
            else:
                end = end_date
            
            # Calculate split point
            total_days = (end - start).days
            train_days = int(total_days * train_ratio)
            
            train_end = start + timedelta(days=train_days - 1)
            test_start = train_end + timedelta(days=1)
            
            # Format as strings
            train_period = (
                start.strftime('%Y-%m-%d'),
                train_end.strftime('%Y-%m-%d')
            )
            test_period = (
                test_start.strftime('%Y-%m-%d'),
                end.strftime('%Y-%m-%d')
            )
        
        return train_period, test_period