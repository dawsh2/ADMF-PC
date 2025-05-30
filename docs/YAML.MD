# ADMF-PC: YAML-Driven Configuration Architecture

## Core Principle: Configuration Over Code

The ADMF-PC system is fundamentally **YAML configuration-driven**. Every aspect of system behavior - from simple backtests to complex multi-phase optimizations - is defined through declarative YAML files, not code.

### Why YAML-Driven?

1. **Reproducibility**: Every experiment is captured in version-controllable configuration
2. **Collaboration**: Researchers share YAML files, not code snippets
3. **Audit Trail**: Complete record of what was tested and when
4. **Non-Technical Access**: Strategists can modify parameters without coding
5. **Consistency**: Same execution path regardless of complexity

## The Coordinator as YAML Interpreter

The Coordinator serves as the **universal YAML interpreter** that:

```python
# Single entry point for all workflows
coordinator = Coordinator()

# YAML file defines everything
result = await coordinator.execute_workflow_from_yaml("configs/my_strategy_test.yaml")

# Or programmatically with identical structure
config_dict = yaml.safe_load(open("configs/my_strategy_test.yaml"))
result = await coordinator.execute_workflow(config_dict)
```

### The Coordinator's Role

**Before Coordinator (Complex, Error-Prone)**:
```python
# Manual wiring - different every time
data_handler = create_data_handler(symbols, start_date, end_date)
indicator_hub = create_indicators(sma_periods, rsi_period)
hmm_classifier = create_hmm_classifier(n_states=3)
risk_manager = create_risk_manager(max_exposure=0.2)
strategy = create_momentum_strategy(lookback=20)
backtest_engine = create_backtest_engine(initial_capital=100000)

# Wire everything together (potential for errors)
wire_components(data_handler, indicator_hub, hmm_classifier, risk_manager, strategy, backtest_engine)

# Run backtest
results = run_backtest()
```

**With Coordinator (Standardized, Reliable)**:
```python
# Single line - YAML defines everything
result = await coordinator.execute_workflow_from_yaml("configs/momentum_strategy.yaml")
```

## Complete YAML Configuration Examples

### 1. Simple Backtest Configuration

```yaml
# configs/simple_backtest.yaml
workflow:
  type: "backtest"
  name: "Simple Momentum Test"
  description: "Testing momentum strategy on tech stocks"

# Data specification
data:
  sources:
    primary:
      type: "csv"
      path: "data/stocks/"
      date_format: "%Y-%m-%d"
  
  symbols: ["AAPL", "GOOGL", "MSFT"]
  timeframe: "1D"
  start_date: "2023-01-01"
  end_date: "2023-12-31"

# Shared infrastructure
infrastructure:
  indicators:
    - name: "sma_fast"
      type: "SMA"
      period: 10
    - name: "sma_slow" 
      type: "SMA"
      period: 30
    - name: "rsi"
      type: "RSI"
      period: 14

# Classification layer
classifiers:
  - name: "hmm_classifier"
    type: "HMM"
    n_states: 3
    regime_labels: ["bull", "bear", "neutral"]
    
    # Risk & Portfolio containers within this classifier
    risk_containers:
      - name: "conservative_risk"
        initial_capital: 100000
        
        position_sizing:
          default:
            type: "percentage"
            percentage: 2.0  # 2% per position
        
        risk_limits:
          - type: "exposure"
            max_exposure_pct: 15  # 15% max total exposure
          - type: "drawdown"
            max_drawdown_pct: 10  # 10% max drawdown
        
        # Strategies within this risk container
        strategies:
          - name: "momentum_strategy"
            class: "MomentumStrategy"
            symbols: ["AAPL", "GOOGL"]  # Strategy-specific symbols
            parameters:
              fast_period: 10
              slow_period: 30
              signal_threshold: 0.02
          
          - name: "mean_reversion_strategy"
            class: "MeanReversionStrategy"
            symbols: ["MSFT"]
            parameters:
              lookback_period: 20
              std_threshold: 2.0

# Execution settings
execution:
  mode: "backtest"
  
  slippage:
    type: "percentage"
    value: 0.001  # 0.1% slippage
  
  commission:
    type: "tiered"
    tiers:
      - max_quantity: 1000
        fixed_amount: 1.0
      - max_quantity: 10000
        fixed_amount: 0.5
      - percentage: 0.0001  # 0.01% for large orders

# Output configuration
output:
  save_results: true
  results_path: "results/simple_backtest/"
  save_equity_curve: true
  save_trade_log: true
  
  metrics:
    - "total_return"
    - "sharpe_ratio"
    - "max_drawdown"
    - "calmar_ratio"
    - "win_rate"
```

### 2. Multi-Phase Optimization Configuration

```yaml
# configs/complex_optimization.yaml
workflow:
  type: "optimization"
  name: "Multi-Phase Strategy Optimization"
  description: "Complete optimization pipeline with validation"

# Same data and infrastructure sections as above
data: 
  sources:
    training:
      type: "csv"
      path: "data/training/"
      start_date: "2020-01-01"
      end_date: "2022-12-31"
    
    validation:
      type: "csv"
      path: "data/validation/"
      start_date: "2023-01-01" 
      end_date: "2023-12-31"
  
  symbols: ["AAPL", "GOOGL", "MSFT", "SPY", "QQQ"]
  timeframe: "1D"

infrastructure:
  indicators:
    - {name: "sma_5", type: "SMA", period: 5}
    - {name: "sma_10", type: "SMA", period: 10}
    - {name: "sma_20", type: "SMA", period: 20}
    - {name: "sma_50", type: "SMA", period: 50}
    - {name: "rsi_14", type: "RSI", period: 14}
    - {name: "rsi_21", type: "RSI", period: 21}

# Multi-phase optimization
optimization:
  # Phase 1: Parameter Grid Search
  phase_1:
    name: "parameter_optimization"
    type: "grid_search"
    objective: "sharpe_ratio"
    
    # Test multiple classifiers
    classifiers:
      - name: "hmm_classifier"
        type: "HMM"
        parameter_space:
          n_states: [2, 3, 4]
          
      - name: "pattern_classifier"
        type: "PatternClassifier"
        parameter_space:
          volatility_window: [15, 20, 25]
          trend_threshold: [0.01, 0.015, 0.02]
    
    # Test multiple risk profiles
    risk_profiles:
      - name: "conservative"
        position_sizing: {type: "percentage", percentage: 1.0}
        max_exposure: 10
        
      - name: "moderate"
        position_sizing: {type: "percentage", percentage: 2.0}
        max_exposure: 20
        
      - name: "aggressive"
        position_sizing: {type: "percentage", percentage: 3.0}
        max_exposure: 30
    
    # Strategy parameter spaces
    strategies:
      - name: "momentum_strategy"
        class: "MomentumStrategy"
        parameter_space:
          fast_period: [5, 10, 15, 20]
          slow_period: [20, 30, 40, 50]
          signal_threshold: [0.01, 0.02, 0.03]
          
      - name: "mean_reversion_strategy"
        class: "MeanReversionStrategy"
        parameter_space:
          lookback_period: [10, 15, 20, 25]
          std_threshold: [1.5, 2.0, 2.5]
  
  # Phase 2: Ensemble Weight Optimization
  phase_2:
    name: "ensemble_optimization"
    type: "signal_replay"  # Uses signals from Phase 1
    objective: "risk_adjusted_return"
    
    # Use best performers from Phase 1
    input_source: "phase_1_results"
    top_k_strategies: 10  # Use top 10 strategies from Phase 1
    
    # Optimize ensemble weights
    weight_optimization:
      algorithm: "genetic"
      population_size: 50
      generations: 100
      
      # Weight constraints
      constraints:
        - type: "sum_to_one"  # Weights sum to 1.0
        - type: "max_weight"
          value: 0.4  # No strategy > 40%
        - type: "min_strategies"
          value: 3    # At least 3 strategies active
  
  # Phase 3: Walk-Forward Validation
  phase_3:
    name: "validation"
    type: "walk_forward"
    
    # Use best ensemble from Phase 2
    input_source: "phase_2_results"
    
    # Walk-forward parameters
    validation:
      training_window: 252  # 1 year
      test_window: 63      # 1 quarter  
      step_size: 21        # 1 month steps
      min_trades: 10       # Minimum trades per period

# Output detailed results
output:
  save_all_phases: true
  
  phase_1_output:
    path: "results/optimization/phase_1/"
    save_top_k: 100
    save_parameter_heatmaps: true
    
  phase_2_output:
    path: "results/optimization/phase_2/"
    save_weight_evolution: true
    save_signal_correlation: true
    
  phase_3_output:
    path: "results/optimization/phase_3/"
    save_walk_forward_results: true
    save_period_breakdown: true
    
  final_output:
    path: "results/optimization/final/"
    generate_report: true
    report_format: ["html", "pdf"]
```

### 3. Live Trading Configuration

```yaml
# configs/live_trading.yaml
workflow:  
  type: "live_trading"
  name: "Production Momentum Strategy"
  description: "Live trading with optimized parameters"

# Real-time data sources
data:
  sources:
    live_feed:
      type: "websocket"
      url: "wss://api.broker.com/market-data"
      auth:
        api_key: "${BROKER_API_KEY}"
        secret: "${BROKER_SECRET}"
    
    reference_data:
      type: "database"
      connection: "postgresql://user:pass@localhost/market_data"
  
  symbols: ["AAPL", "GOOGL", "MSFT"]
  timeframe: "1min"  # Real-time tick aggregation

# Load optimized configuration from previous optimization
optimization_results:
  source: "results/optimization/final/best_config.yaml"
  # This loads the best classifier, risk settings, and strategy parameters

# Live trading specific settings
live_trading:
  broker:
    name: "interactive_brokers"
    account_id: "${IB_ACCOUNT_ID}"
    api_endpoint: "https://api.interactivebrokers.com"
    paper_trading: false  # Set to true for paper trading
  
  risk_overrides:
    # More conservative limits for live trading
    max_daily_loss: 5000
    max_daily_loss_pct: 2.0
    
    # Position limits
    max_position_size: 10000
    max_portfolio_exposure: 15  # More conservative than backtest
    
    # Real-time risk monitoring
    risk_checks:
      - type: "real_time_pnl"
        threshold: -1000  # Stop if down $1000
      - type: "correlation_breach"
        max_correlation: 0.8  # Reduce correlated positions
  
  execution:
    order_type: "adaptive"  # Broker's smart routing
    time_in_force: "DAY"
    
    # Market hours only
    trading_hours:
      start: "09:30"
      end: "16:00"
      timezone: "US/Eastern"
    
    # Minimum time between trades
    throttling:
      min_time_between_orders: 60  # seconds
      max_orders_per_minute: 5

# Monitoring and alerts
monitoring:
  dashboard:
    enabled: true
    port: 8080
    auth_required: true
  
  alerts:
    email:
      enabled: true
      recipients: ["trader@company.com", "risk@company.com"]
      triggers:
        - "daily_loss_threshold"
        - "position_limit_breach"
        - "system_error"
    
    slack:
      enabled: true
      webhook: "${SLACK_WEBHOOK_URL}"
      channel: "#trading-alerts"

# Logging for compliance
logging:
  level: "INFO"
  
  outputs:
    - type: "file"
      path: "logs/live_trading.log"
      rotation: "daily"
      retention: "30_days"
    
    - type: "database"
      table: "trading_audit_log"
      connection: "postgresql://user:pass@localhost/compliance"
  
  audit_events:
    - "order_submitted"
    - "order_filled" 
    - "risk_limit_triggered"
    - "position_opened"
    - "position_closed"
    - "system_start"
    - "system_stop"
```

## How the Coordinator Interprets YAML

### 1. Configuration Parsing and Validation

```python
class Coordinator:
    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            # Load and validate YAML
            self.config = self._load_and_validate_yaml(config_path)
        else:
            self.config = {}
    
    def _load_and_validate_yaml(self, config_path: str) -> Dict[str, Any]:
        """Load YAML and validate against schema"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate against schema
        self._validate_config_schema(config)
        
        # Expand environment variables
        config = self._expand_environment_variables(config)
        
        return config
    
    async def execute_workflow_from_yaml(self, yaml_path: str) -> WorkflowResult:
        """Execute workflow directly from YAML file"""
        config_dict = self._load_and_validate_yaml(yaml_path)
        workflow_config = WorkflowConfig.from_dict(config_dict)
        return await self.execute_workflow(workflow_config)
```

### 2. Container Factory Driven by YAML

```python
class BacktestContainerFactory:
    @staticmethod
    def create_from_yaml(yaml_config: Dict[str, Any]) -> BacktestContainer:
        """Create standardized container from YAML specification"""
        
        # Parse YAML sections
        data_config = yaml_config.get('data', {})
        infrastructure_config = yaml_config.get('infrastructure', {})
        classifiers_config = yaml_config.get('classifiers', [])
        execution_config = yaml_config.get('execution', {})
        
        # Create container with unique ID
        container_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        container = BacktestContainer(container_id)
        
        # 1. Create data layer from YAML
        data_handler = DataHandlerFactory.create_from_config(data_config)
        container.add_component('data_handler', data_handler)
        
        # 2. Create indicator hub from YAML
        indicator_hub = IndicatorHubFactory.create_from_config(infrastructure_config.get('indicators', []))
        container.add_component('indicator_hub', indicator_hub)
        
        # 3. Create classifier containers from YAML
        for classifier_config in classifiers_config:
            classifier_container = ClassifierContainerFactory.create_from_config(classifier_config)
            container.add_subcontainer(classifier_config['name'], classifier_container)
            
            # 4. Create risk & portfolio containers within classifiers
            for risk_config in classifier_config.get('risk_containers', []):
                risk_container = RiskPortfolioContainerFactory.create_from_config(risk_config)
                classifier_container.add_subcontainer(risk_config['name'], risk_container)
                
                # 5. Create strategies within risk containers
                for strategy_config in risk_config.get('strategies', []):
                    strategy = StrategyFactory.create_from_config(strategy_config)
                    risk_container.add_component(strategy_config['name'], strategy)
        
        # 6. Create execution engine from YAML
        execution_engine = ExecutionEngineFactory.create_from_config(execution_config)
        container.add_component('execution_engine', execution_engine)
        
        # 7. Wire event buses (standardized pattern)
        container.wire_event_flows()
        
        return container
```

### 3. Multi-Phase Orchestration from YAML

```python
class OptimizationCoordinator:
    async def execute_optimization_from_yaml(self, yaml_config: Dict[str, Any]) -> OptimizationResult:
        """Execute multi-phase optimization from YAML"""
        
        optimization_config = yaml_config.get('optimization', {})
        
        results = OptimizationResult()
        
        # Phase 1: Parameter optimization
        if 'phase_1' in optimization_config:
            phase_1_config = optimization_config['phase_1']
            phase_1_results = await self._execute_phase_1(phase_1_config, yaml_config)
            results.add_phase_result('phase_1', phase_1_results)
        
        # Phase 2: Ensemble optimization (uses Phase 1 results)
        if 'phase_2' in optimization_config:
            phase_2_config = optimization_config['phase_2']
            phase_2_results = await self._execute_phase_2(phase_2_config, results.get_phase_result('phase_1'))
            results.add_phase_result('phase_2', phase_2_results)
        
        # Phase 3: Validation (uses Phase 2 results)  
        if 'phase_3' in optimization_config:
            phase_3_config = optimization_config['phase_3']
            phase_3_results = await self._execute_phase_3(phase_3_config, results.get_phase_result('phase_2'))
            results.add_phase_result('phase_3', phase_3_results)
        
        return results
```

## Benefits of YAML-Driven Architecture

### 1. **Complete Reproducibility**
```bash
# Share exact experiment
git add configs/my_experiment.yaml
git commit -m "Momentum strategy test with HMM classifier"

# Anyone can reproduce exactly
coordinator.execute_workflow_from_yaml("configs/my_experiment.yaml")
```

### 2. **Non-Technical Collaboration**
```yaml
# Strategist can modify without coding
strategies:
  - name: "momentum_strategy"
    parameters:
      fast_period: 15  # Changed from 10
      slow_period: 35  # Changed from 30
```

### 3. **Environment-Specific Configurations**
```yaml
# configs/development.yaml - Paper trading
broker:
  paper_trading: true
  
# configs/production.yaml - Live trading  
broker:
  paper_trading: false
  account_id: "${PROD_ACCOUNT_ID}"
```

### 4. **Configuration Inheritance and Templates**
```yaml
# configs/base_strategy.yaml
defaults: &defaults
  data:
    timeframe: "1D"
    symbols: ["AAPL", "GOOGL", "MSFT"]
  
  execution:
    slippage: {type: "percentage", value: 0.001}
    commission: {type: "fixed", amount: 1.0}

# configs/momentum_test.yaml
<<: *defaults  # Inherit base configuration

workflow:
  type: "backtest"
  
strategies:
  - name: "momentum_strategy"
    class: "MomentumStrategy"
    # Only specify strategy-specific parameters
```

## Summary: YAML as the Single Source of Truth

The ADMF-PC system treats YAML configuration as the **single source of truth** for all trading experiments. The Coordinator serves as the universal interpreter that can take any valid YAML configuration and create the appropriate container structure to execute it.

This approach provides:

- **Consistency**: Same YAML structure works for backtest, optimization, and live trading
- **Auditability**: Every experiment is captured in version control
- **Collaboration**: Researchers share configs, not code
- **Flexibility**: Change any aspect without coding
- **Reproducibility**: Same YAML always produces same results

The complexity that was previously scattered across multiple modules is now centralized in the Coordinator's YAML interpretation logic, making the entire system more maintainable and reliable.
