"""
Example demonstrating trace level presets in ADMF-PC.

Shows how to use trace levels to control event tracing overhead
for different scenarios.
"""

import yaml

# Example 1: Optimization workflow with minimal tracing
optimization_config = """
workflow: parameter_optimization
trace_level: minimal  # Minimal overhead for optimization

data:
  symbols: [SPY]
  start_date: '2023-01-01'
  end_date: '2023-12-31'
  
strategies:
  - type: momentum
    parameter_grid:
      fast_period: [5, 10, 15, 20]
      slow_period: [20, 30, 40, 50]
"""

# Example 2: Debugging with full tracing
debug_config = """
workflow: simple_backtest
trace_level: debug  # Full event tracing for debugging

data:
  symbols: [SPY]
  start_date: '2023-12-01'
  end_date: '2023-12-31'
  
strategies:
  - type: momentum
    parameters:
      fast_period: 10
      slow_period: 20
"""

# Example 3: Production with no tracing overhead
production_config = """
workflow: live_trading_simulation
trace_level: none  # Zero overhead for production

data:
  symbols: [SPY, QQQ]
  mode: live_stream
  
strategies:
  - type: momentum
    parameters:
      fast_period: 10
      slow_period: 20
      
risk:
  max_position_size: 0.1
  max_drawdown: 0.05
"""

# Example 4: Custom trace level configuration
custom_config = """
workflow: adaptive_ensemble

# Instead of trace_level preset, specify custom settings
execution:
  enable_event_tracing: true
  trace_settings:
    max_events: 5000  # Custom limit
    retention_policy: trade_complete
    container_settings:
      # Portfolio containers get more tracing
      portfolio_*:
        enabled: true
        max_events: 10000
      # Data containers get minimal tracing  
      data_*:
        enabled: true
        max_events: 100
      # Strategy containers disabled
      strategy_*:
        enabled: false
"""

# Example 5: Per-phase trace levels in multi-phase workflow
multi_phase_config = """
workflow: walk_forward_optimization

# Base trace level for all phases
trace_level: minimal

# Phase-specific overrides
phases:
  - name: parameter_sweep
    # Use even more minimal tracing for sweep
    config_override:
      execution:
        trace_settings:
          max_events: 50
          
  - name: validation
    # Use normal tracing for validation phase
    config_override:
      trace_level: normal  # Override for this phase
"""

def main():
    """Demonstrate different trace level configurations."""
    
    print("=== Trace Level Examples ===\n")
    
    configs = [
        ("Optimization (Minimal)", optimization_config),
        ("Debugging (Full)", debug_config),
        ("Production (None)", production_config),
        ("Custom Settings", custom_config),
        ("Multi-Phase", multi_phase_config)
    ]
    
    for name, config_str in configs:
        print(f"\n{name}:")
        print("-" * len(name))
        
        config = yaml.safe_load(config_str)
        
        # Show trace level or custom settings
        if 'trace_level' in config:
            print(f"Trace Level: {config['trace_level']}")
        elif 'execution' in config and config['execution'].get('enable_event_tracing'):
            print("Custom trace configuration:")
            trace_settings = config['execution'].get('trace_settings', {})
            print(f"  Max Events: {trace_settings.get('max_events', 'default')}")
            print(f"  Retention: {trace_settings.get('retention_policy', 'default')}")
            if 'container_settings' in trace_settings:
                print("  Container-specific settings configured")
        else:
            print("No tracing configured")

if __name__ == "__main__":
    main()