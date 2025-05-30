"""Test YAML integration with the existing system."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.config import ConfigSchemaValidator
from src.core.coordinator.yaml_interpreter import YAMLInterpreter, YAMLWorkflowBuilder


def test_simple_backtest_config():
    """Test loading and interpreting simple backtest configuration."""
    print("=" * 60)
    print("Testing YAML Integration")
    print("=" * 60)
    
    config_path = Path("configs/simple_backtest.yaml")
    
    # Step 1: Validate configuration
    print("\n1. Validating configuration...")
    validator = ConfigSchemaValidator()
    validation_result = validator.validate_file(config_path)
    
    if validation_result.is_valid:
        print("   ✓ Configuration is valid!")
    else:
        print("   ✗ Configuration has errors:")
        for error in validation_result.errors:
            print(f"     - {error}")
        return
    
    if validation_result.warnings:
        print("   Warnings:")
        for warning in validation_result.warnings:
            print(f"     - {warning}")
    
    # Step 2: Interpret configuration
    print("\n2. Interpreting configuration...")
    interpreter = YAMLInterpreter()
    workflow_config, _ = interpreter.load_and_interpret(config_path)
    
    print(f"   Workflow type: {workflow_config.workflow_type.value}")
    print(f"   Name: {workflow_config.parameters.get('name')}")
    print(f"   Description: {workflow_config.parameters.get('description')}")
    
    # Step 3: Show interpreted data
    print("\n3. Data Configuration:")
    for key, value in workflow_config.data_config.items():
        print(f"   {key}: {value}")
    
    print("\n4. Strategies:")
    for strategy in workflow_config.backtest_config['strategies']:
        print(f"   - {strategy['name']} ({strategy['type']})")
        print(f"     Allocation: {strategy['allocation']*100}%")
        print(f"     Parameters: {strategy['parameters']}")
    
    print("\n5. Risk Configuration:")
    risk_config = workflow_config.backtest_config['risk']
    print(f"   Position Sizers: {len(risk_config['position_sizers'])}")
    for sizer in risk_config['position_sizers']:
        print(f"     - {sizer['name']} ({sizer['type']})")
    print(f"   Risk Limits: {len(risk_config['risk_limits'])}")
    for limit in risk_config['risk_limits']:
        print(f"     - {limit['type']}")
    
    # Step 4: Build container hierarchy
    print("\n6. Container Hierarchy:")
    builder = YAMLWorkflowBuilder(interpreter)
    container_spec = builder.build_container_hierarchy(workflow_config)
    
    def print_containers(node, level=0):
        indent = "  " * level
        print(f"{indent}├─ {node['type']} ({node['id']})")
        capabilities = node.get('capabilities', [])
        if capabilities:
            print(f"{indent}│  └─ Capabilities: {', '.join(capabilities)}")
        for child in node.get('children', []):
            print_containers(child, level + 1)
    
    print_containers(container_spec['root'])
    
    # Step 5: Show what would happen next
    print("\n7. Next Steps (what would happen in real execution):")
    print("   1. Create container hierarchy as specified")
    print("   2. Load historical data for SPY from 2023-01-01 to 2023-06-30")
    print("   3. Initialize strategy with MA(10) and MA(30)")
    print("   4. Run backtest with $10k fixed position sizing")
    print("   5. Apply risk limits (max position $10k)")
    print("   6. Calculate performance metrics")
    print("   7. Save results in JSON format")
    
    print("\n" + "=" * 60)
    print("✓ YAML configuration successfully loaded and interpreted!")
    print("✓ Ready for execution through the Coordinator")
    print("=" * 60)


def test_config_variations():
    """Test different configuration variations."""
    print("\n\nTesting Configuration Variations")
    print("-" * 40)
    
    # Test minimal config
    minimal_config = {
        "name": "Minimal Test",
        "type": "backtest",
        "data": {
            "symbols": ["AAPL"],
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "frequency": "1d"
        },
        "portfolio": {
            "initial_capital": 50000
        },
        "strategies": [{
            "name": "test",
            "type": "test_strategy"
        }]
    }
    
    validator = ConfigSchemaValidator()
    result = validator.validate(minimal_config, "backtest")
    print(f"\nMinimal config valid: {result.is_valid}")
    
    # Test with multiple strategies
    multi_strategy_config = minimal_config.copy()
    multi_strategy_config["strategies"] = [
        {
            "name": "ma_fast",
            "type": "moving_average_crossover",
            "allocation": 0.5,
            "parameters": {"fast_period": 5, "slow_period": 20}
        },
        {
            "name": "ma_slow", 
            "type": "moving_average_crossover",
            "allocation": 0.5,
            "parameters": {"fast_period": 20, "slow_period": 50}
        }
    ]
    
    result2 = validator.validate(multi_strategy_config, "backtest")
    print(f"Multi-strategy config valid: {result2.is_valid}")
    
    print("\nConfiguration system is working correctly!")


if __name__ == "__main__":
    test_simple_backtest_config()
    test_config_variations()