"""
Type Flow Analysis demonstration for ADMF-PC.

This example shows how to use the new type flow analysis system
to validate adapter configurations and ensure proper event routing.
"""

from dataclasses import dataclass
from typing import Dict, Any, Set, Type

# Import type flow analysis components
from src.core.events import (
    # Semantic events
    MarketDataEvent, IndicatorEvent, TradingSignal, OrderEvent, FillEvent,
    create_caused_event, indicator_to_signal, signal_to_order,
    
    # Type flow analysis
    TypeFlowValidator, EventTypeRegistry, TypeFlowAnalyzer,
    create_flow_visualization, validate_and_visualize,
    
    # Traditional events
    EventType
)

from src.core.communication import AdapterFactory


# Mock container classes for demonstration
@dataclass
class MockContainer:
    """Mock container for demonstration."""
    name: str
    role: str = "unknown"
    
    def receive_event(self, event):
        print(f"Container {self.name} received {type(event).__name__}")


class MockDataContainer(MockContainer):
    """Mock data container that produces market data."""
    
    def __init__(self, name: str):
        super().__init__(name, "data_source")
    
    def produces_events(self) -> Set[Type]:
        return {MarketDataEvent}


class MockStrategyContainer(MockContainer):
    """Mock strategy container that consumes market data and produces signals."""
    
    def __init__(self, name: str):
        super().__init__(name, "strategy")
    
    def produces_events(self) -> Set[Type]:
        return {TradingSignal}
    
    def can_handle_event_type(self, event_type: Type) -> bool:
        return event_type in {MarketDataEvent, IndicatorEvent}


class MockRiskContainer(MockContainer):
    """Mock risk container that consumes signals and produces orders."""
    
    def __init__(self, name: str):
        super().__init__(name, "risk_manager")
    
    def produces_events(self) -> Set[Type]:
        return {OrderEvent}
    
    def can_handle_event_type(self, event_type: Type) -> bool:
        return event_type in {TradingSignal}


class MockExecutionContainer(MockContainer):
    """Mock execution container that consumes orders and produces fills."""
    
    def __init__(self, name: str):
        super().__init__(name, "execution_engine")
    
    def produces_events(self) -> Set[Type]:
        return {FillEvent}
    
    def can_handle_event_type(self, event_type: Type) -> bool:
        return event_type in {OrderEvent}


def demo_semantic_events():
    """Demonstrate semantic event creation and lineage."""
    print("=== Semantic Events Demo ===")
    
    # Create a market data event
    market_data = MarketDataEvent(
        symbol="AAPL",
        price=150.0,
        volume=1000,
        source_container="data_feed",
        source_component="market_data_reader"
    )
    print(f"Created: {market_data}")
    
    # Create an indicator event caused by market data
    indicator = create_caused_event(
        market_data,
        IndicatorEvent,
        symbol="AAPL",
        indicator_name="RSI",
        value=0.7,
        source_container="indicator_engine"
    )
    print(f"Created: {indicator}")
    print(f"Causation chain: {market_data.event_id} -> {indicator.event_id}")
    
    # Transform indicator to signal
    signal = indicator_to_signal(
        indicator,
        source_container="momentum_strategy"
    )
    print(f"Transformed to: {signal}")
    print(f"Signal action: {signal.action}, strength: {signal.strength}")
    
    # Transform signal to order
    order = signal_to_order(
        signal,
        quantity=100,
        source_container="risk_manager"
    )
    print(f"Transformed to: {order}")
    print(f"Complete lineage: {market_data.event_id} -> {indicator.event_id} -> {signal.event_id} -> {order.event_id}")
    print()


def demo_type_flow_validation():
    """Demonstrate type flow validation."""
    print("=== Type Flow Validation Demo ===")
    
    # Create mock containers
    containers = {
        "data_feed": MockDataContainer("data_feed"),
        "momentum_strategy": MockStrategyContainer("momentum_strategy"),
        "risk_manager": MockRiskContainer("risk_manager"),
        "execution_engine": MockExecutionContainer("execution_engine")
    }
    
    # Valid pipeline configuration
    valid_config = [
        {
            "type": "pipeline",
            "name": "trading_pipeline",
            "containers": ["data_feed", "momentum_strategy", "risk_manager", "execution_engine"]
        }
    ]
    
    # Invalid pipeline configuration (wrong order)
    invalid_config = [
        {
            "type": "pipeline", 
            "name": "broken_pipeline",
            "containers": ["data_feed", "risk_manager", "momentum_strategy", "execution_engine"]
        }
    ]
    
    # Test valid configuration
    print("Testing valid configuration:")
    validator = TypeFlowValidator()
    
    for adapter_config in valid_config:
        result = validator.validate_adapter_config(adapter_config, containers)
        if result.valid:
            print(f"✓ {adapter_config['name']}: Valid")
        else:
            print(f"✗ {adapter_config['name']}: Invalid")
            for error in result.errors:
                print(f"  Error: {error}")
        for warning in result.warnings:
            print(f"  Warning: {warning}")
    
    print()
    
    # Test invalid configuration
    print("Testing invalid configuration:")
    for adapter_config in invalid_config:
        result = validator.validate_adapter_config(adapter_config, containers)
        if result.valid:
            print(f"✓ {adapter_config['name']}: Valid")
        else:
            print(f"✗ {adapter_config['name']}: Invalid")
            for error in result.errors:
                print(f"  Error: {error}")
        for warning in result.warnings:
            print(f"  Warning: {warning}")
    
    print()


def demo_flow_visualization():
    """Demonstrate flow visualization."""
    print("=== Flow Visualization Demo ===")
    
    # Create containers
    containers = {
        "data_feed": MockDataContainer("data_feed"),
        "momentum_strategy": MockStrategyContainer("momentum_strategy"),
        "risk_manager": MockRiskContainer("risk_manager"),
        "execution_engine": MockExecutionContainer("execution_engine")
    }
    
    # Create adapter configuration
    adapters_config = [
        {
            "type": "pipeline",
            "name": "main_pipeline", 
            "containers": ["data_feed", "momentum_strategy", "risk_manager", "execution_engine"]
        },
        {
            "type": "broadcast",
            "name": "monitoring",
            "source": "momentum_strategy",
            "targets": ["risk_manager", "logger", "monitor"]
        }
    ]
    
    # Add missing containers for broadcast demo
    containers["logger"] = MockContainer("logger", "logger")
    containers["monitor"] = MockContainer("monitor", "monitor")
    
    # Generate text visualization
    print("Text Visualization:")
    text_viz = create_flow_visualization(containers, adapters_config, "text")
    print(text_viz)
    print()
    
    # Generate Mermaid diagram
    print("Mermaid Diagram:")
    mermaid_viz = create_flow_visualization(containers, adapters_config, "mermaid")
    print(mermaid_viz)
    print()
    
    # Generate comprehensive report
    print("Comprehensive Report:")
    report = create_flow_visualization(containers, adapters_config, "report")
    print(report)
    print()


def demo_adapter_factory_validation():
    """Demonstrate adapter factory with type validation."""
    print("=== Adapter Factory Validation Demo ===")
    
    # Create containers
    containers = {
        "data_feed": MockDataContainer("data_feed"),
        "momentum_strategy": MockStrategyContainer("momentum_strategy"),
        "risk_manager": MockRiskContainer("risk_manager"),
        "execution_engine": MockExecutionContainer("execution_engine")
    }
    
    # Create adapter factory with validation enabled
    factory = AdapterFactory(enable_type_validation=True)
    
    # Test configuration validation
    adapters_config = [
        {
            "type": "pipeline",
            "name": "main_pipeline",
            "containers": ["data_feed", "momentum_strategy", "risk_manager", "execution_engine"]
        }
    ]
    
    print("Validating adapter configuration...")
    is_valid = factory.validate_configuration(adapters_config, containers)
    print(f"Configuration valid: {is_valid}")
    
    if is_valid:
        print("\nCreating adapters...")
        try:
            adapters = factory.create_adapters_from_config(adapters_config, containers)
            print(f"Successfully created {len(adapters)} adapters")
            
            # Get configuration report
            print("\nConfiguration Report:")
            report = factory.get_configuration_report(adapters_config, containers)
            print(report)
            
        except Exception as e:
            print(f"Error creating adapters: {e}")
    
    print()


def demo_validation_and_visualization():
    """Demonstrate combined validation and visualization."""
    print("=== Combined Validation and Visualization Demo ===")
    
    # Create containers
    containers = {
        "data_feed": MockDataContainer("data_feed"),
        "momentum_strategy": MockStrategyContainer("momentum_strategy"),
        "risk_manager": MockRiskContainer("risk_manager"),
        "execution_engine": MockExecutionContainer("execution_engine")
    }
    
    # Create adapter configuration
    adapters_config = [
        {
            "type": "pipeline",
            "name": "trading_pipeline",
            "containers": ["data_feed", "momentum_strategy", "risk_manager", "execution_engine"]
        }
    ]
    
    # Validate and visualize
    print("Combined validation and visualization:")
    combined_report = validate_and_visualize(
        adapters_config, 
        containers, 
        execution_mode="full_backtest"
    )
    print(combined_report)


if __name__ == "__main__":
    print("ADMF-PC Type Flow Analysis Demonstration")
    print("=" * 50)
    print()
    
    demo_semantic_events()
    demo_type_flow_validation()
    demo_flow_visualization()
    demo_adapter_factory_validation()
    demo_validation_and_visualization()
    
    print("Demo completed successfully! 🎉")