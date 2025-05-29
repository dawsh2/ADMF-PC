"""
Example usage of the Risk & Portfolio module.

Shows how to create a Risk & Portfolio container that manages
multiple strategy components.
"""

from decimal import Decimal
from datetime import datetime
from typing import Dict, Any

from ..core.containers import UniversalScopedContainer
from ..core.components import ComponentFactory
from .capabilities import RiskPortfolioCapability
from .protocols import Signal, SignalType, OrderSide


def create_risk_portfolio_container(config: Dict[str, Any]) -> UniversalScopedContainer:
    """
    Create a Risk & Portfolio container with strategies.
    
    This demonstrates the correct architecture:
    - Risk & Portfolio container manages multiple strategies
    - Strategies only generate signals
    - Risk & Portfolio converts signals to orders
    """
    # Create the Risk & Portfolio container
    risk_container = UniversalScopedContainer(
        container_id=config['container_id'],
        parent_container=config.get('parent_container')
    )
    
    # Apply Risk & Portfolio capability
    factory = ComponentFactory()
    factory.apply_capability(
        risk_container,
        RiskPortfolioCapability(),
        {
            'initial_capital': config.get('initial_capital', 100000),
            'position_sizers': config.get('position_sizers', [
                {
                    'name': 'default',
                    'type': 'percentage',
                    'percentage': 2.0  # 2% per position
                }
            ]),
            'risk_limits': config.get('risk_limits', [
                {
                    'type': 'position',
                    'max_position': 10000
                },
                {
                    'type': 'exposure',
                    'max_exposure_pct': 20  # 20% max exposure
                },
                {
                    'type': 'drawdown',
                    'max_drawdown_pct': 10,  # 10% max drawdown
                    'reduce_at_pct': 8  # Start reducing at 8%
                }
            ])
        }
    )
    
    # Create strategy components within the Risk & Portfolio container
    strategies = config.get('strategies', [])
    for strategy_config in strategies:
        # Each strategy is a simple component that generates signals
        strategy = factory.create_component({
            'name': strategy_config['name'],
            'class': strategy_config['class'],
            'capabilities': ['lifecycle', 'events'],
            'container': risk_container,
            **strategy_config.get('params', {})
        })
        
        # Strategies would implement signal generation logic
        # For now, we'll add a simple signal generator
        def create_signal_generator(strat_name, symbols):
            def generate_signal(market_data: Dict[str, Any]) -> Signal:
                """Generate a signal based on market data."""
                # Simplified signal generation
                return Signal(
                    signal_id=f"{strat_name}_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    strategy_id=strat_name,
                    symbol=market_data['symbol'],
                    signal_type=SignalType.ENTRY,
                    side=OrderSide.BUY,  # Simplified
                    strength=0.8,
                    metadata={
                        'reason': 'example_signal',
                        'confidence': 0.8
                    }
                )
            return generate_signal
        
        strategy.generate_signal = create_signal_generator(
            strategy_config['name'],
            strategy_config.get('symbols', [])
        )
    
    return risk_container


def example_workflow():
    """
    Example workflow showing Risk & Portfolio in action.
    """
    # Configuration for a conservative Risk & Portfolio container
    config = {
        'container_id': 'conservative_risk_portfolio',
        'initial_capital': 100000,
        'position_sizers': [
            {
                'name': 'default',
                'type': 'percentage',
                'percentage': 2.0  # 2% per position
            },
            {
                'name': 'momentum',
                'type': 'volatility',
                'risk_per_trade': 1.0  # 1% risk per trade
            }
        ],
        'risk_limits': [
            {
                'type': 'position',
                'max_position': 5000  # Max 5000 shares per position
            },
            {
                'type': 'exposure',
                'max_exposure_pct': 15  # Max 15% total exposure
            },
            {
                'type': 'concentration',
                'max_position_pct': 5,  # Max 5% in any position
                'max_sector_pct': 20  # Max 20% in any sector
            },
            {
                'type': 'daily_loss',
                'max_daily_loss': 2000,  # Max $2000 daily loss
                'max_daily_loss_pct': 2  # Max 2% daily loss
            }
        ],
        'strategies': [
            {
                'name': 'momentum_strategy',
                'class': 'MomentumStrategy',
                'symbols': ['AAPL', 'GOOGL', 'MSFT'],
                'params': {
                    'lookback': 20,
                    'threshold': 0.02
                }
            },
            {
                'name': 'mean_reversion_strategy',
                'class': 'MeanReversionStrategy',
                'symbols': ['SPY'],
                'params': {
                    'lookback': 10,
                    'num_std': 2.0
                }
            }
        ]
    }
    
    # Create the Risk & Portfolio container
    risk_container = create_risk_portfolio_container(config)
    
    # Simulate signal generation and processing
    print("\n=== Risk & Portfolio Container Example ===\n")
    
    # Example 1: Process a signal that passes all checks
    signal1 = Signal(
        signal_id="sig_001",
        timestamp=datetime.now(),
        strategy_id="momentum_strategy",
        symbol="AAPL",
        signal_type=SignalType.ENTRY,
        side=OrderSide.BUY,
        strength=0.85,
        metadata={
            'price': 150.00,
            'confidence': 0.85
        }
    )
    
    print("Processing Signal 1 (should pass):")
    print(f"  Strategy: {signal1.strategy_id}")
    print(f"  Symbol: {signal1.symbol}")
    print(f"  Side: {signal1.side}")
    
    order1 = risk_container.process_signal(signal1.__dict__)
    if order1:
        print(f"  ✓ Order created: {order1['quantity']} shares")
    else:
        print("  ✗ Signal rejected")
    
    # Example 2: Process a signal that may be rejected due to exposure
    signal2 = Signal(
        signal_id="sig_002",
        timestamp=datetime.now(),
        strategy_id="momentum_strategy",
        symbol="GOOGL",
        signal_type=SignalType.ENTRY,
        side=OrderSide.BUY,
        strength=0.90,
        metadata={
            'price': 2800.00,
            'confidence': 0.90
        }
    )
    
    print("\nProcessing Signal 2 (may hit exposure limit):")
    print(f"  Strategy: {signal2.strategy_id}")
    print(f"  Symbol: {signal2.symbol}")
    print(f"  Side: {signal2.side}")
    
    order2 = risk_container.process_signal(signal2.__dict__)
    if order2:
        print(f"  ✓ Order created: {order2['quantity']} shares")
    else:
        print("  ✗ Signal rejected (likely due to risk limits)")
    
    # Show portfolio state
    print("\n=== Portfolio State ===")
    portfolio_state = risk_container.get_portfolio_state()
    print(f"Cash: ${portfolio_state['cash']:,.2f}")
    print(f"Positions: {portfolio_state['position_count']}")
    print(f"Total Value: ${portfolio_state['total_value']:,.2f}")
    
    # Show risk metrics
    print("\n=== Risk Metrics ===")
    risk_metrics = risk_container.get_risk_metrics()
    print(f"Total Exposure: {risk_metrics.get('total_exposure_pct', 0):.1f}%")
    print(f"Current Drawdown: {risk_metrics.get('current_drawdown_pct', 0):.1f}%")
    print(f"Max Position Size: {risk_metrics.get('max_position_pct', 0):.1f}%")


if __name__ == "__main__":
    example_workflow()