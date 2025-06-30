#!/usr/bin/env python3
"""
Example demonstrating the structured event system.

Shows how the new event system eliminates string parsing and enables
flexible subscriptions based on any combination of criteria.
"""

import sys
sys.path.append('.')

from src.core.events.structured_events import (
    create_structured_signal_event,
    SubscriptionDescriptor,
    create_subscription_filter
)
from src.core.events import Event, EventType
from datetime import datetime


def demonstrate_structured_events():
    """Show the difference between old and new event formats."""
    
    print("=== Structured Events Example ===\n")
    
    # Old way - everything encoded in strategy_id
    print("OLD WAY - String-based strategy_id:")
    print("  strategy_id: 'SPY_1m_sma_crossover_grid_5_20'")
    print("  - Must parse string to extract parameters")
    print("  - Fragile matching logic")
    print("  - Name explosion in grid search\n")
    
    # New way - structured data
    print("NEW WAY - Structured event:")
    signal = create_structured_signal_event(
        symbol='SPY',
        timeframe='1m',
        direction='long',
        strength=0.8,
        strategy_type='sma_crossover',
        parameters={'fast_period': 5, 'slow_period': 20},
        metadata={'price': 450.25, 'fast_sma': 449.80, 'slow_sma': 449.50}
    )
    
    print(f"  Event payload:")
    for key, value in signal.payload.items():
        if key != 'strategy_id':  # Skip legacy field
            print(f"    {key}: {value}")
    print()


def demonstrate_subscriptions():
    """Show flexible subscription patterns."""
    
    print("=== Flexible Subscriptions ===\n")
    
    # Create some test events
    events = [
        create_structured_signal_event(
            symbol='SPY', timeframe='1m', direction='long', strength=0.8,
            strategy_type='sma_crossover', 
            parameters={'fast_period': 5, 'slow_period': 20}
        ),
        create_structured_signal_event(
            symbol='SPY', timeframe='1m', direction='short', strength=0.6,
            strategy_type='sma_crossover',
            parameters={'fast_period': 10, 'slow_period': 30}
        ),
        create_structured_signal_event(
            symbol='AAPL', timeframe='5m', direction='long', strength=0.7,
            strategy_type='rsi_threshold',
            parameters={'period': 14, 'threshold': 30}
        ),
        create_structured_signal_event(
            symbol='SPY', timeframe='1m', direction='long', strength=0.9,
            strategy_type='macd_crossover',
            parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
        )
    ]
    
    # Example 1: Subscribe to specific strategy and parameters
    print("1. Subscribe to SPY SMA(5,20) crossover:")
    desc1 = SubscriptionDescriptor({
        'symbol': 'SPY',
        'strategy_type': 'sma_crossover',
        'parameters': {'fast_period': 5, 'slow_period': 20}
    })
    
    for i, event in enumerate(events):
        if desc1.matches(event):
            print(f"   ✓ Event {i} matches")
    print()
    
    # Example 2: Subscribe to all SMA crossovers on SPY
    print("2. Subscribe to all SPY SMA crossovers:")
    desc2 = SubscriptionDescriptor({
        'symbol': 'SPY',
        'strategy_type': 'sma_crossover'
    })
    
    for i, event in enumerate(events):
        if desc2.matches(event):
            print(f"   ✓ Event {i} matches")
    print()
    
    # Example 3: Subscribe to any strategy with fast_period=5
    print("3. Subscribe to any strategy with fast_period=5:")
    desc3 = SubscriptionDescriptor({
        'parameters.fast_period': 5
    })
    
    for i, event in enumerate(events):
        if desc3.matches(event):
            print(f"   ✓ Event {i} matches")
    print()
    
    # Example 4: Subscribe to multiple strategy types
    print("4. Subscribe to multiple strategy types on SPY:")
    desc4 = SubscriptionDescriptor({
        'symbol': 'SPY',
        'strategy_type': ['sma_crossover', 'macd_crossover']
    })
    
    for i, event in enumerate(events):
        if desc4.matches(event):
            print(f"   ✓ Event {i} matches")
    print()


def demonstrate_portfolio_subscriptions():
    """Show how portfolios can subscribe to signals."""
    
    print("=== Portfolio Subscription Examples ===\n")
    
    # Portfolio managing specific strategies
    print("Portfolio A - Manages two specific strategy configurations:")
    portfolio_a_strategies = [
        {'type': 'sma_crossover', 'params': {'fast_period': 5, 'slow_period': 20}},
        {'type': 'rsi_threshold', 'params': {'period': 14, 'threshold': 30}}
    ]
    
    # In real code, this would use create_portfolio_subscription_filter
    print("  Subscribed to:")
    for strat in portfolio_a_strategies:
        print(f"    - {strat['type']} with {strat['params']}")
    print()
    
    # Portfolio managing all momentum strategies
    print("Portfolio B - Manages all momentum strategies on SPY:")
    print("  Subscription criteria:")
    print("    symbol: 'SPY'")
    print("    strategy_type: ['momentum', 'macd_crossover', 'rsi_threshold']")
    print()
    
    # Portfolio with partial parameter matching
    print("Portfolio C - Manages all strategies with RSI period 14:")
    print("  Subscription criteria:")
    print("    parameters.rsi_period: 14")
    print("    OR")
    print("    parameters.period: 14  (for strategies using generic 'period')")
    print()


def demonstrate_benefits():
    """Show the benefits of structured events."""
    
    print("=== Benefits of Structured Events ===\n")
    
    print("1. **No String Parsing**")
    print("   - Parameters are directly accessible")
    print("   - No regex or split operations")
    print("   - Type-safe access to data\n")
    
    print("2. **Flexible Filtering**")
    print("   - Subscribe by symbol, strategy, parameters, or any combination")
    print("   - Partial matching (e.g., any strategy with fast_period=5)")
    print("   - List membership (e.g., strategy in ['sma', 'ema'])\n")
    
    print("3. **Clean Grid Search**")
    print("   - No more 'sma_crossover_grid_5_20_3_15' names")
    print("   - Strategy type stays clean: 'sma_crossover'")
    print("   - Parameters stored separately\n")
    
    print("4. **Natural Composition**")
    print("   - Ensemble strategies work without special handling")
    print("   - Sub-strategies publish with their own clean types")
    print("   - Portfolio can subscribe to ensemble OR components\n")
    
    print("5. **Better Analytics**")
    print("   - Easy to query by parameter values")
    print("   - Group by strategy type across parameters")
    print("   - Compare parameter sensitivity\n")


if __name__ == '__main__':
    demonstrate_structured_events()
    demonstrate_subscriptions()
    demonstrate_portfolio_subscriptions()
    demonstrate_benefits()
    
    print("=== Migration Path ===\n")
    print("The system supports BOTH formats during migration:")
    print("- New strategies use structured events")
    print("- Legacy strategies still work with string IDs")
    print("- Subscriptions handle both formats transparently")
    print("- Gradual migration without breaking changes")
    print("\nSee subscription_helpers.py for migration utilities.")