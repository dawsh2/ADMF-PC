#!/usr/bin/env python3
"""
Debug Signal Flow - Comprehensive End-to-End Signal Tracing

This script runs a backtest with detailed logging to trace:
1. Signal generation by strategy
2. Event subscription and delivery
3. Portfolio signal processing
4. Order creation and routing
5. Fill processing and portfolio updates

The goal is to validate that ALL portfolios receive and process signals correctly.
"""

import subprocess
import re
import sys
from collections import defaultdict

def run_debug_backtest():
    """Run backtest with detailed signal flow logging."""
    print("🔍 DEBUGGING SIGNAL FLOW END-TO-END")
    print("=" * 80)
    
    # Run the backtest
    result = subprocess.run([
        'python', 'main.py', '--config', 'config/multi_portfolio_trades_test.yaml'
    ], capture_output=True, text=True)
    
    output = result.stdout + result.stderr
    return output

def analyze_signal_flow(output):
    """Analyze the complete signal flow from the logs."""
    
    print("\n📊 SIGNAL FLOW ANALYSIS")
    print("-" * 60)
    
    # Track different stages - Updated for correct architecture
    strategies_called = defaultdict(int)
    signals_generated = defaultdict(list)
    features_received_by_strategies = defaultdict(int)  # Strategies should receive FEATURES
    signals_received_by_portfolios = defaultdict(int)   # Portfolios should receive SIGNALS
    orders_created = defaultdict(list)
    fills_processed = defaultdict(list)
    portfolio_subscriptions = set()
    
    # Parse the logs
    for line in output.split('\n'):
        
        # Track strategy calls
        if 'Calling strategy with features' in line:
            match = re.search(r'Portfolio (\w+)', line)
            if match:
                portfolio = match.group(1)
                strategies_called[portfolio] += 1
        
        # Track signal generation
        if 'Generated SHORT signal' in line or 'Generated LONG signal' in line:
            signal_type = 'SHORT' if 'SHORT' in line else 'LONG'
            price_match = re.search(r'price=([0-9.]+)', line)
            rsi_match = re.search(r'rsi=([0-9.]+)', line)
            if price_match and rsi_match:
                signals_generated['global'].append({
                    'type': signal_type,
                    'price': float(price_match.group(1)),
                    'rsi': float(rsi_match.group(1))
                })
        
        # Track features received by STRATEGIES (not portfolios!)
        if 'Received event payload keys' in line and 'strategy' in line.lower():
            # Features should go to strategy services, not portfolios
            strategy_match = re.search(r'strategy.*service', line)
            if strategy_match:
                features_received_by_strategies['strategy_services'] += 1
        
        # Track SIGNALS received by portfolios
        if 'Received SIGNAL event' in line or 'Processing signal' in line:
            match = re.search(r'Portfolio (\w+)', line)
            if match:
                portfolio = match.group(1)
                signals_received_by_portfolios[portfolio] += 1
        
        # Track portfolio subscriptions
        if 'Initialized portfolio' in line:
            match = re.search(r'portfolio (\w+)', line)
            if match:
                portfolio = match.group(1)
                portfolio_subscriptions.add(portfolio)
        
        # Track order creation
        if 'published ORDER event' in line:
            match = re.search(r'Portfolio (\w+).*for (\w+) (\w+)', line)
            if match:
                portfolio = match.group(1)
                direction = match.group(2)
                symbol = match.group(3)
                orders_created[portfolio].append({
                    'direction': direction,
                    'symbol': symbol
                })
        
        # Track fill processing
        if 'processed fill for' in line:
            match = re.search(r'Portfolio (\w+) processed fill for (\w+): (\w+) (\d+)', line)
            if match:
                portfolio = match.group(1)
                symbol = match.group(2)
                side = match.group(3)
                quantity = int(match.group(4))
                fills_processed[portfolio].append({
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity
                })
    
    # Analysis results
    print(f"📋 Portfolio Subscriptions: {len(portfolio_subscriptions)}")
    for portfolio in sorted(portfolio_subscriptions):
        print(f"   - {portfolio}")
    
    print(f"\n🎯 Signals Generated Globally: {len(signals_generated['global'])}")
    for i, signal in enumerate(signals_generated['global'][:5], 1):
        print(f"   {i}. {signal['type']} @ ${signal['price']:.2f} (RSI: {signal['rsi']:.1f})")
    if len(signals_generated['global']) > 5:
        print(f"   ... and {len(signals_generated['global']) - 5} more")
    
    print(f"\n📥 Features Received by Strategy Services:")
    if features_received_by_strategies:
        for service in sorted(features_received_by_strategies.keys()):
            count = features_received_by_strategies[service]
            print(f"   - {service}: {count} FEATURES events")
    else:
        print("   - ❌ NO strategy services receiving FEATURES events")
    
    print(f"\n🎯 Signals Received by Portfolios:")
    if signals_received_by_portfolios:
        for portfolio in sorted(signals_received_by_portfolios.keys()):
            count = signals_received_by_portfolios[portfolio]
            print(f"   - {portfolio}: {count} SIGNALS events")
    else:
        print("   - ❌ NO portfolios receiving SIGNALS events")
    
    print(f"\n🔄 Strategy Calls by Portfolio:")
    for portfolio in sorted(strategies_called.keys()):
        count = strategies_called[portfolio]
        print(f"   - {portfolio}: {count} strategy calls")
    
    print(f"\n📤 Orders Created by Portfolio:")
    for portfolio in sorted(orders_created.keys()):
        orders = orders_created[portfolio]
        print(f"   - {portfolio}: {len(orders)} orders")
        for order in orders:
            print(f"     └─ {order['direction']} {order['symbol']}")
    
    print(f"\n📈 Fills Processed by Portfolio:")
    for portfolio in sorted(fills_processed.keys()):
        fills = fills_processed[portfolio]
        print(f"   - {portfolio}: {len(fills)} fills")
        for fill in fills:
            print(f"     └─ {fill['side']} {fill['quantity']} {fill['symbol']}")
    
    # Identify issues
    print(f"\n🚨 POTENTIAL ISSUES:")
    
    # Check if strategy services are receiving features (this is correct architecture)
    if not features_received_by_strategies:
        print(f"   ❌ NO strategy services receiving FEATURES events (major architecture issue)")
    else:
        print(f"   ✅ Strategy services receiving FEATURES events correctly")
    
    # Check if portfolios are receiving signals
    expected_portfolios = {f'c{i:04d}' for i in range(9)}
    missing_signals = expected_portfolios - set(signals_received_by_portfolios.keys())
    if missing_signals:
        print(f"   ❌ Portfolios not receiving SIGNALS: {missing_signals}")
    else:
        print(f"   ✅ All portfolios receiving SIGNALS events")
    
    # Check signal to order conversion
    total_signals = len(signals_generated['global'])
    total_orders = sum(len(orders) for orders in orders_created.values())
    if total_orders == 0 and total_signals > 0:
        print(f"   ❌ Generated {total_signals} signals but NO orders created")
    elif total_orders < total_signals:
        print(f"   ⚠️  Generated {total_signals} signals but only {total_orders} orders")
    
    # Check order to fill conversion
    total_fills = sum(len(fills) for fills in fills_processed.values())
    if total_fills < total_orders:
        print(f"   ⚠️  Created {total_orders} orders but only {total_fills} fills")
    
    if not any([missing_signals, total_orders == 0]):
        print(f"   ✅ Signal flow appears to be working correctly")
    
    return {
        'portfolio_subscriptions': portfolio_subscriptions,
        'signals_generated': signals_generated,
        'features_received_by_strategies': features_received_by_strategies,
        'signals_received_by_portfolios': signals_received_by_portfolios,
        'strategies_called': strategies_called,
        'orders_created': orders_created,
        'fills_processed': fills_processed
    }

def analyze_strategy_parameters(output):
    """Analyze which strategy parameters are being used."""
    print(f"\n🎛️  STRATEGY PARAMETER ANALYSIS")
    print("-" * 60)
    
    # Extract strategy configurations
    strategy_configs = defaultdict(list)
    
    for line in output.split('\n'):
        if "'strategy_params':" in line:
            # Extract the strategy config from the line
            match = re.search(r"'strategy_params': ({[^}]+})", line)
            if match:
                config_str = match.group(1)
                # Extract key parameters
                name_match = re.search(r"'name': '([^']+)'", config_str)
                rsi_long_match = re.search(r"'rsi_threshold_long': (\d+)", config_str)
                rsi_short_match = re.search(r"'rsi_threshold_short': (\d+)", config_str)
                
                if name_match:
                    name = name_match.group(1)
                    rsi_long = int(rsi_long_match.group(1)) if rsi_long_match else 'unknown'
                    rsi_short = int(rsi_short_match.group(1)) if rsi_short_match else 'unknown'
                    
                    strategy_configs[name].append({
                        'rsi_long': rsi_long,
                        'rsi_short': rsi_short
                    })
    
    for strategy_name, configs in strategy_configs.items():
        print(f"📋 {strategy_name}:")
        for config in configs[:1]:  # Show first config as example
            print(f"   - RSI Long Threshold: {config['rsi_long']}")
            print(f"   - RSI Short Threshold: {config['rsi_short']}")
        print(f"   - Used by {len(configs)} portfolios")
        print()

if __name__ == "__main__":
    # Run the debug analysis
    print("Starting signal flow debug analysis...")
    
    output = run_debug_backtest()
    flow_data = analyze_signal_flow(output)
    analyze_strategy_parameters(output)
    
    print(f"\n📋 SUMMARY:")
    print(f"   - Portfolios initialized: {len(flow_data['portfolio_subscriptions'])}")
    print(f"   - Global signals generated: {len(flow_data['signals_generated']['global'])}")
    print(f"   - Strategy services receiving features: {len(flow_data['features_received_by_strategies'])}")
    print(f"   - Portfolios receiving signals: {len(flow_data['signals_received_by_portfolios'])}")
    print(f"   - Portfolios with strategy calls: {len(flow_data['strategies_called'])}")
    print(f"   - Portfolios creating orders: {len(flow_data['orders_created'])}")
    print(f"   - Portfolios processing fills: {len(flow_data['fills_processed'])}")
    
    print(f"\n🎯 NEXT STEPS:")
    if len(flow_data['features_received_by_strategies']) == 0:
        print("   1. ❌ CRITICAL: Strategy services not receiving FEATURES events")
        print("   2. Check symbol container → strategy service event routing")
        print("   3. Verify feature computation and broadcasting")
    elif len(flow_data['signals_received_by_portfolios']) == 0:
        print("   1. ❌ Strategy services receiving features but portfolios not getting signals")
        print("   2. Check strategy service → portfolio event routing")
        print("   3. Verify signal generation logic")
    elif len(flow_data['strategies_called']) < len(flow_data['portfolio_subscriptions']):
        print("   1. Portfolios receiving signals but not all are calling strategies")
        print("   2. Check signal processing in portfolio containers")
        print("   3. Verify signal conditions are being met")
    else:
        print("   1. ✅ Event flow appears to be working correctly")
        print("   2. Focus on strategy parameter tuning for more signals")