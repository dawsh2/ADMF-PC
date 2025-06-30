#!/usr/bin/env python3
"""
Proper fix: Portfolio just manages state, Risk makes ALL decisions.

Portfolio's ONLY job on receiving a SIGNAL:
1. Update last known prices
2. Pass ALL state to risk manager
3. Create orders based on risk decisions
"""

PORTFOLIO_SIGNAL_HANDLER = """
def process_signal(self, event):
    '''Portfolio receives signal, delegates ALL decisions to risk.'''
    signal = event.payload
    symbol = signal['symbol']
    
    # Extract price from signal metadata
    price = Decimal(str(signal.get('metadata', {}).get('price', 0)))
    if not price:
        logger.warning(f"Signal missing price for {symbol}")
        return
        
    # Update last known price for this symbol
    self._last_prices[symbol] = price
    
    # Prepare portfolio state for risk manager
    portfolio_state = {
        'positions': self._positions,
        'cash': self._cash_balance,
        'pending_orders': self._pending_orders,
        'last_prices': self._last_prices,
        'strategy_risk_rules': self._strategy_risk_rules
    }
    
    # Call risk manager with EVERYTHING
    risk_decisions = self.risk_manager.evaluate_signal(
        signal=signal,
        portfolio_state=portfolio_state,
        timestamp=event.timestamp
    )
    
    # Execute risk decisions
    for decision in risk_decisions:
        if decision['action'] == 'create_order':
            self._create_order_from_decision(decision)
        elif decision['action'] == 'update_position_metadata':
            # Update things like highest_price for trailing stops
            position = self._positions.get(decision['symbol'])
            if position:
                position.metadata.update(decision['updates'])
"""

RISK_MANAGER_EVALUATE = """
def evaluate_signal(self, signal, portfolio_state, timestamp):
    '''Risk manager makes ALL decisions based on signal and state.'''
    decisions = []
    symbol = signal['symbol']
    price = signal['metadata']['price']
    
    # Check ALL positions for exit conditions (not just signal symbol)
    for pos_symbol, position in portfolio_state['positions'].items():
        if position.quantity == 0:
            continue
            
        # Get risk rules for this position's strategy
        strategy_id = position.metadata.get('strategy_id')
        risk_rules = portfolio_state['strategy_risk_rules'].get(strategy_id, {})
        
        # Update position tracking (highest price for trailing stop)
        if position.quantity > 0 and price > position.metadata.get('highest_price', 0):
            decisions.append({
                'action': 'update_position_metadata',
                'symbol': pos_symbol,
                'updates': {'highest_price': price}
            })
        
        # Check exit conditions
        exit_signal = check_exit_conditions(
            position={
                'symbol': pos_symbol,
                'quantity': position.quantity,
                'average_price': position.average_price,
                'metadata': position.metadata
            },
            current_price=portfolio_state['last_prices'].get(pos_symbol, price),
            risk_rules=risk_rules
        )
        
        if exit_signal.should_exit:
            decisions.append({
                'action': 'create_order',
                'symbol': pos_symbol,
                'side': 'SELL' if position.quantity > 0 else 'BUY',
                'quantity': abs(position.quantity),
                'order_type': 'MARKET',
                'reason': exit_signal.reason,
                'exit_type': exit_signal.exit_type,
                'strategy_id': strategy_id + '_exit'
            })
            # Don't process entry for this symbol if exiting
            if pos_symbol == symbol:
                return decisions
    
    # Check if we should enter based on the signal
    entry_decision = self.validate_entry(signal, portfolio_state)
    if entry_decision['approved']:
        decisions.append({
            'action': 'create_order',
            'symbol': symbol,
            'side': 'BUY' if signal['direction'] > 0 else 'SELL',
            'quantity': entry_decision['size'],
            'order_type': 'MARKET',
            'strategy_id': signal['strategy_id']
        })
    
    return decisions
"""

KEY_POINTS = """
Key Architecture Points:

1. Portfolio stores:
   - Current positions with entry prices
   - Last known price for EACH symbol
   - Highest price seen (for trailing stops)
   - Strategy risk rules

2. Risk Manager:
   - Receives ALL state on EVERY signal
   - Checks ALL positions for exits (not just signal symbol)
   - Returns a list of decisions
   - Handles stop loss, trailing stop, take profit

3. Portfolio executes decisions:
   - Creates orders as instructed
   - Updates metadata as instructed
   - Does NOT make any risk decisions itself

4. Critical: Risk checks ALL positions on every signal, using:
   - Entry price from position
   - Current price from last_prices dict
   - Risk rules from strategy config
"""

print("This properly separates concerns:")
print("- Portfolio: state management only")
print("- Risk: ALL entry/exit decisions")
print("- Stop losses work because risk checks all positions on every signal")