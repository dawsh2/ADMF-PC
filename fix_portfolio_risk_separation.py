#!/usr/bin/env python3
"""
Fix to properly separate portfolio state management from risk decisions.

The portfolio should:
1. Receive signals with price data
2. Call risk manager for ALL decisions (entry and exit)
3. Create orders based on risk decisions
4. NOT subscribe to BAR events
"""

# This is a documentation file showing the needed changes

NEEDED_CHANGES = """
1. In portfolio/state.py process_event() for SIGNAL events:
   - Extract price from signal metadata
   - For new signals: call risk manager for entry decision
   - For existing positions: call risk manager to check exit conditions
   - Create orders based on risk decisions

2. Remove BAR event handling from portfolio/state.py:
   - Delete the entire BAR/TICK event handler (lines 718-805)
   - Portfolio should ONLY process SIGNAL, FILL, and ORDER events

3. Create or update risk manager to handle:
   - Entry decisions: validate_signal()
   - Exit decisions: check_position_exit()
   - Both methods should be stateless, taking position state as input

4. Update topology.py:
   - Do NOT subscribe portfolio to BAR events
   - Ensure portfolio has reference to risk manager

5. Ensure signals contain price data:
   - Strategies already include price in metadata
   - Portfolio can use this for risk checks
"""

EXAMPLE_FLOW = """
# On SIGNAL event:
def process_signal(self, event):
    signal = event.payload
    symbol = signal['symbol']
    price = signal['metadata']['price']  # Price from signal
    
    # Check existing positions for exits
    if symbol in self._positions:
        position = self._positions[symbol]
        position_state = {
            'symbol': symbol,
            'quantity': position.quantity,
            'entry_price': position.average_price,
            'current_price': price,
            'bars_held': position.metadata.get('bars_held', 0)
        }
        
        # Call risk manager for exit decision
        exit_decision = self.risk_manager.check_position_exit(
            position_state,
            self._strategy_risk_rules.get(position.metadata.get('strategy_id'))
        )
        
        if exit_decision.should_exit:
            # Create exit order
            self._create_exit_order(position, exit_decision)
            return
    
    # For new signals, check entry
    entry_decision = self.risk_manager.validate_signal(
        signal,
        self.get_portfolio_state(),
        {'current_price': price}
    )
    
    if entry_decision.approved:
        # Create entry order
        self._create_entry_order(signal, entry_decision)
"""

print("This fix would properly separate concerns:")
print("- Portfolio: manages state")
print("- Risk Manager: makes entry/exit decisions")
print("- No race conditions from BAR events")
print("- Stop losses checked on every signal (which comes every bar)")