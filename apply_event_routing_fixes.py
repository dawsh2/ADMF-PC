#!/usr/bin/env python3
"""
Apply event routing fixes to prevent cycles and duplicate signal processing.
"""

import re
from pathlib import Path

def apply_strategy_container_fixes():
    """Fix StrategyContainer event routing."""
    file_path = Path("src/execution/containers.py")
    content = file_path.read_text()
    
    # Fix 1: Update _configure_external_events to check if sub-container
    old_pattern = r'''ext_config\['publishes'\]\.extend\(\[
            \{
                'events': \['SIGNAL'\],
                'scope': 'PARENT',  # Send to PortfolioContainer
                'tier': 'standard'
            \}
        \]\)'''
    
    new_code = '''# Check if this is a sub-container
        is_sub_container = '_' in config.get('container_id', '')
        
        if is_sub_container:
            # Sub-containers publish to parent only
            ext_config['publishes'].extend([
                {
                    'events': ['SIGNAL'],
                    'scope': 'PARENT',  # Only to parent container
                    'tier': 'standard'
                }
            ])
        # Main strategy container uses internal bus for parent communication'''
    
    # Fix 2: Add deduplication to StrategyContainer
    dedup_init = '''        # State management
        self._current_indicators: Dict[str, Any] = {}
        self._current_market_data: Dict[str, Any] = {}  # Store latest market data
        
        # Deduplication tracking
        self._processed_signals: Set[str] = set()
        self._signal_cleanup_interval = 300  # 5 minutes'''
    
    # Fix 3: Update _emit_signals to use internal bus
    emit_fix = '''            # ALWAYS use internal communication to parent
            if self.parent_container:
                logger.info(f"🚀 StrategyContainer publishing SIGNAL to parent via internal bus")
                self.publish_event(signal_event, target_scope="parent")
            else:
                # Only use external if truly orphaned (shouldn't happen)
                logger.warning(f"StrategyContainer has no parent, using external routing")
                from ..core.events.hybrid_interface import CommunicationTier
                self.publish_external(signal_event, tier=CommunicationTier.STANDARD)'''
    
    print("Fixes to apply to StrategyContainer:")
    print("1. Make external SIGNAL publishing conditional (sub-containers only)")
    print("2. Add signal deduplication tracking")
    print("3. Use internal bus for parent communication")
    print("4. Add deduplication to _handle_sub_container_signal")

def apply_portfolio_container_fixes():
    """Fix PortfolioContainer event routing."""
    print("\nFixes to apply to PortfolioContainer:")
    print("1. Remove external SIGNAL publishing configuration")
    print("2. Use internal bus only for forwarding signals to parent")
    print("3. Add signal deduplication")
    
    portfolio_config_fix = '''        # Portfolio container only subscribes externally, doesn't publish signals externally
        config['external_events']['subscribes'] = config['external_events'].get('subscribes', []) + [
            {
                'source': '*',
                'events': ['BAR', 'INDICATORS'],
                'tier': 'fast'
            }
        ]'''
    
    portfolio_forward_fix = '''                # Forward to parent (risk container) via INTERNAL bus only
                if self.parent_container:
                    logger.info(f"📤 PortfolioContainer forwarding {len(unique_signals)} signals to parent")
                    self.publish_event(allocated_event, target_scope="parent")'''

def apply_risk_container_fixes():
    """Fix RiskContainer event routing."""
    print("\nFixes to apply to RiskContainer:")
    print("1. Remove external SIGNAL subscription")
    print("2. Add time-window based signal deduplication") 
    print("3. Only publish ORDERs externally")
    
    risk_config_fix = '''        # RiskContainer only publishes ORDERs externally, not subscribes to SIGNALs
        config['external_events']['publishes'] = config['external_events'].get('publishes', []) + [
            {
                'events': ['ORDER'],
                'scope': 'GLOBAL',
                'tier': 'standard'
            }
        ]
        
        # Remove external SIGNAL subscription - signals come via internal bus
        config['external_events']['subscribes'] = config['external_events'].get('subscribes', []) + [
            {
                'source': '*',
                'events': ['FILL'],
                'tier': 'standard'
            }
        ]'''
    
    risk_dedup_fix = '''        self.risk_manager = None
        self._processed_signals: Set[str] = set()
        self._signal_window = {}  # Track signals by timestamp for deduplication'''

def main():
    print("Event Routing Fix Summary")
    print("=" * 50)
    print("\nPROBLEM: Event cycles causing duplicate signal processing")
    print("ROOT CAUSE: Signals being published both internally and externally")
    print("\nSOLUTION: Use internal bus for parent-child, external only for cross-hierarchy")
    
    apply_strategy_container_fixes()
    apply_portfolio_container_fixes()
    apply_risk_container_fixes()
    
    print("\n" + "=" * 50)
    print("MANUAL STEPS REQUIRED:")
    print("1. Edit src/execution/containers.py")
    print("2. Apply the fixes shown above to each container")
    print("3. Add deduplication logic to prevent duplicate processing")
    print("4. Test with: python main.py --config config/multi_strategy_test.yaml --bars 50")
    print("\nEXPECTED RESULT:")
    print("- Only 1 order per signal (not 3-4)")
    print("- Portfolio stays within risk limits")
    print("- No overleveraging")

if __name__ == "__main__":
    main()