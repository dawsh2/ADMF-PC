#!/usr/bin/env python3
"""Add detailed logging to trace execution path."""

# Add temporary debug logging to risk manager
risk_manager_file = "src/risk/strategy_risk_manager.py"

print("=== Adding Debug Logging ===")

# Read the file
with open(risk_manager_file, 'r') as f:
    content = f.read()

# Check if we already have debug logging
if "EXIT_MEMORY_DEBUG" in content:
    print("Debug logging already added")
else:
    # Add debug logging at the start of evaluate_signal
    debug_code = '''
        # EXIT_MEMORY_DEBUG - Temporary logging
        logger.info(f"[EXIT_MEMORY_DEBUG] evaluate_signal called")
        logger.info(f"[EXIT_MEMORY_DEBUG] exit_memory_enabled: {portfolio_state.get('exit_memory_enabled')}")
        logger.info(f"[EXIT_MEMORY_DEBUG] exit_memory contents: {portfolio_state.get('exit_memory', {})}")
        '''
    
    # Insert after the function definition
    insert_pos = content.find('decisions = []')
    if insert_pos > 0:
        new_content = content[:insert_pos] + debug_code + '\n        ' + content[insert_pos:]
        
        # Write back
        with open(risk_manager_file, 'w') as f:
            f.write(new_content)
        
        print("✓ Added debug logging to risk manager")
        print("")
        print("Now run your backtest and check the logs for [EXIT_MEMORY_DEBUG]")
        print("This will show if exit memory is being used at all")
    else:
        print("❌ Could not find insertion point")