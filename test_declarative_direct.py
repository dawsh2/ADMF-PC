"""
Direct test of declarative modules.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    # Test sequencer_declarative import
    print("Importing sequencer_declarative...")
    from core.coordinator.sequencer_declarative import DeclarativeSequencer
    print("✅ DeclarativeSequencer imported successfully")
    
    # Check container lifecycle
    seq = DeclarativeSequencer()
    print(f"✅ Created sequencer with {len(seq.sequence_patterns)} patterns")
    
    # Check that it has proper execution methods
    if hasattr(seq, '_execute_topology'):
        print("✅ Has _execute_topology method")
    if hasattr(seq, '_collect_phase_results'):
        print("✅ Has _collect_phase_results method")
    if hasattr(seq, '_process_results'):
        print("✅ Has _process_results method")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()