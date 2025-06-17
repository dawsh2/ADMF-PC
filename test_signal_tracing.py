#!/usr/bin/env python3
"""
Test signal tracing with historical data to verify the setup.
"""

import subprocess
import time
import re

def test_signal_tracing():
    """Test if signals are being traced with historical data."""
    print("🧪 Testing signal tracing with historical data...")
    print("=" * 60)
    
    # Run with historical data (no --alpaca flag)
    proc = subprocess.Popen([
        'python', 'main.py',
        '--config', 'config/duckdb_ensemble_example.yaml',
        '--bars', '50'  # Enough bars to generate signals
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Track what we find
    bars_seen = 0
    signals_seen = 0
    tracer_setup = False
    workspace = None
    
    start_time = time.time()
    timeout = 30  # 30 second timeout
    
    while time.time() - start_time < timeout:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                break
            continue
            
        line = line.strip()
        
        # Look for tracer setup
        if "MultiStrategyTracer" in line and ("attached" in line or "setup" in line):
            tracer_setup = True
            print(f"✅ Tracer setup: {line}")
            
        # Look for workspace creation
        if "Workspace:" in line and "duckdb_ensemble" in line:
            match = re.search(r'Workspace: (\S+)', line)
            if match:
                workspace = match.group(1)
                print(f"📁 Workspace: {workspace}")
        
        # Track bars
        if "Bar #" in line or "BAR event" in line:
            bars_seen += 1
            
        # Look for signals
        signal_patterns = [
            "SIGNAL.*generated",
            "Published.*SIGNAL", 
            "📈 SIGNAL:",
            "Processing signal",
            "SIGNAL event",
            "Generated signal"
        ]
        
        for pattern in signal_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                signals_seen += 1
                print(f"🎉 Signal #{signals_seen}: {line}")
                break
                
    # Terminate process
    if proc.poll() is None:
        proc.terminate()
        proc.wait(timeout=5)
        
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY:")
    print(f"  - Tracer setup: {'✅ Yes' if tracer_setup else '❌ No'}")
    print(f"  - Workspace: {workspace or 'Not found'}")
    print(f"  - Bars processed: {bars_seen}")
    print(f"  - Signals generated: {signals_seen}")
    
    if signals_seen > 0:
        print("\n✅ Signal tracing is working correctly!")
        if workspace:
            print(f"\n💡 Check signals in workspace:")
            print(f"   cd workspaces/{workspace}")
            print(f"   ls -la")
    else:
        print("\n⚠️ No signals detected - possible issues:")
        print("  1. Strategies may not be generating signals")
        print("  2. Tracing may not be configured correctly")
        print("  3. Not enough data for signal generation")

if __name__ == "__main__":
    test_signal_tracing()