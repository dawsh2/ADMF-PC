#!/usr/bin/env python3
"""
Monitor live signal generation from the ensemble.
Shows bar accumulation and signal generation status.
"""

import subprocess
import sys
import time
import re
from datetime import datetime

def monitor_live_trading():
    """Monitor live trading with detailed status updates."""
    print("🔴 LIVE SIGNAL MONITORING")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    print("Monitoring for:")
    print("  - Bar accumulation (need 30 bars)")
    print("  - Signal generation")
    print("  - Signal tracing")
    print("=" * 60)
    
    try:
        # Start the process
        proc = subprocess.Popen([
            'python', 'main.py', 
            '--alpaca', 
            '--config', 'config/duckdb_ensemble_example.yaml', 
            '--bars', '40'  # Allow up to 40 bars to see signals
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Track state
        connected = False
        bar_count = 0
        signals_generated = 0
        start_time = time.time()
        last_bar_time = None
        
        # Read output line by line
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
                
            line = line.strip()
            if not line:
                continue
            
            # Track connection
            if "✅ Connected to Alpaca WebSocket" in line:
                connected = True
                print(f"✅ CONNECTED at {datetime.now()}")
                
            # Track bars - multiple patterns
            bar_patterns = [
                r"ENSEMBLE BARS.*has (\d+)",  # From ensemble warm-up
                r"Bar count.*: (\d+)",         # From tracer
                r"Live bars streamed: (\d+)",  # From data handler
                r"BAR event #(\d+)",           # From event publishing
            ]
            
            for pattern in bar_patterns:
                match = re.search(pattern, line)
                if match:
                    new_count = int(match.group(1))
                    if new_count > bar_count:
                        bar_count = new_count
                        current_time = datetime.now()
                        
                        if last_bar_time:
                            time_between = (current_time - last_bar_time).total_seconds()
                            print(f"📊 Bar #{bar_count} received at {current_time.strftime('%H:%M:%S')} "
                                  f"(+{time_between:.1f}s)")
                        else:
                            print(f"📊 Bar #{bar_count} received at {current_time.strftime('%H:%M:%S')}")
                            
                        last_bar_time = current_time
                        
                        # Check if we should see signals soon
                        if bar_count >= 30 and signals_generated == 0:
                            print("🎯 READY FOR SIGNALS - 30 bar warm-up complete!")
                        
                        break
            
            # Track signal generation
            signal_patterns = [
                r"SIGNAL.*event.*published",
                r"Published.*SIGNAL",
                r"📈 SIGNAL:",
                r"Processing signal from",
                r"MultiStrategyTracer received SIGNAL event",
                r"Generated signal.*strength",
            ]
            
            for pattern in signal_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    signals_generated += 1
                    print(f"🎉 SIGNAL #{signals_generated} GENERATED!")
                    print(f"   Details: {line}")
                    break
            
            # Show critical errors
            if "Error" in line and "ENSEMBLE BARS" not in line:
                print(f"❌ ERROR: {line}")
                
            # Show connection issues
            if "connection limit" in line:
                print(f"⚠️ CONNECTION LIMIT HIT")
                break
                
        # Summary
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("📊 SESSION SUMMARY:")
        print(f"  - Duration: {elapsed:.1f} seconds")
        print(f"  - Connected: {'Yes' if connected else 'No'}")
        print(f"  - Bars received: {bar_count}")
        print(f"  - Signals generated: {signals_generated}")
        
        if bar_count < 30:
            print(f"\n⏳ Need {30 - bar_count} more bars for signal generation")
            print("   (During market hours, this takes ~30 minutes)")
        elif signals_generated == 0:
            print("\n🤔 Enough bars but no signals - check strategy logic")
        else:
            print(f"\n✅ Success! Generated {signals_generated} signals")
            
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped by user")
        if proc.poll() is None:
            proc.terminate()
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    monitor_live_trading()