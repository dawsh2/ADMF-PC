#!/usr/bin/env python3
"""Quick live data test with market open."""

import subprocess
import sys
import time
import threading

def run_with_timeout():
    """Run the command and capture output."""
    try:
        # Start the process
        proc = subprocess.Popen([
            'python', 'main.py', 
            '--alpaca', 
            '--config', 'config/duckdb_ensemble_example.yaml', 
            '--bars', '3'
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Read output line by line with timeout
        output_lines = []
        start_time = time.time()
        
        while time.time() - start_time < 20:  # 20 second timeout
            line = proc.stdout.readline()
            if line:
                output_lines.append(line.strip())
                print(line.strip())  # Print as we go
                
                # Check for key indicators
                if "✅ Connected to Alpaca WebSocket" in line:
                    print("🎉 CONNECTION SUCCESS!")
                    
                if "📊 Live bars streamed:" in line or "BAR event" in line:
                    print("🎉 LIVE DATA SUCCESS!")
                    
                if "'tuple' object has no attribute 'symbol'" in line:
                    print("❌ FOUND THE BUG - need to fix bar processing")
                    break
                    
                if "connection limit exceeded" in line:
                    print("⚠️ Hit connection limit")
                    break
                    
            elif proc.poll() is not None:
                # Process finished
                break
                
        # Terminate if still running
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)
            
        return output_lines
        
    except Exception as e:
        print(f"Error: {e}")
        return []

if __name__ == "__main__":
    print("🔴 Testing live Alpaca with market OPEN...")
    print("Looking for connection success and live data...")
    print("=" * 50)
    
    output_lines = run_with_timeout()
    
    print("=" * 50)
    print("🔍 SUMMARY:")
    
    has_connection = any("Connected to Alpaca WebSocket" in line for line in output_lines)
    has_data = any("Live bars streamed" in line or "BAR event" in line for line in output_lines)
    has_error = any("tuple.*symbol" in line for line in output_lines)
    has_limit = any("connection limit" in line for line in output_lines)
    
    if has_connection:
        print("✅ Connection: SUCCESS")
    else:
        print("❌ Connection: FAILED")
        
    if has_data:
        print("✅ Live Data: SUCCESS") 
    elif has_connection:
        print("⏳ Live Data: Connected but waiting...")
    else:
        print("❌ Live Data: NO CONNECTION")
        
    if has_error:
        print("❌ Bug: Bar processing error found")
    elif has_connection:
        print("✅ Bug: Fixed or no data yet")
        
    if has_limit:
        print("⚠️ Issue: Connection limit")