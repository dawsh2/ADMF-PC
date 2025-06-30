"""Force debug logging to see filter activity"""
import logging
import subprocess
import os

# Set all loggers to DEBUG before running
logging.basicConfig(level=logging.DEBUG)

# Also set environment variable
env = os.environ.copy()
env['PYTHONUNBUFFERED'] = '1'
env['LOG_LEVEL'] = 'DEBUG'

print("Running with forced DEBUG logging...")
print("Looking specifically for filter evaluation logs...\n")

# Run the command with explicit debug
cmd = [
    "python3", "main.py", 
    "--config", "config/test_keltner_filter_working.yaml",
    "--signal-generation",
    "--dataset", "train",
    "--log-level", "DEBUG"  # Force debug logging
]

# Run with real-time output capture
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                          text=True, bufsize=1, env=env)

filter_count = 0
signal_count = 0
line_count = 0

print("=== Real-time Filter Activity ===")
while True:
    line = process.stdout.readline()
    if not line:
        if process.poll() is not None:
            break
        continue
    
    line_count += 1
    
    # Look for filter-specific logs
    if 'config_filter' in line and 'evaluated' in line:
        filter_count += 1
        print(f"FILTER EVAL {filter_count}: {line.strip()}")
    elif 'Added filter for' in line:
        print(f"FILTER ADDED: {line.strip()}")
    elif 'rejected by filter' in line:
        print(f"FILTER REJECT: {line.strip()}")
    elif 'Publishing signal' in line and 'test_keltner' in line:
        signal_count += 1
        print(f"SIGNAL PUB {signal_count}: {line.strip()}")
    elif 'Compiled filter' in line:
        print(f"FILTER COMPILE: {line.strip()}")
    
    # Stop after processing enough lines to avoid too much output
    if line_count > 10000:
        print("\n... (stopping after 10000 lines)")
        process.terminate()
        break

process.wait()

print(f"\n=== Summary ===")
print(f"Total lines processed: {line_count}")
print(f"Filter evaluations found: {filter_count}")
print(f"Signals published: {signal_count}")

if filter_count == 0:
    print("\n⚠️  No filter evaluations found!")
    print("This suggests:")
    print("1. The ComponentState might not be executing strategies")
    print("2. The filter might not be connected properly")
    print("3. The logging might still be suppressed somehow")