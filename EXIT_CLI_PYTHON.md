# How to Exit Python from CLI

## If you're running a Python script (python main.py)

**To stop the running script:**
```bash
# Press Ctrl+C (interrupt the program)
^C
```

This will stop the script and return you to the bash prompt.

## If you're in a virtual environment (venv)

You'll see something like `(venv)` or `(.venv)` in your prompt:
```bash
(venv) user@machine:~/ADMF-PC$ 
```

**You don't need to "kill" the venv.** Just:

1. First stop any running Python script with Ctrl+C
2. Then deactivate the virtual environment:
```bash
deactivate
```

Your prompt will return to normal:
```bash
user@machine:~/ADMF-PC$ 
```

## Complete Restart Process

1. **Stop the current Python script:**
   ```bash
   # Press Ctrl+C
   ```

2. **Clear Python cache:**
   ```bash
   ./clear_python_cache.sh
   ```

3. **If using venv, you can either:**
   - Option A: Just run again (venv stays active)
   ```bash
   python main.py --config config/bollinger/test.yaml
   ```
   
   - Option B: Full restart with deactivate/reactivate
   ```bash
   deactivate
   source venv/bin/activate  # or whatever your venv path is
   python main.py --config config/bollinger/test.yaml
   ```

## The Fix is Applied!

I've added OHLC data to the Bollinger Bands strategy. Now when you run:
```bash
python main.py --config config/bollinger/test.yaml
```

The risk manager will have access to:
- `high` price to check if take profit was hit
- `low` price to check if stop loss was hit

This should fix the incorrect exit prices and reduce trades from 453 to ~416!