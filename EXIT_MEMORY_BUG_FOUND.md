# Exit Memory Bug Found!

## The Problem

Exit memory is NOT working because of a **signal representation mismatch**:

1. **Strategy returns**: `signal_value` as numeric (-1, 0, 1)
2. **Strategy state converts to**: `SignalDirection` enum (LONG, SHORT, FLAT)  
3. **Portfolio tracks**: Numeric value from signal direction
4. **Risk manager compares**: Direction enum vs stored numeric value

## The Bug Flow

1. Bollinger strategy returns `signal_value = -1` (short)
2. Strategy state converts: `-1` → `SignalDirection.SHORT`
3. Portfolio receives SIGNAL event with `direction = "SHORT"`
4. Portfolio maps: `"SHORT"` → `-1.0` and stores in `_last_signal_values`
5. On stop loss, exit memory stores `-1.0`
6. Next signal comes with `direction = "SHORT"`
7. Risk manager maps: `"SHORT"` → `-1.0`
8. Compares: `-1.0 == -1.0` → Signal unchanged, blocks re-entry ✓

Wait... that should work! Let me trace more carefully...

## The Real Issue

Looking at the trace output:
- Signal at bar 16646: -1
- Stop loss at bar 16648: NO SIGNAL DATA
- Re-entry at bar 16649: NO SIGNAL DATA  
- Next signal at bar 16650: 0 (FLAT)

The problem is:
1. Last known signal before stop loss was -1 (SHORT)
2. Exit memory correctly stores -1
3. But at bar 16649, the strategy is called and returns `signal_value = 0` (FLAT)
4. The system processes this as a new FLAT signal
5. FLAT (0) != SHORT (-1), so exit memory is cleared
6. Re-entry is allowed!

## The Root Cause

The Bollinger strategy is returning FLAT signals between entry/exit conditions, but these FLAT signals are clearing the exit memory! The strategy should maintain its signal until explicitly changed.

## Immediate Re-entries Explained

1. Position opens when price > upper band (signal = -1)
2. Price drops, stop loss triggers
3. Next bar, price is between bands, strategy returns 0 (FLAT)
4. System sees signal change from -1 to 0, clears exit memory
5. But position is already closed, so no exit order needed
6. Next bar, if price > upper band again, signal = -1, new position opens

## The Fix Options

1. **Fix the strategy**: Make Bollinger bands "sticky" - maintain signal until opposite condition
2. **Fix exit memory logic**: Don't clear memory on FLAT signals after risk exits
3. **Fix sparse storage**: Ensure signals are forward-filled properly

Option 2 is the quickest fix - exit memory should only clear when signal goes to opposite direction, not FLAT.