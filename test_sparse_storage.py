"""Test temporal sparse storage directly."""

from src.core.events.storage.temporal_sparse_storage import TemporalSparseStorage
import logging

logging.basicConfig(level=logging.INFO)

# Create storage
storage = TemporalSparseStorage()

# Simulate signals that stay the same for multiple bars
print("Simulating 50 bars of signals...")

# Bars 0-19: No signal (warming up)
for i in range(20):
    # No signals during warmup
    storage._bar_index += 1

# Bars 20-30: Long signal
print("\n--- Bars 20-30: Long signals ---")
for i in range(20, 31):
    was_change = storage.process_signal(
        symbol="SPY",
        direction="long",
        strategy_id="ma_crossover",
        timestamp=f"2024-01-{i+1:02d}T10:00:00",
        price=520.0 + i * 0.1
    )
    print(f"Bar {i}: long signal - {'STORED' if was_change else 'skipped'}")

# Bars 31-40: Short signals  
print("\n--- Bars 31-40: Short signals ---")
for i in range(31, 41):
    was_change = storage.process_signal(
        symbol="SPY",
        direction="short",
        strategy_id="ma_crossover",
        timestamp=f"2024-01-{i+1:02d}T10:00:00",
        price=521.0 - i * 0.1
    )
    print(f"Bar {i}: short signal - {'STORED' if was_change else 'skipped'}")

# Bars 41-50: Long again
print("\n--- Bars 41-50: Long signals ---")
for i in range(41, 51):
    was_change = storage.process_signal(
        symbol="SPY",
        direction="long",
        strategy_id="ma_crossover",
        timestamp=f"2024-02-{i-40:02d}T10:00:00",
        price=519.0 + i * 0.05
    )
    print(f"Bar {i}: long signal - {'STORED' if was_change else 'skipped'}")

# Show results
print("\n=== SPARSE STORAGE RESULTS ===")
print(f"Total bars processed: {storage._bar_index}")
print(f"Signal changes stored: {len(storage._changes)}")
print(f"Compression ratio: {storage._bar_index / len(storage._changes):.1f}x")

print("\n=== SIGNAL RANGES ===")
ranges = storage.get_signal_ranges()
for r in ranges:
    print(f"Bars {r['start_bar']}-{r['end_bar']}: "
          f"{'LONG' if r['signal'] == 1 else 'SHORT'} "
          f"(duration: {r['end_bar'] - r['start_bar'] + 1} bars)")

# Save and show file size
filepath = storage.save(tag="test")
print(f"\nSaved to: {filepath}")

# Demonstrate reconstruction
print("\n=== RECONSTRUCTION TEST ===")
test_bars = [15, 25, 35, 45]
reconstructed = storage.reconstruct_signals(test_bars)
for bar_idx, signals in reconstructed.items():
    signal_str = "No signal" if not signals else ", ".join(
        f"{k}: {'LONG' if v == 1 else 'SHORT'}" for k, v in signals.items()
    )
    print(f"Bar {bar_idx}: {signal_str}")