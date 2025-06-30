#!/usr/bin/env python3
"""Check the structure of Keltner trace files"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to a sample trace file
trace_file = Path("config/keltner/results/20250622_180858/traces/keltner_bands/SPY_5m_compiled_strategy_0.parquet")

logger.info(f"Loading trace file: {trace_file}")
df = pd.read_parquet(trace_file)

logger.info(f"\nDataFrame shape: {df.shape}")
logger.info(f"\nColumn names: {list(df.columns)}")
logger.info(f"\nData types:\n{df.dtypes}")
logger.info(f"\nFirst 10 rows:")
print(df.head(10))

logger.info(f"\nLast 10 rows:")
print(df.tail(10))

# Check for non-zero signals
if 'signal' in df.columns:
    non_zero_signals = df[df['signal'] != 0]
    logger.info(f"\nNon-zero signals: {len(non_zero_signals)} out of {len(df)}")
    if len(non_zero_signals) > 0:
        logger.info("\nSample non-zero signals:")
        print(non_zero_signals.head(10))
        
        # Signal value distribution
        logger.info("\nSignal value counts:")
        print(df['signal'].value_counts().sort_index())

# Check unique values in each column
logger.info("\nUnique values per column:")
for col in df.columns:
    n_unique = df[col].nunique()
    if n_unique < 20:
        logger.info(f"{col}: {n_unique} unique values - {df[col].unique()[:10]}")
    else:
        logger.info(f"{col}: {n_unique} unique values")