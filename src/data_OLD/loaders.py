"""
Data loaders for ADMF-PC.

This module provides various data loading implementations for different
data sources and formats.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import logging

from .models import Bar, Timeframe
from ..core.logging import StructuredLogger


class DataLoader(ABC):
    """Abstract base class for data loaders."""
    
    @abstractmethod
    def load(self, symbol: str, **kwargs) -> pd.DataFrame:
        """
        Load data for a symbol.
        
        Args:
            symbol: Symbol to load data for
            **kwargs: Additional loader-specific parameters
            
        Returns:
            DataFrame with OHLCV data
        """
        pass
    
    @abstractmethod
    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validate loaded data.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid
        """
        pass


class CSVLoader(DataLoader):
    """
    Loads market data from CSV files.
    
    Supports various CSV formats and performs data validation
    and normalization.
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        date_column: str = "Date",
        date_format: Optional[str] = None
    ):
        """
        Initialize CSV loader.
        
        Args:
            data_dir: Directory containing CSV files
            date_column: Name of date column
            date_format: Date parsing format (None for auto)
        """
        self.data_dir = Path(data_dir)
        self.date_column = date_column
        self.date_format = date_format
        self._logger = StructuredLogger("CSVLoader")
        
        # Column mappings for normalization
        self.column_mappings = {
            "open": ["open", "Open", "OPEN", "o", "O"],
            "high": ["high", "High", "HIGH", "h", "H"],
            "low": ["low", "Low", "LOW", "l", "L"],
            "close": ["close", "Close", "CLOSE", "c", "C"],
            "volume": ["volume", "Volume", "VOLUME", "v", "V"]
        }
    
    def load(self, symbol: str, **kwargs) -> pd.DataFrame:
        """Load data from CSV file."""
        # Find CSV file
        csv_path = self._find_csv_file(symbol)
        if not csv_path:
            raise FileNotFoundError(f"No CSV file found for {symbol}")
        
        self._logger.info(f"Loading data from {csv_path}")
        
        # Load CSV with various options
        df = self._load_csv_with_options(csv_path)
        
        # Normalize columns
        df = self._normalize_columns(df)
        
        # Parse dates
        df = self._parse_dates(df)
        
        # Validate data
        if not self.validate(df):
            raise ValueError(f"Invalid data in {csv_path}")
        
        # Sort by date
        df.sort_index(inplace=True)
        
        # Handle missing data
        df = self._handle_missing_data(df)
        
        self._logger.info(
            f"Loaded {len(df)} rows for {symbol}",
            start=df.index[0],
            end=df.index[-1]
        )
        
        return df
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate OHLCV data."""
        try:
            # Check required columns
            required = ["open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required):
                return False
            
            # Check data types
            for col in ["open", "high", "low", "close", "volume"]:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    return False
            
            # Check OHLC relationships
            invalid_bars = (
                (df["high"] < df["low"]) |
                (df["high"] < df["open"]) |
                (df["high"] < df["close"]) |
                (df["low"] > df["open"]) |
                (df["low"] > df["close"]) |
                (df["volume"] < 0)
            )
            
            if invalid_bars.any():
                self._logger.warning(
                    f"Found {invalid_bars.sum()} invalid bars"
                )
                return False
            
            # Check for duplicate dates
            if df.index.duplicated().any():
                self._logger.warning("Found duplicate dates")
                return False
            
            return True
            
        except Exception as e:
            self._logger.error(f"Validation error: {e}")
            return False
    
    def _find_csv_file(self, symbol: str) -> Optional[Path]:
        """Find CSV file for symbol."""
        # Try various naming conventions
        patterns = [
            f"{symbol}.csv",
            f"{symbol.lower()}.csv",
            f"{symbol.upper()}.csv",
            f"{symbol}.CSV",
            f"{symbol}_daily.csv",
            f"{symbol}_1d.csv"
        ]
        
        for pattern in patterns:
            path = self.data_dir / pattern
            if path.exists():
                return path
        
        return None
    
    def _load_csv_with_options(self, path: Path) -> pd.DataFrame:
        """Load CSV with various parsing options."""
        # Try different delimiters
        for delimiter in [",", ";", "\t", "|"]:
            try:
                df = pd.read_csv(
                    path,
                    delimiter=delimiter,
                    thousands=",",
                    decimal=".",
                    engine="python"
                )
                
                if len(df.columns) >= 5:  # Minimum for OHLCV
                    return df
                    
            except Exception:
                continue
        
        # Default load
        return pd.read_csv(path)
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to standard format."""
        # Create mapping from actual to standard names
        column_map = {}
        
        for standard, variants in self.column_mappings.items():
            for col in df.columns:
                if col in variants or col.lower() in [v.lower() for v in variants]:
                    column_map[col] = standard
                    break
        
        # Rename columns
        df = df.rename(columns=column_map)
        
        # Keep only required columns plus date
        keep_cols = ["open", "high", "low", "close", "volume"]
        
        # Find date column
        date_cols = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
        if date_cols:
            keep_cols.append(date_cols[0])
            if date_cols[0] != self.date_column:
                df = df.rename(columns={date_cols[0]: self.date_column})
        
        # Filter columns
        available_cols = [col for col in keep_cols if col in df.columns]
        df = df[available_cols]
        
        return df
    
    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse date column and set as index."""
        if self.date_column in df.columns:
            # Parse dates
            if self.date_format:
                df[self.date_column] = pd.to_datetime(
                    df[self.date_column],
                    format=self.date_format
                )
            else:
                df[self.date_column] = pd.to_datetime(
                    df[self.date_column]
                )
            
            # Set as index
            df.set_index(self.date_column, inplace=True)
        else:
            # Try to infer from index
            if not isinstance(df.index, pd.DatetimeIndex):
                # Assume index is dates
                df.index = pd.to_datetime(df.index)
        
        return df
    
    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data points."""
        # Forward fill for price data (more appropriate than interpolation)
        price_cols = ["open", "high", "low", "close"]
        df[price_cols] = df[price_cols].ffill()
        
        # Zero fill for volume
        df["volume"] = df["volume"].fillna(0)
        
        # Drop any remaining rows with NaN
        initial_len = len(df)
        df = df.dropna()
        
        if len(df) < initial_len:
            self._logger.warning(
                f"Dropped {initial_len - len(df)} rows with missing data"
            )
        
        return df


class MemoryOptimizedCSVLoader(CSVLoader):
    """
    Memory-efficient CSV loader for large datasets.
    
    Uses chunked reading and data type optimization.
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        chunk_size: int = 10000,
        optimize_types: bool = True
    ):
        """
        Initialize memory-optimized loader.
        
        Args:
            data_dir: Directory containing CSV files
            chunk_size: Rows to read per chunk
            optimize_types: Whether to optimize data types
        """
        super().__init__(data_dir)
        self.chunk_size = chunk_size
        self.optimize_types = optimize_types
    
    def load(self, symbol: str, **kwargs) -> pd.DataFrame:
        """Load data with memory optimization."""
        csv_path = self._find_csv_file(symbol)
        if not csv_path:
            raise FileNotFoundError(f"No CSV file found for {symbol}")
        
        # Read in chunks
        chunks = []
        
        for chunk in pd.read_csv(
            csv_path,
            chunksize=self.chunk_size,
            parse_dates=[self.date_column],
            index_col=self.date_column
        ):
            # Normalize columns
            chunk = self._normalize_columns(chunk)
            
            # Optimize types if requested
            if self.optimize_types:
                chunk = self._optimize_data_types(chunk)
            
            chunks.append(chunk)
        
        # Combine chunks
        df = pd.concat(chunks, ignore_index=False)
        
        # Validate
        if not self.validate(df):
            raise ValueError(f"Invalid data in {csv_path}")
        
        # Sort by date
        df.sort_index(inplace=True)
        
        return df
    
    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency."""
        # Price data - use float32 instead of float64
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)
        
        # Volume - use appropriate integer type
        if "volume" in df.columns:
            max_vol = df["volume"].max()
            if max_vol < np.iinfo(np.uint32).max:
                df["volume"] = df["volume"].astype(np.uint32)
            else:
                df["volume"] = df["volume"].astype(np.uint64)
        
        return df


class MultiFileLoader(DataLoader):
    """
    Loads data from multiple files and concatenates them.
    
    Useful for loading data split across multiple time periods.
    """
    
    def __init__(self, base_loader: DataLoader):
        """
        Initialize multi-file loader.
        
        Args:
            base_loader: Underlying loader for individual files
        """
        self.base_loader = base_loader
        self._logger = StructuredLogger("MultiFileLoader")
    
    def load(self, symbol: str, file_pattern: str = "{symbol}_{year}.csv") -> pd.DataFrame:
        """
        Load data from multiple files matching pattern.
        
        Args:
            symbol: Symbol to load
            file_pattern: Pattern for file names
            
        Returns:
            Combined DataFrame
        """
        dfs = []
        
        # Try loading files for recent years
        import datetime
        current_year = datetime.datetime.now().year
        
        for year in range(current_year - 10, current_year + 1):
            filename = file_pattern.format(symbol=symbol, year=year)
            
            try:
                df = self.base_loader.load(filename)
                dfs.append(df)
                self._logger.info(f"Loaded {len(df)} rows from {filename}")
            except FileNotFoundError:
                # File doesn't exist for this year
                continue
            except Exception as e:
                self._logger.warning(f"Failed to load {filename}: {e}")
        
        if not dfs:
            raise FileNotFoundError(f"No data files found for {symbol}")
        
        # Combine all dataframes
        combined = pd.concat(dfs)
        
        # Remove duplicates (in case of overlapping data)
        combined = combined[~combined.index.duplicated(keep='last')]
        
        # Sort by date
        combined.sort_index(inplace=True)
        
        self._logger.info(
            f"Loaded total of {len(combined)} rows for {symbol}",
            files=len(dfs),
            start=combined.index[0],
            end=combined.index[-1]
        )
        
        return combined
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate combined data."""
        return self.base_loader.validate(df)