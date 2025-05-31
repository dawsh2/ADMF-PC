"""
Data loaders using Protocol+Composition - NO INHERITANCE!

Simple classes that implement DataLoader protocol through duck typing.
Enhanced through capabilities, not inheritance.
"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import pandas as pd
import numpy as np

from .models import Bar, Timeframe, ValidationResult


class SimpleCSVLoader:
    """
    Simple CSV loader - NO INHERITANCE!
    Implements DataLoader protocol through duck typing.
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        date_column: str = "Date",
        date_format: Optional[str] = None
    ):
        self.data_dir = Path(data_dir)
        self.date_column = date_column
        self.date_format = date_format
        
        # Column mappings for normalization
        self.column_mappings = {
            "open": ["open", "Open", "OPEN", "o", "O"],
            "high": ["high", "High", "HIGH", "h", "H"],
            "low": ["low", "Low", "LOW", "l", "L"],
            "close": ["close", "Close", "CLOSE", "c", "C"],
            "volume": ["volume", "Volume", "VOLUME", "v", "V"]
        }
    
    # Implements DataLoader protocol
    def load(self, symbol: str, **kwargs) -> pd.DataFrame:
        """Load data from CSV file - implements protocol method."""
        csv_path = self._find_csv_file(symbol)
        if not csv_path:
            raise FileNotFoundError(f"No CSV file found for {symbol}")
        
        # Load CSV
        df = self._load_csv_with_options(csv_path)
        
        # Normalize columns
        df = self._normalize_columns(df)
        
        # Parse dates
        df = self._parse_dates(df)
        
        # Validate
        if not self.validate(df):
            raise ValueError(f"Invalid data in {csv_path}")
        
        # Sort by date
        df.sort_index(inplace=True)
        
        # Handle missing data
        df = self._handle_missing_data(df)
        
        return df
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate OHLCV data - implements protocol method."""
        try:
            # Check required columns
            required = ["open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required):
                return False
            
            # Check data types
            for col in required:
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
            
            return not invalid_bars.any()
            
        except Exception:
            return False
    
    # Private helper methods - no inheritance complexity
    def _find_csv_file(self, symbol: str) -> Optional[Path]:
        """Find CSV file for symbol."""
        patterns = [
            f"{symbol}.csv",
            f"{symbol.lower()}.csv",
            f"{symbol.upper()}.csv",
            f"{symbol}_daily.csv"
        ]
        
        for pattern in patterns:
            path = self.data_dir / pattern
            if path.exists():
                return path
        
        return None
    
    def _load_csv_with_options(self, path: Path) -> pd.DataFrame:
        """Load CSV with various parsing options."""
        for delimiter in [",", ";", "\t", "|"]:
            try:
                df = pd.read_csv(path, delimiter=delimiter, engine="python")
                if len(df.columns) >= 5:
                    return df
            except Exception:
                continue
        return pd.read_csv(path)
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names."""
        column_map = {}
        
        for standard, variants in self.column_mappings.items():
            for col in df.columns:
                if col in variants or col.lower() in [v.lower() for v in variants]:
                    column_map[col] = standard
                    break
        
        df = df.rename(columns=column_map)
        
        # Keep only required columns plus date
        keep_cols = ["open", "high", "low", "close", "volume"]
        
        # Find date column
        date_cols = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
        if date_cols and date_cols[0] != self.date_column:
            df = df.rename(columns={date_cols[0]: self.date_column})
            keep_cols.append(self.date_column)
        
        available_cols = [col for col in keep_cols if col in df.columns]
        return df[available_cols]
    
    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse date column and set as index."""
        if self.date_column in df.columns:
            if self.date_format:
                df[self.date_column] = pd.to_datetime(df[self.date_column], format=self.date_format)
            else:
                df[self.date_column] = pd.to_datetime(df[self.date_column], utc=True)
            df.set_index(self.date_column, inplace=True)
        else:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)
        
        return df
    
    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data points."""
        # Forward fill for price data
        price_cols = ["open", "high", "low", "close"]
        df[price_cols] = df[price_cols].ffill()
        
        # Zero fill for volume
        df["volume"] = df["volume"].fillna(0)
        
        # Drop remaining NaN rows
        df = df.dropna()
        
        return df


class MemoryEfficientCSVLoader:
    """
    Memory-efficient CSV loader - NO INHERITANCE!
    Simple class enhanced through capabilities.
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        chunk_size: int = 10000,
        optimize_types: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.optimize_types = optimize_types
        self.base_loader = SimpleCSVLoader(data_dir)
    
    def load(self, symbol: str, **kwargs) -> pd.DataFrame:
        """Load data with memory optimization."""
        csv_path = self.base_loader._find_csv_file(symbol)
        if not csv_path:
            raise FileNotFoundError(f"No CSV file found for {symbol}")
        
        chunks = []
        
        for chunk in pd.read_csv(csv_path, chunksize=self.chunk_size):
            # Normalize columns
            chunk = self.base_loader._normalize_columns(chunk)
            
            # Optimize types
            if self.optimize_types:
                chunk = self._optimize_data_types(chunk)
            
            chunks.append(chunk)
        
        # Combine chunks
        df = pd.concat(chunks, ignore_index=False)
        
        # Validate
        if not self.base_loader.validate(df):
            raise ValueError(f"Invalid data in {csv_path}")
        
        df.sort_index(inplace=True)
        return df
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Delegate to base loader."""
        return self.base_loader.validate(df)
    
    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency."""
        # Price data - use float32
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


class MultiFileLoader:
    """
    Multi-file loader - NO INHERITANCE!
    Composes other loaders instead of inheriting.
    """
    
    def __init__(self, base_loader_class=SimpleCSVLoader, **loader_kwargs):
        self.base_loader = base_loader_class(**loader_kwargs)
    
    def load(self, symbol: str, file_pattern: str = "{symbol}_{year}.csv") -> pd.DataFrame:
        """Load data from multiple files matching pattern."""
        dfs = []
        
        import datetime
        current_year = datetime.datetime.now().year
        
        for year in range(current_year - 10, current_year + 1):
            filename = file_pattern.format(symbol=symbol, year=year)
            
            try:
                # Try to load with modified symbol name
                df = self.base_loader.load(filename.replace('.csv', ''))
                dfs.append(df)
            except FileNotFoundError:
                continue
            except Exception:
                continue
        
        if not dfs:
            raise FileNotFoundError(f"No data files found for {symbol}")
        
        # Combine all dataframes
        combined = pd.concat(dfs)
        combined = combined[~combined.index.duplicated(keep='last')]
        combined.sort_index(inplace=True)
        
        return combined
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Delegate to base loader."""
        return self.base_loader.validate(df)


class DatabaseLoader:
    """
    Example database loader - NO INHERITANCE!
    Shows how to implement DataLoader protocol for different data sources.
    """
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        # In real implementation, would set up database connection
    
    def load(self, symbol: str, **kwargs) -> pd.DataFrame:
        """Load data from database - implements DataLoader protocol."""
        # Placeholder implementation
        query = f"SELECT * FROM market_data WHERE symbol = '{symbol}'"
        # df = pd.read_sql(query, self.connection)
        # For now, just raise
        raise NotImplementedError("Database loader not yet implemented")
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate database data."""
        # Same validation as CSV
        required = ["open", "high", "low", "close", "volume"]
        return all(col in df.columns for col in required)


# Data loader factory - creates loaders without inheritance
def create_data_loader(loader_type: str, **config) -> Any:
    """
    Factory function to create data loaders.
    No inheritance needed - just returns appropriate class.
    """
    loaders = {
        'csv': SimpleCSVLoader,
        'memory_csv': MemoryEfficientCSVLoader,
        'multi_file': MultiFileLoader,
        'database': DatabaseLoader
    }
    
    loader_class = loaders.get(loader_type)
    if not loader_class:
        raise ValueError(f"Unknown loader type: {loader_type}")
    
    return loader_class(**config)
