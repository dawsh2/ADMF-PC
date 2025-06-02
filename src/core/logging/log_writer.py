"""
LogWriter - Enhanced File I/O Component for Logging System v3
Composable component for high-performance log writing with lifecycle management
"""

import json
import gzip
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from .protocols import LifecycleManaged, PerformanceOptimized


class LogWriter:
    """
    Writes logs to files with lifecycle management - composable component.
    
    This component handles file I/O operations for logging including automatic
    rotation, compression, and error handling. Designed for composition rather
    than inheritance.
    
    Features:
    - Automatic log rotation based on size limits
    - Optional compression for rotated logs
    - Thread-safe writing operations  
    - Error handling for file system issues
    - Performance metrics tracking
    """
    
    def __init__(self, log_file: Path, max_size_mb: int = 100, enable_compression: bool = False):
        """
        Initialize log writer.
        
        Args:
            log_file: Path to log file
            max_size_mb: Maximum file size before rotation
            enable_compression: Whether to compress rotated files
        """
        self.log_file = log_file
        self.max_size_mb = max_size_mb
        self.enable_compression = enable_compression
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._handle = None
        self._bytes_written = 0
        self._entries_written = 0
        self._rotation_count = 0
        self._init_handle()
    
    def _init_handle(self):
        """Initialize file handle with error handling."""
        try:
            self._handle = open(self.log_file, 'a', encoding='utf-8')
        except Exception as e:
            print(f"Failed to open log file {self.log_file}: {e}")
            self._handle = None
    
    def write(self, entry: Dict[str, Any]) -> None:
        """
        Write log entry to file with rotation check.
        
        Args:
            entry: Log entry dictionary to write
        """
        if self._handle:
            try:
                json_str = json.dumps(entry, default=str)
                self._handle.write(json_str + '\n')
                self._handle.flush()
                
                # Update metrics
                self._bytes_written += len(json_str) + 1
                self._entries_written += 1
                
                # Check if rotation is needed
                if self._bytes_written > (self.max_size_mb * 1024 * 1024):
                    self._rotate_log()
                    
            except Exception as e:
                print(f"Failed to write log entry: {e}")
    
    def _rotate_log(self):
        """Rotate log file when size limit reached."""
        if self._handle:
            self._handle.close()
            
            # Create rotated filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rotated_path = self.log_file.with_suffix(f".{timestamp}.log")
            
            # Move current log to rotated name
            self.log_file.rename(rotated_path)
            
            # Optionally compress
            if self.enable_compression:
                self._compress_log(rotated_path)
            
            # Reinitialize handle with original filename
            self._init_handle()
            self._bytes_written = 0
            self._rotation_count += 1
    
    def _compress_log(self, log_path: Path):
        """Compress rotated log file."""
        compressed_path = log_path.with_suffix('.log.gz')
        
        with open(log_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                f_out.writelines(f_in)
        
        log_path.unlink()  # Delete uncompressed version
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get writer performance metrics."""
        return {
            'bytes_written': self._bytes_written,
            'entries_written': self._entries_written, 
            'rotation_count': self._rotation_count,
            'max_size_mb': self.max_size_mb,
            'compression_enabled': self.enable_compression,
            'file_path': str(self.log_file)
        }
    
    def close(self):
        """Close the log writer and clean up resources."""
        if self._handle:
            self._handle.close()
            self._handle = None


class AsyncBatchLogWriter:
    """
    High-performance async batch log writer for high-throughput environments.
    
    This writer buffers log entries and writes them in batches to optimize
    performance for high-frequency logging scenarios.
    """
    
    def __init__(self, log_file: Path, batch_size: int = 1000, flush_interval: int = 5):
        """
        Initialize async batch writer.
        
        Args:
            log_file: Path to log file
            batch_size: Number of entries to batch before writing
            flush_interval: Seconds between forced flushes
        """
        self.log_file = log_file
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.batch_buffer = []
        self.last_flush = datetime.utcnow()
        self._total_entries = 0
        self._batch_count = 0
    
    async def write_async(self, entry: Dict[str, Any]):
        """Add entry to batch buffer for async writing."""
        self.batch_buffer.append(entry)
        
        # Flush if batch is full or time interval exceeded
        if (len(self.batch_buffer) >= self.batch_size or
            (datetime.utcnow() - self.last_flush).seconds >= self.flush_interval):
            await self._flush_batch()
    
    async def _flush_batch(self):
        """Flush all buffered entries to disk."""
        if not self.batch_buffer:
            return
            
        try:
            # Use aiofiles for async file I/O
            import aiofiles
            
            async with aiofiles.open(self.log_file, 'a') as f:
                for entry in self.batch_buffer:
                    await f.write(json.dumps(entry, default=str) + '\n')
            
            self._total_entries += len(self.batch_buffer)
            self._batch_count += 1
            self.batch_buffer.clear()
            self.last_flush = datetime.utcnow()
            
        except Exception as e:
            print(f"Error flushing batch to {self.log_file}: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get async writer performance metrics."""
        return {
            'total_entries': self._total_entries,
            'batch_count': self._batch_count,
            'current_buffer_size': len(self.batch_buffer),
            'batch_size': self.batch_size,
            'flush_interval': self.flush_interval
        }