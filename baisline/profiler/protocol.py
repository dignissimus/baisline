from typing import Protocol, Optional, Any

class FlopProfiler(Protocol):
    total_flops: float
    
    def __enter__(self) -> "FlopProfiler": ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...

class MemoryProfiler(Protocol):
    peak_memory_mb: float
    total_bytes_read: int
    total_bytes_written: int
    
    def __enter__(self) -> "MemoryProfiler": ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...
