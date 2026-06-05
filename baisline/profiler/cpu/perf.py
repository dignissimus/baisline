import os
from typing import Optional

class PerfDRAMProfiler:
    """
    Stub for CPU memory profiler using Linux perf.
    """
    def __init__(self, events: str = "LLC-load-misses,LLC-store-misses", process_specific: bool = True):
        self.peak_memory_mb: float = 0.0
        self.total_bytes_read: int = 0
        self.total_bytes_written: int = 0

    def __enter__(self) -> "PerfDRAMProfiler":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
