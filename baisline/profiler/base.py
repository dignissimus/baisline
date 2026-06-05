import time
from dataclasses import dataclass, field
from typing import Optional
from .protocol import FlopProfiler, MemoryProfiler

@dataclass
class Profiler:
    """
    Container for FLOP and Memory profilers.
    Orchestrates their lifecycles and tracks total wall-clock time.
    """
    flop_profiler: Optional[FlopProfiler] = None
    memory_profiler: Optional[MemoryProfiler] = None
    total_time: float = 0.0
    _start_time: float = field(init=False, default=0.0)

    def __enter__(self) -> "Profiler":
        self._start_time = time.perf_counter()
        if self.flop_profiler:
            self.flop_profiler.__enter__()
        if self.memory_profiler:
            self.memory_profiler.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.memory_profiler:
            self.memory_profiler.__exit__(exc_type, exc_val, exc_tb)
        if self.flop_profiler:
            self.flop_profiler.__exit__(exc_type, exc_val, exc_tb)
        self.total_time += time.perf_counter() - self._start_time
            
    @property
    def total_flops(self) -> float:
        return self.flop_profiler.total_flops if self.flop_profiler else 0.0

    @property
    def peak_memory_mb(self) -> float:
        return self.memory_profiler.peak_memory_mb if self.memory_profiler else 0.0

    @property
    def total_bytes_read(self) -> int:
        return self.memory_profiler.total_bytes_read if self.memory_profiler else 0

    @property
    def total_bytes_written(self) -> int:
        return self.memory_profiler.total_bytes_written if self.memory_profiler else 0
