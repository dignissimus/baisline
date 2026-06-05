import torch
import torch.nn as nn
from typing import Optional, List, Set
from torch.profiler import profile, ProfilerActivity, _ExperimentalConfig

class CuptiProfiler:
    """
    Standalone GPU memory profiler using CUPTI counters.
    """
    def __init__(self, additional_metrics: Optional[List[str]] = None):
        assert torch.cuda.is_available(), "CUDA must be available to use CuptiProfiler"
        
        self.metrics: Set[str] = {"dram__bytes_read.sum", "dram__bytes_write.sum"}
        if additional_metrics:
            self.metrics.update(additional_metrics)
            
        self._experimental_config = _ExperimentalConfig(
            profiler_metrics=list(self.metrics),
            profiler_measure_per_kernel=True
        )
        
        self._torch_prof: Optional[profile] = None
        self.peak_memory_mb: float = 0.0
        self.total_bytes_read: int = 0
        self.total_bytes_written: int = 0

    def __enter__(self) -> "CuptiProfiler":
        self._torch_prof = profile(
            activities=[ProfilerActivity.CUDA],
            experimental_config=self._experimental_config
        )
        self._torch_prof.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._torch_prof:
            self._torch_prof.__exit__(exc_type, exc_val, exc_tb)
            for event in self._torch_prof.events():
                if event.device_type == torch.profiler.DeviceType.CUDA:
                    metrics = event.extra_fields.cupti_metrics
                    self.total_bytes_read += int(metrics.get("dram__bytes_read.sum", 0))
                    self.total_bytes_written += int(metrics.get("dram__bytes_write.sum", 0))
            self._torch_prof = None
        
        current_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
        if current_memory > self.peak_memory_mb:
            self.peak_memory_mb = current_memory
