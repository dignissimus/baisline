from dataclasses import dataclass, field
import json
from typing import Dict, List, Optional

@dataclass
class BenchmarkReport:
    model_name: str = "Unknown"
    val_loss: Optional[float] = None
    total_flops: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    total_bytes_read: Optional[int] = None
    total_bytes_written: Optional[int] = None
    total_time_seconds: Optional[float] = None
    history: List[Dict] = field(default_factory=list) # List of {"cum_flops": X, "metrics": {...}}

    @property
    def arithmetic_intensity(self) -> Optional[float]:
        if self.total_flops is None:
            return None
        
        read = self.total_bytes_read or 0
        write = self.total_bytes_written or 0
        total_bytes = read + write
        
        if total_bytes == 0:
            return None
        return self.total_flops / total_bytes

    @property
    def tflops_per_second(self) -> Optional[float]:
        if self.total_flops is None or self.total_time_seconds is None or self.total_time_seconds == 0:
            return None
        return (self.total_flops / 1e12) / self.total_time_seconds

    def to_dict(self) -> Dict:
        """Convert report to a dictionary."""
        return {
            "model_name": self.model_name,
            "val_loss": self.val_loss,
            "total_flops": self.total_flops,
            "peak_memory_mb": self.peak_memory_mb,
            "total_bytes_read": self.total_bytes_read,
            "total_bytes_written": self.total_bytes_written,
            "total_time_seconds": self.total_time_seconds,
            "arithmetic_intensity": self.arithmetic_intensity,
            "tflops_per_second": self.tflops_per_second,
            "history": self.history
        }

    def export_report(self, path: str):
        """Export the report to a JSON file."""
        data = self.to_dict()
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Report exported to {path}")
