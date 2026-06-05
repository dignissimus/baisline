import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode
import time

class PytorchFlopProfiler:
    def __init__(self, model: nn.Module):
        self.model = model
        self.total_flops: float = 0.0
        self._fcm = FlopCounterMode()

    def __enter__(self) -> "PytorchFlopProfiler":
        self._fcm.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.total_flops += float(self._fcm.get_total_flops())
        self._fcm.__exit__(exc_type, exc_val, exc_tb)
