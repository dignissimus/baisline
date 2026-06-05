from dataclasses import dataclass
from typing import Optional

@dataclass
class FlopBudget:
    max_flops: Optional[float] 

    @classmethod
    def from_pflops(cls, pflops):
        return FlopBudget(pflops * 10 ** 12)

    @classmethod
    def from_tflops(cls, tflops):
        return FlopBudget(tflops * 10 ** 12)

    @classmethod
    def from_gflops(cls, gflops):
        return FlopBudget(gflops * 10 ** 9)

@dataclass
class MemoryBudget:
    ...

@dataclass
class ComputeBudget:
    flop_budget: FlopBudget = None
    memory_budget: MemoryBudget = None
