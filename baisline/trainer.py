import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from typing import Optional, Callable, Dict, Any
from tqdm import tqdm
from .budget import ComputeBudget
from .profiler import Profiler, FlopProfiler, MemoryProfiler
from .reports import BenchmarkReport
from .exceptions import BudgetExceededException

from .strategy import ModelTrainer, HuggingFaceModelTrainer

class BudgetTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        budget: ComputeBudget,
        model_trainer: ModelTrainer,
        val_dataloader: Optional[DataLoader] = None,
        flop_profiler: Optional[FlopProfiler] = None,
        memory_profiler: Optional[MemoryProfiler] = None,
        model_name: str = "MyModel"
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.budget = budget
        self.model_trainer = model_trainer
        self.val_dataloader = val_dataloader
        self.model_name = model_name
        
        self.profiler = Profiler(flop_profiler=flop_profiler, memory_profiler=memory_profiler)
        self.history = []

    def train_step(self, batch, pbar, step_idx, history_interval, eval_interval, eval_fn):
        with self.profiler:
            loss_val = self.model_trainer.step(self.model, batch, self.optimizer)
        
        if self.budget.flop_budget and self.budget.flop_budget.max_flops:
            if self.profiler.total_flops >= self.budget.flop_budget.max_flops:
                raise BudgetExceededException("FLOP budget exceeded.")

        # TODO: Use unit of appropriate magnitude for flops
        pbar.set_postfix({
            "loss": f"{loss_val:.4f}", 
            "TFLOPS": f"{self.profiler.total_flops/1e12:.4f}"
        })

        if history_interval and step_idx % history_interval == 0:
            metrics = {"loss": loss_val}
            if eval_fn and eval_interval and step_idx % eval_interval == 0:
                metrics.update(eval_fn(self.model))
            
            self.history.append({
                "step": step_idx,
                "cum_flops": self.profiler.total_flops,
                "metrics": metrics
            })

    def fit(
        self,
        epochs: int = 1,
        eval_interval: int = 1,
        eval_fn: Optional[Callable[[nn.Module], Dict[str, Any]]] = None,
        history_interval: Optional[int] = None
    ) -> BenchmarkReport:
        self.model.train()
        step_idx = 0
        if history_interval is None:
            history_interval = eval_interval
            
        try:
            for epoch in range(epochs):
                pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch} {self.model_name}")
                for batch in pbar:
                    self.train_step(batch, pbar, step_idx, history_interval, eval_interval, eval_fn)
                    step_idx += 1
        except BudgetExceededException:
            print(f"\nCompute budget exhausted for {self.model_name}.")

        report = BenchmarkReport(
            model_name=self.model_name,
            total_flops=self.profiler.total_flops,
            peak_memory_mb=self.profiler.peak_memory_mb,
            total_bytes_read=self.profiler.total_bytes_read,
            total_bytes_written=self.profiler.total_bytes_written,
            total_time_seconds=self.profiler.total_time,
            history=self.history
        )
        
        return report
