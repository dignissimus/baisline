import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from baisline import ComputeBudget, FlopBudget, Profiler, PytorchFlopProfiler

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)
dataset = TensorDataset(torch.randn(100, 10))
train_loader = DataLoader(dataset, batch_size=10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

budget = ComputeBudget(flop_budget=FlopBudget(max_flops=1e7))
profiler = Profiler(flop_profiler=PytorchFlopProfiler())

def run_with_manual_check():
    print("Running with explicit budget check...")
    for epoch in range(10):
        for batch in train_loader:
            if profiler.total_flops >= budget.flop_budget.max_flops:
                print("Budget exhausted!")
                return

            with profiler:
                optimizer.zero_grad()
                loss = model(batch[0]).mean()
                loss.backward()
                optimizer.step()

if __name__ == "__main__":
    run_with_manual_check()
    print(f"\nFinal FLOPs: {profiler.total_flops}")
    print(f"Total Time: {profiler.total_time}s")
