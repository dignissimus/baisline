import torch
import torch.nn as nn
from baisline import ComputeBudget, FlopBudget, Profiler, PytorchFlopProfiler, PerfDRAMProfiler
import os

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

x_data = torch.randn(100, 10)
y_data = torch.randint(0, 2, (100,))
dataset = list(zip(x_data, y_data))

budget = ComputeBudget(flop_budget=FlopBudget(max_flops=1e7))
flop_prof = PytorchFlopProfiler()
profiler = Profiler(flop_profiler=flop_prof)

epochs = 2
for epoch in range(epochs):
    for i, (inputs, targets) in enumerate(dataset):
        with profiler:
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(0)) 
            loss = nn.functional.cross_entropy(outputs, targets.unsqueeze(0))
            loss.backward()
            optimizer.step()
            
        if profiler.total_flops >= budget.flop_budget.max_flops:
            print(f"Budget exhausted at epoch {epoch}, step {i}!")
            break
    
    if profiler.total_flops >= budget.flop_budget.max_flops:
        break

print("\n--- Manual API Profiling Results ---")
print(f"Total FLOPs: {profiler.total_flops}")
print(f"Total Time: {profiler.total_time}s")
