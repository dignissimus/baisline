import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from baisline import BudgetTrainer, ComputeBudget, FlopBudget, UnsupervisedModelTrainer, PytorchFlopProfiler

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
    def forward(self, x):
        return self.net(x)

model = MyModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

dataset = TensorDataset(torch.randn(1000, 10))
train_loader = DataLoader(dataset, batch_size=32)

budget = ComputeBudget(flop_budget=FlopBudget(max_flops=1e9)) 
strategy = UnsupervisedModelTrainer(loss_fn=lambda x: x.mean())
trainer = BudgetTrainer(
    model=model,
    train_dataloader=train_loader,
    optimizer=optimizer,
    budget=budget,
    model_trainer=strategy,
    flop_profiler=PytorchFlopProfiler()
)

if __name__ == "__main__":
    results = trainer.fit()
    print(f"Total FLOPs spent: {results.total_flops}")
    print(f"Peak memory usage: {results.peak_memory_mb} MB")
    results.export_report("testcases/out/simple_mlp_report.json")
