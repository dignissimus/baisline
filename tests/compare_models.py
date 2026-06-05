import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from baisline import BudgetTrainer, ComputeBudget, FlopBudget, SupervisedModelTrainer, PytorchFlopProfiler
from baisline.visualize import plot_flops_vs_metric

class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.net(x)

class LargeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.net(x)

x_train = torch.randn(1000, 10)
y_train = torch.randint(0, 2, (1000,))
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32)

x_val = torch.randn(200, 10)
y_val = torch.randint(0, 2, (200,))

def compute_metrics(model: nn.Module):
    model.eval()
    with torch.no_grad():
        outputs = model(x_val)
        loss = nn.functional.cross_entropy(outputs, y_val).item()
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == y_val).float().mean().item()
    model.train()
    return {"val_loss": loss, "accuracy": acc}

reports = []
models = {"Small MLP": SmallMLP(), "Large MLP": LargeMLP()}

for name, model in models.items():
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    budget = ComputeBudget(flop_budget=FlopBudget(max_flops=5e7)) 
    
    trainer = BudgetTrainer(
        model=model,
        train_dataloader=train_loader,
        optimizer=optimizer,
        budget=budget,
        model_trainer=SupervisedModelTrainer(loss_fn=nn.functional.cross_entropy),
        flop_profiler=PytorchFlopProfiler(),
        model_name=name
    )
    
    report = trainer.fit(epochs=5, eval_interval=10, eval_fn=compute_metrics)
    reports.append(report)

plot_flops_vs_metric(reports, metric_name="val_loss", save_path="testcases/out/flops_vs_val_loss.png", use_log_scale=True)
plot_flops_vs_metric(reports, metric_name="accuracy", save_path="testcases/out/flops_vs_accuracy.png", use_log_scale=True)
