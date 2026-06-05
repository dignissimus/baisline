import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from baisline import BudgetTrainer, ComputeBudget, SupervisedModelTrainer, FlopBudget, PytorchFlopProfiler
from baisline.visualize import plot_flops_vs_metric, plot_steps_vs_metric
import os

class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class LargeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)

def compute_metrics(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            break 
    model.train()
    return {"accuracy": correct / total}

def train_model(model_class, name, budget_gflops):
    print(f"\nTraining {name}...")
    model = model_class()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    budget = ComputeBudget(flop_budget=FlopBudget.from_gflops(budget_gflops))
    
    trainer = BudgetTrainer(
        model=model,
        train_dataloader=train_loader,
        optimizer=optimizer,
        budget=budget,
        model_trainer=SupervisedModelTrainer(loss_fn=F.cross_entropy),
        flop_profiler=PytorchFlopProfiler(model),
        model_name=name
    )
    
    return trainer.fit(epochs=5, eval_interval=1, eval_fn=compute_metrics, history_interval=1)

if __name__ == "__main__":
    if not os.path.exists('./data'):
        os.makedirs('./data')

    budget_gflops = 25.0 
    
    report_small = train_model(SmallMLP, "Small MLP", budget_gflops)
    report_large = train_model(LargeMLP, "Large MLP", budget_gflops)
    
    plot_flops_vs_metric(
        [report_small, report_large], 
        metric_name="loss", 
        save_path="testcases/out/flops_vs_loss_mnist_log.png", 
        use_log_scale=True
    )
    plot_flops_vs_metric(
        [report_small, report_large], 
        metric_name="loss", 
        save_path="testcases/out/flops_vs_loss_mnist_linear.png", 
        use_log_scale=False
    )
    
    plot_flops_vs_metric(
        [report_small, report_large], 
        metric_name="accuracy", 
        save_path="testcases/out/flops_vs_accuracy_mnist_log.png", 
        use_log_scale=True
    )
    plot_flops_vs_metric(
        [report_small, report_large], 
        metric_name="accuracy", 
        save_path="testcases/out/flops_vs_accuracy_mnist_linear.png", 
        use_log_scale=False
    )

    plot_steps_vs_metric(
        [report_small, report_large],
        metric_name="loss",
        save_path="testcases/out/steps_vs_loss_mnist.png"
    )
    plot_steps_vs_metric(
        [report_small, report_large],
        metric_name="accuracy",
        save_path="testcases/out/steps_vs_accuracy_mnist.png"
    )

    print("\nComparison completed. Charts generated in 'testcases/out/':")
    print(" - FLOPs vs Loss: flops_vs_loss_mnist_log.png, flops_vs_loss_mnist_linear.png")
    print(" - FLOPs vs Accuracy: flops_vs_accuracy_mnist_log.png, flops_vs_accuracy_mnist_linear.png")
    print(" - Steps vs Loss/Accuracy: steps_vs_loss_mnist.png, steps_vs_accuracy_mnist.png")
    print("Small MLP FLOPs:", report_small.total_flops)
    print("Large MLP FLOPs:", report_large.total_flops)
