from typing import Any, Callable, Optional, Protocol
import torch
import torch.nn as nn

# TODO: rename file to trainer

class ModelTrainer(Protocol):
    def step(self, model: nn.Module, batch: Any, optimizer: torch.optim.Optimizer) -> float:
        """Executes a single training step and returns the loss as a float."""
        ...

class HuggingFaceModelTrainer:
    def step(self, model: nn.Module, batch: Any, optimizer: torch.optim.Optimizer) -> float:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
        loss.backward()
        optimizer.step()
        return loss.item()

class SupervisedModelTrainer:
    def __init__(self, loss_fn: Callable):
        self.loss_fn = loss_fn

    def step(self, model: nn.Module, batch: Any, optimizer: torch.optim.Optimizer) -> float:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        return loss.item()

class UnsupervisedModelTrainer:
    def __init__(self, loss_fn: Callable):
        self.loss_fn = loss_fn

    def step(self, model: nn.Module, batch: Any, optimizer: torch.optim.Optimizer) -> float:
        optimizer.zero_grad()
        # TODO: check batch[0]
        inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
        outputs = model(inputs)
        loss = self.loss_fn(outputs)
        loss.backward()
        optimizer.step()
        return loss.item()
