# src/train.py
from typing import Dict, Callable, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PreprocessFn = Callable[[torch.Tensor], torch.Tensor]


def train_classifier(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    preprocess_fn: Optional[PreprocessFn] = None,
) -> Dict:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for X, y in train_loader:
            if preprocess_fn is not None:
                X = preprocess_fn(X)

            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X.size(0)
            _, preds = outputs.max(1)
            total += y.size(0)
            correct += (preds == y).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate_classifier(
            model,
            val_loader,
            device,
            criterion=criterion,
            preprocess_fn=preprocess_fn,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

    return history


@torch.no_grad()
def evaluate_classifier(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    preprocess_fn: Optional[PreprocessFn] = None,
):
    model.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    total = 0
    correct = 0
    running_loss = 0.0

    for X, y in data_loader:
        if preprocess_fn is not None:
            X = preprocess_fn(X)

        X = X.to(device)
        y = y.to(device)

        outputs = model(X)
        loss = criterion(outputs, y)

        running_loss += loss.item() * X.size(0)
        _, preds = outputs.max(1)
        total += y.size(0)
        correct += (preds == y).sum().item()

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc
