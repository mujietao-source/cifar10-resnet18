from pathlib import Path
import csv

import torch
import torch.nn as nn
from torch.optim import Adam,AdamW

from data import get_dataloaders
from model import ResNet18


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def evaluate(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)

            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def train(
    epochs=15,
    batch_size=128,
    learning_rate=0.001,
    experiment_name="baseline",
    optimizer_name="adam",
    weight_decay=0.0
):
    device = get_device()
    print("Device:", device)

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size
    )

    model = ResNet18(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "adam":
        optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    elif optimizer_name == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}"
        )

    output_dir = Path("outputs") / experiment_name
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    best_val_accuracy = 0.0
    best_epoch = 0

    history = []

    for epoch in range(1, epochs + 1):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)

            loss = criterion(logits, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * images.size(0)

            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = correct / total

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        print(
            f"Epoch [{epoch:02d}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_accuracy * 100:.2f}%"
        )

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                output_dir / "best_model.pth"
            )

            print(
                f"  → Best model saved: "
                f"{best_val_accuracy * 100:.2f}%"
            )

    history_path = output_dir / "history.csv"

    with open(history_path, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy"
            ]
        )

        writer.writeheader()
        writer.writerows(history)

    print("\nTraining finished.")

    print(
        f"Best Validation Accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        f"Best Epoch: {best_epoch}"
    )

    print(
        f"Model saved to: "
        f"{output_dir / 'best_model.pth'}"
    )

    print(
        f"History saved to: "
        f"{history_path}"
    )


if __name__ == "__main__":
    train(
        epochs=25,
        batch_size=128,
        learning_rate=0.001,
        experiment_name="adamw_wd1e-4",
        optimizer_name="adamw",
        weight_decay=1e-4
    )