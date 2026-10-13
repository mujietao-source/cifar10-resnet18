from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


HISTORY_PATH = Path("outputs") / "epoch25" / "history.csv"

OUTPUT_DIR = Path("outputs") / "final"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


history = pd.read_csv(HISTORY_PATH)

print(history.head())


plt.figure(figsize=(8, 6))

plt.plot(
    history["epoch"],
    history["train_accuracy"] * 100,
    label="Train Accuracy"
)

plt.plot(
    history["epoch"],
    history["val_accuracy"] * 100,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.title(
    "ResNet18 Training and Validation Accuracy"
)

plt.legend()
plt.grid()

plt.tight_layout()

accuracy_path = (
    OUTPUT_DIR /
    "accuracy_curve.png"
)

plt.savefig(
    accuracy_path,
    dpi=200
)

plt.show()


plt.figure(figsize=(8, 6))

plt.plot(
    history["epoch"],
    history["train_loss"],
    label="Train Loss"
)

plt.plot(
    history["epoch"],
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "ResNet18 Training and Validation Loss"
)

plt.legend()
plt.grid()

plt.tight_layout()

loss_path = (
    OUTPUT_DIR /
    "loss_curve.png"
)

plt.savefig(
    loss_path,
    dpi=200
)

plt.show()


print("\nSaved:")
print(accuracy_path)
print(loss_path)