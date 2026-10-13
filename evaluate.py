from pathlib import Path

import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from data import get_dataloaders
from model import ResNet18


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


MODEL_PATH = Path("outputs") / "epoch25" / "best_model.pth"

OUTPUT_DIR = Path("outputs") / "final"


def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def evaluate():

    device = get_device()

    print("Device:", device)
    print("Model:", MODEL_PATH)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    _, _, test_loader = get_dataloaders(
        batch_size=128
    )

    model = ResNet18(
        num_classes=10
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    model.load_state_dict(
        state_dict
    )

    model = model.to(device)

    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            predictions = logits.argmax(
                dim=1
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    print("\n========== Final Test Result ==========")

    print(
        f"Test Accuracy:   "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Macro Precision: "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Macro Recall:    "
        f"{recall * 100:.2f}%"
    )

    print(
        f"Macro F1:        "
        f"{f1 * 100:.2f}%"
    )


    print(
        "\n========== Classification Report =========="
    )

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    print(report)


    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    print(
        "\n========== Confusion Matrix =========="
    )

    print(cm)

    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )

    disp.plot(
        ax=ax,
        xticks_rotation=45,
        cmap="Blues",
        colorbar=False
    )

    plt.title(
        "CIFAR-10 Final Model Confusion Matrix"
    )

    plt.tight_layout()

    confusion_matrix_path = (
        OUTPUT_DIR /
        "confusion_matrix.png"
    )

    plt.savefig(
        confusion_matrix_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()

    print(
        "\nConfusion matrix saved to:"
    )

    print(
        confusion_matrix_path
    )


if __name__ == "__main__":
    evaluate()