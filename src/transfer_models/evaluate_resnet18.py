from pathlib import Path
import sys

# Add src folder to Python path
SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


import pandas as pd
import matplotlib.pyplot as plt

import torch

from config import (
    DEVICE,
    MODEL_DIR,
    RESULT_DIR,
    FIGURE_DIR,
    CLASS_NAMES,
)

from dataset import get_dataloaders
from metrics import calculate_classification_metrics
from resnet18_model import get_resnet18


MODEL_NAME = "resnet18"


def predict(model, data_loader, device):
    """
    Run prediction on test set.
    """
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)

            outputs = model(images)
            probs = softmax(outputs)
            _, preds = torch.max(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    return all_labels, all_preds, all_probs


def save_confusion_matrix(cm, class_names, output_path):
    """
    Save confusion matrix figure.
    """
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("ResNet18 Confusion Matrix")
    plt.colorbar()

    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                horizontalalignment="center",
                verticalalignment="center"
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE)

    print("=" * 50)
    print("Evaluating ResNet18 on Test Set")
    print("=" * 50)
    print(f"Device: {device}")

    _, _, test_loader = get_dataloaders()

    model_path = MODEL_DIR / "resnet18_best.pth"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Cannot find model file: {model_path}. "
            "Please run train_resnet18.py first."
        )

    # pretrained=False here avoids downloading weights again.
    # The trained weights are loaded from resnet18_best.pth.
    model = get_resnet18(
        pretrained=False,
        fine_tune_last_block=True
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))

    y_true, y_pred, y_probs = predict(
        model=model,
        data_loader=test_loader,
        device=device
    )

    overall_metrics, per_class_metrics, cm = calculate_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=CLASS_NAMES
    )

    print("\nOverall metrics:")
    for key, value in overall_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nPer-class metrics:")
    per_class_df = pd.DataFrame(per_class_metrics)
    print(per_class_df)

    overall_df = pd.DataFrame([{
        "model": MODEL_NAME,
        "loss": "CrossEntropyLoss",
        **overall_metrics
    }])

    overall_path = RESULT_DIR / "resnet18_test_metrics.csv"
    overall_df.to_csv(overall_path, index=False)

    per_class_path = RESULT_DIR / "resnet18_per_class_metrics.csv"
    per_class_df.to_csv(per_class_path, index=False)

    prediction_rows = []

    for idx, (true_label, pred_label, probs) in enumerate(zip(y_true, y_pred, y_probs)):
        row = {
            "sample_index": idx,
            "true_label": true_label,
            "true_class": CLASS_NAMES[true_label],
            "pred_label": pred_label,
            "pred_class": CLASS_NAMES[pred_label],
            "error_distance": abs(pred_label - true_label),
        }

        for class_index, prob in enumerate(probs):
            row[f"prob_{class_index}_{CLASS_NAMES[class_index]}"] = prob

        prediction_rows.append(row)

    predictions_df = pd.DataFrame(prediction_rows)
    predictions_path = RESULT_DIR / "resnet18_test_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{name}" for name in CLASS_NAMES],
        columns=[f"pred_{name}" for name in CLASS_NAMES]
    )

    cm_csv_path = RESULT_DIR / "resnet18_confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path)

    cm_fig_path = FIGURE_DIR / "resnet18_confusion_matrix.png"
    save_confusion_matrix(
        cm=cm,
        class_names=CLASS_NAMES,
        output_path=cm_fig_path
    )

    print("\nSaved files:")
    print(f"Overall metrics: {overall_path}")
    print(f"Per-class metrics: {per_class_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Confusion matrix CSV: {cm_csv_path}")
    print(f"Confusion matrix figure: {cm_fig_path}")


if __name__ == "__main__":
    main()