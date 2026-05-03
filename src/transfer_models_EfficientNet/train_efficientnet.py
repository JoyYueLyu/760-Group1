from pathlib import Path
import sys

# Add src folder to Python path
SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import time
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    DEVICE,
    NUM_EPOCHS,
    RANDOM_SEED,
    MODEL_DIR,
    RESULT_DIR,
    FIGURE_DIR,
)

from dataset import get_dataloaders
from efficientnet_model import get_efficientnet_b0



MODEL_NAME = "efficientnet"
LEARNING_RATE = 0.0001
PATIENCE = 5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate_one_epoch(model, data_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def save_training_curves(history):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    epochs = history["epoch"]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("EfficientNet-B0 Transfer Learning Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "efficientnet_loss_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("EfficientNet-B0 Transfer Learning Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "efficientnet_accuracy_curve.png", dpi=300)
    plt.close()


def main():
    set_seed(RANDOM_SEED)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE)

    print("=" * 50)
    print("Training EfficientNet-B0 Transfer Learning Model")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Early stopping patience: {PATIENCE}")

    train_loader, val_loader, test_loader = get_dataloaders()

    print("\nDataLoader info:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    model = get_efficientnet_b0(
        pretrained=True,
        fine_tune_last_block=True
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = optim.Adam(
        trainable_params,
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_without_improvement = 0

    best_model_path = MODEL_DIR / "efficientnet_best.pth"

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        val_loss, val_acc = evaluate_one_epoch(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            epochs_without_improvement = 0

            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to: {best_model_path}")
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement for "
                f"{epochs_without_improvement}/{PATIENCE} epochs"
            )

        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    end_time = time.time()
    training_time = end_time - start_time

    history_df = pd.DataFrame(history)
    history_path = RESULT_DIR / "efficientnet_training_history.csv"
    history_df.to_csv(history_path, index=False)

    save_training_curves(history)

    summary = {
        "model": MODEL_NAME,
        "loss": "CrossEntropyLoss",
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "pretrained": True,
        "fine_tune_last_block": True,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "epochs_trained": len(history["epoch"]),
        "training_time_seconds": training_time,
    }

    summary_df = pd.DataFrame([summary])
    summary_path = RESULT_DIR / "efficientnet_training_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 50)
    print("EfficientNet-B0 transfer learning training completed")
    print("=" * 50)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Epochs trained: {len(history['epoch'])}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Training history saved to: {history_path}")
    print(f"Training summary saved to: {summary_path}")
    print(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main()