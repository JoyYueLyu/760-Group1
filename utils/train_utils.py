from pathlib import Path
import copy

import torch
import pandas as pd
from tqdm.auto import tqdm

from utils.metrics import calculate_metrics


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    all_labels = []
    all_preds = []

    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())

    epoch_loss = running_loss / len(train_loader.dataset)
    metrics = calculate_metrics(all_labels, all_preds)

    return epoch_loss, metrics


def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())

    epoch_loss = running_loss / len(val_loader.dataset)
    metrics = calculate_metrics(all_labels, all_preds)

    return epoch_loss, metrics, all_labels, all_preds


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs=30,
    save_path=None,
    monitor_metric="qwk",
    patience=7,
    min_delta=1e-4,
):
    """
    Train model with early stopping.

    The best model is selected based on a validation metric.

    Args:
        model: PyTorch model.
        train_loader: training dataloader.
        val_loader: validation dataloader.
        criterion: loss function.
        optimizer: optimizer.
        device: cuda or cpu.
        num_epochs: maximum number of epochs.
        save_path: path to save best model.
        monitor_metric: validation metric to monitor.
                        Recommended: "qwk", "macro_f1", or "accuracy".
        patience: early stopping patience.
        min_delta: minimum improvement required to reset patience.

    Returns:
        model: best model loaded.
        history_df: training history.
    """
    from pathlib import Path
    import copy
    import torch
    import pandas as pd

    history = []

    best_score = -float("inf")
    best_model_state = None
    epochs_without_improvement = 0
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        print()
        print(f"Epoch {epoch}/{num_epochs}")
        print("-" * 60)

        train_loss, train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_metrics, _, _ = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        current_score = val_metrics[monitor_metric]

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Train Macro F1: {train_metrics['macro_f1']:.4f} | "
            f"Train QWK: {train_metrics['qwk']:.4f}"
        )

        print(
            f"Val Loss:   {val_loss:.4f} | "
            f"Val Acc:   {val_metrics['accuracy']:.4f} | "
            f"Val Macro F1: {val_metrics['macro_f1']:.4f} | "
            f"Val QWK:   {val_metrics['qwk']:.4f} | "
            f"Val MAE:   {val_metrics['mae']:.4f}"
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_epoch_so_far": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
        }

        for key, value in train_metrics.items():
            row[f"train_{key}"] = value

        for key, value in val_metrics.items():
            row[f"val_{key}"] = value

        history.append(row)

        improved = current_score > best_score + min_delta

        if improved:
            best_score = current_score
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

            print(f"New best model found at epoch {epoch}.")
            print(f"Best val {monitor_metric}: {best_score:.4f}")

            if save_path is not None:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_model_state, save_path)
                print(f"Saved best model to: {save_path}")

        else:
            epochs_without_improvement += 1
            print(
                f"No improvement in val {monitor_metric}. "
                f"Patience: {epochs_without_improvement}/{patience}"
            )

        if epochs_without_improvement >= patience:
            print()
            print("=" * 60)
            print("Early stopping triggered")
            print("=" * 60)
            print(f"Best epoch: {best_epoch}")
            print(f"Best val {monitor_metric}: {best_score:.4f}")
            break

    history_df = pd.DataFrame(history)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history_df






def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate model on validation or test set.
    """
    loss, metrics, y_true, y_pred = validate_one_epoch(
        model=model,
        val_loader=data_loader,
        criterion=criterion,
        device=device,
    )

    return loss, metrics, y_true, y_pred


