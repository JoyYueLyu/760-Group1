from pathlib import Path
import sys
import argparse

# Add src folder to Python path
SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from config import (
    DEVICE,
    DATA_DIR,
    SPLIT_CSV,
    MODEL_DIR,
    RESULT_DIR,
    FIGURE_DIR,
    IMAGE_SIZE,
    CLASS_NAMES,
)

from transfer_models_EfficientNet.efficientnet_model import get_efficientnet_b0


MODEL_VARIANTS = {
    "standard": {
        "model_path": MODEL_DIR / "efficientnet_best.pth",
        "output_prefix": "efficientnet",
        "display_name": "EfficientNet-B0 + Standard CE",
    },
    "classweighted": {
        "model_path": MODEL_DIR / "efficientnet_classweighted_best.pth",
        "output_prefix": "efficientnet_classweighted",
        "display_name": "EfficientNet-B0 + Class-weighted CE",
    },
    "ordinal": {
        "model_path": MODEL_DIR / "efficientnet_ordinal_best.pth",
        "output_prefix": "efficientnet_ordinal",
        "display_name": "EfficientNet-B0 + Ordinal-aware Loss",
    },
}


class GradCAM:
    """
    Grad-CAM for CNN-based models.

    It uses:
    - activations from the target convolution layer
    - gradients of the predicted class score
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = self.target_layer.register_forward_hook(
            self.save_activation
        )

        self.backward_hook = self.target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for one image.

        Args:
            input_tensor: shape [1, 3, H, W]
            target_class: class index. If None, use predicted class.

        Returns:
            cam: normalized Grad-CAM heatmap
            pred_class: predicted class index
            confidence: predicted probability
        """

        self.model.zero_grad()

        outputs = self.model(input_tensor)
        probs = torch.softmax(outputs, dim=1)

        pred_class = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred_class].item())

        if target_class is None:
            target_class = pred_class

        score = outputs[:, target_class]
        score.backward()

        # Global average pooling over gradients
        weights = torch.mean(
            self.gradients,
            dim=(2, 3),
            keepdim=True
        )

        # Weighted sum of activations
        cam = torch.sum(weights * self.activations, dim=1)

        # ReLU: only positive influence
        cam = F.relu(cam)

        # Resize to input image size
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, pred_class, confidence


def get_eval_transform():
    """
    Use ImageNet normalization because EfficientNet-B0 is pretrained on ImageNet.
    """

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_image_path(row):
    """
    Support different split CSV formats:
    1. absolute path
    2. relative_path
    3. folder + filename
    """

    if "path" in row and pd.notna(row["path"]):
        path = Path(str(row["path"]))

        if path.exists():
            return path

        # If absolute path is from another machine, rebuild from MedicalExpert-I
        parts = list(path.parts)

        if "MedicalExpert-I" in parts:
            index = parts.index("MedicalExpert-I")
            relative_part = Path(*parts[index + 1:])
            candidate = DATA_DIR / relative_part

            if candidate.exists():
                return candidate

    if "relative_path" in row and pd.notna(row["relative_path"]):
        candidate = DATA_DIR / str(row["relative_path"])

        if candidate.exists():
            return candidate

    if "folder" in row and "filename" in row:
        candidate = DATA_DIR / str(row["folder"]) / str(row["filename"])

        if candidate.exists():
            return candidate

    if "filename" in row and "label" in row and "class_name" in row:
        folder = f"{int(row['label'])}{row['class_name']}"
        candidate = DATA_DIR / folder / str(row["filename"])

        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Cannot find image path for row:\n"
        f"{row}"
    )


def load_image_for_gradcam(image_path, transform, device):
    image = Image.open(image_path).convert("RGB")

    resized_image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    return resized_image, input_tensor


def create_overlay(image, cam, alpha=0.45):
    """
    Overlay Grad-CAM heatmap on original image.
    """

    image_np = np.array(image).astype(np.float32) / 255.0

    heatmap = plt.cm.jet(cam)
    heatmap = heatmap[:, :, :3]

    overlay = (1 - alpha) * image_np + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)

    return overlay


def select_samples(test_df, samples_per_class):
    """
    Select fixed number of test samples from each class.
    """

    selected_rows = []

    test_df = test_df.copy()
    test_df["label"] = test_df["label"].astype(int)

    for label in sorted(test_df["label"].unique()):
        class_df = test_df[test_df["label"] == label]
        selected_rows.append(class_df.head(samples_per_class))

    selected_df = pd.concat(selected_rows, ignore_index=True)

    return selected_df


def save_gradcam_grid(results, output_path, title):
    """
    Save original image and Grad-CAM overlay side by side.
    """

    num_images = len(results)

    fig, axes = plt.subplots(
        num_images,
        2,
        figsize=(8, 3 * num_images)
    )

    if num_images == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle(title, fontsize=14)

    for i, result in enumerate(results):
        image = result["image"]
        overlay = result["overlay"]

        true_class = result["true_class"]
        pred_class = result["pred_class"]
        confidence = result["confidence"]

        axes[i, 0].imshow(image, cmap="gray")
        axes[i, 0].set_title(f"Original | True: {true_class}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title(
            f"Grad-CAM | Pred: {pred_class} ({confidence:.2f})"
        )
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variant",
        type=str,
        default="standard",
        choices=["standard", "classweighted", "ordinal"],
        help="Which EfficientNet-B0 model variant to visualize."
    )

    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=1,
        help="Number of test images per class to visualize."
    )

    args = parser.parse_args()

    variant_config = MODEL_VARIANTS[args.variant]

    model_path = variant_config["model_path"]
    output_prefix = variant_config["output_prefix"]
    display_name = variant_config["display_name"]

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    gradcam_dir = FIGURE_DIR / "gradcam"
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE)

    print("=" * 50)
    print("Generating Grad-CAM Visualizations")
    print("=" * 50)
    print(f"Model variant: {display_name}")
    print(f"Device: {device}")
    print(f"Model path: {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Cannot find model file: {model_path}"
        )

    if not SPLIT_CSV.exists():
        raise FileNotFoundError(
            f"Cannot find split file: {SPLIT_CSV}. "
            "Please run create_split.py first."
        )

    df = pd.read_csv(SPLIT_CSV)
    test_df = df[df["split"] == "test"].copy()

    selected_df = select_samples(
        test_df=test_df,
        samples_per_class=args.samples_per_class
    )

    model = get_efficientnet_b0(
        pretrained=False,
        fine_tune_last_block=True
    ).to(device)

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    model.eval()

    # Last convolutional layer of EfficientNet-B0
    target_layer = model.features[-1][0]

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer
    )

    transform = get_eval_transform()

    results = []
    prediction_rows = []

    for index, row in selected_df.iterrows():
        image_path = get_image_path(row)

        true_label = int(row["label"])
        true_class = CLASS_NAMES[true_label]

        image, input_tensor = load_image_for_gradcam(
            image_path=image_path,
            transform=transform,
            device=device
        )

        cam, pred_label, confidence = gradcam.generate(
            input_tensor=input_tensor,
            target_class=None
        )

        pred_class = CLASS_NAMES[pred_label]

        overlay = create_overlay(
            image=image,
            cam=cam,
            alpha=0.45
        )

        results.append({
            "image": image,
            "overlay": overlay,
            "true_class": true_class,
            "pred_class": pred_class,
            "confidence": confidence,
        })

        prediction_rows.append({
            "variant": args.variant,
            "image_path": str(image_path),
            "true_label": true_label,
            "true_class": true_class,
            "pred_label": pred_label,
            "pred_class": pred_class,
            "confidence": confidence,
            "correct": true_label == pred_label,
            "error_distance": abs(pred_label - true_label),
        })

    gradcam.remove_hooks()

    grid_path = gradcam_dir / f"{output_prefix}_gradcam_examples.png"

    save_gradcam_grid(
        results=results,
        output_path=grid_path,
        title=f"Grad-CAM Examples: {display_name}"
    )

    predictions_df = pd.DataFrame(prediction_rows)

    prediction_path = (
        RESULT_DIR / f"{output_prefix}_gradcam_predictions.csv"
    )

    predictions_df.to_csv(prediction_path, index=False)

    print("\nSaved files:")
    print(f"Grad-CAM figure: {grid_path}")
    print(f"Grad-CAM predictions: {prediction_path}")


if __name__ == "__main__":
    main()
