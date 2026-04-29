from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as T

from config import SPLIT_CSV, FIGURE_DIR, LABEL_MAP, IMAGE_SIZE
from transforms import get_train_transform


# ImageNet normalization values used in transforms.py
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def denormalize(tensor):
    """
    Convert normalized tensor back to displayable image.
    Input tensor shape: [3, H, W]
    Output shape: [H, W, 3]
    """
    tensor = tensor.clone()

    for channel, mean, std in zip(tensor, MEAN, STD):
        channel.mul_(std).add_(mean)

    tensor = tensor.clamp(0, 1)
    image = tensor.permute(1, 2, 0).numpy()

    return image


def main():
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(
            f"Cannot find split file: {SPLIT_CSV}. "
            "Please run create_split.py first."
        )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SPLIT_CSV)

    # Only use training set because augmentation is applied only to training data
    train_df = df[df["split"] == "train"].reset_index(drop=True)

    train_transform = get_train_transform()
    resize_only = T.Resize((IMAGE_SIZE, IMAGE_SIZE))

    rows = 5
    cols = 4

    plt.figure(figsize=(14, 16))

    plot_index = 1

    for label in range(5):
        class_df = train_df[train_df["label"] == label]

        if class_df.empty:
            continue

        sample = class_df.iloc[0]
        img_path = Path(sample["path"])

        original_image = Image.open(img_path).convert("RGB")

        # Column 1: original resized image
        resized_original = resize_only(original_image)

        plt.subplot(rows, cols, plot_index)
        plt.imshow(resized_original)
        plt.title(f"{label}: {LABEL_MAP[label]}\nOriginal")
        plt.axis("off")
        plot_index += 1

        # Columns 2-4: augmented versions of the same image
        for aug_id in range(1, 4):
            augmented_tensor = train_transform(original_image)
            augmented_image = denormalize(augmented_tensor)

            plt.subplot(rows, cols, plot_index)
            plt.imshow(augmented_image)
            plt.title(f"Augmented {aug_id}")
            plt.axis("off")
            plot_index += 1

    plt.tight_layout()

    output_path = FIGURE_DIR / "augmentation_examples.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("=" * 50)
    print("Augmentation Visualization")
    print("=" * 50)
    print(f"Saved augmentation examples to: {output_path}")


if __name__ == "__main__":
    main()