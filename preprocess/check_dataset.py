from pathlib import Path
from collections import Counter
import random

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


CLASS_FOLDERS = {
    "0Normal": 0,
    "1Doubtful": 1,
    "2Mild": 2,
    "3Moderate": 3,
    "4Severe": 4,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_image_paths(data_dir):
    """
    Collect image paths and labels from class folders.

    Args:
        data_dir: Path to MedicalExpert-I folder.

    Returns:
        pandas DataFrame with columns: image_path, label, class_name
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    records = []

    for class_name, label in CLASS_FOLDERS.items():
        class_dir = data_dir / class_name

        if not class_dir.exists():
            print(f"Warning: missing folder: {class_dir}")
            continue

        for img_path in class_dir.rglob("*"):
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append({
                    "image_path": str(img_path),
                    "label": label,
                    "class_name": class_name
                })

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError(
            "No images found. Please check your dataset path and image file extensions."
        )

    return df


def print_dataset_summary(df):
    """
    Print total image count and class distribution.
    """
    print("=" * 60)
    print("Dataset Summary")
    print("=" * 60)

    print(f"Total images: {len(df)}")
    print()

    class_counts = (
        df.groupby(["label", "class_name"])
        .size()
        .reset_index(name="count")
        .sort_values("label")
    )

    print("Class distribution:")
    print(class_counts.to_string(index=False))

    return class_counts


def check_image_sizes(df, max_images=None):
    """
    Check image sizes.

    Args:
        df: DataFrame containing image paths.
        max_images: If set, only check first N images for speed.

    Returns:
        pandas DataFrame with image_path, width, height, mode
    """
    records = []

    if max_images is not None:
        df_check = df.head(max_images)
    else:
        df_check = df

    for _, row in df_check.iterrows():
        img_path = row["image_path"]

        try:
            with Image.open(img_path) as img:
                width, height = img.size
                mode = img.mode

            records.append({
                "image_path": img_path,
                "label": row["label"],
                "class_name": row["class_name"],
                "width": width,
                "height": height,
                "mode": mode
            })

        except Exception as e:
            print(f"Could not open image: {img_path}")
            print(f"Error: {e}")

    size_df = pd.DataFrame(records)

    print()
    print("=" * 60)
    print("Image Size Summary")
    print("=" * 60)

    if not size_df.empty:
        print("Most common image sizes:")
        print(
            size_df.groupby(["width", "height"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(10)
            .to_string(index=False)
        )

        print()
        print("Image modes:")
        print(size_df["mode"].value_counts().to_string())

    return size_df


def plot_class_distribution(class_counts, save_path=None):
    """
    Plot class distribution bar chart.
    """
    plt.figure(figsize=(8, 5))

    labels = class_counts["class_name"]
    counts = class_counts["count"]

    plt.bar(labels, counts)
    plt.title("Class Distribution of KL Grades")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=30)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved class distribution figure to: {save_path}")

    plt.show()


def show_random_samples(df, samples_per_class=3, save_path=None, seed=42):
    """
    Show random image samples from each class.
    """
    random.seed(seed)

    num_classes = len(CLASS_FOLDERS)
    fig, axes = plt.subplots(
        num_classes,
        samples_per_class,
        figsize=(samples_per_class * 3, num_classes * 3)
    )

    for row_idx, (class_name, label) in enumerate(CLASS_FOLDERS.items()):
        class_df = df[df["label"] == label]

        if len(class_df) == 0:
            continue

        sample_df = class_df.sample(
            n=min(samples_per_class, len(class_df)),
            random_state=seed
        )

        for col_idx in range(samples_per_class):
            ax = axes[row_idx, col_idx]

            if col_idx < len(sample_df):
                img_path = sample_df.iloc[col_idx]["image_path"]

                with Image.open(img_path) as img:
                    ax.imshow(img, cmap="gray")

                ax.set_title(f"{class_name}\nLabel {label}")
            else:
                ax.set_title(f"{class_name}\nNo image")

            ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved sample images figure to: {save_path}")

    plt.show()


def run_dataset_check(
    data_dir,
    output_dir="outputs/figures",
    samples_per_class=3
):
    """
    Run full dataset checking process.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    df = collect_image_paths(data_dir)
    class_counts = print_dataset_summary(df)
    size_df = check_image_sizes(df)

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_class_distribution(
        class_counts,
        save_path=output_dir / "class_distribution.png"
    )

    show_random_samples(
        df,
        samples_per_class=samples_per_class,
        save_path=output_dir / "random_samples.png"
    )

    # Save CSV summaries
    Path("outputs/results").mkdir(parents=True, exist_ok=True)

    df.to_csv("outputs/results/dataset_images.csv", index=False)
    class_counts.to_csv("outputs/results/class_counts.csv", index=False)
    size_df.to_csv("outputs/results/image_sizes.csv", index=False)

    print()
    print("=" * 60)
    print("Saved output files")
    print("=" * 60)
    print("outputs/results/dataset_images.csv")
    print("outputs/results/class_counts.csv")
    print("outputs/results/image_sizes.csv")
    print("outputs/figures/class_distribution.png")
    print("outputs/figures/random_samples.png")

    return df, class_counts, size_df


if __name__ == "__main__":
    DATA_DIR = "data/raw/Digital Knee X-ray Images/MedicalExpert-I"

    run_dataset_check(
        data_dir=DATA_DIR,
        output_dir="outputs/figures",
        samples_per_class=3
    )

