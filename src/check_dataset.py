from pathlib import Path
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

from config import DATA_DIR, FIGURE_DIR, RESULT_DIR, LABEL_MAP


def collect_image_info():
    records = []

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

    for class_folder in sorted(DATA_DIR.iterdir()):
        if not class_folder.is_dir():
            continue

        # Folder names should start with 0, 1, 2, 3, or 4
        if not class_folder.name[0].isdigit():
            continue

        label = int(class_folder.name[0])

        for img_path in class_folder.glob("*.png"):
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    mode = img.mode

                records.append({
                    "path": str(img_path),
                    "filename": img_path.name,
                    "folder": class_folder.name,
                    "label": label,
                    "class_name": LABEL_MAP[label],
                    "width": width,
                    "height": height,
                    "mode": mode
                })

            except Exception as e:
                print(f"Could not read image: {img_path}")
                print(e)

    return pd.DataFrame(records)


def save_class_distribution(df):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    class_order = ["Normal", "Doubtful", "Mild", "Moderate", "Severe"]

    counts = (
        df["class_name"]
        .value_counts()
        .reindex(class_order)
    )

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title("Class Distribution of MedicalExpert-I Dataset")
    plt.xlabel("KL Grade")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=30)
    plt.tight_layout()

    output_path = FIGURE_DIR / "class_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved class distribution figure to: {output_path}")


def save_sample_images(df):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))

    for label in range(5):
        sample = df[df["label"] == label].iloc[0]
        img = Image.open(sample["path"]).convert("L")

        plt.subplot(1, 5, label + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"{label}: {LABEL_MAP[label]}")
        plt.axis("off")

    plt.tight_layout()

    output_path = FIGURE_DIR / "sample_images.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved sample images figure to: {output_path}")


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    df = collect_image_info()

    print("=" * 50)
    print("Dataset Check: MedicalExpert-I")
    print("=" * 50)

    print(f"\nDataset path:\n{DATA_DIR}")
    print(f"\nTotal images found: {len(df)}")

    print("\nClass distribution:")
    print(df.groupby(["label", "class_name"]).size())

    print("\nImage modes:")
    print(df["mode"].value_counts())

    print("\nImage size summary:")
    print(df[["width", "height"]].describe())

    print("\nFirst 10 images:")
    print(df.head(10))

    summary_path = RESULT_DIR / "dataset_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nSaved dataset summary to: {summary_path}")

    save_class_distribution(df)
    save_sample_images(df)


if __name__ == "__main__":
    main()