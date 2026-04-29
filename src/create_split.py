from pathlib import Path
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_DIR, PROCESSED_DIR, RESULT_DIR, SPLIT_CSV, LABEL_MAP, RANDOM_SEED


def collect_images():
    records = []

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

    for class_folder in sorted(DATA_DIR.iterdir()):
        if not class_folder.is_dir():
            continue

        if not class_folder.name[0].isdigit():
            continue

        label = int(class_folder.name[0])

        for img_path in class_folder.glob("*.png"):
            records.append({
                "path": str(img_path),
                "filename": img_path.name,
                "folder": class_folder.name,
                "label": label,
                "class_name": LABEL_MAP[label],
            })

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("No images found. Please check DATA_DIR and image format.")

    return df


def create_stratified_split(df):
    # First split: 70% train, 30% temporary
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=RANDOM_SEED
    )

    # Second split: temporary 30% -> validation 10%, test 20%
    # Since temp is 30%, validation should be 1/3 of temp and test should be 2/3 of temp
    val_df, test_df = train_test_split(
        temp_df,
        test_size=2 / 3,
        stratify=temp_df["label"],
        random_state=RANDOM_SEED
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    split_df = pd.concat([train_df, val_df, test_df], axis=0)
    split_df = split_df.reset_index(drop=True)

    return split_df


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    df = collect_images()

    print("=" * 50)
    print("Creating Train / Validation / Test Split")
    print("=" * 50)

    print(f"\nTotal images: {len(df)}")

    print("\nOriginal class distribution:")
    print(df.groupby(["label", "class_name"]).size())

    split_df = create_stratified_split(df)

    split_df.to_csv(SPLIT_CSV, index=False)

    print(f"\nSaved split file to: {SPLIT_CSV}")

    print("\nSplit counts:")
    print(split_df["split"].value_counts())

    print("\nSplit distribution by class:")
    split_distribution = split_df.groupby(["split", "label", "class_name"]).size()
    print(split_distribution)

    # Save split distribution as a CSV for presentation/report evidence
    split_distribution_df = (
        split_df
        .groupby(["split", "label", "class_name"])
        .size()
        .reset_index(name="count")
    )

    split_distribution_path = RESULT_DIR / "split_distribution.csv"
    split_distribution_df.to_csv(split_distribution_path, index=False)

    print(f"\nSaved split distribution to: {split_distribution_path}")


if __name__ == "__main__":
    main()