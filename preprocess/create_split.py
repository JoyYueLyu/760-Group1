from pathlib import Path
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split


CLASS_FOLDERS = {
    "0Normal": 0,
    "1Doubtful": 1,
    "2Mild": 2,
    "3Moderate": 3,
    "4Severe": 4,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_cleaned_images(cleaned_data_dir):
    cleaned_data_dir = Path(cleaned_data_dir)

    if not cleaned_data_dir.exists():
        raise FileNotFoundError(f"Cleaned dataset folder not found: {cleaned_data_dir}")

    records = []

    for class_name, label in CLASS_FOLDERS.items():
        class_dir = cleaned_data_dir / class_name

        if not class_dir.exists():
            print(f"Warning: missing class folder: {class_dir}")
            continue

        for img_path in class_dir.rglob("*"):
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append({
                    "image_path": str(img_path),
                    "label": label,
                    "class_name": class_name,
                    "file_name": img_path.name,
                })

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("No images found in cleaned dataset folder.")

    df = df.sort_values(["label", "image_path"]).reset_index(drop=True)

    return df


def create_stratified_split(
    df,
    train_size=0.70,
    val_size=0.10,
    test_size=0.20,
    random_state=42,
):
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    train_df, temp_df = train_test_split(
        df,
        test_size=val_size + test_size,
        stratify=df["label"],
        random_state=random_state,
        shuffle=True,
    )

    relative_test_size = test_size / (val_size + test_size)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df["label"],
        random_state=random_state,
        shuffle=True,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    return split_df


def copy_split_images(split_df, output_split_dir):
    output_split_dir = Path(output_split_dir)

    for _, row in split_df.iterrows():
        src_path = Path(row["image_path"])

        split_name = row["split"]
        class_name = row["class_name"]

        dst_dir = output_split_dir / split_name / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        dst_path = dst_dir / src_path.name

        # Avoid overwrite if duplicate file names exist
        if dst_path.exists():
            dst_path = dst_dir / f"{src_path.stem}_{row.name}{src_path.suffix}"

        shutil.copy2(src_path, dst_path)


def summarize_split(split_df):
    total_summary = (
        split_df
        .groupby("split")
        .size()
        .reset_index(name="total_count")
        .sort_values("split")
    )

    class_summary = (
        split_df
        .groupby(["split", "label", "class_name"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "label"])
    )

    print("=" * 60)
    print("Total images by split")
    print("=" * 60)
    print(total_summary.to_string(index=False))

    print()
    print("=" * 60)
    print("Class distribution by split")
    print("=" * 60)
    print(class_summary.to_string(index=False))

    return total_summary, class_summary


def create_split_folders(
    cleaned_data_dir,
    output_split_dir,
    results_dir="outputs/results",
    train_size=0.70,
    val_size=0.10,
    test_size=0.20,
    random_state=42,
    overwrite=False,
):
    cleaned_data_dir = Path(cleaned_data_dir)
    output_split_dir = Path(output_split_dir)
    results_dir = Path(results_dir)

    if output_split_dir.exists():
        if overwrite:
            print(f"Removing existing split folder: {output_split_dir}")
            shutil.rmtree(output_split_dir)
        else:
            raise FileExistsError(
                f"Output split folder already exists: {output_split_dir}\n"
                f"Set overwrite=True if you want to recreate it, or manually delete it."
            )

    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Collecting cleaned images")
    print("=" * 60)

    df = collect_cleaned_images(cleaned_data_dir)

    print(f"Total cleaned images found: {len(df)}")

    print()
    print("=" * 60)
    print("Creating stratified split")
    print("=" * 60)

    split_df = create_stratified_split(
        df=df,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )

    print()
    print("=" * 60)
    print("Copying images into train / val / test folders")
    print("=" * 60)

    copy_split_images(split_df, output_split_dir)

    total_summary, class_summary = summarize_split(split_df)

    split_df.to_csv(results_dir / "split_folder_records.csv", index=False)
    total_summary.to_csv(results_dir / "split_folder_total_summary.csv", index=False)
    class_summary.to_csv(results_dir / "split_folder_class_summary.csv", index=False)

    print()
    print("=" * 60)
    print("Saved output")
    print("=" * 60)
    print(f"Split image folders: {output_split_dir}")
    print(results_dir / "split_folder_records.csv")
    print(results_dir / "split_folder_total_summary.csv")
    print(results_dir / "split_folder_class_summary.csv")

    return split_df, total_summary, class_summary


if __name__ == "__main__":
    PROJECT_ROOT = Path.cwd()

    CLEANED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "cleaned_images" / "MedicalExpert-I"
    OUTPUT_SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "split_images"
    RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

    create_split_folders(
        cleaned_data_dir=CLEANED_DATA_DIR,
        output_split_dir=OUTPUT_SPLIT_DIR,
        results_dir=RESULTS_DIR,
        train_size=0.70,
        val_size=0.10,
        test_size=0.20,
        random_state=42,
        overwrite=False,
    )


