from pathlib import Path
import shutil
from PIL import Image
import pandas as pd


CLASS_FOLDERS = {
    "0Normal": 0,
    "1Doubtful": 1,
    "2Mild": 2,
    "3Moderate": 3,
    "4Severe": 4,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_image_metadata(data_dir):
    """
    Collect image paths, labels, class names, image sizes, and image modes.

    Args:
        data_dir: path to MedicalExpert-I folder.

    Returns:
        pandas DataFrame
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    records = []

    for class_name, label in CLASS_FOLDERS.items():
        class_dir = data_dir / class_name

        if not class_dir.exists():
            print(f"Warning: missing class folder: {class_dir}")
            continue

        for img_path in class_dir.rglob("*"):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    mode = img.mode

                records.append({
                    "image_path": str(img_path),
                    "label": label,
                    "class_name": class_name,
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "file_name": img_path.name,
                })

            except Exception as e:
                print(f"Could not read image: {img_path}")
                print(f"Error: {e}")

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("No images found. Please check dataset path.")

    return df


def is_double_knee_candidate(width, height, width_threshold=500, aspect_ratio_threshold=3.0):
    """
    Detect likely double-knee images.

    In this dataset:
    - normal single-knee images are mostly 300 x 162
    - double-knee images are mostly 640 x 161

    We use a safe rule:
    - image width >= 500, OR
    - width / height >= 3.0

    Args:
        width: image width
        height: image height
        width_threshold: threshold for wide images
        aspect_ratio_threshold: threshold for very wide images

    Returns:
        True if likely double-knee image, otherwise False.
    """
    if height == 0:
        return True

    aspect_ratio = width / height

    return (width >= width_threshold) or (aspect_ratio >= aspect_ratio_threshold)


def get_unique_destination_path(destination_path):
    """
    Avoid overwriting files if a file with the same name already exists.
    """
    destination_path = Path(destination_path)

    if not destination_path.exists():
        return destination_path

    stem = destination_path.stem
    suffix = destination_path.suffix
    parent = destination_path.parent

    counter = 1

    while True:
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


def copy_clean_images(clean_df, cleaned_data_dir):
    """
    Copy clean single-knee images to processed folder.

    Args:
        clean_df: DataFrame containing clean image rows.
        cleaned_data_dir: output folder.

    Returns:
        DataFrame with an additional cleaned_image_path column.
    """
    cleaned_data_dir = Path(cleaned_data_dir)
    copied_records = []

    for _, row in clean_df.iterrows():
        src_path = Path(row["image_path"])

        class_name = row["class_name"]
        dst_class_dir = cleaned_data_dir / class_name
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        dst_path = dst_class_dir / src_path.name
        dst_path = get_unique_destination_path(dst_path)

        shutil.copy2(src_path, dst_path)

        new_row = row.to_dict()
        new_row["cleaned_image_path"] = str(dst_path)
        copied_records.append(new_row)

    return pd.DataFrame(copied_records)


def create_cleaned_dataset(
    raw_data_dir,
    cleaned_data_dir,
    results_dir="outputs/results",
    overwrite=False,
    width_threshold=500,
    aspect_ratio_threshold=3.0,
):
    """
    Create a cleaned single-knee dataset.

    This function:
    1. Reads image metadata from raw dataset.
    2. Detects likely double-knee images.
    3. Copies only clean single-knee images into processed folder.
    4. Saves metadata CSV files.

    Important:
    - This function does NOT delete raw images.
    - It only creates a cleaned copy under data/processed/.

    Args:
        raw_data_dir: path to original MedicalExpert-I folder.
        cleaned_data_dir: path to output cleaned MedicalExpert-I folder.
        results_dir: folder for CSV outputs.
        overwrite: if True, delete existing cleaned_data_dir before copying.
        width_threshold: width threshold for double-knee detection.
        aspect_ratio_threshold: aspect ratio threshold for double-knee detection.

    Returns:
        all_df, clean_copied_df, removed_df, summary_df
    """
    raw_data_dir = Path(raw_data_dir)
    cleaned_data_dir = Path(cleaned_data_dir)
    results_dir = Path(results_dir)

    results_dir.mkdir(parents=True, exist_ok=True)

    if cleaned_data_dir.exists() and overwrite:
        print(f"Removing existing cleaned folder: {cleaned_data_dir}")
        shutil.rmtree(cleaned_data_dir)

    cleaned_data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Collecting image metadata")
    print("=" * 60)

    all_df = collect_image_metadata(raw_data_dir)

    all_df["is_double_knee_candidate"] = all_df.apply(
        lambda row: is_double_knee_candidate(
            width=row["width"],
            height=row["height"],
            width_threshold=width_threshold,
            aspect_ratio_threshold=aspect_ratio_threshold,
        ),
        axis=1,
    )

    clean_df = all_df[all_df["is_double_knee_candidate"] == False].copy()
    removed_df = all_df[all_df["is_double_knee_candidate"] == True].copy()

    print(f"Original images: {len(all_df)}")
    print(f"Clean single-knee images: {len(clean_df)}")
    print(f"Removed double-knee candidates: {len(removed_df)}")

    print()
    print("=" * 60)
    print("Copying clean images")
    print("=" * 60)

    clean_copied_df = copy_clean_images(clean_df, cleaned_data_dir)

    print(f"Copied clean images to: {cleaned_data_dir}")

    original_counts = (
        all_df.groupby(["label", "class_name"])
        .size()
        .reset_index(name="original_count")
    )

    clean_counts = (
        clean_copied_df.groupby(["label", "class_name"])
        .size()
        .reset_index(name="cleaned_count")
    )

    removed_counts = (
        removed_df.groupby(["label", "class_name"])
        .size()
        .reset_index(name="removed_count")
    )

    summary_df = original_counts.merge(
        clean_counts,
        on=["label", "class_name"],
        how="left"
    ).merge(
        removed_counts,
        on=["label", "class_name"],
        how="left"
    )

    summary_df["cleaned_count"] = summary_df["cleaned_count"].fillna(0).astype(int)
    summary_df["removed_count"] = summary_df["removed_count"].fillna(0).astype(int)

    summary_df = summary_df.sort_values("label")

    print()
    print("=" * 60)
    print("Cleaning Summary")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    all_df.to_csv(results_dir / "all_image_metadata.csv", index=False)
    clean_copied_df.to_csv(results_dir / "cleaned_dataset.csv", index=False)
    removed_df.to_csv(results_dir / "removed_double_knee_candidates.csv", index=False)
    summary_df.to_csv(results_dir / "cleaning_summary.csv", index=False)

    print()
    print("=" * 60)
    print("Saved output files")
    print("=" * 60)
    print(results_dir / "all_image_metadata.csv")
    print(results_dir / "cleaned_dataset.csv")
    print(results_dir / "removed_double_knee_candidates.csv")
    print(results_dir / "cleaning_summary.csv")

    return all_df, clean_copied_df, removed_df, summary_df


if __name__ == "__main__":
    RAW_DATA_DIR = "data/raw/Digital Knee X-ray Images/MedicalExpert-I"
    CLEANED_DATA_DIR = "data/processed/cleaned_images/MedicalExpert-I"
    RESULTS_DIR = "outputs/results"

    create_cleaned_dataset(
        raw_data_dir=RAW_DATA_DIR,
        cleaned_data_dir=CLEANED_DATA_DIR,
        results_dir=RESULTS_DIR,
        overwrite=True,
        width_threshold=500,
        aspect_ratio_threshold=3.0,
    )

