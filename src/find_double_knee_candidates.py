from pathlib import Path
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import math

from config import DATA_DIR, FIGURE_DIR, RESULT_DIR, LABEL_MAP


WIDTH_THRESHOLD = 450
ASPECT_RATIO_THRESHOLD = 2.7
IMAGES_PER_PAGE = 20


def collect_candidates():
    records = []

    for class_folder in sorted(DATA_DIR.iterdir()):
        if not class_folder.is_dir():
            continue

        if not class_folder.name[0].isdigit():
            continue

        label = int(class_folder.name[0])

        for img_path in class_folder.glob("*.png"):
            with Image.open(img_path) as img:
                width, height = img.size
                aspect_ratio = width / height

            is_candidate = (
                width >= WIDTH_THRESHOLD
                or aspect_ratio >= ASPECT_RATIO_THRESHOLD
            )

            if is_candidate:
                records.append({
                    "path": str(img_path),
                    "filename": img_path.name,
                    "folder": class_folder.name,
                    "label": label,
                    "class_name": LABEL_MAP[label],
                    "width": width,
                    "height": height,
                    "aspect_ratio": round(aspect_ratio, 3),
                    "remove": ""
                })

    return pd.DataFrame(records)


def save_candidate_pages(df):
    if df.empty:
        print("No double-knee candidates found.")
        return

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    num_pages = math.ceil(len(df) / IMAGES_PER_PAGE)

    for page in range(num_pages):
        page_df = df.iloc[
            page * IMAGES_PER_PAGE : (page + 1) * IMAGES_PER_PAGE
        ]

        cols = 5
        rows = math.ceil(len(page_df) / cols)

        plt.figure(figsize=(15, rows * 3))

        for i, (_, row) in enumerate(page_df.iterrows()):
            img = Image.open(row["path"]).convert("L")

            plt.subplot(rows, cols, i + 1)
            plt.imshow(img, cmap="gray")
            plt.title(
                f'{row["class_name"]}\n{row["filename"]}\n{row["width"]}x{row["height"]}',
                fontsize=8
            )
            plt.axis("off")

        plt.tight_layout()

        output_path = FIGURE_DIR / f"double_knee_candidates_page_{page + 1}.png"
        plt.savefig(output_path, dpi=200)
        plt.close()

        print(f"Saved: {output_path}")


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    df = collect_candidates()

    print("=" * 50)
    print("Double-knee Candidate Detection")
    print("=" * 50)

    print(f"Dataset path: {DATA_DIR}")
    print(f"Number of candidates: {len(df)}")

    if not df.empty:
        print("\nCandidates by class:")
        print(df.groupby(["label", "class_name"]).size())

        output_csv = RESULT_DIR / "double_knee_candidates.csv"
        df.to_csv(output_csv, index=False)
        print(f"\nSaved candidate list to: {output_csv}")

        save_candidate_pages(df)


if __name__ == "__main__":
    main()