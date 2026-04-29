from pathlib import Path
import pandas as pd
import shutil

from config import RESULT_DIR, PROJECT_ROOT


CANDIDATE_CSV = RESULT_DIR / "double_knee_candidates.csv"
EXCLUDED_DIR = PROJECT_ROOT / "data" / "excluded" / "double_knee"


def main():
    if not CANDIDATE_CSV.exists():
        raise FileNotFoundError(
            f"Cannot find {CANDIDATE_CSV}. "
            "Please run find_double_knee_candidates.py first."
        )

    df = pd.read_csv(CANDIDATE_CSV)

    if "remove" not in df.columns:
        raise ValueError("The CSV file must contain a 'remove' column.")

    remove_df = df[df["remove"].astype(str).str.lower().str.strip() == "yes"]

    print("=" * 50)
    print("Moving confirmed double-knee images")
    print("=" * 50)

    print(f"Images marked for removal: {len(remove_df)}")

    EXCLUDED_DIR.mkdir(parents=True, exist_ok=True)

    moved_records = []

    for _, row in remove_df.iterrows():
        src_path = Path(row["path"])

        if not src_path.exists():
            print(f"File not found, skipping: {src_path}")
            continue

        class_folder = row["folder"]
        target_folder = EXCLUDED_DIR / class_folder
        target_folder.mkdir(parents=True, exist_ok=True)

        target_path = target_folder / src_path.name

        shutil.move(str(src_path), str(target_path))

        moved_records.append({
            "original_path": str(src_path),
            "new_path": str(target_path),
            "label": row["label"],
            "class_name": row["class_name"],
            "filename": row["filename"]
        })

        print(f"Moved: {src_path.name}")

    moved_df = pd.DataFrame(moved_records)
    moved_log = RESULT_DIR / "removed_double_knee_log.csv"
    moved_df.to_csv(moved_log, index=False)

    print(f"\nSaved removal log to: {moved_log}")
    print(f"Excluded images are stored in: {EXCLUDED_DIR}")


if __name__ == "__main__":
    main()