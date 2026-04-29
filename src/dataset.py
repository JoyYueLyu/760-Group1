from pathlib import Path
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from config import SPLIT_CSV, BATCH_SIZE
from transforms import get_train_transform, get_eval_transform


class KneeXrayDataset(Dataset):
    def __init__(self, split: str, transform=None):
        """
        PyTorch Dataset for knee X-ray classification.

        Args:
            split: one of "train", "val", or "test"
            transform: torchvision transforms
        """
        if split not in ["train", "val", "test"]:
            raise ValueError("split must be one of: train, val, test")

        if not SPLIT_CSV.exists():
            raise FileNotFoundError(
                f"Cannot find split file: {SPLIT_CSV}. "
                "Please run create_split.py first."
            )

        self.df = pd.read_csv(SPLIT_CSV)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        if self.df.empty:
            raise ValueError(f"No images found for split: {split}")

        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        img_path = Path(row["path"])
        label = int(row["label"])

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Convert all images to RGB.
        # This handles RGB and I;16 images consistently.
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders():
    """
    Create train, validation, and test dataloaders.
    """
    train_dataset = KneeXrayDataset(
        split="train",
        transform=get_train_transform()
    )

    val_dataset = KneeXrayDataset(
        split="val",
        transform=get_eval_transform()
    )

    test_dataset = KneeXrayDataset(
        split="test",
        transform=get_eval_transform()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()

    print("=" * 50)
    print("PyTorch DataLoader Check")
    print("=" * 50)

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    images, labels = next(iter(train_loader))

    print("\nOne training batch:")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels: {labels}")