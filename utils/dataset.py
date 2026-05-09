from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ImageNet normalization values.
# DenseNet121 is pretrained on ImageNet, so we use ImageNet mean and std.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(
    image_size=224,
    use_augmentation=True,
):
    """
    Transform for training images.

    Training transform can include data augmentation.
    Validation and test transforms should not use random augmentation.
    """
    transform_list = [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
    ]

    if use_augmentation:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
            ),
        ])

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])

    return transforms.Compose(transform_list)


def get_eval_transform(image_size=224):
    """
    Transform for validation and test images.

    No random augmentation should be used for validation/test.
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])


def get_imagefolder_datasets(
    split_root,
    image_size=224,
    use_augmentation=True,
):
    """
    Create train, validation, and test datasets using ImageFolder.

    Expected folder structure:

    split_root/
        train/
            0Normal/
            1Doubtful/
            2Mild/
            3Moderate/
            4Severe/
        val/
            ...
        test/
            ...
    """
    split_root = Path(split_root)

    train_dir = split_root / "train"
    val_dir = split_root / "val"
    test_dir = split_root / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train folder not found: {train_dir}")

    if not val_dir.exists():
        raise FileNotFoundError(f"Validation folder not found: {val_dir}")

    if not test_dir.exists():
        raise FileNotFoundError(f"Test folder not found: {test_dir}")

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=get_train_transform(
            image_size=image_size,
            use_augmentation=use_augmentation,
        ),
    )

    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=get_eval_transform(image_size=image_size),
    )

    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=get_eval_transform(image_size=image_size),
    )

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(
    split_root,
    image_size=224,
    batch_size=32,
    num_workers=0,
    use_augmentation=True,
):
    """
    Create train, validation, and test dataloaders.

    num_workers=0 is safer on Windows and Jupyter Notebook.
    If running from a .py file, you can try num_workers=2 or 4.
    """
    train_dataset, val_dataset, test_dataset = get_imagefolder_datasets(
        split_root=split_root,
        image_size=image_size,
        use_augmentation=use_augmentation,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def print_dataset_info(train_dataset, val_dataset, test_dataset):
    """
    Print basic dataset information.
    """
    print("=" * 60)
    print("Dataset Information")
    print("=" * 60)

    print(f"Train images: {len(train_dataset)}")
    print(f"Val images:   {len(val_dataset)}")
    print(f"Test images:  {len(test_dataset)}")

    print()
    print("Class to index mapping:")
    print(train_dataset.class_to_idx)

    print()
    print("Classes:")
    print(train_dataset.classes)


def get_class_counts(dataset):
    """
    Count images in each class from an ImageFolder dataset.
    """
    class_counts = {class_name: 0 for class_name in dataset.classes}

    for _, label in dataset.samples:
        class_name = dataset.classes[label]
        class_counts[class_name] += 1

    return class_counts


