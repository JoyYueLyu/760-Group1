from torchvision import transforms
from torchvision.transforms import InterpolationMode

from config import IMAGE_SIZE


def get_train_transform():
    """
    Transform for training set.
    Data augmentation is only applied to the training set.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        # Mild augmentation for medical X-ray images
        transforms.RandomRotation(
            degrees=5,
            interpolation=InterpolationMode.BILINEAR,
            fill=0
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.05,
            contrast=0.05
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_eval_transform():
    """
    Transform for validation and test sets.
    No random augmentation is applied here.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])