from torchvision import transforms

from config import IMAGE_SIZE


def get_train_transform():
    """
    Transform for training set.
    Data augmentation is only applied to the training set.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        # Small augmentation for medical images
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(p=0.5),

        transforms.ToTensor(),

        # ImageNet normalization.
        # Useful for pretrained models such as ResNet, EfficientNet, DenseNet.
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_eval_transform():
    """
    Transform for validation and test sets.
    No random augmentation here.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])