import torch.nn as nn
from torchvision import models

from config import NUM_CLASSES


def get_resnet18(num_classes=NUM_CLASSES, pretrained=True, fine_tune_last_block=True):
    """
    ResNet18 transfer learning model for 5-class KL grading.
    This is separate from baseline CNN code.
    """

    try:
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=pretrained)

    # Freeze all pretrained layers first
    for param in model.parameters():
        param.requires_grad = False

    # Replace final classification layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Train final layer
    for param in model.fc.parameters():
        param.requires_grad = True

    # Fine-tune last ResNet block
    if fine_tune_last_block:
        for param in model.layer4.parameters():
            param.requires_grad = True

    return model