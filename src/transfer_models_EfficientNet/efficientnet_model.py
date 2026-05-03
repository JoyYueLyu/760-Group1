import torch.nn as nn
from torchvision import models

from config import NUM_CLASSES


def get_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=True, fine_tune_last_block=True):
    """
    EfficientNet-B0 transfer learning model for 5-class KL grading.
    This is separate from baseline CNN code.
    """

    try:
        from torchvision.models import EfficientNet_B0_Weights
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
    except Exception:
        model = models.efficientnet_b0(pretrained=pretrained)

    # Freeze all pretrained layers first
    for param in model.parameters():
        param.requires_grad = False

    # Replace final classification layer
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # Train final layer
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Fine-tune last EfficientNet block
    if fine_tune_last_block:
        for param in model.features[-1].parameters():
            param.requires_grad = True

    return model
