import torch.nn as nn
from torchvision import models

from src.config import NUM_CLASSES


def get_densenet121(num_classes=NUM_CLASSES, pretrained=True, fine_tune_last_block=True):
    """
    DenseNet121 transfer learning model for 5-class KL grading.
    """

    try:
        from torchvision.models import DenseNet121_Weights
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
    except Exception:
        model = models.densenet121(pretrained=pretrained)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    for param in model.classifier.parameters():
        param.requires_grad = True

    if fine_tune_last_block:
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True
        for param in model.features.norm5.parameters():
            param.requires_grad = True

    return model
