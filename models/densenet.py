import torch.nn as nn
from torchvision import models


def get_densenet121(
    num_classes=5,
    pretrained=True,
    dropout=0.3,
):
    """
    Create DenseNet121 for 5-class KL grade classification.

    Args:
        num_classes: number of output classes.
        pretrained: whether to use ImageNet pretrained weights.
        dropout: dropout rate in the classifier head.

    Returns:
        DenseNet121 model.
    """
    if pretrained:
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = models.densenet121(weights=weights)

    in_features = model.classifier.in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )

    return model


def freeze_features(model):
    """
    Freeze DenseNet feature extractor.
    Only classifier head will be trainable.
    """
    for param in model.features.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def unfreeze_all(model):
    """
    Unfreeze all layers for full fine-tuning.
    """
    for param in model.parameters():
        param.requires_grad = True

    return model


def count_trainable_parameters(model):
    """
    Count trainable and total parameters.
    """
    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    return trainable_params, total_params

