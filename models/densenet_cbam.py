import torch
import torch.nn as nn
from torchvision import models


class ChannelAttention(nn.Module):
    """
    Channel Attention Module.

    It learns which feature channels are important.
    """

    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()

        hidden_channels = max(in_channels // reduction_ratio, 1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        max_pool = torch.amax(x, dim=(2, 3), keepdim=True)

        avg_out = self.shared_mlp(avg_pool)
        max_out = self.shared_mlp(max_pool)

        attention = self.sigmoid(avg_out + max_out)

        return x * attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.

    It learns where the important image regions are.
    """

    def __init__(self, kernel_size=7):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)

        pooled = torch.cat([avg_pool, max_pool], dim=1)

        attention = self.sigmoid(self.conv(pooled))

        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Channel attention first, then spatial attention.
    """

    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super().__init__()

        self.channel_attention = ChannelAttention(
            in_channels=in_channels,
            reduction_ratio=reduction_ratio,
        )

        self.spatial_attention = SpatialAttention(
            kernel_size=spatial_kernel_size,
        )

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class DenseNet121CBAM(nn.Module):
    """
    DenseNet121 with one CBAM module after the final feature block.

    This is a lightweight CBAM experiment:
    - DenseNet121 pretrained backbone
    - CBAM after model.features
    - classifier head for 5 KL grades
    """

    def __init__(
        self,
        num_classes=5,
        pretrained=True,
        dropout=0.3,
        cbam_reduction_ratio=16,
        cbam_spatial_kernel_size=7,
    ):
        super().__init__()

        if pretrained:
            weights = models.DenseNet121_Weights.IMAGENET1K_V1
        else:
            weights = None

        base_model = models.densenet121(weights=weights)

        self.features = base_model.features

        # DenseNet121 final feature channels = 1024
        self.cbam = CBAM(
            in_channels=1024,
            reduction_ratio=cbam_reduction_ratio,
            spatial_kernel_size=cbam_spatial_kernel_size,
        )

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.relu(x)

        x = self.cbam(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x


def get_densenet121_cbam(
    num_classes=5,
    pretrained=True,
    dropout=0.3,
    cbam_reduction_ratio=16,
    cbam_spatial_kernel_size=7,
):
    model = DenseNet121CBAM(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        cbam_reduction_ratio=cbam_reduction_ratio,
        cbam_spatial_kernel_size=cbam_spatial_kernel_size,
    )

    return model


def count_trainable_parameters(model):
    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    return trainable_params, total_params

