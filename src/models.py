import torch
import torch.nn as nn

from config import NUM_CLASSES


class BaselineCNN(nn.Module):
    """
    Simple CNN baseline for knee OA KL grade classification.

    Input:
        [batch_size, 3, 224, 224]

    Output:
        [batch_size, 5]
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super(BaselineCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 3 -> 16
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 2: 16 -> 32
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 3: 32 -> 64
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 4: 64 -> 128
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Convert feature map to fixed-size vector
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = BaselineCNN()

    dummy_input = torch.randn(16, 3, 224, 224)
    output = model(dummy_input)

    print("=" * 50)
    print("Baseline CNN Test")
    print("=" * 50)
    print(model)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)