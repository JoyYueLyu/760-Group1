import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalCrossEntropyLoss(nn.Module):
    """
    Ordinal-aware loss for KL grade classification.
    1. Standard CrossEntropyLoss for 5-class classification.
    2. An ordinal penalty based on the distance between the expected predicted grade
       and the true grade.
    """

    def __init__(self, num_classes=5, alpha=0.2):
        super().__init__()

        self.num_classes = num_classes
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

        class_values = torch.arange(num_classes, dtype=torch.float32)
        self.register_buffer("class_values", class_values)

    def forward(self, logits, targets):
        """
        Args:
            logits: model outputs, shape [batch_size, num_classes]
            targets: true labels, shape [batch_size]

        Returns:
            total loss = CE loss + alpha * ordinal MSE loss
        """
        ce = self.ce_loss(logits, targets)

        probabilities = F.softmax(logits, dim=1)

        expected_grade = torch.sum(
            probabilities * self.class_values.to(logits.device),
            dim=1
        )

        targets_float = targets.float()

        ordinal_mse = torch.mean(
            (expected_grade - targets_float) ** 2
        )

        total_loss = ce + self.alpha * ordinal_mse

        return total_loss

