import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalAwareWeightedLoss(nn.Module):
    """
    Ordinal-aware weighted loss for KL grade classification.

    This loss combines:
    1. Cross-entropy loss
    2. Optional class weights for class imbalance
    3. Ordinal distance penalty

    The ordinal penalty is based on the expected distance between
    the predicted probability distribution and the true class.

    Example:
        true label = 1
        predicting class 2 is less severe than predicting class 4
    """

    def __init__(self, num_classes, class_weights=None, alpha=1.0):
        super(OrdinalAwareWeightedLoss, self).__init__()

        self.num_classes = num_classes
        self.class_weights = class_weights
        self.alpha = alpha

    def forward(self, logits, targets):
        """
        Args:
            logits: model outputs, shape [batch_size, num_classes]
            targets: true labels, shape [batch_size]

        Returns:
            scalar loss
        """

        # Standard cross-entropy per sample
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            reduction="none"
        )

        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=1)

        # Class index tensor: [0, 1, 2, 3, 4]
        class_indices = torch.arange(
            self.num_classes,
            device=logits.device
        ).float()

        # Shape: [1, num_classes]
        class_indices = class_indices.unsqueeze(0)

        # Shape: [batch_size, 1]
        targets_float = targets.float().unsqueeze(1)

        # Ordinal distance between each class and the true label
        # Normalized by num_classes - 1
        distances = torch.abs(class_indices - targets_float)
        distances = distances / (self.num_classes - 1)

        # Expected ordinal distance under predicted probability distribution
        expected_distance = torch.sum(probs * distances, dim=1)

        # Final loss:
        # CE loss is increased when probability is assigned to far-away classes
        loss = ce_loss * (1.0 + self.alpha * expected_distance)

        return loss.mean()