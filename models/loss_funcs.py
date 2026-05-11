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


class ChenOrdinalLoss(nn.Module):
    """
    Chen-style adjustable ordinal loss for KL grading.

    This loss uses an adjustable ordinal penalty matrix.
    Larger prediction distances receive higher penalties.

    The model still outputs 5-class logits.
    """

    def __init__(
        self,
        num_classes=5,
        penalty_distance=2,
        square_loss=True,
        normalize=False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.penalty_distance = penalty_distance
        self.square_loss = square_loss
        self.normalize = normalize

        penalty_matrix = self._create_penalty_matrix(
            num_classes=num_classes,
            penalty_distance=penalty_distance,
            normalize=normalize,
        )

        self.register_buffer("penalty_matrix", penalty_matrix)

    def _create_penalty_matrix(self, num_classes, penalty_distance, normalize):
        """
        Create Chen-style modified ordinal penalty matrix.

        Diagonal values are 0.
        Off-diagonal values increase with ordinal distance.

        Example with penalty_distance=2:
        distance 1 -> 3
        distance 2 -> 5
        distance 3 -> 7
        distance 4 -> 9

        This follows the simplified modified matrix idea:
        W_bar[i, j] = 0 if i == j
        W_bar[i, j] = distance * penalty_distance + 1 if i != j
        """
        class_indices = torch.arange(num_classes, dtype=torch.float32)

        distance_matrix = torch.abs(
            class_indices.unsqueeze(0) - class_indices.unsqueeze(1)
        )

        penalty_matrix = distance_matrix * penalty_distance + 1.0

        # Correct prediction should have zero penalty
        penalty_matrix[distance_matrix == 0] = 0.0

        if normalize:
            max_value = penalty_matrix.max()
            if max_value > 0:
                penalty_matrix = penalty_matrix / max_value

        return penalty_matrix

    def forward(self, logits, targets):
        """
        Args:
            logits: model outputs, shape [batch_size, num_classes]
            targets: true labels, shape [batch_size]

        Returns:
            ordinal loss
        """
        probabilities = F.softmax(logits, dim=1)

        # Select penalty row according to true labels
        # Shape: [batch_size, num_classes]
        target_penalties = self.penalty_matrix[targets]

        # Weighted sum of prediction probabilities and ordinal penalties
        loss_per_sample = torch.sum(
            probabilities * target_penalties,
            dim=1
        )

        if self.square_loss:
            loss_per_sample = loss_per_sample ** 2

        loss = torch.mean(loss_per_sample)

        return loss

class HybridChenOrdinalLoss(nn.Module):
    """
    Hybrid Chen-style ordinal loss.

    Loss = CrossEntropyLoss + alpha * Chen-style ordinal penalty

    CrossEntropyLoss keeps normal classification learning.
    Chen-style ordinal penalty discourages large KL-grade mistakes.
    """

    def __init__(
        self,
        num_classes=5,
        alpha=0.1,
        penalty_distance=2,
        square_loss=False,
        normalize=True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.alpha = alpha
        self.penalty_distance = penalty_distance
        self.square_loss = square_loss
        self.normalize = normalize

        self.ce_loss = nn.CrossEntropyLoss()

        penalty_matrix = self._create_penalty_matrix(
            num_classes=num_classes,
            penalty_distance=penalty_distance,
            normalize=normalize,
        )

        self.register_buffer("penalty_matrix", penalty_matrix)

    def _create_penalty_matrix(self, num_classes, penalty_distance, normalize):
        class_indices = torch.arange(num_classes, dtype=torch.float32)

        distance_matrix = torch.abs(
            class_indices.unsqueeze(0) - class_indices.unsqueeze(1)
        )

        # Chen-style modified penalty:
        # distance 1 -> 3, distance 2 -> 5, distance 3 -> 7, distance 4 -> 9
        penalty_matrix = distance_matrix * penalty_distance + 1.0

        # Correct prediction has zero ordinal penalty
        penalty_matrix[distance_matrix == 0] = 0.0

        if normalize:
            max_value = penalty_matrix.max()
            if max_value > 0:
                penalty_matrix = penalty_matrix / max_value

        return penalty_matrix

    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)

        probabilities = F.softmax(logits, dim=1)

        penalty_matrix = self.penalty_matrix.to(logits.device)
        targets = targets.long()

        target_penalties = penalty_matrix[targets]

        penalty_per_sample = torch.sum(
            probabilities * target_penalties,
            dim=1
        )

        if self.square_loss:
            penalty_per_sample = penalty_per_sample ** 2

        ordinal_penalty = torch.mean(penalty_per_sample)

        total_loss = ce + self.alpha * ordinal_penalty

        return total_loss

