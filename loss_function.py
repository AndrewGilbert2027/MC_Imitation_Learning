import torch
import torch.nn.functional as F

def combined_loss(predictions, targets, binary_indices=list(range(11)), alpha=1.0):
    """
    Computes the combined loss for binary values (including camera directions).
    
    Args:
        predictions: Model outputs (batch_size, feature_vector_size).
        targets: Ground truth values (batch_size, feature_vector_size).
        binary_indices: Indices of binary values in the feature vector. [0-10]
        alpha: Weight for additional loss components (if needed).
    
    Returns:
        Total loss (scalar).
    """
    # Binary Cross-Entropy Loss for all binary values
    binary_preds = predictions[:, binary_indices]
    binary_targets = targets[:, binary_indices]
    binary_loss = F.binary_cross_entropy_with_logits(binary_preds, binary_targets)

    # Combine losses (only binary loss in this case)
    total_loss = binary_loss
    return total_loss
