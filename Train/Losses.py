import torch


def negative_log_likelihood(z, jacobian):
    
    """
    z has shape (B, C) or (B, N, C).
    jacobian is the total forward log-absolute-determinant of the entire
    invertible network, summed across all coupling blocks, shape (batch_size,).
    It is not the Jacobian matrix itself.
    N is the node count for 3D z; otherwise N = 1.
    
    """
    N = z.shape[1] if z.dim() == 3 else 1
    loss_ml = torch.mean(z**2) / 2 - torch.mean(jacobian) / N

    return loss_ml


def reconstruction_loss(observed_outputs, predicted_outputs):
    """Compare observations with forward-model outputs from generated inputs.

    observed_outputs and predicted_outputs must have the same shape, e.g.
    (batch_size, output_features).
    The MSE averages over all samples and output features.
    predicted_outputs come from applying the forward model to inputs generated
    by the inverse flow. Keep gradients enabled through both operations.
    """
    mse_loss = torch.mean((observed_outputs - predicted_outputs)**2)
    return mse_loss



