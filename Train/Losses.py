import torch


def negative_log_likelihood(z, jacobian):
    """Return Gaussian NLL per node, omitting the additive Gaussian constant.

    z has shape (B, C) or (B, N, C). jacobian has shape (B,) and contains
    each sample's total forward log-absolute-determinant.
    """
    node_count = z.shape[1] if z.dim() == 3 else 1
    latent_energy = 0.5 * z.square().flatten(start_dim=1).sum(dim=1)
    return (latent_energy - jacobian).mean() / node_count


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


def hybrid_flow_loss(nll, reconstruction, *, epsilon, weight_bounds):
    """Return the combined loss and the detached reconstruction weight."""
    min_weight, max_weight = weight_bounds
    if epsilon <= 0 or not 0 < min_weight <= max_weight:
        raise ValueError("Expected positive epsilon and ordered positive weight bounds.")

    nll_magnitude = nll.detach().abs().clamp_min(epsilon)
    reconstruction_magnitude = reconstruction.detach().clamp_min(epsilon)
    weight = (nll_magnitude / reconstruction_magnitude).clamp(min_weight, max_weight)

    return nll + weight * reconstruction, weight
