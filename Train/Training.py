"""Training loops for forward and inverse models."""

import torch
import Train.Losses as losses
from torch.utils.data import DataLoader


def train_forward_model(model, train_dataset, validation_dataset, settings):
    """Fit a model on (input, target) pairs using mean squared error.

    The caller places the model on the desired device and dtype. 
    """
    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError("Training and validation datasets must both be nonempty.")
    if settings.epochs < 1:
        raise ValueError("Expected at least one training epoch.")
    if settings.lr_mode not in ("fixed", "scheduled"):
        raise ValueError(f"Unsupported learning-rate mode: {settings.lr_mode!r}.")
    if settings.lr_mode == "scheduled" and (
        settings.lr_step_epochs < 1 or not 0 < settings.lr_decay_factor <= 1
    ):
        raise ValueError("Expected a positive step interval and a decay factor in (0, 1].")


    # Model parameters, new or pretrained model

    parameter = next(model.parameters())
    device = parameter.device
    dtype = parameter.dtype
    if settings.pretrained_checkpoint is not None:
        weights = torch.load(settings.pretrained_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(weights)

    # Data Loaders and batch size   

    train_loader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=settings.batch_size)

    #  Optimizer and learning rate

    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    scheduler = None
    if settings.lr_mode == "scheduled":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=settings.lr_step_epochs, gamma=settings.lr_decay_factor
        )

    # 

    def batch_mse(inputs, targets):
        targets = targets.to(parameter)
        predictions = model(inputs.to(parameter))
        batch_size = targets.shape[0]
        return losses.reconstruction_loss(predictions, targets), batch_size

    # Training

    history = {"train_mse": [], "validation_mse": [], "learning_rate": []}
    for _ in range(settings.epochs):
        model.train()
        train_error = torch.zeros((), device=device, dtype=dtype)
        train_count = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, count = batch_mse(inputs, targets)
            loss.backward()
            optimizer.step()
            train_error += loss.detach() * count
            train_count += count

        model.eval()
        validation_error = torch.zeros((), device=device, dtype=dtype)
        validation_count = 0
        with torch.no_grad():
            for inputs, targets in validation_loader:
                loss, count = batch_mse(inputs, targets)
                validation_error += loss * count
                validation_count += count

        history["train_mse"].append(train_error.item() / train_count)
        history["validation_mse"].append(validation_error.item() / validation_count)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step()

    return history



def train_cnf_model(nf_model, forward_model, train_dataset, validation_dataset, settings):
    
    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError("Training and validation datasets must both be nonempty.")
    if settings.epochs < 1:
        raise ValueError("Expected at least one training epoch.")
    if settings.lr_mode not in ("fixed", "scheduled"):
        raise ValueError(f"Unsupported learning-rate mode: {settings.lr_mode!r}.")
    if settings.lr_mode == "scheduled" and (
        settings.lr_step_epochs < 1 or not 0 < settings.lr_decay_factor <= 1
    ):
        raise ValueError("Expected a positive step interval and a decay factor in (0, 1].")



    # Model parameters, new or pretrained model

    parameter = next(nf_model.parameters())
    device = parameter.device
    dtype = parameter.dtype
    if settings.pretrained_checkpoint is not None:
        weights = torch.load(settings.pretrained_checkpoint, map_location="cpu", weights_only=True)
        nf_model.load_state_dict(weights)

    # Data Loaders and batch size   

    train_loader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=settings.batch_size)

    #  Optimizer and learning rate

    optimizer = torch.optim.Adam(
        nf_model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    scheduler = None
    if settings.lr_mode == "scheduled":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=settings.lr_step_epochs, gamma=settings.lr_decay_factor
        )

    # Reconstruction Samples
    input_sample, _ = train_dataset[0]
    input_shape = input_sample.shape

    
    z_reconstruct = torch.randn(
    (settings.batch_size, *input_shape), device=device, dtype=dtype)

    z_reconstruct_validation = torch.randn((len(validation_dataset), *input_shape), 
                                          device=device, dtype=dtype)

    # Training

    history = {
        "train_loss": [], "train_nll_loss": [],
        "train_reconstruction_loss": [], "train_hybrid_loss": [],
        "validation_loss": [], "validation_nll_loss": [],
        "validation_reconstruction_loss": [], "validation_hybrid_loss": [],
        "learning_rate": [],
    }

    for epoch in range(settings.epochs):
        use_hybrid = settings.use_hybrid_loss and epoch >= settings.reconstruction_start_epoch

        nf_model.train()
        train_error = torch.zeros((), device=device, dtype=dtype)
        train_nll = torch.zeros_like(train_error)
        train_reconstruction = torch.zeros_like(train_error)
        count_train = 0
        for inputs, targets in train_loader:
            batch_size = inputs.shape[0]
            optimizer.zero_grad(set_to_none=True)
            conditions = [targets] * nf_model.NBlocks
            z, jacobian, _, _ = nf_model(inputs, conditions, isInverse=False)

            nll_loss_train = losses.negative_log_likelihood(z, jacobian)
            loss_train = nll_loss_train
            if use_hybrid:
                x_reconstruct, _, _, _ = nf_model(
                    z_reconstruct[:batch_size], conditions, isInverse=True
                )
                y_reconstruct = forward_model(x_reconstruct)
                reconstruction_loss_train = losses.reconstruction_loss(targets, y_reconstruct)
                loss_train, _ = losses.hybrid_flow_loss(
                    nll_loss_train, reconstruction_loss_train,
                    epsilon=settings.loss_balance_epsilon,
                    weight_bounds=settings.loss_balance_weight_bounds,
                )
                train_reconstruction += reconstruction_loss_train.detach() * batch_size

            loss_train.backward()
            optimizer.step()
            train_error += loss_train.detach() * batch_size
            train_nll += nll_loss_train.detach() * batch_size
            count_train += batch_size

        # Validation

        nf_model.eval()
        validation_error = torch.zeros((), device=device, dtype=dtype)
        validation_nll = torch.zeros_like(validation_error)
        validation_reconstruction = torch.zeros_like(validation_error)
        validation_count = 0
        with torch.no_grad():
            for inputs, targets in validation_loader:
                batch_size = inputs.shape[0]
                conditions = [targets] * nf_model.NBlocks
                z, jacobian, _, _ = nf_model(inputs, conditions, isInverse=False)
                nll_loss_validation = losses.negative_log_likelihood(z, jacobian)
                loss_validation = nll_loss_validation

                if use_hybrid:
                    latent = z_reconstruct_validation[validation_count:validation_count + batch_size]
                    x_reconstruct, _, _, _ = nf_model(latent, conditions, isInverse=True)
                    y_reconstruct = forward_model(x_reconstruct)
                    reconstruction_loss_validation = losses.reconstruction_loss(targets, y_reconstruct)
                    loss_validation, _ = losses.hybrid_flow_loss(
                        nll_loss_validation, reconstruction_loss_validation,
                        epsilon=settings.loss_balance_epsilon,
                        weight_bounds=settings.loss_balance_weight_bounds,
                    )
                    validation_reconstruction += reconstruction_loss_validation * batch_size

                validation_error += loss_validation * batch_size
                validation_nll += nll_loss_validation * batch_size
                validation_count += batch_size

        train_mean = train_error.item() / count_train
        validation_mean = validation_error.item() / validation_count
        history["train_loss"].append(train_mean)
        history["train_nll_loss"].append(train_nll.item() / count_train)
        history["train_reconstruction_loss"].append(
            train_reconstruction.item() / count_train if use_hybrid else None
        )
        history["train_hybrid_loss"].append(train_mean if use_hybrid else None)
        history["validation_loss"].append(validation_mean)
        history["validation_nll_loss"].append(validation_nll.item() / validation_count)
        history["validation_reconstruction_loss"].append(
            validation_reconstruction.item() / validation_count if use_hybrid else None
        )
        history["validation_hybrid_loss"].append(validation_mean if use_hybrid else None)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        if scheduler is not None:
            scheduler.step()

    return history