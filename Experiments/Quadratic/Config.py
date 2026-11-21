"""Settings for the quadratic inverse experiment."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class TrainingConfig:
    """Adam settings for one model."""

    epochs: int = 1000
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6

    # Load this model state dict before starting fine-tuning. None starts a fresh model.
    pretrained_checkpoint: str | None = None

    # The rate is constant in fixed mode and decays by steps in scheduled mode.
    lr_mode: Literal["fixed", "scheduled"] = "fixed"
    lr_step_epochs: int = 100
    lr_decay_factor: float = 0.5

    # Use only NLL throughout inverse training when this is False.
    use_hybrid_loss: bool = True

    # Use NLL alone until this zero-based inverse-training epoch.
    # A value of 0 enables reconstruction loss from the first epoch.
    reconstruction_start_epoch: int = 50

    # Balance the reconstruction loss against the NLL magnitude each batch.
    loss_balance_epsilon: float = 1e-6
    loss_balance_weight_bounds: tuple[float, float] = (1e-3, 100.0)






@dataclass(frozen=True)
class QuadraticConfig:


    seed: int = 42
    device: Literal["auto", "cpu", "cuda"] = "auto"
    dtype: Literal["float32", "float64"] = "float32"

    # Per-sample shapes, excluding the batch dimension.
    input_shape: tuple[int, ...] = (2,)
    output_shape: tuple[int, ...] = (1,)


    # Samples for forward and inverse model
    train_samples: int = 512
    validation_samples: int = 128

    # Forward MLP with one activation per hidden layer and a linear output.
    forward_model_hidden_layers: tuple[int, ...] = (10, 20, 10)
    forward_model_activations: tuple[Literal["LeakyReLU", "ReLU", "Tanh", "Sigmoid", "Identity"],
                                      ...] = ("Tanh", "Tanh", "LeakyReLU")

    # Conditional flow architecture.
    cnf_num_blocks: int = 4
    cnf_hidden_layers: tuple[int, ...] = (10, 50, 10)
    cnf_activation: tuple[Literal["LeakyReLU", "ReLU", "Tanh", "Sigmoid", "Identity"],
                           ...] = ("LeakyReLU", "LeakyReLU", "LeakyReLU", "LeakyReLU")

    # Adam settings for the two training stages.
    forward_training: TrainingConfig = field(default_factory=TrainingConfig)
    inverse_training: TrainingConfig = field(default_factory=TrainingConfig)

    
