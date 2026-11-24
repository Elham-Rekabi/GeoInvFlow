"""Latin hypercube datasets for the quadratic inverse experiment."""

import numpy as np
import torch
from scipy.stats import qmc
from torch.utils.data import TensorDataset

from Data_Generation.Toy_Examples.Toy_Functions import quadratic_function
from Experiments.Quadratic.Config import QuadraticConfig



def make_datasets(config: QuadraticConfig) -> tuple[TensorDataset, TensorDataset]:
    """Return the training and validation splits."""
    device = config.device
    dtype = config.dtype

    bounds = np.asarray(config.input_bounds, dtype=float)
    lower, upper = bounds.T

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    sampler = qmc.LatinHypercube(d=len(lower), rng=config.seed)

    def sample_dataset(sample_count: int) -> TensorDataset:
        samples = qmc.scale(sampler.random(n=sample_count), lower, upper)
        inputs = torch.as_tensor(samples, dtype=dtype, device=device)
        targets = quadratic_function(inputs[:, :1], inputs[:, 1:])
        return TensorDataset(inputs, targets)

    return sample_dataset(config.train_samples), sample_dataset(config.validation_samples)
