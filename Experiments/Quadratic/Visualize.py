"""Animate saved quadratic CNFs with python -m Experiments.Quadratic.Visualize."""

from pathlib import Path

import numpy as np
import torch

from Data_Generation.Toy_Examples.Toy_Functions import quadratic_function
from Experiments.Quadratic.Config import QuadraticConfig
from Experiments.Quadratic.Model_Setup import build_cnf
from Visualization.Scalar_Visualization import animate_conditional_inverse


def visualize_saved_model(config: QuadraticConfig, checkpoint_path, observations):
    """Load a CNF checkpoint on CPU and save the animation beside it."""
    checkpoint_path = Path(checkpoint_path)
    model = build_cnf(config, config.input_shape[0], config.output_shape[0]).to(dtype=config.dtype)
    weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    model.eval()

    return animate_conditional_inverse(
        model, quadratic_function, observations, config.input_bounds,
        checkpoint_path.with_name("conditional_inverse.gif"), seed=config.seed,
    )


if __name__ == "__main__":
    config = QuadraticConfig()
    observations = np.linspace(0.25, 4.0, 25)
    output_dir = Path(__file__).resolve().parent / "outputs"
    for mode in ("nll", "hybrid"):
        path = visualize_saved_model(config, output_dir / mode / "cnf_model.pt", observations)
        print(f"Saved {path}", flush=True)
