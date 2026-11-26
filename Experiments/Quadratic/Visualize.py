"""Visualize saved quadratic CNFs with python -m Experiments.Quadratic.Visualize."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from Data_Generation.Toy_Examples.Toy_Functions import quadratic_function
from Experiments.Quadratic.Config import QuadraticConfig
from Experiments.Quadratic.Model_Setup import build_cnf
from Visualization.Scalar_Visualization import (
    animate_conditional_inverse,
    plot_conditional_density,
    plot_conditional_distribution,
)


def _load_saved_model(config, checkpoint_path):
    model = build_cnf(config, config.input_shape[0], config.output_shape[0]).to(dtype=config.dtype)
    weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    return model.eval()


def visualize_saved_model(config: QuadraticConfig, checkpoint_path, observations):
    """Load a CNF checkpoint on CPU and save the animation beside it."""
    checkpoint_path = Path(checkpoint_path)
    model = _load_saved_model(config, checkpoint_path)
    return animate_conditional_inverse(
        model, quadratic_function, observations, config.input_bounds,
        checkpoint_path.with_name("conditional_inverse.gif"), seed=config.seed,
    )


def main():
    config = QuadraticConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    plots = parser.add_mutually_exclusive_group()
    plots.add_argument("--density", action="store_true", help="Plot conditional PDFs for the reference and saved flows.")
    plots.add_argument("--conditional", action="store_true", help="Compare the reference and both saved flows.")
    plots.add_argument("--compare", action="store_true", help="Show the sample histogram and direct PDF in one figure (default).")
    plots.add_argument("--animate", action="store_true", help="Generate the conditional inverse GIFs.")
    parser.add_argument("--y", type=float, help="Override the configured observations with one value.")
    parser.add_argument("--half-width", type=float, default=0.01, help="Reference observation band half-width.")
    parser.add_argument("--cmap", choices=("magma", "jet"), default="magma", help="Color palette for density plots.")
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parent / "outputs"

    if not args.animate:
        models = {
            "NLL": _load_saved_model(config, output_dir / "nll" / "cnf_model.pt"),
            "Hybrid": _load_saved_model(config, output_dir / "hybrid" / "cnf_model.pt"),
        }
        observations = config.conditional_plot_observations if args.y is None else (args.y,)
        if args.conditional:
            quantity, plot, options = "distribution", plot_conditional_distribution, {}
            suffix = ""
        else:
            quantity = "density" if args.density else "density_comparison"
            plot = plot_conditional_density
            options = {"cmap": args.cmap, "include_histogram": not args.density}
            suffix = "_jet" if args.cmap == "jet" else ""

        for observation in observations:
            figure = plot(
                models, quadratic_function, observation, config.input_bounds,
                half_width=args.half_width, seed=config.seed, **options,
            )
            prefix = quantity if quantity == "density_comparison" else f"conditional_{quantity}"
            path = output_dir / f"{prefix}_y{observation:g}{suffix}.png"
            try:
                figure.savefig(path, dpi=160)
            finally:
                plt.close(figure)
            print(f"Saved {path}", flush=True)
        return

    # Use smaller observation steps near the minimum and the highest level sets.
    observations = np.concatenate((
        np.linspace(0.1, 0.5, 10, endpoint=False),
        np.linspace(0.5, 3.0, 20, endpoint=False),
        np.linspace(3.0, 4.0, 10),
    ))
    for mode in ("nll", "hybrid"):
        path = visualize_saved_model(config, output_dir / mode / "cnf_model.pt", observations)
        print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
