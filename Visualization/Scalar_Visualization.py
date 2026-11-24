"""Visualization for scalar forward models."""

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
import torch


def plot_forward_surface(model, train_data, validation_data, input_bounds, *, grid_size=60):
    """Plot predictions from an eval-mode model and observed targets from both splits.

    The model accepts inputs shaped (N, 2) and returns one scalar per input.
    The datasets expose their input and target tensors through `TensorDataset.tensors`.
    """
    parameter = next(model.parameters())
    coordinates = [
        torch.linspace(lower, upper, grid_size, device=parameter.device, dtype=parameter.dtype)
        for lower, upper in input_bounds
    ]
    x1, x2 = torch.meshgrid(*coordinates, indexing="xy")
    grid = torch.stack((x1.reshape(-1), x2.reshape(-1)), dim=1)
    with torch.inference_mode():
        predicted = model(grid).reshape_as(x1).cpu().numpy()

    figure, axis = plt.subplots(
        figsize=(8, 6), subplot_kw={"projection": "3d"}, layout="constrained"
    )
    axis.plot_surface(
        x1.cpu().numpy(), x2.cpu().numpy(), predicted,
        color="steelblue", alpha=0.35, label="Predicted surface",
    )
    for dataset, label, color, marker in (
        (train_data, "Training targets", "tab:orange", "o"),
        (validation_data, "Validation targets", "tab:purple", "^"),
    ):
        inputs, targets = (tensor.detach().cpu().numpy() for tensor in dataset.tensors)
        axis.scatter(
            inputs[:, 0], inputs[:, 1], targets.ravel(), label=label,
            color=color, marker=marker, s=14, depthshade=False,
        )
    axis.set(xlabel="x₁", ylabel="x₂", zlabel="Output", title="Forward-model predictions")
    axis.legend()
    return figure


def plot_conditional_inverse(
    cnf_model, forward_function, observation, input_bounds, *,
    sample_count=500, seed=42, grid_size=100,
):
    """Visualize one scalar condition using an eval-mode two-input CNF.

    forward_function takes the two coordinate tensors and returns their true outputs.
    Colors compare the true output errors before and after decoding.
    """
    if sample_count < 1 or grid_size < 2:
        raise ValueError("Expected at least one sample and a grid size of at least two.")

    parameter = next(cnf_model.parameters())
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latent = torch.randn(sample_count, 2, generator=generator, device="cpu").to(parameter)
    condition = latent.new_full((sample_count, 1), observation)
    x1, x2 = torch.meshgrid(
        *(torch.linspace(lo, hi, grid_size, device=parameter.device, dtype=parameter.dtype)
          for lo, hi in input_bounds),
        indexing="xy",
    )
    with torch.inference_mode():
        decoded, _, _, _ = cnf_model(latent, [condition] * cnf_model.NBlocks, isInverse=True)
        outputs = forward_function(decoded[:, 0], decoded[:, 1])
        surface = forward_function(x1, x2)
        latent_outputs = forward_function(latent[:, 0], latent[:, 1])
        latent_error = (latent_outputs - observation).abs().cpu().numpy()
        decoded_error = (outputs - observation).abs().cpu().numpy()

    latent_style = dict(c=latent_error, cmap="jet", s=12, alpha=0.8)
    decoded_style = dict(
        c=decoded_error, cmap="jet", s=12, alpha=0.8,
        vmin=float(decoded_error.min()), vmax=float(decoded_error.max()),
    )
    latent, decoded, outputs = (tensor.cpu().numpy() for tensor in (latent, decoded, outputs))
    x1, x2, surface = (tensor.cpu().numpy() for tensor in (x1, x2, surface))

    figure = plt.figure(figsize=(15, 5), layout="constrained")
    figure.suptitle(f"Conditional inverse at y = {observation:g} · {sample_count} samples")
    latent_axis = figure.add_subplot(131)
    latent_points = latent_axis.scatter(*latent.T, **latent_style)
    latent_axis.set(
        title=f"Raw Gaussian baseline · MAE {latent_error.mean():.3g}",
        xlabel="z₁", ylabel="z₂", aspect="equal",
    )

    input_axis = figure.add_subplot(132)
    decoded_points = input_axis.scatter(*decoded.T, **decoded_style)
    input_axis.set(title=f"Decoded inputs · MAE {decoded_error.mean():.3g}",
                   xlabel="x₁", ylabel="x₂", aspect="equal")

    surface_axis = figure.add_subplot(133, projection="3d")
    surface_axis.plot_surface(x1, x2, surface, color="steelblue", alpha=0.23)
    surface_axis.scatter(*decoded.T, outputs, depthshade=False, **decoded_style)
    surface_axis.set(title="Decoded inputs on the true forward surface",
                     xlabel="x₁", ylabel="x₂", zlabel="y")

    if surface.min() < observation < surface.max():
        level = input_axis.contour(x1, x2, surface, levels=[observation], colors="black")
        reference_lines, _ = level.legend_elements()
        input_axis.legend(reference_lines, ["True level set"])
        for segment in level.allsegs[0]:
            surface_axis.plot(segment[:, 0], segment[:, 1], observation, color="black")
    figure.colorbar(
        latent_points, ax=latent_axis, shrink=0.8, label="Raw Gaussian absolute output error"
    )
    figure.colorbar(
        decoded_points, ax=[input_axis, surface_axis], shrink=0.8, label="Decoded absolute output error"
    )
    return figure


def animate_conditional_inverse(
    cnf_model, forward_function, observations, input_bounds, output_path, *,
    sample_count=500, seed=42, frame_ms=200,
):
    """Save a looping GIF with fixed axes and the same latent draws in every frame."""
    if frame_ms < 1:
        raise ValueError("Expected a positive frame duration.")
    observations = tuple(observations)
    if not observations:
        raise ValueError("Provide at least one observation to animate.")

    def make_frame(observation):
        return plot_conditional_inverse(
            cnf_model, forward_function, float(observation), input_bounds,
            sample_count=sample_count, seed=seed,
        )

    # Collect the autoscaled ranges over the entire sweep before rendering any frames.
    axis_names = ("xlim", "ylim", "zlim")
    panel_limits = [{}, {}, {}]
    for observation in observations:
        figure = make_frame(observation)
        try:
            for axis, limits in zip(figure.axes[:3], panel_limits):
                for name in axis_names[:3 if axis.name == "3d" else 2]:
                    lower, upper = getattr(axis, f"get_{name}")()
                    previous_lower, previous_upper = limits.get(name, (lower, upper))
                    limits[name] = (min(lower, previous_lower), max(upper, previous_upper))
        finally:
            plt.close(figure)

    for limits in panel_limits:
        for name, fraction in (("xlim", 0.2), ("ylim", 0.1)):
            lower, upper = limits[name]
            padding = fraction * (upper - lower)
            limits[name] = (lower - padding, upper + padding)

    frames = []
    positions = None
    for observation in observations:
        figure = make_frame(observation)
        try:
            for axis, limits in zip(figure.axes[:3], panel_limits):
                axis.set(**limits)
            if positions is None:
                figure.canvas.draw()
                positions = [axis.get_position(original=True).frozen() for axis in figure.axes]
            figure.set_layout_engine("none")
            for axis, position in zip(figure.axes, positions):
                axis.set_position(position)
            with BytesIO() as buffer:
                figure.savefig(buffer, format="png", dpi=100)
                buffer.seek(0)
                with Image.open(buffer) as image:
                    frames.append(image.convert("RGB"))
        finally:
            plt.close(figure)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path, format="GIF", save_all=True, append_images=frames[1:],
        duration=frame_ms, loop=0,
    )
    return output_path
