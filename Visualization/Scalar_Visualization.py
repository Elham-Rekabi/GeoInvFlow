"""Visualization for scalar forward models."""

from io import BytesIO
from pathlib import Path

from matplotlib.colors import PowerNorm
import matplotlib.pyplot as plt
import numpy as np
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


def _inverse_surface_grid(input_bounds, grid_size, parameter):
    return torch.meshgrid(
        *(torch.linspace(lo, hi, grid_size, device=parameter.device, dtype=parameter.dtype)
          for lo, hi in input_bounds),
        indexing="xy",
    )


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
    x1, x2 = _inverse_surface_grid(input_bounds, grid_size, parameter)
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
    sample_count=500, seed=42, frame_ms=200, grid_size=100,
):
    """Save a looping GIF with fixed axes and the same latent draws in every frame."""
    if frame_ms < 1:
        raise ValueError("Expected a positive frame duration.")
    observations = tuple(float(value) for value in observations)
    if not observations:
        raise ValueError("Provide at least one observation to animate.")

    if grid_size < 2:
        raise ValueError("Expected a grid size of at least two.")

    parameter = next(cnf_model.parameters())
    with torch.inference_mode():
        grid = _inverse_surface_grid(input_bounds, grid_size, parameter)
        surface = forward_function(*grid)
        lower, upper = surface.amin().item(), surface.amax().item()
    for observation in observations:
        if not lower <= observation <= upper:
            raise ValueError(
                f"Observation y={observation:g} is outside the plotted forward surface "
                f"range [{lower:.9g}, {upper:.9g}] within input_bounds."
            )

    output_limits = (lower, upper)

    def make_frame(observation):
        return plot_conditional_inverse(
            cnf_model, forward_function, observation, input_bounds,
            sample_count=sample_count, seed=seed, grid_size=grid_size,
        )

    # Collect the autoscaled ranges over the entire sweep before rendering any frames.
    panel_limits = [{}, {}, {}]
    for observation in observations:
        figure = make_frame(observation)
        try:
            for axis, limits in zip(figure.axes[:3], panel_limits):
                for name in ("xlim", "ylim"):
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

    panel_limits[2]["zlim"] = output_limits

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





def plot_conditional_distribution(
    models, forward_function, observation, input_bounds, *,
    half_width=0.01, reference_count=2_000_000, sample_count=50_000, bins=80, seed=42,
):
    """Compare probability per cell under a uniform input prior and saved flows.

    The reference conditions on |f(x) - observation| <= half_width to approximate
    the singular conditional on the exact level set. Flow draws use the exact
    observation. Cell probabilities retain the full sample count as denominator.
    """
    reference, _ = _conditional_reference(
        forward_function, observation, input_bounds, half_width, reference_count, seed,
    )
    lower, upper = np.asarray(input_bounds).T
    panels, edges, masses = _conditional_histograms(
        models, observation, input_bounds, half_width, reference, sample_count, bins, seed,
    )
    max_mass = max(mass.max() for mass in masses)
    x1, x2 = np.meshgrid(*(np.linspace(lo, hi, 401) for lo, hi in input_bounds))
    surface = forward_function(x1, x2)

    figure, axes = plt.subplots(
        1, len(panels), figsize=(4.3 * len(panels), 4.5),
        sharex=True, sharey=True, squeeze=False, layout="constrained",
    )
    figure.suptitle(f"Conditional distribution at y = {observation:g}")
    for axis, (label, samples), mass in zip(axes[0], panels, masses):
        mesh = axis.pcolormesh(*edges, mass.T, cmap="viridis", vmin=0, vmax=max_mass)
        if surface.min() < observation < surface.max():
            axis.contour(x1, x2, surface, levels=[observation], colors="white",
                         linestyles="dashed", linewidths=0.8, alpha=0.8)
        inside = np.all((samples >= lower) & (samples <= upper), axis=1).mean()
        axis.set(title=f"{label}\n{len(samples):,} draws · {1 - inside:.1%} outside bounds",
                 xlabel="x₁", aspect="equal")
    axes[0, 0].set_ylabel("x₂")
    figure.colorbar(mesh, ax=axes[0].tolist(), label="Probability per cell")
    return figure


def plot_conditional_density(
    models, forward_function, observation, input_bounds, *,
    half_width=0.01, reference_count=2_000_000, grid_size=501, seed=42, cmap="magma", include_histogram=False,
):
    """Evaluate flow PDFs and the finite-band reference PDF on the input plane."""
    reference, reference_area = _conditional_reference(
        forward_function, observation, input_bounds, half_width, reference_count, seed,
    )
    reference_density = 1 / reference_area

    x1, x2 = np.meshgrid(*(np.linspace(lo, hi, grid_size) for lo, hi in input_bounds))
    grid = torch.from_numpy(np.column_stack((x1.ravel(), x2.ravel())))
    densities = {}
    with torch.inference_mode():
        for label, model in models.items():
            parameter = next(model.parameters())
            log_densities = []
            for chunk in grid.split(8192):
                points = chunk.to(parameter)
                condition = points.new_full((len(points), 1), observation)
                latent, logdet, _, _ = model(points, [condition] * model.NBlocks)
                log_densities.append(logdet - 0.5 * latent.square().sum(dim=1) - np.log(2 * np.pi))
            densities[label] = torch.cat(log_densities).exp().cpu().numpy().reshape(x1.shape)

    peaks = [reference_density] + [density.max() for density in densities.values()]
    if include_histogram:
        _, edges, masses = _conditional_histograms(
            models, observation, input_bounds, half_width, reference, 50_000, 80, seed,
        )
        cell_areas = np.outer(np.diff(edges[0]), np.diff(edges[1]))
        histograms = [mass / cell_areas for mass in masses]
        peaks.extend(histogram.max() for histogram in histograms)
    max_density = max(peaks)
    # Spread out low densities while retaining a common scale in PDF units.
    norm = PowerNorm(gamma=0.35, vmin=0, vmax=max_density)
    cmap = plt.get_cmap(cmap)
    surface = forward_function(x1, x2)
    figure, axes = plt.subplots(
        2 if include_histogram else 1, 1 + len(densities),
        figsize=(4.3 * (1 + len(densities)), 8 if include_histogram else 4.5),
        sharex=True, sharey=True, squeeze=False, layout="constrained",
    )
    title = "Conditional distribution and density" if include_histogram else "Conditional probability density"
    figure.suptitle(f"{title} at y = {observation:g}")
    reference_axis = axes[0, 0]
    reference_axis.set_facecolor(cmap(norm(0)))
    reference_axis.contourf(
        x1, x2, surface, levels=[observation - half_width, observation + half_width],
        colors=[cmap(norm(reference_density))],
    )
    reference_axis.set(title=f"Reference · |f(x) − y| ≤ {half_width:g}", ylabel="x₂")
    for axis, (label, density) in zip(axes[0, 1:], densities.items()):
        axis.pcolormesh(x1, x2, density, shading="nearest", cmap=cmap, norm=norm)
        axis.set_title(label)

    if include_histogram:
        reference_axis.set_ylabel("Direct PDF\nx₂")
        axes[1, 0].set_ylabel("Sample histogram\nx₂")
        for axis, histogram in zip(axes[1], histograms):
            axis.pcolormesh(*edges, histogram.T, cmap=cmap, norm=norm)

    for axis in axes.flat:
        if surface.min() < observation < surface.max():
            axis.contour(x1, x2, surface, levels=[observation], colors="white",
                         linestyles="dashed", linewidths=0.6, alpha=0.65)
        axis.set(aspect="equal", xlim=input_bounds[0], ylim=input_bounds[1])
    for axis in axes[-1]:
        axis.set_xlabel("x₁")
    figure.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.ravel().tolist(),
        label="Probability density per unit area (power scale)",
    )
    return figure



def _conditional_reference(forward_function, observation, input_bounds, half_width, reference_count, seed):
    if not np.isfinite(observation) or not np.isfinite(half_width) or half_width <= 0:
        raise ValueError("Expected a finite observation and a positive finite half-width.")

    lower, upper = np.asarray(input_bounds).T
    candidates = np.random.default_rng(seed).uniform(lower, upper, (reference_count, 2))
    reference = candidates[np.abs(forward_function(*candidates.T) - observation) <= half_width]
    if len(reference) == 0:
        raise ValueError("No reference samples in the observation band. Check y or increase the half-width.")
    return reference, np.prod(upper - lower) * len(reference) / reference_count


def _conditional_histograms(models, observation, input_bounds, half_width, reference, sample_count, bins, seed):
    panels = [(f"Reference · |f(x) − y| ≤ {half_width:g}", reference)]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latent = torch.randn(sample_count, 2, generator=generator)
    with torch.inference_mode():
        for label, model in models.items():
            parameter = next(model.parameters())
            condition = parameter.new_full((sample_count, 1), observation)
            decoded, _, _, _ = model(latent.to(parameter), [condition] * model.NBlocks, isInverse=True)
            panels.append((label, decoded.cpu().numpy()))

    edges = [np.linspace(lo, hi, bins + 1) for lo, hi in input_bounds]
    masses = [np.histogram2d(*samples.T, bins=edges)[0] / len(samples) for _, samples in panels]
    return panels, edges, masses