"""Live plots of training and validation losses."""

from time import monotonic

import matplotlib.pyplot as plt


def loss_figure(metrics, live_plot, *, preview_path=None):
    """Create a loss figure and a callback that updates it from epoch history."""
    figure, axes = plt.subplots(
        len(metrics), 1, sharex=True, squeeze=False,
        figsize=(8, 2.8 * len(metrics)), layout="constrained")


    curves = []
    for axis, (metric, label) in zip(axes[:, 0], metrics):
        for split in ("train", "validation"):
            line, = axis.plot([], [], label=split.capitalize())
            curves.append((line, f"{split}_{metric}"))
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")
    axes[-1, 0].set_xlabel("Epoch")

    show_live = live_plot and figure.canvas.required_interactive_framework is not None
    if show_live:
        figure.show()

    last_preview = None

    def update(history):
        nonlocal last_preview
        for line, key in curves:
            values = history[key]
            line.set_data(
                range(1, len(values) + 1),
                [float("nan") if value is None else value for value in values],
            )
        for axis in axes[:, 0]:
            axis.relim()
            axis.autoscale_view()
        if show_live:
            figure.canvas.draw_idle()
            figure.canvas.flush_events()
        elif live_plot and preview_path is not None:
            now = monotonic()
            if last_preview is None or now - last_preview >= 2:
                temporary_path = preview_path.with_suffix(".png.tmp")
                figure.savefig(temporary_path, format="png", dpi=100)
                temporary_path.replace(preview_path)
                if last_preview is None:
                    print(f"Open {preview_path.resolve()} to view training progress.", flush=True)
                last_preview = now

    return figure, update
