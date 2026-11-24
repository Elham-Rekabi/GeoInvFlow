"""Reusable file logging for model training."""

from pathlib import Path


class TrainingLog:
    """Start a fresh log and provide callbacks for the training loops."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def report(self, message, *, console=True):
        if console:
            print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as log_file:
            log_file.write(message + "\n")

    def log_iterations(self, stage, *, console=False):
        def on_iteration_end(epoch, iteration, metrics):
            values = ", ".join(f"{name}={value:.6g}" for name, value in metrics.items())
            self.report(f"{stage} iteration {iteration} (epoch {epoch}) .... {values}", console=console)

        return on_iteration_end

    def log_epochs(self, stage, total_epochs, update_plot=None, *, console=False):
        def on_epoch_end(history):
            epoch = len(history["learning_rate"])
            metrics = ", ".join(
                f"{name}={values[-1]:.6g}"
                for name, values in history.items()
                if values[-1] is not None
            )
            self.report(f"{stage} epoch {epoch}/{total_epochs} .... {metrics}", console=console)
            if update_plot is not None:
                update_plot(history)

        return on_epoch_end
