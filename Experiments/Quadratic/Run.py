"""Run the quadratic experiment with python -m Experiments.Quadratic.Run."""

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from Data_Generation.Toy_Examples.Toy_Functions import quadratic_function
from Experiments.Quadratic.Config import QuadraticConfig
from Experiments.Quadratic.Data import make_datasets
from Experiments.Quadratic.Model_Setup import build_cnf
from Flows.FC_Layers import FC_layer
from Train.Training import train_cnf_model, train_forward_model
from Train.Training_Log import TrainingLog
from Visualization.Scalar_Visualization import plot_conditional_inverse, plot_forward_surface
from Visualization.Loss_Visualization import loss_figure


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def run_experiment(
    config: QuadraticConfig, *, output_dir=DEFAULT_OUTPUT_DIR, live_plot=True):

    """Return the trained forward model, CNF, and their training histories."""

    output_dir = Path(output_dir)
    log = TrainingLog(output_dir / "training.log")

    log.report("Preparing the quadratic experiment...")
    torch.manual_seed(config.seed)
    train_data, validation_data = make_datasets(config)
    train_inputs, train_targets = train_data.tensors

    input_features = train_inputs.shape[-1]
    output_features = train_targets.shape[-1]

    # forward Model (MLP)

    forward_model = FC_layer(input_features,
        config.forward_model_hidden_layers,
        output_features,
        config.forward_model_activations).to(train_inputs)

    cnf_model = build_cnf(config, input_features, output_features).to(train_inputs)

    forward_path = output_dir / "forward_loss.png"
    forward_figure, update_forward = loss_figure(
        [("mse", "Forward MSE")], live_plot, preview_path=forward_path)

    try:
        log.report(f"Training the forward model for {config.forward_training.epochs} epochs...")
        forward_history = train_forward_model(
            forward_model, train_data, validation_data, config.forward_training,
            on_epoch_end=log.log_epochs(
                "Forward", config.forward_training.epochs,
                update_forward if live_plot else None, console=True,
            ),
        )

        checkpoint_path = output_dir / "forward_model.pt"
        #torch.save(forward_model.state_dict(), checkpoint_path)
        log.report(f"Forward model weights saved to {checkpoint_path}")
        if not live_plot:
            update_forward(forward_history)
        forward_figure.savefig(forward_path, dpi=150)
    finally:
        if not live_plot or forward_figure.canvas.required_interactive_framework is None:
            plt.close(forward_figure)

    log.report(f"Forward validation MSE = {forward_history['validation_mse'][-1]:.6g}")

    surface = plot_forward_surface(forward_model, train_data, validation_data, config.input_bounds)
    try:
        surface_path = output_dir / "forward_surface.png"
        surface.savefig(surface_path, dpi=150)
        log.report(f"Forward surface saved to {surface_path}")
        if live_plot and surface.canvas.required_interactive_framework is not None:
            surface.show()
            surface.canvas.flush_events()
    finally:
        if not live_plot or surface.canvas.required_interactive_framework is None:
            plt.close(surface)

    cnf_metrics = [("nll_loss", "CNF NLL")]
    if config.inverse_training.use_hybrid_loss:
        cnf_metrics.extend([
            ("reconstruction_loss", "Reconstruction MSE"),
            ("hybrid_loss", "Hybrid loss")])


    cnf_path = output_dir / "cnf_loss.png"
    cnf_figure, update_cnf = loss_figure(cnf_metrics, live_plot, preview_path=cnf_path)
    try:
        log.report(f"Training the CNF for {config.inverse_training.epochs} epochs...")
        cnf_history = train_cnf_model(
            cnf_model, train_data, validation_data, config.inverse_training,
            forward_model=forward_model,
            on_epoch_end=log.log_epochs(
                "CNF", config.inverse_training.epochs,
                update_cnf if live_plot else None, console=True,
            ),
        )
        checkpoint_path = output_dir / "cnf_model.pt"
        #torch.save(cnf_model.state_dict(), checkpoint_path)
        log.report(f"CNF model weights saved to {checkpoint_path}")
        if not live_plot:
            update_cnf(cnf_history)
        cnf_figure.savefig(cnf_path, dpi=150)
    finally:
        if not live_plot or cnf_figure.canvas.required_interactive_framework is None:
            plt.close(cnf_figure)

    log.report(f"CNF validation NLL = {cnf_history['validation_nll_loss'][-1]:.6g}")
    for key, label in (
        ("validation_reconstruction_loss", "CNF validation reconstruction MSE"),
        ("validation_hybrid_loss", "CNF validation hybrid loss")):
        value = cnf_history[key][-1]
        if value is not None:
            log.report(f"{label} = {value:.6g}")

    inverse_figure = plot_conditional_inverse(
        cnf_model, quadratic_function, config.inverse_plot_observation,
        config.input_bounds, seed=config.seed)

    inverse_path = output_dir / "conditional_inverse.png"
    try:
        inverse_figure.savefig(inverse_path, dpi=150)
        log.report(f"Conditional inverse figure saved to {inverse_path}")
        if live_plot and inverse_figure.canvas.required_interactive_framework is not None:
            inverse_figure.show()
            inverse_figure.canvas.flush_events()
    finally:
        if not live_plot or inverse_figure.canvas.required_interactive_framework is None:
            plt.close(inverse_figure)

    log.report(f"Loss figures saved to {output_dir}")

    return forward_model, cnf_model, forward_history, cnf_history


if __name__ == "__main__":

    config = QuadraticConfig()
    output_dirs = {mode: DEFAULT_OUTPUT_DIR / mode for mode in ("nll", "hybrid")}
    for output_dir in output_dirs.values():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Experiment output folder: {output_dir}", flush=True)

    for mode, output_dir in output_dirs.items():
        inverse_training = replace(config.inverse_training, use_hybrid_loss=(mode == "hybrid"))
        print(f"Starting quadratic experiment with {mode.upper()} loss.", flush=True)
        run_experiment(
            replace(config, inverse_training=inverse_training),
            output_dir=output_dir)
    if plt.get_fignums():
        plt.show(block=True)
