"""Model construction shared by quadratic training and visualization."""

from Experiments.Quadratic.Config import QuadraticConfig
from Flows.FC_Cond_Invertible_Network import Conditional_Invertible_Network


def build_cnf(config: QuadraticConfig, input_features: int, output_features: int):
    hidden_layers = config.cnf_hidden_layers
    block_sizes = [
        [input_features, output_features, hidden_layers, hidden_layers]
        for _ in range(config.cnf_num_blocks)
    ]
    block_layers = [
        {
            "Activation_Functions_Shift": config.cnf_activation,
            "Activation_Functions_Scale": config.cnf_activation,
            "isBatch_Norm": False,
        }
        for _ in block_sizes
    ]
    return Conditional_Invertible_Network(input_features, block_sizes, block_layers)
