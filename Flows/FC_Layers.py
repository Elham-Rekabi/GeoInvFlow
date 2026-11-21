import torch
import torch.nn as nn
import numpy as np
from torch.nn import init



class FC_layer(nn.Module):

    def __init__(self, input_size, hidden_size, output_size , activation_function, isBatch_Norm = False):
        super (FC_layer, self).__init__()

        if len(activation_function) != len(hidden_size) + 1:
            raise ValueError(
                f"(expected {len(hidden_size) + 1}, got {len(activation_function)})."
            )

        activation_factories = {
            'LeakyReLU': lambda: nn.LeakyReLU(negative_slope=0.1),
            'ReLU': nn.ReLU,
            'Tanh': nn.Tanh,
            'Sigmoid': nn.Sigmoid,
            'Identity': nn.Identity,
        }
        for name in activation_function:
            if name not in activation_factories:
                raise ValueError(
                    f"Unsupported activation {name!r}. "
                    f"Choose from {', '.join(activation_factories)}."
                )

        self.isBatch_Norm = isBatch_Norm
        self.activation_function = activation_function

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.FCLayers = nn.ModuleList()
        self.BatchNormLayers = nn.ModuleList() ###
        self.ActivationFunction = nn.ModuleList()


        input_layer_size = input_size
        count = 0
        
        for hidden_layer_size in hidden_size:

            self.FCLayers.append(nn.Linear(input_layer_size, hidden_layer_size, bias = True))

            self.ActivationFunction.append(activation_factories[self.activation_function[count]]())


            if self.isBatch_Norm == True:

                self.BatchNormLayers.append(nn.BatchNorm1d(hidden_layer_size))
                init.constant_(self.BatchNormLayers[count].weight, 1)
                init.constant_(self.BatchNormLayers[count].bias, 0)

            input_layer_size = hidden_layer_size
            count+=1

        self.FCLayers.append(nn.Linear(input_layer_size, output_size, bias = True))

        if self.isBatch_Norm == True:
            self.BatchNormLayers.append(nn.BatchNorm1d(output_size))
            init.constant_(self.BatchNormLayers[-1].weight, 1)
            init.constant_(self.BatchNormLayers[-1].bias, 0)

        self.ActivationFunction.append(activation_factories[self.activation_function[-1]]())


    def forward(self, x):

        NFCLayers = len(self.FCLayers)

        for i in range(0, NFCLayers):

            FCLayer = self.FCLayers[i]
            activation = self.ActivationFunction[i]

            x = FCLayer(x)
            x = activation(x)

            if self.isBatch_Norm:
                batch_norm = self.BatchNormLayers[i]
                flattened = x.reshape(-1, x.shape[-1])
                # BatchNorm needs multiple rows to estimate training statistics.
                if flattened.shape[0] > 1 or not batch_norm.training:
                    x = batch_norm(flattened).reshape(x.shape)

        return x

