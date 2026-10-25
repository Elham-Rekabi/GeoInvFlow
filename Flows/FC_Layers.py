import torch 
import torch.nn as nn
import numpy as np 
from torch.nn import init



class FC_layer(nn.Module):

    def __init__(self, input_size, hidden_size, output_size , activation_function, isBatch_Norm = False): 
        super (FC_layer, self).__init__()

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

            if self.activation_function[count] == 'LeakyReLU':
                leakyrelu_negslope = 0.1
                self.ActivationFunction.append(nn.LeakyReLU(negative_slope=leakyrelu_negslope)) 
            elif self.activation_function[count] == 'ReLU':
                self.ActivationFunction.append(nn.ReLU())
            elif self.activation_function[count] == 'Tanh':
                self.ActivationFunction.append(nn.Tanh())
            elif self.activation_function[count] == 'Sigmoid':
                self.ActivationFunction.append(nn.Sigmoid())


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

        if self.activation_function[-1] == 'LeakyReLU':
            self.ActivationFunction.append(nn.LeakyReLU(negative_slope=leakyrelu_negslope ))
        elif self.activation_function[-1] == 'ReLU':
            self.ActivationFunction.append(nn.ReLU())
        elif self.activation_function[-1] == 'Tanh':
            self.ActivationFunction.append(nn.Tanh())
        elif self.activation_function[-1] == 'Sigmoid':
            self.ActivationFunction.append(nn.Sigmoid())


    def forward(self, x):

        NFCLayers = len(self.FCLayers)

        for i in range(0, NFCLayers):

            FCLayer = self.FCLayers[i]  
            activation = self.ActivationFunction[i]

            x = FCLayer(x)
            x = activation(x)

            if self.isBatch_Norm == True:
                BatchNormLayer = self.BatchNormLayers[i]
                B, C = x.size()
                if B > 1:
                    x = x.view(-1, C)
                    x = BatchNormLayer(x) ####
                    #x = (x-x.mean(dim =0))/(x.std(dim =0)+1e-5)
                    x = x.view(B, C)
                
        return x
        