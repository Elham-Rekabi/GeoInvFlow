import torch 
import numpy as np
import torch.nn as nn
from Invertible_Block import Affine_Coupling_Block


class Invertible_Network(nn.Module):
    def  __init__(self, Dim, BlocksSizes, BlockLayers):
        super(Invertible_Network, self).__init__()
        
        self.BlocksSizes = BlocksSizes
        self.BlockLayers = BlockLayers
        NBlocks = len(BlocksSizes)
        self.NBlocks = NBlocks

        self.Blocks = nn.ModuleList()
        
        for i in range(0, NBlocks):

            in_channels_inv_block = BlocksSizes[i][0]
            in_channels_cond_block = BlocksSizes[i][1]
            s_net_hidden_layers_channels = BlocksSizes[i][2]
            t_net_hidden_layers_channels = BlocksSizes[i][3]
            out_channels_inv_block = BlocksSizes[i][4]
            activation_functions_shift = BlockLayers[i]["Activation_Functions_Shift"]
            activation_functions_scale= BlockLayers[i]["Activation_Functions_Scale"]
            isBatchNorm = BlockLayers[i]["isBatch_Norm"]

            # Invertible Blocks
            Conditional_Affine_Coupling_Block = Affine_Coupling_Block(in_channels_inv_block, in_channels_cond_block,
                                                                            s_net_hidden_layers_channels, t_net_hidden_layers_channels, out_channels_inv_block, activation_functions_shift, activation_functions_scale, isBatchNorm)
            self.Blocks.append(Conditional_Affine_Coupling_Block)


    def forward(self, x, Cond_block_outputs, isInverse = False): 

        NBlocks = self.NBlocks

        if isInverse == False: # Forward pass
            
            Output = []
            Jacobian = []

            for i in range(0, NBlocks):

                Conditional_Affine_Coupling_Block = self.Blocks[i]
                c = Cond_block_outputs[i].clone()

                # Invertible block
                x, jacobian = Conditional_Affine_Coupling_Block(x, c, isInverse)
                
                Output.append(x)
                Jacobian.append(jacobian.unsqueeze(1))

            jacobian = torch.sum(torch.cat(Jacobian, dim = -1), dim = -1)
            return x, jacobian, Output, Jacobian

        else: # Inverse pass

            Output = []
            InvJacobian = []
            #x = x.detach()
            for i in range(0, NBlocks):

                Conditional_Affine_Coupling_Block = self.Blocks[NBlocks-i-1]
                c = Cond_block_outputs[NBlocks-i-1].clone()

                # Invertible block
                x, invJacobian = Conditional_Affine_Coupling_Block(x, c, isInverse)

                Output.append(x)
                InvJacobian.append(invJacobian.unsqueeze(1))

            invJacobian = torch.sum(torch.cat(InvJacobian, dim = -1), dim = -1)

            return x,invJacobian, Output, InvJacobian
         
