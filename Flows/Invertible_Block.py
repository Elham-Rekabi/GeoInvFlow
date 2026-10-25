import torch 
import torch.nn as nn
import numpy as np 
from torch.nn import init
from FC_Layers import FC_layer

class Affine_Coupling_Block(nn.Module):

    def __init__(self, in_channels_inv_block, in_channels_cond_block, s_net_hidden_layers_channels, t_net_hidden_layers_channels, out_channels_inv_block, activation_functions_shift, activation_functions_scale, isBatchNorm = False):

        super(Affine_Coupling_Block, self).__init__()

        self.in_channels_inv_block = in_channels_inv_block
        self.in_channels_cond_block = in_channels_cond_block
        self.out_channels_inv_block = out_channels_inv_block
        self.activation_functions_shift = activation_functions_shift
        self.activation_functions_scale = activation_functions_scale
        self.isBatch_Norm = isBatchNorm
      
        self.s1_t1_net_in_channels = in_channels_inv_block // 2 + in_channels_cond_block
        self.s2_t2_net_in_channels = in_channels_inv_block - in_channels_inv_block // 2 + in_channels_cond_block
        self.s1_t1_net_out_channels = out_channels_inv_block // 2 
        self.s2_t2_net_out_channels = out_channels_inv_block - out_channels_inv_block // 2 
        self.s1_network = FC_layer(self.s1_t1_net_in_channels, s_net_hidden_layers_channels, self.s1_t1_net_out_channels, self.activation_functions_scale, isBatch_Norm = self.isBatch_Norm)
        self.t1_network = FC_layer(self.s1_t1_net_in_channels, t_net_hidden_layers_channels, self.s1_t1_net_out_channels, self.activation_functions_shift, isBatch_Norm = self.isBatch_Norm)
        self.s2_network = FC_layer(self.s2_t2_net_in_channels, s_net_hidden_layers_channels, self.s2_t2_net_out_channels, self.activation_functions_scale, isBatch_Norm = self.isBatch_Norm)
        self.t2_network = FC_layer(self.s2_t2_net_in_channels, t_net_hidden_layers_channels, self.s2_t2_net_out_channels, self.activation_functions_shift, isBatch_Norm = self.isBatch_Norm)

        

    def forward (self, x, c, isInverse):

        if isInverse == False:

            x = x.unsqueeze(0) if x.dim() == 2 else x
            c = c.unsqueeze(0) if c.dim() == 2 else c
            B, N, _ = x.size()

            x_columns_no = x.size()[-1]
            node_per_channel = int(x_columns_no / self.in_channels_inv_block)

            u1 = x.narrow(-1, 0, self.in_channels_inv_block//2)
            u2 = x.narrow(-1, self.in_channels_inv_block//2, self.in_channels_inv_block - self.in_channels_inv_block//2)

            u2_c = torch.cat([u2, c], -1)

            s2_net = self.s2_network(u2_c)
            t2_net = self.t2_network(u2_c)


            alpha = 0.636*2

            v1 = (torch.exp(alpha * torch.atan(s2_net)))*u1 + t2_net
            v1_c = torch.cat([v1, c], -1)


            s1_net = self.s1_network(v1_c)

            t1_net = self.t1_network(v1_c)

            v2 = (torch.exp(alpha * torch.atan(s1_net)))*u2 + t1_net


            output = torch.cat([v1, v2], -1)

            ###############################################################
            # Jacobians
            Jacobian = torch.sum(alpha * torch.atan(s1_net)+ alpha * torch.atan(s2_net), dim = (1, 2))
            self.Jacobian = Jacobian 

            return output, Jacobian

        else:
        
            x = x.unsqueeze(0) if x.dim() == 2 else x
            c = c.unsqueeze(0) if c.dim() == 2 else c
            B, N, _ = x.size()

            x_columns_no = x.size()[-1]
            node_per_channel = int(x_columns_no / self.in_channels_inv_block)


            v1 = x.narrow(-1, 0, self.in_channels_inv_block//2)
            v2 = x.narrow(-1, self.in_channels_inv_block//2, self.in_channels_inv_block - self.in_channels_inv_block//2)

            v1_c = torch.cat([v1, c], -1)

            s1_net = self.s1_network(v1_c)
            t1_net = self.t1_network(v1_c)

            alpha = 0.636*2

            u2 = (v2 - t1_net) * (torch.exp(-1*alpha*torch.atan(s1_net)))
                    
            u2_c = torch.cat([u2, c], -1)

            s2_net = self.s2_network(u2_c)
            t2_net = self.t2_network(u2_c)

            u1 = (v1 - t2_net) * (torch.exp(-1*alpha*torch.atan(s2_net)))

            output = torch.cat([u1, u2], -1)

            ###############################################################
            # Jacobians
            InvJacobian = -1*torch.sum(alpha * torch.atan(s1_net)+ alpha * torch.atan(s2_net), dim = (1, 2))
            self.InvJacobian = InvJacobian 
            ######
            
            return output, InvJacobian
