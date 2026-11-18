import torch 
import torch.nn as nn
from .FC_Layers import FC_layer

class Affine_Coupling_Block(nn.Module):

    def __init__(self, in_channels_inv_block, in_channels_cond_block, s_net_hidden_layers_channels, t_net_hidden_layers_channels, activation_functions_shift, activation_functions_scale, isBatchNorm = False):

        super(Affine_Coupling_Block, self).__init__()

        if in_channels_inv_block < 2:
            raise ValueError("An affine coupling block requires at least two input channels.")

        self.in_channels_inv_block = in_channels_inv_block
        self.in_channels_cond_block = in_channels_cond_block
        self.activation_functions_shift = activation_functions_shift
        self.activation_functions_scale = activation_functions_scale
        self.isBatch_Norm = isBatchNorm
      
        self.u1_channels = in_channels_inv_block // 2
        self.u2_channels = in_channels_inv_block - self.u1_channels

        # s1/t1 take [v1, c] and transform u2.
        self.s1_t1_net_in_channels = self.u1_channels + in_channels_cond_block
        self.s1_t1_net_out_channels = self.u2_channels
        # s2/t2 take [u2, c] and transform u1.
        self.s2_t2_net_in_channels = self.u2_channels + in_channels_cond_block
        self.s2_t2_net_out_channels = self.u1_channels
        self.s1_network = FC_layer(self.s1_t1_net_in_channels, s_net_hidden_layers_channels, self.s1_t1_net_out_channels, self.activation_functions_scale, isBatch_Norm = self.isBatch_Norm)
        self.t1_network = FC_layer(self.s1_t1_net_in_channels, t_net_hidden_layers_channels, self.s1_t1_net_out_channels, self.activation_functions_shift, isBatch_Norm = self.isBatch_Norm)
        self.s2_network = FC_layer(self.s2_t2_net_in_channels, s_net_hidden_layers_channels, self.s2_t2_net_out_channels, self.activation_functions_scale, isBatch_Norm = self.isBatch_Norm)
        self.t2_network = FC_layer(self.s2_t2_net_in_channels, t_net_hidden_layers_channels, self.s2_t2_net_out_channels, self.activation_functions_shift, isBatch_Norm = self.isBatch_Norm)

        

    def forward(self, x, c, isInverse=False):
        """Transform x (batch, features) with c (batch, condition_features).

        Returns the transformed batch and one log-determinant per sample.
        """
        if x.dim() != 2 or x.shape[1] != self.in_channels_inv_block:
            raise ValueError("x must have shape (batch_size, input_features).")
        if (c.dim() != 2 or c.shape[0] != x.shape[0]
                or c.shape[1] != self.in_channels_cond_block):
            raise ValueError("c must have shape (batch_size, condition_features).")

        if isInverse == False:


            u1 = x.narrow(-1, 0, self.u1_channels)
            u2 = x.narrow(-1, self.u1_channels, self.u2_channels)

            u2_c = torch.cat([u2, c], dim=-1)

            s2_net = self.s2_network(u2_c)
            t2_net = self.t2_network(u2_c)


            alpha = 0.636*2

            v1 = (torch.exp(alpha * torch.atan(s2_net)))*u1 + t2_net
            v1_c = torch.cat([v1, c], dim=-1)


            s1_net = self.s1_network(v1_c)

            t1_net = self.t1_network(v1_c)

            v2 = (torch.exp(alpha * torch.atan(s1_net)))*u2 + t1_net


            output = torch.cat([v1, v2], -1)

            ###############################################################
            # Jacobians
            Jacobian = (alpha * torch.atan(s1_net)).sum(dim=-1) + (alpha * torch.atan(s2_net)).sum(dim=-1)
            self.Jacobian = Jacobian 

            return output, Jacobian

        else:
        


            v1 = x.narrow(-1, 0, self.u1_channels)
            v2 = x.narrow(-1, self.u1_channels, self.u2_channels)

            v1_c = torch.cat([v1, c], dim=-1)

            s1_net = self.s1_network(v1_c)
            t1_net = self.t1_network(v1_c)

            alpha = 0.636*2

            u2 = (v2 - t1_net) * (torch.exp(-1*alpha*torch.atan(s1_net)))
                    
            u2_c = torch.cat([u2, c], dim=-1)

            s2_net = self.s2_network(u2_c)
            t2_net = self.t2_network(u2_c)

            u1 = (v1 - t2_net) * (torch.exp(-1*alpha*torch.atan(s2_net)))

            output = torch.cat([u1, u2], -1)

            ###############################################################
            # Jacobians
            InvJacobian = -((alpha * torch.atan(s1_net)).sum(dim=-1) + (alpha * torch.atan(s2_net)).sum(dim=-1))
            self.InvJacobian = InvJacobian 
            ######
            
            return output, InvJacobian
