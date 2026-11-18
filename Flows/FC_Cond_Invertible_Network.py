"""Conditional fully connected flow using (batch_size, features) tensors."""

import torch
import torch.nn as nn

from .FC_Invertible_Block import Affine_Coupling_Block


class Conditional_Invertible_Network(nn.Module):
    def __init__(self, Dim, BlocksSizes, BlockLayers):
        """Each block size is [features, condition_features, s_hidden, t_hidden].
        """
        super().__init__()
        if not BlocksSizes:
            raise ValueError("At least one coupling block is required.")
        if len(BlocksSizes) != len(BlockLayers):
            raise ValueError("BlocksSizes and BlockLayers must have equal lengths.")
        if any(len(sizes) != 4 for sizes in BlocksSizes):
            raise ValueError("Each block needs four entries: "
                             "[features, condition_features, s_hidden, t_hidden].")

        self.BlocksSizes = BlocksSizes
        self.BlockLayers = BlockLayers
        self.NBlocks = len(BlocksSizes)
        self.Blocks = nn.ModuleList()
        for sizes, settings in zip(BlocksSizes, BlockLayers):
            features, condition_features, s_hidden, t_hidden = sizes
            if features != BlocksSizes[0][0]:
                raise ValueError("All blocks must preserve the same feature count.")
            self.Blocks.append(Affine_Coupling_Block(
                in_channels_inv_block=features,
                in_channels_cond_block=condition_features,
                s_net_hidden_layers_channels=s_hidden,
                t_net_hidden_layers_channels=t_hidden,
                activation_functions_shift=settings["Activation_Functions_Shift"],
                activation_functions_scale=settings["Activation_Functions_Scale"],
                isBatchNorm=settings["isBatch_Norm"],
            ))

    def forward(self, x, Cond_block_outputs, isInverse=False):
        """Return output, total logdet, block outputs, and block logdets.

        x: (batch_size, features).
        Each condition: (batch_size, condition_features), in forward block order.
        Output: (batch_size, features); total logdet: (batch_size,).
        Individual block logdets retain shape (batch_size, 1).
        """
        if x.dim() != 2 or x.shape[1] != self.BlocksSizes[0][0]:
            raise ValueError("x must have shape (batch_size, input_features).")
        if len(Cond_block_outputs) != self.NBlocks:
            raise ValueError("Provide exactly one condition tensor per block.")
        for i, c in enumerate(Cond_block_outputs):
            if (c.dim() != 2 or c.shape[0] != x.shape[0]
                    or c.shape[1] != self.BlocksSizes[i][1]):
                raise ValueError(f"Condition {i} must have shape "
                                 "(batch_size, condition_features).")

        indices = range(self.NBlocks - 1, -1, -1) if isInverse else range(self.NBlocks)
        outputs = []
        block_logdets = []
        for i in indices:
            x, logdet = self.Blocks[i](x, Cond_block_outputs[i], isInverse)
            outputs.append(x)
            block_logdets.append(logdet.unsqueeze(1))
        total_logdet = torch.cat(block_logdets, dim=-1).sum(dim=-1)
        return x, total_logdet, outputs, block_logdets
