# Copyright (c) Facebook, Inc. and its affiliates.
# # Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
"""
Various positional encodings for the transformer.
"""
import math

import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            not_mask = torch.ones((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.float32)

        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = torch.unsqueeze(x_embed,-1)/ dim_t
        pos_y = torch.unsqueeze(y_embed,-1)/ dim_t

        pos_x_0_2 = torch.index_select(pos_x,-1,torch.range(0,127,2,dtype = torch.long,device=pos_x.device) )
        pos_x_1_2 = torch.index_select(pos_x,-1,torch.range(1,128,2,dtype = torch.long,device=pos_x.device) )
        pos_x = torch.stack(
            (pos_x_0_2.sin(), pos_x_1_2.cos()), dim=4
        ).flatten(3)

        pos_y_0_2 = torch.index_select(pos_y,-1,torch.range(0,127,2,dtype = torch.long,device=pos_x.device) )
        pos_y_1_2 = torch.index_select(pos_y,-1,torch.range(1,128,2,dtype = torch.long,device=pos_x.device) )
        pos_y = torch.stack(
            (pos_y_0_2.sin(), pos_y_1_2.cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
    
    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
