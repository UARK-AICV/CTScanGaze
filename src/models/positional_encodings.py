import math

import torch
from torch import nn


class PositionEmbeddingSine3d(nn.Module):
    def __init__(
        self,
        spatial_dim,
        hidden_dim=768,
        temperature=10000,
        normalize=False,
        scale=None,
        flatten=True,
        device="cuda:0",
    ):
        super(PositionEmbeddingSine3d, self).__init__()
        self.num_pos_feats = hidden_dim // 3
        normalize = normalize
        self.h, self.w, self.c = 8, 8, 8
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.device = device
        position_y = torch.arange(self.h).unsqueeze(1)
        position_x = torch.arange(self.w).unsqueeze(1)
        position_z = torch.arange(self.c).unsqueeze(1)
        if normalize:
            eps = 1e-6
            position_y = position_y / (self.h - 1 + eps) * scale
            position_x = position_x / (self.w - 1 + eps) * scale
            position_z = position_z / (self.c - 1 + eps) * scale
        div_term = torch.exp(
            torch.arange(0, self.num_pos_feats, 2)
            * (-math.log(temperature) / self.num_pos_feats)
        )
        pe_y = torch.zeros(self.h, 1, 1, self.num_pos_feats)
        pe_x = torch.zeros(1, self.w, 1, self.num_pos_feats)
        pe_z = torch.zeros(1, 1, self.c, self.num_pos_feats)
        # 100, 128 shape assigned with 100,128 in shape as well
        pe_y[:, 0, 0, 0::2] = torch.sin(position_y * div_term)
        pe_y[:, 0, 0, 1::2] = torch.cos(position_y * div_term)
        pe_x[0, :, 0, 0::2] = torch.sin(position_x * div_term)
        pe_x[0, :, 0, 1::2] = torch.cos(position_x * div_term)
        pe_z[0, 0, :, 0::2] = torch.sin(position_z * div_term)
        pe_z[0, 0, :, 1::2] = torch.cos(position_z * div_term)
        pe_y = pe_y.repeat(1, self.w, self.c, 1)
        pe_x = pe_x.repeat(self.h, 1, self.c, 1)
        pe_z = pe_z.repeat(self.h, self.w, 1, 1)
        self.pos = torch.cat((pe_y, pe_x, pe_z), dim=-1).permute(
            3, 0, 1, 2
        )  # dim, h, w, c
        if flatten:
            self.pos = self.pos.view(hidden_dim, -1).permute(1, 0).unsqueeze(1)
        else:
            self.pos = self.pos.permute(1, 2, 3, 0)
        self.pos = nn.Parameter(self.pos, requires_grad=False)
        del pe_y, pe_x, pe_z, position_y, position_x, position_z

    def forward(self, x):
        # return x.to(self.device) + self.pos.to(self.device)
        return x.to(self.device) + self.pos.to(self.device)[: x.shape[0]]
