"""
Logistic regression (LR), Factorization machine (FM) and Field-aware FM (FFM)
Author: Ruoqiu Zhang (Ecustwaterman, waterteam), 2021.01.05
"""


import torch
import torch.nn as nn


# the basic LR
class LR(nn.Module):
    def __init__(self, dim_features):
        super(LR, self).__init__()
        self.linear_weight = nn.Linear(dim_features, 1, bias=True)

    def forward(self, x):
        output = self.linear_weight(x)
        output = nn.functional.sigmoid(output)
        return output


# the basic FM calculating framework based on pyTorch
class FM(nn.Module):
    def __init__(self, dim_features, dim_embedding):
        super(FM, self).__init__()
        self.linear_weight = nn.Linear(dim_features, 1, bias=True)
        self.embedding_vec = nn.Parameter(torch.randn(dim_features, dim_embedding))

    def forward(self, x):
        # X * W
        linear_part = self.linear_weight(x)
        # (X * V)^2
        cross_part_1 = torch.pow(torch.mm(x, self.embedding_vec), 2)
        # (X^2 * V^2)
        cross_part_2 = torch.mm(torch.pow(x, 2), torch.pow(self.embedding_vec, 2))
        # output
        output = linear_part + 0.5*torch.sum(cross_part_1 - cross_part_2)
        output = nn.functional.sigmoid(output)
        return output



