"""
Logistic regression (LR), Factorization machine (FM) and Field-aware FM (FFM)
Author: Ruoqiu Zhang (Ecustwaterman, waterteam), 2021.01.05
"""


import torch
import torch.nn as nn


# the basic LR model built by pyTorch
class LR(nn.Module):
    def __init__(self, dim_features):
        super(LR, self).__init__()
        self.linear_weight = nn.Linear(dim_features, 1, bias=True)
        self.__initialize_params()

    def forward(self, x):
        output = self.linear_weight(x)
        output = nn.functional.sigmoid(output)
        return output

    def __initialize_params(self):
        nn.init.normal_(self.linear_weight.weight, 0.0, 1.0)
        nn.init.constant_(self.linear_weight.bias, 0.0)


# the basic FM model built by pyTorch
class FM(nn.Module):
    def __init__(self, dim_features, dim_embedding):
        super(FM, self).__init__()
        self.linear_weight = nn.Linear(dim_features, 1, bias=True)
        self.embedding_vec = nn.Parameter(torch.randn(dim_features, dim_embedding))
        self.__initialize_params()

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

    def __initialize_params(self):
        nn.init.normal_(self.linear_weight.weight, 0.0, 1.0)
        nn.init.constant_(self.linear_weight.bias, 0.0)