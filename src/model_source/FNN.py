'''
Feedforward Neural Network (FNN) based on Pytorch
Author: Ruoqiu Zhang (Ecustwaterman, waterteam), 2020.12.24
'''
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data


# An simple Example of FNN with three layers (input-hidden-output)
class FnnModel(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(FnnModel, self).__init__()
        self.input_layer = nn.Sequential(
            nn.BatchNorm1d(num_inputs),
            nn.Linear(num_inputs, hidden_size),
            nn.Dropout(0.05),
            nn.ReLU())
        self.hidden_layer = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.20),
            nn.ReLU())
        self.output_layer = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, num_outputs),
            nn.Dropout(0.10),
            nn.Sigmoid())

    def forward(self, inputs):
        inputs = self.input_layer(inputs)
        inputs = self.hidden_layer(inputs)
        outputs = self.output_layer(inputs)
        return outputs

