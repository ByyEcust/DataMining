"""
DeepFM model based on pyTorch
Author: Ruoqiu Zhang (Ecustwaterman, waterteam), 2020.01.14
"""
import torch
import torch.nn as nn


# basic DeepFM model built by pyTorch (Single field of inputs and 2 hidden layers)
class DeepFM(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_embedding, hidden_size=(512, 512)):
        super(DeepFM, self).__init__()
        self.Linear_part = nn.Linear(num_inputs, 1)
        self.FM_part = _fm_layer
        self.Embedding = nn.Sequential(
            nn.Linear(num_inputs, num_embedding),
            nn.BatchNorm1d(num_embedding),
            nn.ReLU())
        self.hidden_layer = nn.Sequential(
            # hidden layer # 1
            nn.Linear(num_inputs, hidden_size[0]),
            nn.BatchNorm1d(hidden_size[0]),
            nn.Dropout(0.25),
            nn.ReLU(),
            # hidden layer # 2
            nn.Linear(num_inputs, hidden_size[1]),
            nn.BatchNorm1d(hidden_size[1]),
            nn.Dropout(0.25),
            nn.ReLU())
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size[1]+num_embedding, num_outputs),
            nn.Dropout(0.05),
            nn.Sigmoid())
        self.__initialize_params()

    def forward(self, x):
        # embedding layer
        embedding_vec = self.Embedding(x)
        # Linear part
        linear_output = self.Linear_part(x)
        # FM part
        fm_output = self.fm(embedding_vec)
        # FNN part
        fnn_output = self.hidden_layer(embedding_vec)
        fnn_output = self.output_layer(fnn_output)
        # y = Linear_part + FM_part + FNN_part
        output = linear_output + fm_output + fnn_output
        return output

    def __initialize_params(self):
        for layer in self.parameters():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1.)
                nn.init.constant_(layer.bias, 0.)
                nn.init.constant_(layer.running_mean, 0.)
                nn.init.constant_(layer.running_var, 1.)


# FM special version for DeepFM
def _fm_layer(x):
    square_of_sum = torch.sum(x, dim=1) ** 2
    sum_of_square = torch.sum(x ** 2, dim=1)
    cross_part = square_of_sum - sum_of_square
    cross_part = 0.5 * cross_part.sum(1).unsqueeze(1)
    return cross_part




