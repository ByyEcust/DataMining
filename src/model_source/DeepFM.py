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
        self.NN_part = nn.Sequential(
            # hidden layer # 1
            nn.Linear(num_embedding, hidden_size[0]),
            nn.BatchNorm1d(hidden_size[0]),
            nn.Dropout(0.25),
            nn.ReLU(),
            # hidden layer # 2
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.BatchNorm1d(hidden_size[1]),
            nn.Dropout(0.25),
            nn.ReLU(),
            # output layer
            nn.Linear(hidden_size[1], num_outputs),
            nn.Dropout(0.05))
        self.__initialize_params()

    def forward(self, x):
        # embedding layer
        embedding_vec = self.Embedding(x)
        # Linear part
        linear_output = self.Linear_part(x)
        # FM part
        fm_output = self.FM_part(x, self.Embedding[0].weight)
        # FNN part
        fnn_output = self.NN_part(embedding_vec)
        # y = Linear_part + FM_part + FNN_part
        output = linear_output + fm_output + fnn_output
        return torch.sigmoid(output.squeeze(1))

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
def _fm_layer(x, emb_vec):
    cross_part_1 = torch.pow(torch.mm(x, emb_vec.t()), 2)
    cross_part_2 = torch.mm(torch.pow(x, 2), torch.pow(emb_vec.t(), 2))
    output = 0.5 * (cross_part_1 - cross_part_2).sum(1).unsqueeze(1)
    return output




