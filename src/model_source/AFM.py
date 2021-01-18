"""
Attentional factorization machine (AFM) model based on pyTorch
Author: Ruoqiu Zhang (Ecustwaterman, waterteam), 2020.01.18
"""
import torch
import torch.nn as nn


# the basic AFM framework based on pyTorch
class AFM(nn.Module):
    def __init__(self, dim_features, dim_outputs, dim_embedding, attention_size, dropouts):
        super(AFM, self).__init__()
        # Linear part x*w
        self.linear_weight = nn.Linear(dim_features, dim_outputs, bias=True)
        # embedding part
        self.embedding_layer = nn.Sequential(
            nn.Linear(dim_features, dim_embedding),
            nn.BatchNorm1d(dim_embedding),
            nn.ReLU())
        # attention network
        self.attention_layer = nn.Sequential(
            nn.Linear(dim_embedding, attention_size),  # attention layer
            nn.ReLU(),
            nn.Linear(attention_size, 1),  # projection layer
            nn.Softmax(dim=1),
            nn.Dropout(dropouts[0]))
        # forward part
        self.fc = nn.Linear(dim_embedding, dim_outputs)
        self.dropouts = dropouts
        self.__initialize_params()

    def forward(self, x):
        rows, cols = self.__cross_term_generation(x.shape[1])
        inner_product = x[:, rows] * x[:, cols]  # get inner product of each pair of features
        embedding_vec = self.embedding_layer(x)  # embedding
        attention_scores = self.attention(embedding_vec)  # get attention scores
        attention_outputs = (attention_scores * inner_product).sum(1).unsqueeze(1)  # aij * (vi*vj) * xi*xj
        attention_outputs = nn.functional.dropout(attention_outputs, p=self.dropouts[1], training=self.training)
        outputs = torch.sigmoid(self.linear_weight(x) + self.fc(attention_outputs))  # linear_part + cross_part
        return outputs

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

    @staticmethod
    def __cross_term_generation(dim_features):
        rows, cols = [], []
        for i in range(dim_features-1):
            for j in range(1, dim_features):
                rows.append(i)
                cols.append(j)
        return rows, cols



