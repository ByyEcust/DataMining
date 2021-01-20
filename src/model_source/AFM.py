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
        self.embedding_vec = nn.Parameter(torch.randn(dim_features, dim_embedding))
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
        inner_product = self.__cross_term_generation(x)  # (batch_size * m(m-1)/2 * dim_emb)
        attention_score = self.attention_layer(inner_product)  # (batch_size * m(m-1)/2 * 1)
        attention_out = torch.sum(attention_score * inner_product, dim=1) # (batch_size * dim_emb)
        attention_out = nn.functional.dropout(attention_out, p=self.dropouts[1], training=self.training)
        cross_term_out = self.fc(attention_out)
        linear_term_out = self.linear_weight(x)
        output = torch.sigmoid(linear_term_out+cross_term_out)
        return output.squeeze(1)

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

    def __cross_term_generation(self, x):
        dim_features, dim_embedding = x.shape[1], self.embedding_vec.shape[1]
        num_cross_term = int(dim_features*(dim_features-1)/2)
        inner_product = torch.empty((x.shape[0], num_cross_term, dim_embedding))
        count = 0
        for i in range(dim_features-1):
            # xi * vi
            p = torch.mm(x[:, i].view(-1, 1), self.embedding_vec[i, :].view(1, -1))
            for j in range(i+1, dim_features):
                # xj * vj
                q = torch.mm(x[:, j].view(-1, 1), self.embedding_vec[j, :].view(1, -1))
                # (vi * vj) xi * xj
                inner_product[:, count, :] = p * q
                count += 1
        return inner_product
