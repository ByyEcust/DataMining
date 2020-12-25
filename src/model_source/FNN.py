'''
Feedforward Neural Network (FNN) based on Pytorch
*** Demo and simple examples of FNN Pytorch Structure ***
    class FNNModel: An simple Example of FNN with three layers
    class FnnModelMultiInputs: An simple example of FNN with multi-inputs and embedding layers
    class FNN: the training framework of FNN model
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


# An simple example of FNN with multi-inputs and embedding layers (input-embedding-hidden-output)
class FnnModelMultiInputs(nn.Module):
    def __init__(self, num_inputs_1, num_inputs_2, num_outputs, hidden_size, embedding_dim=(128, 128)):
        super(FnnModelMultiInputs, self).__init__()
        self.embedding_1 = nn.Sequential(
            nn.BatchNorm1d(num_inputs_1),
            nn.Linear(num_inputs_1, embedding_dim[0]),
            nn.ReLU())
        self.embedding_2 = nn.Sequential(
            nn.BatchNorm1d(num_inputs_2),
            nn.Linear(num_inputs_2, embedding_dim[1]),
            nn.ReLU())
        self.hidden_layer = nn.Sequential(
            nn.BatchNorm1d(embedding_dim[0]+embedding_dim[1]),
            nn.Linear(embedding_dim[0]+embedding_dim[1], hidden_size),
            nn.Dropout(0.30),
            nn.ReLU())
        self.output_layer = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, num_outputs),
            nn.Dropout(0.20),
            nn.ReLU(),
            nn.Sigmoid())

    def forward(self, inputs_1, inputs_2):
        emb_1 = self.embedding_1(inputs_1)
        emb_2 = self.embedding_2(inputs_2)
        emb = torch.cat((emb_1, emb_2), dim=1)
        inputs = self.hidden_layer(emb)
        outputs = self.output_layer(inputs)
        return outputs


# the training framework of FNN model
class FNN(object):
    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    def fit(self, x_train, x_valid=None,
            batch_size=128, epochs=25,
            lr=1e-3, weight_decay=0, lambda_l1=0,
            early_stopping_steps=-1):
        pass

    def predict(self):
        pass

    def save(self, file_name='current'):
        torch.save(self.model.state_dict(), file_name+'_FNN_model.pth')

    def __train_fn__(self, optimizer, data_loader, lambda_l1):
        self.model.train()
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            inputs, targets = inputs.to(self.DEVICE), targets.to(self.DEVICE)
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            if not lambda_l1:
                loss += lambda_l1*self.__l1_loss(self.model)
            loss.backward()
            optimizer.step()

    def __valid_fn__(self, data_loader):
        self.model.eval()
        final_loss = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.DEVICE), targets.to(self.DEVICE)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                final_loss += loss.item()
            final_loss /= len(data_loader)
        return final_loss

    @staticmethod
    def __initialize_params(layers):
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1.)
                nn.init.constant_(layer.bias, 0.)
                nn.init.constant_(layer.running_mean, 0.)
                nn.init.constant_(layer.running_var, 1.)

    @staticmethod
    def __l1_loss(model):
        l1_loss = 0
        for param in model.parameters():
            l1_loss += torch.sum(abs(param))
        return l1_loss

    @staticmethod
    def __data_generation(x, y, batch_size, shuffle):
        x_tensor = torch.tensor(x, dtype=torch.float)
        y_tensor = torch.tensor(y, dtype=torch.float)
        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return data_loader
