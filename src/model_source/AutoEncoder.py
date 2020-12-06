'''
Auto Encoder based on Pytorch
methods:
    * fit: main iteration of epochs
    * predict:
Author: Ruoqiu Zhang (Ecustwaterman, waterteam), 2020.12.05
'''

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data


# class of auto encoder
class AutoEncoder(nn.Module):
    def __init__(self, num_features, num_compress):
        super(AutoEncoder, self).__init__()
        # Definition of encoder & decoder can be various, such as num of layers and hidden size
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_compress))
        self.decoder = nn.Sequential(
            nn.Linear(num_compress, 128),
            nn.ReLU(),
            nn.Linear(128, num_features),
            nn.Sigmoid())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# class of auto encoder training framework
class AutoEncoderTraining(object):
    def __init__(self, model, loss_fn):
        self.model = model  # model is a NN object
        self.loss_fn = loss_fn
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    def fit(self, x_train, x_valid=-1,
            batch_size=128, num_epoch=25,
            lr=1e-3, weight_decay=0, lambda_l1=0,
            early_stopping_steps=0):
        # initialization of NN model
        self.__initialize_params()
        self.model.to(self.DEVICE)
        early_step = 0
        best_loss = np.inf
        # definition optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8)
        # generation data loader
        train_loader = self.__data_generation(x_train, x_train, batch_size, shuffle=True)
        if not x_valid == -1:
            valid_loader = self.__data_generation(x_valid, x_valid, batch_size, shuffle=True)
        # epoch iteration
        for epoch in range(num_epoch):
            self.__train_fn(optimizer, train_loader, lambda_l1)
            if not x_valid == -1:
                train_loss = self.__valid_fn(self.model, train_loader)
                valid_loss = self.__valid_fn(self.model, valid_loader)
                print('EPOCH: %d train_loss: %f' % (epoch, train_loss))
                print('EPOCH: %d valid_loss: %f' % (epoch, valid_loss))
            else:
                valid_loss = valid_fn(self.model, train_loader)
                print('EPOCH: %d train_loss: %f' %(epoch, valid_loss))
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(self.model.state_dict(), 'auto_encoder_model.pth')
            elif early_stopping_steps != -1:
                early_step += 1
                if early_step >= early_stopping_steps:
                    break

    def predict(self, test_data):
        test_data = torch.tensor(test_data, dtype=torch.float)
        self.model.eval()
        with torch.no_grad():
            encoded, decoded = self.model(test_data)
        encoded = np.concatenate(encoded.detach().cpu().numpy())
        decoded = np.concatenate(decoded.detach().cpu().numpy())
        return encoded, decoded

    def __train_fn(self, optimizer, data_loader, lambda_l1):
        self.model.train()
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            inputs, targets = inputs.to(self.DEVICE), targets.to(self.DEVICE)
            _, outputs = self.model(inputs)
            # L1 regularization
            if not lambda_l1:
                l1_loss = 0
                for param in self.model.parameters():
                    l1_loss += torch.sum(abs(param))
                loss = self.loss_fn(outputs, targets) + lambda_l1*l1_loss
            else:
                loss = self.loss_fn(outputs, targets)
            optimizer.step()

    def __valid_fn(self, data_loader):
        self.model.eval()
        final_loss = 0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(self.DEVICE), targets.to(self.DEVICE)
            _, outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            final_loss += loss.item()
        final_loss /= len(data_loader)
        return final_loss

    def __initialize_params(self):
        for layer in self.model.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.)

    def __data_generation(self, x, y, batch_size, shuffle):
        x_tensor = torch.tensor(x, dtype=torch.float)
        y_tensor = torch.tensor(y, dtype=torch.float)
        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return data_loader












