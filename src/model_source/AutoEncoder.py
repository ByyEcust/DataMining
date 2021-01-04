"""
Auto Encoder based on Pytorch
*** Demo and simple examples of naive Auto-Encoder Pytorch Structure
    class AutoEncoder: An example of auto-encoder pytorch class
    class AutoEncoderTraining: class of auto-encoder training framework
Author: Ruoqiu Zhang (Ecustwaterman, waterteam), 2020.12.05
"""
import copy

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

    def fit(self, x_train, x_valid=None,
            batch_size=128, num_epoch=25,
            lr=1e-3, weight_decay=0, lambda_l1=0,
            early_stopping_steps=-1):
        # initialize
        early_step = 0
        best_loss = np.inf
        best_model_params = self.model.state_dict()
        self.__initialize_params(self.model.encoder)
        self.__initialize_params(self.model.decoder)
        self.model.to(self.DEVICE)
        # definition optimizer and scheduler
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
        # generation data loader
        train_loader = self.__data_generation(x_train, x_train, batch_size, shuffle=True)
        print('+++ initial loss of train data: ' + str(self.__initial_loss(x_train)))
        if x_valid is not None:
            valid_loader = self.__data_generation(x_valid, x_valid, batch_size, shuffle=True)
            print('+++ initial loss of valid data: ' + str(self.__initial_loss(x_valid)))
        # epoch iteration
        for epoch in range(num_epoch):
            self.__train_fn(optimizer, train_loader, lambda_l1)
            train_loss = self.__valid_fn(train_loader)
            print('EPOCH: %d train_loss: %f' % (epoch, train_loss))
            if x_valid is not None:
                valid_loss = self.__valid_fn(valid_loader)
                print('EPOCH: %d valid_loss: %f' % (epoch, valid_loss))
            else:
                valid_loss = train_loss
            scheduler.step(valid_loss)
            # early-stopping
            if valid_loss < best_loss:
                best_loss = valid_loss
                early_step = 0
                best_model_params = copy.deepcopy(self.model.state_dict())
            elif early_stopping_steps != -1:
                early_step += 1
                if early_step >= early_stopping_steps:
                    break
        self.model.load_state_dict(best_model_params)

    def predict(self, test_data):
        print('+++ initial loss of prediction data: ' + str(self.__initial_loss(test_data)))
        test_data = torch.tensor(test_data, dtype=torch.float)
        test_data = test_data.to(self.DEVICE)
        self.model.eval()
        with torch.no_grad():
            encoded, decoded = self.model(test_data)
        encoded = encoded.detach().cpu().numpy()
        decoded = decoded.detach().cpu().numpy()
        return encoded, decoded

    def save(self, file_name='current'):
        torch.save(self.model.state_dict(), file_name + '_auto_encoder_model.pth')

    def __train_fn(self, optimizer, data_loader, lambda_l1):
        self.model.train()
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            inputs, targets = inputs.to(self.DEVICE), targets.to(self.DEVICE)
            _, outputs = self.model(inputs)
            # L1 regularization
            if not lambda_l1:
                loss = self.loss_fn(outputs, targets) + lambda_l1*self.__l1_loss(self.model)
            else:
                loss = self.loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

    def __valid_fn(self, data_loader):
        self.model.eval()
        final_loss = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.DEVICE), targets.to(self.DEVICE)
                _, outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                final_loss += loss.item()
            final_loss /= len(data_loader)
        return final_loss

    def __initial_loss(self, data):
        data_mean = np.mean(data, axis=0)
        data_predict = np.tile(data_mean, data.shape[0]).reshape(data.shape[0], data.shape[1])
        data_predict_tensor = torch.Tensor(data_predict)
        data_tensor = torch.Tensor(data)
        initial_loss = self.loss_fn(data_predict_tensor, data_tensor)
        return initial_loss.item()

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



