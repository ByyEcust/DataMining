"""
Logistic regression (LR), Factorization machine (FM)
Author: Ruoqiu Zhang (Ecustwaterman, waterteam), 2021.01.05
"""


import copy
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data


# the basic LR model built by pyTorch
class LR(nn.Module):
    def __init__(self, dim_features, dim_outputs):
        super(LR, self).__init__()
        self.linear_weight = nn.Linear(dim_features, dim_outputs, bias=True)
        self.__initialize_params()

    def forward(self, x):
        output = self.linear_weight(x)
        output = torch.sigmoid(output)
        output = output.squeeze(-1)
        return output

    def __initialize_params(self):
        nn.init.normal_(self.linear_weight.weight, 0.0, 1.0)
        nn.init.constant_(self.linear_weight.bias, 0.0)


# the basic FM model built by pyTorch
class FM(nn.Module):
    def __init__(self, dim_features, dim_outputs, dim_embedding):
        super(FM, self).__init__()
        self.linear_weight = nn.Linear(dim_features, dim_outputs, bias=True)
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
        output = linear_part + 0.5*(cross_part_1 - cross_part_2).sum(1).unsqueeze(1)
        output = torch.sigmoid(output)
        output = output.squeeze(-1)
        return output

    def __initialize_params(self):
        nn.init.normal_(self.linear_weight.weight, 0.0, 1.0)
        nn.init.constant_(self.linear_weight.bias, 0.0)


# the training framework of FNN model
class FNN(object):
    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    def fit(self, x_train, y_train,
            x_valid=None, y_valid=None,
            batch_size=128, epochs=25,
            lr=1e-3, weight_decay=0, lambda_l1=0,
            early_stopping_steps=-1):
        # initialize
        early_step = 0
        best_loss = np.inf
        self.model.to(self.DEVICE)
        # define optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
        # generation of data loader
        train_loader = self.__data_generation(x_train, y_train, batch_size=batch_size, shuffle=True)
        if x_valid is not None:
            valid_loader = self.__data_generation(x_valid, y_valid, batch_size=batch_size, shuffle=False)
        # epoch iteration
        best_model_params = copy.deepcopy(self.model.state_dict())
        for epoch in range(epochs):
            self.__train_fn__(optimizer, train_loader, lambda_l1)
            train_loss = self.__valid_fn__(train_loader)
            print('EPOCH: %d train_loss: %f' % (epoch, train_loss))
            if x_valid is not None:
                valid_loss = self.__valid_fn__(valid_loader)
                print('EPOCH: %d valid_loss: %f' % (epoch, valid_loss))
            else:
                valid_loss = train_loss
            scheduler.step(valid_loss)
            # early stopping
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
        # avoid lack of RAM in cuda device
        self.model.to('cpu')
        self.model.eval()
        test_data = torch.Tensor(test_data)
        test_data = test_data.to('cpu')
        with torch.no_grad():
            output = self.model(test_data)
        output = output.detach().cpu().numpy()
        return output

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
