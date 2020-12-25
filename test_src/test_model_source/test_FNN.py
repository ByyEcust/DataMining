"""
This is a test file and a quick start of FNN based on pytorch
Author: Ruoqiu Zhang (Ecustwaterman, waterteam), 2020.12.25
"""

import numpy as np
import torch.nn as nn
from src.model_source.FNN import FnnModel, FNN

# data generation
x_train, y_train = np.random.randn(256, 128), np.random.randn(256, 1)
x_valid, y_valid = np.random.randn(256, 128), np.random.randn(256, 1)

# model definition
model = FnnModel(num_inputs=x_train.shape[1], num_outputs=y_train.shape[1], hidden_size=128)

# training
fnn_model = FNN(model=model, loss_fn=nn.MSELoss())
fnn_model.fit(x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid,
              batch_size=128, epochs=25, lr=1e-3,
              weight_decay=1e-5, lambda_l1=0, early_stopping_steps=-1)

# prediction
valid_pre = fnn_model.predict(x_valid)
print(valid_pre[:5])

# model saving
fnn_model.save('test_model')






