"""
This is a test file and a quick start of Auto-Encoder based on pytorch
Author: Ruoqiu Zhang (Ecustwaterman, waterteam), 2020.12.23
"""

from src.model_source.AutoEncoder import AutoEncoder, AutoEncoderTraining
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# test data generation
df_train = pd.DataFrame(np.random.randn(256, 64), columns=['feat_' + str(i) for i in range(64)])
df_valid = pd.DataFrame(np.random.randn(256, 64), columns=['feat_' + str(i) for i in range(64)])

# NN model defined
model = AutoEncoder(num_features=df_train.shape[1], num_compress=8)

# Auto Encoder training framework defined
auto_encoder = AutoEncoderTraining(model=model, loss_fn=torch.nn.MSELoss())

# training
auto_encoder.fit(x_train=df_train.values, x_valid=df_valid.values)

# res obtaining
encoded, decoded = auto_encoder.predict(df_valid.values)

# model saving
auto_encoder.save()

