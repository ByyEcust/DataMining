'''
This is a test file and a quick start of Auto-Encoder based on pytorch
Author: Ruoqiu Zhang (Ecustwaterman, waterteam), 2020.12.23
'''

from AutoEncoder import AutoEncoder, AutoEncoderTraining
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# test data generation
df = pd.DataFrame(np.random.randn(256, 64), columns=['feat_' + str(i) for i in range(64)])

# NN model defined
model = AutoEncoder(num_features=df.shape[1], num_compress=8)

# Auto Encoder training framework defined
auto_encoder = AutoEncoderTraining(model=model, loss_fn=torch.nn.MSELoss())

# training
auto_encoder.fit(df.values)

# res obtaining
encoded, decoded = auto_encoder.predict(df.values)

