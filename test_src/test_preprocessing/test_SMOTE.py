"""
This is a test file and a quick start of SMOTE
Author: Ruoqiu Zhang (Ecustwaterman, waterteam), 2021.02.04
"""
import numpy as np
from sklearn import datasets
from src.preprocessing.SMOTE import SMOTE


# data generation
cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target
pos_samples = X[np.concatenate(np.argwhere(y == 1)), :]
print('the shape of pos_samples: ' + str(pos_samples.shape))

# class definition
smote = SMOTE(num_generation=20, random_state=2020)

# over-sampling
generated_samples = smote.fit(pos_samples)

# data shape
print('the shape of generated samples: ' + str(generated_samples.shape))
