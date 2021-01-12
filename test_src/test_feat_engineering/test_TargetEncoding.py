"""
This is a test file and a quick start of Target encoding
Author: Ruoqiu Zhang (Ecustwaterman, waterteam), 2021.01.06
"""
import numpy as np
from sklearn import datasets
from src.feat_engineering.TargetEncoding import TargetEncoding

# data generation
boston = datasets.load_boston()
x = cate_feat = ['A'] * 200 + ['B'] * 200 + ['C'] * 106
x = np.array(x)
y = boston.target

# target encoding
te = TargetEncoding(num_outer_folds=5, num_inner_folds=5)
label_pre_train = te.fit(x, y)

# obtain target encoded
label_pre = te.transform(x)
print(label_pre[:5])
print(label_pre[201:206])
print(label_pre[501:506])