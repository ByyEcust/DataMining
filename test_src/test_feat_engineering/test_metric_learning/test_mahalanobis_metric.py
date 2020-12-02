# Author Yaoyao Bao
# yaoyaobao@mail.ecust.edu.cn

import numpy as np
import torch
from src.feat_engineering.metric_learning.mahalanobis_metric import MaxCorr


class TestMaxCorr:
    def test1(self):
        X = np.random.rand(100, 10)
        Y = np.random.rand(100, 5)
        agent = MaxCorr(2)
        x_trans = agent.fit_transform(X, Y, verbose=None, early_stop=20)
        np.testing.assert_array_almost_equal(x_trans.shape, np.array([100, 2]), decimal=2,
                                             err_msg="MaxCorr dimension reduction failed!")
        np.testing.assert_array_almost_equal(torch.tensor(agent.trans_mat.shape).data.numpy(), np.array([2, 10]),
                                             decimal=2, err_msg="MaxCorr dimension reduction failed!")
