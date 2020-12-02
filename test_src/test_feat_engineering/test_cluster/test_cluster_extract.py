# Author Yaoyao Bao
# yaoyaobao@mail.ecust.edu.cn

import numpy as np
from sklearn.cluster import KMeans
from src.feat_engineering.cluster.cluster_extract import ClusterExtract


class TestClusterExtract:
    def test1(self):
        np.random.seed(0)
        X = np.vstack((np.random.randn(3, 2), np.random.randn(3, 2) + 10))
        fc_agent = ClusterExtract(KMeans, n_clusters=2)
        tmp_feat = fc_agent.fit_transform(X)
        true_res = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])
        np.testing.assert_array_almost_equal(tmp_feat.values, true_res, decimal=2,
                                             err_msg="Cluster Feat Extract By KMeans Failed.")

