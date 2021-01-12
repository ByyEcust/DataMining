"""
This is a test file and a quick start of Target encoding
Author: Ruoqiu Zhang (Ecustwaterman, waterteam), 2021.01.06
"""
import numpy as np
from sklearn import datasets
from src.feat_engineering.TargetEncoding import TargetEncoding


class TestTargetEncoding:
    def test_fit_transform(self):
        # data generation
        boston = datasets.load_boston()
        x = np.array(['A'] * 200 + ['B'] * 200 + ['C'] * 106)
        y = boston.target

        # target encoding
        te = TargetEncoding(num_outer_folds=5, num_inner_folds=5)
        label_pre1 = te.fit_transform(x, y)
        label_pre2 = te.transform(x)
        np.testing.assert_array_almost_equal(label_pre1[0], 23.33726682, decimal=2, err_msg="target encoding error")
        np.testing.assert_array_almost_equal(label_pre1[201], 25.02934193, decimal=2, err_msg="target encoding error")
        np.testing.assert_array_almost_equal(label_pre1[501], 15.61133666, decimal=2, err_msg="target encoding error")
        np.testing.assert_array_almost_equal(label_pre2[0], 23.25702477, decimal=2, err_msg="target encoding error")
        np.testing.assert_array_almost_equal(label_pre2[201], 25.40436784, decimal=2, err_msg="target encoding error")
        np.testing.assert_array_almost_equal(label_pre2[501], 15.73335422, decimal=2, err_msg="target encoding error")
