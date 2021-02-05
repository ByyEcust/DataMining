"""
SMOTE regular version for over-sampling
***
    The current version regards all features as numerical features;
    All categorical features should be encoded by encoder (one-hot, target encoder or Auto-encoder);
***
Author: Ruoqiu Zhang (Ecustwaterman, waterteam), 2021.02.04
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


class SMOTE(object):
    def __init__(self, num_generation=10):
        """
        :param num_generation:  the number of generated pos-samples
        :param random_state:    random seed
        """
        self.num_generation = num_generation

    def fit(self, X, verbose=10):
        """
        :param verbose:     num of verbose to print (int)
        :param X:           pos-samples (numpy array-like)
        :return:            generated samples (numpy array-like)
        """
        self.X_idx = list(range(X.shape[0]))
        self.__calculate_neighbors(X)
        X_generation = np.zeros((self.num_generation, X.shape[1]))
        # main loop
        for i in range(self.num_generation):
            x = self.__random_choice(X)  # random select one pos-sample
            neighbor_idx = self.__find_neighbors(x)  # find the nearest sample idx of x
            x_neighbor = X[neighbor_idx, :]
            x_generation = self.__sample_generation(x, x_neighbor)
            X_generation[i, :] = x_generation
            if not i % verbose:
                print('the number of generated samples is %d' % i)
        return X_generation

    def __calculate_neighbors(self, x):
        self.neighbor = NearestNeighbors(n_neighbors=2, n_jobs=-1)
        self.neighbor.fit(x)

    def __find_neighbors(self, x):
        return int(self.neighbor.kneighbors(x.reshape(1, -1), return_distance=False)[0][1:])

    def __random_choice(self, X):
        x_idx = np.random.choice(self.X_idx, size=1)
        return X[int(x_idx), :]

    @staticmethod
    def __sample_generation(x, x_neighbor):
        num_features = len(x)
        rebuilt_sample = np.zeros(num_features)
        for feature in range(num_features):
            if x[feature] < x_neighbor[feature]:
                rebuilt_sample[feature] = np.random.uniform(x[feature], x_neighbor[feature], 1)
            else:
                rebuilt_sample[feature] = np.random.uniform(x_neighbor[feature], x[feature], 1)
        return rebuilt_sample



