import numpy as np


class FFS:
    def __init__(self, X, X_t, dim):
        self.X = X
        self.X_t = X_t
        self.dim = dim

    def get_new_features(self):
        var_array = np.var(self.X, axis=0)
        index = np.argsort(-var_array)
        # print(self.X[:, index[:self.dim]].shape)
        return self.X[:, index[:self.dim]], self.X_t[:, index[:self.dim]]
