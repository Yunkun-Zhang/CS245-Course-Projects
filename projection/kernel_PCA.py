from dataset import load_data
import numpy as np
import scipy


class kernelPCA:
    def __init__(self, data, dim, kernel_name="linear",
                 p_c=0, p_d=1, g_sigma=1, s_alpha=1, s_c=0):
        self.kernel_name = kernel_name
        self.data = data
        self.dim = dim
        self.p_c = p_c
        self.p_d = p_d
        self.g_sigma = g_sigma
        self.s_alpha = s_alpha
        self.s_c = s_c
        assert dim <= data.shape[1]

    def _decentralize(self):
        self.data -= np.mean(self.data, axis=0)
        print("Decentralization done.")

    def _get_kernel_matrix(self):
        if self.kernel_name == "linear":
            K = np.matmul(self.data, self.data.T)
        elif self.kernel_name == "poly":
            K = np.power((np.matmul(self.data, self.data.T) + self.p_c), self.p_d)
        elif self.kernel_name == "gaussion":
            D2 = np.sum(self.data * self.data, axis=1, keepdims=True) \
                 + np.sum(self.data * self.data, axis=1, keepdims=True).T \
                 - 2 * np.dot(self.data, self.data.T)
            K = np.exp(-D2 / (2 * self.g_sigma ** 2))
        elif self.kernel_name == "sigmoid":
            K = np.tanh(self.s_alpha * np.dot(self.data, self.data.T) + self.s_c)
        else:
            raise ValueError("Not implemented yet")
        print("Compute kernel matrix done.")

        return K

    def compute(self):
        self._decentralize()
        K = self._get_kernel_matrix()
        e_vals, e_vecs = scipy.sparse.linalg.eigs(K, self.dim, which="LM")
        print("Eigenvalue decomposition done.")
        return np.matmul(K, e_vecs)


def processed_data_with_kernalPCA(
        dim,
        kernel_name="linear",
        p_c=0, p_d=1,  # parameters for poly kernel
        g_sigma=1,  # parameters for gaussion kernel
        s_alpha=1, s_c=0  # parameters for sigmoid kernel
):
    X_train, X_test, y_train, y_test = load_data()
    len_train = X_train.shape[0]
    X_all = np.concatenate((X_train, X_test), axis=0)

    kpca_instance = kernelPCA(X_all, dim, kernel_name,
                              p_c=p_c, p_d=p_d,
                              g_sigma=g_sigma,
                              s_alpha=s_alpha, s_c=s_c)
    reduced_data = kpca_instance.compute()

    return np.real(reduced_data[:len_train]), \
           np.real(reduced_data[len_train:]), \
           y_train, y_test


if __name__ == "__main__":
    a, b, c, d = processed_data_with_kernalPCA(dim=4)
    print(a[0])