import numpy as np
import torch
import scipy
import sys

sys.path.append("..")
from dataset import load_data


class kernelPCA:
    def __init__(self, data, dim, device, kernel_name="linear",
                 p_c=0, p_d=1, g_sigma=1, s_alpha=1, s_c=0, ):
        self.kernel_name = kernel_name
        self.data = data
        self.dim = dim
        self.p_c = p_c
        self.p_d = p_d
        self.g_sigma = g_sigma
        self.s_alpha = s_alpha
        self.s_c = s_c
        assert dim <= data.shape[1]

        self.device = device

    def _decentralize(self):
        self.data -= np.mean(self.data, axis=0)
        self.data = torch.tensor(self.data).type(torch.float32).to(self.device)
        print("Decentralization done.")

    def _get_kernel_matrix(self):
        if self.kernel_name == "linear":
            K = torch.mm(self.data, self.data.t())
        elif self.kernel_name == "poly":
            K = torch.pow((torch.mm(self.data, self.data.t()) + self.p_c), self.p_d)
        elif self.kernel_name == "gaussian":
            K = torch.exp(-torch.square(torch.cdist(self.data, self.data)) / (2 * self.g_sigma ** 2))
        elif self.kernel_name == "sigmoid":
            K = torch.tanh(self.s_alpha * torch.mm(self.data, self.data.t()) + self.s_c)
        else:
            raise ValueError("Not implemented yet")
        print("Compute kernel matrix done.")

        return K

    def compute(self):
        self._decentralize()
        K = self._get_kernel_matrix()
        e_vals, e_vecs = torch.lobpcg(K, k=self.dim, largest=True)
        print("Eigenvalue decomposition done.")
        return torch.mm(K, e_vecs).to('cpu').numpy()


if __name__ == "__main__":
    X, _, _, _ = load_data("../data")
    kpca = kernelPCA(X[:10000], 5)
    kpca.compute()
