import torch
import openTSNE
import sys
# import plot_utils
import numpy as np
from sklearn.manifold import TSNE

sys.path.append("..")
from dataset import load_data
import matplotlib.pyplot as plt
from projection.kernel_PCA import kernelPCA


class myTSNE:
    def __init__(self, data, dim, tol, lr, max_iter,device):
        self.device=device
        self.data = torch.tensor(data).type(torch.float32).to(self.device)
        self.dim = dim
        self.KL_criterion = torch.nn.KLDivLoss(size_average=False)
        self.tol = tol
        self.lr = lr
        self.max_iter = max_iter

    def _distance_matrix(self, matrix):
        # (i,i) indicating the distance within this matrix
        return torch.square(torch.cdist(matrix, matrix))

    def _conditional_prob(self, D2_matrix):
        # given matrix, calculate p(*|i)
        # D2_matrix: (#sample, #sample)
        up = torch.pow(D2_matrix + 1, -1)  # (#sample, #sample)
        down = (torch.sum(up, dim=1))  # (#sample)
        res = torch.div(up, down)
        # for i in range(D2_matrix.shape[0]):
        #    res[i][i] = 0
        return res  # (#sample, #sample)

    def _KL(self, cp_true, cp_pred):
        return self.KL_criterion(torch.log(cp_pred), cp_true)

    def compute(self):
        reduced_matrix = torch.randn(self.data.shape[0], self.dim).to(self.device)
        reduced_matrix.requires_grad = True
        cp_true = self._conditional_prob(self._distance_matrix(self.data))
        iter = 1
        prev_loss = None
        optimizer = torch.optim.AdamW([reduced_matrix], lr=self.lr)

        while True:
            optimizer.zero_grad()
            cp_pred = self._conditional_prob(self._distance_matrix(reduced_matrix))
            loss = self._KL(cp_true, cp_pred)
            print("ITER: ", iter)
            print("Loss: ", loss)
            loss.backward()
            optimizer.step()
            if iter >= self.max_iter or \
                    (prev_loss is not None and abs(loss - prev_loss) <= self.tol):
                break
            prev_loss = loss
            iter += 1
        return reduced_matrix.detach().to("cpu").numpy()


if __name__ == "__main__":
    X, X_t, y, y_t = load_data("../data")
    # tsne = myTSNE(data=X[:10000], dim=2, tol=1e-7, lr=1, max_iter=500)
    # output = tsne.compute()
    output = openTSNE.TSNE(verbose=True).fit(np.concatenate([X, X_t], axis=0))
    # plot_utils.plot(output, np.concatenate([y, y_t], axis=0))
    # plt.show()
