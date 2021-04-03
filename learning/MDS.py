import torch
import sys
import plot_utils
import matplotlib.pyplot as plt
from sklearn.manifold import MDS as sMDS

sys.path.append("..")
from dataset import load_data


class myMDS:
    def __init__(self, data, dim, tol, lr, max_iter, learning=False):
        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        self.data = torch.tensor(data).type(torch.float32).to(self.device)
        self.dim = dim
        self.tol = tol
        self.lr = lr
        self.max_iter = max_iter
        self.learning = learning
        self.num_sample = self.data.shape[0]
        self.D = self._distance_matrix(self.data)
        self.D2 = torch.square(self.D)

    def _process_D2(self, matrix):
        diag = torch.diag(matrix)
        matrix_diag = torch.diag_embed(diag)
        return matrix - matrix_diag

    def _distance_matrix(self, matrix):
        # (i,i) indicating the distance within this matrix
        return torch.cdist(matrix, matrix)

    def _get_B(self):
        H = ((torch.eye(self.num_sample) -
              torch.ones(self.num_sample, self.num_sample) / self.num_sample)) \
            .to(self.device)
        B = -0.5 * torch.mm(torch.mm(H, self.D2), H)
        return B

    def _non_learning(self):
        e_vals, e_vecs = torch.lobpcg(self._get_B(), k=self.dim, largest=True)
        Lambda_sqrt = torch.eye(self.dim).to(self.device) * torch.sqrt(e_vals)
        res = torch.mm(e_vecs, Lambda_sqrt)
        return res.to("cpu").numpy()

    def _learning(self):
        reduced_matrix = torch.rand([self.num_sample, self.dim]).to(self.device)
        reduced_matrix.requires_grad = True
        prev_loss = None
        optimizer = torch.optim.AdamW([reduced_matrix], lr=self.lr)
        for i in range(self.max_iter):
            optimizer.zero_grad()
            D_pred = self._distance_matrix(reduced_matrix)
            loss = self._loss(D_pred, self.D)
            print("ITER: ", i)
            print("Loss: ", loss)
            loss.backward()
            optimizer.step()
            if prev_loss is not None and abs(loss - prev_loss) < self.tol:
                break
            prev_loss = loss
        return reduced_matrix.detach().to("cpu").numpy()

    def _loss(self, D_1, D_2):
        diff2 = torch.pow(D_1 - D_2, 2)
        return torch.sum(diff2) / (self.num_sample ** 2)

    def compute(self):
        if self.learning:
            res = self._learning()
        else:
            res = self._non_learning()
        return res


if __name__ == "__main__":
    X, _, y, _ = load_data("../data")
    mds = myMDS(X[:10000], 2, 1e-5, 1, 100, False)
    output = mds.compute()

    # clf2 = sMDS(2)
    # output = clf2.fit_transform(X[:1000])
    plot_utils.plot(output, y[:10000])
    plt.show()
