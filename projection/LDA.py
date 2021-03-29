from dataset import load_data
import numpy as np
import scipy


class LDA:
    def __init__(self, data, dim, labels, ori_dim=2048, classes=50):
        self.data = data
        self.dim = dim
        self.labels = labels
        self.classes = classes
        self.ori_dim = ori_dim

    def _process_data(self):
        self.global_mu = np.mean(self.data, 0)
        self.parts = [np.empty([0, self.ori_dim]) for _ in range(self.classes)]
        for feature, label in zip(self.data, self.labels):
            self.parts[label[0] - 1] = np.concatenate(
                (self.parts[label[0] - 1], feature.reshape((1, -1))), axis=0)
        self.loacl_mus = [np.mean(x, 0) for x in self.parts]

    def compute(self):
        S_w = np.zeros([self.ori_dim, self.ori_dim])
        S_b = np.zeros([self.ori_dim, self.ori_dim])
        for i in range(self.classes):
            S_b += self.parts[i].shape[0] * \
                   np.matmul((self.loacl_mus[i] - self.global_mu).reshape(-1, 1),
                             (self.loacl_mus[i] - self.global_mu).reshape(1, -1))

            S_w += np.matmul((self.parts[i] - self.loacl_mus[i]).T,
                             (self.parts[i] - self.loacl_mus[i]))
        S_wm1S_b =np.matmul(np.linalg.pinv(S_w),S_b)
        e_vals, e_vecs = scipy.sparse.linalg.eigs(S_wm1S_b, self.dim, which="LM")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    LDA_ins = LDA(X_train, 4, y_train, 2048, 50)
    LDA_ins.process_data()