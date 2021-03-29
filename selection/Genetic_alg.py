import numpy as np
from dataset import load_data
from SVM import runSVM


def fun(mask, X, X_t, y, y_t, svm_weight, features_weight, svm_C=5, svm_k='rbf'):
    X = X[:, mask != 0]
    X_t = X_t[:, mask != 0]
    SVM_score = runSVM(svm_C, svm_k, X, y, X_t, y_t)
    return svm_weight*SVM_score + features_weight*np.sum(mask)


class GA:
    def __init__(self, dim, size, X, X_t, y, y_t, MR=0.5, fit_fun=fun, svm_C=5, svm_k='rbf', SVM_weight=0.75,
                 features_weight=0.25):
        self.MR = MR
        self.dim = dim  # 维度
        self.size = size  # 总群个数
        self.fit_fun = fit_fun
        self.X = X
        self.X_t = X_t
        self.y = y
        self.y_t = y_t
        self.svm_C = svm_C
        self.svm_k = svm_k
        self.SVM_weight = SVM_weight
        self.features_weight = features_weight
        # initialize all chromosomes
        self.unit_list = np.random.randint(0, 2, (size, dim))

    # 变异
    def mutation(self):
        size = self.unit_list.shape[0]
        prob = np.random.uniform(0, 1, size)
        pos = np.random.randint(0, self.dim, size)
        pos_all = np.eyes(self.dim)[pos]
        prob = prob > self.MR
        mask = np.repeat(prob, self.dim).reshape(size, self.dim)
        self.unit_list += mask*pos_all
        self.unit_list %= 2

    # 交叉
    def crossover(self):
        crosspoint = self.dim/2
        size = self.unit_list.shape[0]
        pairs = np.arange(size)
        np.random.shuffle(pairs)
        pairs = pairs.reshape(2, -1)
        for f, m in pairs:
            father = self.unit_list[f]
            mother = self.unit_list[m]
            child1 = np.append(father[0: crosspoint], mother[crosspoint:])
            child2 = np.append(mother[0: crosspoint], father[crosspoint:])
            self.unit_list[f] = child1
            self.unit_list[m] = child2

    # 选择
    def selection(self, svm_weight=1, feature_weight=1):
        score_list = np.array([self.fit_fun(unit, self.X, self.X_t, self.y, self.y_t, self.SVM_weight, self.features_weight,
                                            self.svm_C, self.svm_k) for unit in self.unit_list])
        print("current score list: ", score_list)
        index_list = np.argsort(-score_list)[:self.dim/2]
        self.unit_list = self.unit_list[index_list]

    def update(self):
        while self.unit_list.shape[0] != 1:
            self.selection()
            self.crossover()
            self.mutation()
        return self.unit_list[0]
