import numpy as np
from SVM import runSVM
from multiprocessing import Pool


def get_random_by_rate(rate, size):
    raw = np.random.uniform(0, 1, size)
    return (raw < rate).astype(int)

class GA:
    def __init__(self, dim, size, X, X_t, y, y_t, max_iter=5, IR=0.3, CR=0.5, MR=0.5, svm_C=5, svm_k='rbf', SVM_weight=0.75,
                 features_weight=0.25):
        self.MR = MR
        self.CR = CR
        self.max_iter = max_iter
        self.dim = dim  # 维度
        self.size = size  # 总群个数
        self.X = X
        self.X_t = X_t
        self.y = y
        self.y_t = y_t
        self.svm_C = svm_C
        self.svm_k = svm_k
        self.SVM_weight = SVM_weight
        self.features_weight = features_weight
        # initialize all chromosomes
        self.initialize_by_rate(IR)
        self.scores = np.zeros(size)

    def initialize_by_rate(self, rate):
        raw = np.random.uniform(0, 1, (self.size, self.dim))
        self.unit_list = (raw < rate).astype(int)

    def get_score_by_id(self, id, X, X_t, y, y_t, svm_weight, features_weight, svm_C=5, svm_k='rbf', seed=None):
        mask = self.unit_list[id]
        X = X[:, mask != 0]
        X_t = X_t[:, mask != 0]
        print(f"current feature dim: {X.shape[1]}")
        np.random.seed(seed)
        select_image = get_random_by_rate(0.1, X.shape[0])
        X = X[select_image == 1, :]
        print(f"current X length: {X.shape[0]}")
        SVM_score = runSVM(svm_C, svm_k, X, y, X_t, y_t)
        print("score is: ", SVM_score)
        return id, svm_weight*SVM_score + 1/(features_weight*np.sum(mask))

    def apply_score(self, id, score):
        self.scores[id] = score

    # 变异
    def mutation(self):
        size = self.unit_list.shape[0]
        prob = np.random.uniform(0, 1, size)
        pos = np.random.randint(0, self.dim, size)
        pos_all = np.eye(self.dim)[pos].astype(int)
        prob = prob < self.MR  # Mutation with a possibility of MR
        mask = np.repeat(prob, self.dim).reshape(size, self.dim)
        self.unit_list += mask*pos_all
        self.unit_list %= 2

    # 交叉
    def crossover(self):
        new_unit_list = []
        for unit in self.unit_list:
            child = unit
            if np.random.rand() < self.CR:  # Crossover with a possibility of CR
                mother = self.unit_list[np.random.randint(self.size)]
                cross_point = np.random.randint(0, self.dim)
                child[cross_point:] = mother[cross_point:]
            new_unit_list.append(child)
        self.unit_list = np.array(new_unit_list)

    # 选择
    def selection(self, iter_num):
        ps = Pool(self.size)
        result = []
        for id in range(self.size):
            result.append(ps.apply_async(self.get_score_by_id, args=(id, self.X, self.X_t, self.y, self.y_t, self.SVM_weight,
                                                       self.features_weight, self.svm_C, self.svm_k, iter_num)))
        ps.close()
        ps.join()
        for i in result:
            self.scores[i.get()[0]] = i.get()[1]
        idx = np.random.choice(np.arange(self.size), self.size, replace=True, p=self.scores/self.scores.sum())
        self.unit_list = self.unit_list[idx]

    def update(self):
        for i in range(self.max_iter):
            self.selection(i)
            self.crossover()
            self.mutation()
        # find the best one in the final list of chromosomes
        ps = Pool(self.size)
        result = []
        for id in range(self.size):
            result.append(ps.apply_async(self.get_score_by_id, args=(id, self.X, self.X_t, self.y, self.y_t, self.SVM_weight,
                                                       self.features_weight, self.svm_C, self.svm_k)))
        ps.close()
        ps.join()
        for i in result:
            self.scores[i.get()[0]] = i.get()[1]
        max_id = np.argmax(self.scores)
        return self.unit_list[max_id], np.max(self.scores)
