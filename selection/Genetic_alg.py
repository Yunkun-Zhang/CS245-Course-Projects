import numpy as np
from SVM import runSVM
from multiprocessing import Pool


def get_random_by_rate(rate, size):
    raw = np.random.uniform(0, 1, size)
    return (raw < rate).astype(int)


class GA:
    def __init__(self, dim, size, X, X_t, y, y_t, max_iter=5, IR=0.3, CR=0.5, MR=0.5, svm_C=5, svm_k='linear',
                 svm_weight=0.9, features_weight=0.1):
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
        self.svm_weight = svm_weight
        self.features_weight = features_weight
        # initialize all chromosomes
        self.initialize_by_rate(IR)
        self.scores = np.zeros(size)

    def initialize_by_rate(self, rate):
        raw = np.random.uniform(0, 1, (self.size, self.dim))
        self.unit_list = (raw < rate).astype(int)

    def get_score_by_id(self, id, sample_rate=0.05, only_svm=False, seed=None):
        mask = self.unit_list[id]
        X = self.X[:, mask != 0]
        X_t = self.X_t[:, mask != 0]
        y = self.y
        print(f"current feature dim: {X.shape[1]}")
        if sample_rate < 1:
            np.random.seed(seed)
            select_image = get_random_by_rate(sample_rate, X.shape[0])
            X = X[select_image == 1, :]
            y = self.y[select_image == 1]
        print(f"current X length: {X.shape[0]}")
        SVM_score = runSVM(self.svm_C, self.svm_k, X, y, X_t, self.y_t)
        if only_svm:
            return id, SVM_score
        with open("ga_result.txt", 'a') as f:
            f.write(f"{id}\t{SVM_score}\t{self.svm_weight*SVM_score + self.features_weight/np.sum(mask)}\n")
        print("score is: ", SVM_score)
        return id, self.svm_weight*SVM_score + self.features_weight/np.sum(mask)

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
        for index in range(len(self.unit_list)):
            child = self.unit_list[index]
            if np.random.rand() < self.CR:  # Crossover with a possibility of CR
                # find a strictly different node
                new_index = index
                while index == new_index:
                    new_index = np.random.randint(self.size)
                mother = self.unit_list[new_index]
                cross_point = np.random.randint(0, self.dim)
                child[cross_point:] = mother[cross_point:]
            new_unit_list.append(child)
        self.unit_list = np.array(new_unit_list)

    # 选择
    def selection(self, iter_num):
        ps = Pool(self.size)
        result = []
        for id in range(self.size):
            result.append(ps.apply_async(self.get_score_by_id, args=(id, 1, False, iter_num)))
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
            result.append(ps.apply_async(self.get_score_by_id, args=(id, 1, True, None)))
        ps.close()
        ps.join()
        for i in result:
            self.scores[i.get()[0]] = i.get()[1]
        max_id = np.argmax(self.scores)
        return self.unit_list[max_id], np.max(self.scores)
