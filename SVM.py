import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from multiprocessing import Pool
from dataset import load_data


def runSVM(C, kernel, X_train, y_train, X_test, y_test):
    svc = SVC(C=C, kernel=kernel, gamma='auto')
    svc.fit(X_train, y_train)
    score = svc.score(X_test, y_test)
    return score


def runKFold(X, y, C, kernel, K=5):
    kf = KFold(n_splits=K)
    s = 0
    for (X_train, X_test), (y_train, y_test) in zip(kf.split(X), kf.split(y)):
        s += runSVM(C, kernel, X[X_train], y[y_train], X[X_test], y[y_test])
        print('.', end='')
    return s / K


def get_best_C(kernel, X, y):
    crange = [0.001, 0.01, 0.1, 1, 10, 100]
    result = []
    ps = Pool(len(crange))
    for c in crange:
        result.append((c, ps.apply_async(runKFold, args=(X, y, c, kernel, 5)).get()))
    ps.close()
    ps.join()
    return result


if __name__ == '__main__':
    X, X_t, y, y_t = load_data()
    print(runKFold(X, y, 1, 'linear'))
