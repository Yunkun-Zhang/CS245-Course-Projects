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
        score = runSVM(C, kernel, X[X_train], y[y_train], X[X_test], y[y_test])
        s += score
        with open('C.txt', 'a') as f:
            f.write(f'{C}, {score}\n')
    return s / K


def get_best_C(kernel, X, y):
    crange = [0.01, 0.1, 1, 10, 100]
    result = []
    ps = Pool(len(crange))
    for c in crange:
        result.append((c, ps.apply_async(runKFold, args=(X, y, c, kernel, 5)).get()))
    ps.close()
    ps.join()
    return result


if __name__ == '__main__':
    X, X_t, y, y_t = load_data()
    res = get_best_C('linear', X, y)
    print(res)
    res.sort(key=lambda x: x[1], reverse=True)
    c = res[0][0]
    score = runSVM(res[0][0], 'linear', X, y, X_t, y_t)
    with open('C.txt', 'a') as f:
        f.write(f'{c}, score = {score}\n')
