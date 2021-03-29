from sklearn.svm import SVC
from multiprocessing import Pool
from dataset import load_data


def runSVM(C, kernel, X_train, y_train, X_test, y_test):
    svc = SVC(C=C, kernel=kernel, gamma='auto')
    svc.fit(X_train, y_train)
    score = svc.score(X_test, y_test)
    return score


def get_best_C(kernel, X_train, y_train, X_test, y_test):
    crange = [0.001, 0.01, 0.1, 1, 10, 100]
    ps = Pool(len(crange))
    for c in crange:
        ps.apply_async(runSVM, args=(c, kernel, X_train, y_train, X_test, y_test))
    ps.close()
    ps.join()


if __name__ == '__main__':
    X, X_t, y, y_t = load_data()
    # runSVM(50, 'rbf', X, y, X_t, y_t)
    get_best_C('rbf', X, y, X_t, y_t)
