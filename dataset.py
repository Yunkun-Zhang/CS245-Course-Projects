import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def read(path='Animals_with_Attributes2/Features/ResNet101'):
    X = pd.read_csv(path + '/AwA2-features.txt', sep=' ', names=[f'feature{i}' for i in range(2048)])
    y = pd.read_csv(path + '/AwA2-labels.txt', names=['label'])
    return X, y


def divide_data(save_path='data', test=0.4):
    X, y = read()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test)
    np.save(save_path + '/X_train', X_train)
    np.save(save_path + '/X_test', X_test)
    np.save(save_path + '/y_train', y_train)
    np.save(save_path + '/y_test', y_test)


def load_data(path='data'):
    X_train = np.load(path + '/X_train.npy')
    X_train = X_train / np.max(X_train)
    X_test = np.load(path + '/X_test.npy')
    X_test = X_test / np.max(X_test)
    y_train = np.squeeze(np.load(path + '/y_train.npy'))
    y_test = np.squeeze(np.load(path + '/y_test.npy'))
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    divide_data()
    x, _, y, _ = load_data()
