from dataset import load_data
from SVM import runSVM
from selection.Genetic_alg import GA
from projection.auto_encoder import AE, VAE
import argparse

# params
svm_C = 5
svm_k = 'rbf'
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--method', default='ae')
parser.add_argument('-d', '--dim', default=64)
parser.add_argument('-b', '--batch', default=128)
args = parser.parse_args()

print('Loading data...')
X, X_t, y, y_t = load_data()


def runAE(auto_encoder_dim):
    print('Training auto encoder...')
    ae = AE(auto_encoder_dim)
    ae.build(X.shape[1])
    ae.auto_encoder.summary()
    ae.fit(X, validation_data=X_t)
    X_ae, Xt_ae = ae.predict(X, X_t)
    score = runSVM(svm_C, svm_k, X_ae, y, Xt_ae, y_t)
    with open('result.txt', 'a') as f:
        f.write(f'AE score: {score} (dim={auto_encoder_dim}, kernel={svm_k})\n')


def runVAE(auto_encoder_dim):
    print('Training validation auto encoder...')
    vae = VAE(auto_encoder_dim)
    vae.build(X.shape[1])
    vae.auto_encoder.summary()
    vae.fit(X, validation_data=X_t)
    X_vae, Xt_vae = vae.predict(X, X_t)
    score = runSVM(svm_C, svm_k, X_vae, y, Xt_vae, y_t)
    with open('result.txt', 'a') as f:
        f.write(f'VAE score: {score} (dim={auto_encoder_dim}, kernel={svm_k})\n')


def runGA():
    print("Starting GA")
    ga = GA(2048, 2, X, X_t, y, y_t)
    res = ga.update()
    score = runSVM(svm_C, svm_k, X[:, res != 0], y, X_t[:, res != 0], y_t)
    with open('result.txt', 'a') as f:
        f.write(f'VAE score: {score} (kernel={svm_k})\n')


if __name__ == '__main__':
    args.method = 'vae'
    if args.method == 'ae':
        runAE(args.dim)
    elif args.method == 'vae':
        runVAE(args.dim)
    elif args.method == 'ga':
        runGA()
