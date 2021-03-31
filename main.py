from dataset import load_data
from SVM import runSVM
from selection.Genetic_alg import GA
from selection.variation_based import FFS
# from projection.auto_encoder import AE, VAE
# from projection.kernel_PCA import kernelPCA
# from projection.LDA import LDA
import numpy as np
import argparse

# params
svm_C = 0.5
svm_k = 'linear'
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--method', default='ae')
parser.add_argument('-d', '--dim', type=int, default=64)
parser.add_argument('-b', '--batch', type=int, default=128)
parser.add_argument('-k', '--kernel', default=128)
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
    ga = GA(2048, 10, X, X_t, y, y_t, max_iter=50, IR=0.3)
    mask, score = ga.update()
    # X_ga = X[:, mask != 0]
    # Xt_ga = X_t[:, mask != 0]
    # score = runSVM(svm_C, svm_k, X_ga, y.T, Xt_ga, y_t.T)
    with open('result.txt', 'a') as f:
        f.write(f'GA score: {score} (mask={mask}, dim = {mask.sum()}, kernel={svm_k})\n')


def runFFS():
    print("starting Forward Feature Selection(On Variance)")
    target_dim = 200
    ffs = FFS(X, X_t, target_dim)
    X_ffs, Xt_ffs = ffs.get_new_features()
    score = runSVM(svm_C, svm_k, X_ffs, y, Xt_ffs, y_t)
    with open('result.txt', 'a') as f:
        f.write(f'FFS score: {score} (dim={target_dim}, kernel={svm_k})\n')


def runKernelPCA(kpca_dim, kernel_name):
    print(f"Starting Kernel-[{kernel_name}]-PCA")
    len_train = X.shape[1]
    X_all = np.concatenate([X, X_t], axis=0)
    kpca = kernelPCA(X_all, kpca_dim, kernel_name)
    X_kpca_all = kpca.compute()
    X_kpca, Xt_kpca = X_kpca_all[:len_train], X[len_train:]
    score = runSVM(svm_C, svm_k, X_kpca, y, Xt_kpca, y_t)
    with open('result.txt', 'a') as f:
        f.write(f'Kernal-[{kernel_name}]-PCA score: {score} (dim={kpca_dim}, kernel={svm_k})\n')


def runLDA(lda_dim):
    print("Starting LDA")
    lda = LDA(X, lda_dim, y, X.shape[1])
    X_lda, Xt_lda = lda.compute(X, X_t)
    score = runSVM(svm_C, svm_k, X_lda, y, Xt_lda, y_t)
    with open('result.txt', 'a') as f:
        f.write(f'LDA score: {score} (dim={lda_dim}, kernel={svm_k})\n')


if __name__ == '__main__':
    if args.method == 'ae':
        runAE(args.dim)
    elif args.method == 'vae':
        runVAE(args.dim)
    elif args.method == 'ga':
        runGA()
    elif args.method == 'ffs':
        runFFS()
    elif args.method == 'kpca':
        runKernelPCA(args.dim,args.kernel)
    elif args == "lda":
        runLDA(args.dim)
