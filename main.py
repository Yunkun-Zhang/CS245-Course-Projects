from dataset import load_data
from SVM import runSVM
from selection.Genetic_alg import GA
from projection.auto_encoder import AE, VAE

# params
auto_encoder_dim = 16
svm_C = 5
svm_k = 'rbf'

print('Loading data...')
X, X_t, y, y_t = load_data()

vae = VAE(auto_encoder_dim)
vae.build(X.shape[1])
vae.fit(X, validation_data=X_t)
X_vae, Xt_vae = vae.predict(X, X_t)

score = runSVM(svm_C, svm_k, X_vae, y, Xt_vae, y_t)
print("VAE score:", score)
'''
print('Training auto encoder...')
ae = AE(auto_encoder_dim)
ae.build(X.shape[1])
ae.auto_encoder.summary()
ae.fit(X, validation_data=X_t)
X_ae, Xt_ae = ae.predict(X, X_t)

score = runSVM(svm_C, svm_k, X_ae, y, Xt_ae, y_t)
print("Auto encoder score:", score)

print("Starting GA")
ga = GA(2048, 2, X, X_t, y, y_t)
res = ga.update()
print("final mask: ", res)
'''