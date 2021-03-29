from dataset import load_data
from SVM import runSVM
from auto_encoder import AE

# params
auto_encoder_dim = 64
svm_C = 5
svm_k = 'rbf'

print('Loading data...')
X, X_t, y, y_t = load_data()

print('Training auto encoder...')
ae = AE(auto_encoder_dim)
ae.build(X.shape[1])
ae.auto_encoder.summary()
ae.fit(X, y, validation_data=(X_t, y_t))
X, X_t = ae.predict(X, X_t)

print('Training SVM...')
score = runSVM(svm_C, svm_k, X, y, X_t, y_t)
print("Score:", score)

with open('result.txt', 'a') as f:
    f.write(f'{auto_encoder_dim}, /4, {svm_C}: {score}\n')
