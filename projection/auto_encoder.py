import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda


class AE:
    def __init__(self, dim=2):
        self.dim = dim
        self.auto_encoder = None
        self.encoder = None

    def build(self, input_dim, intermediate_dim=512):
        img = Input(shape=input_dim)
        if isinstance(intermediate_dim, int):
            intermediate_dim = [intermediate_dim]

        # encoder
        encoded = Dense(intermediate_dim[0], activation='relu')(img)
        for dim in intermediate_dim[1:]:
            encoded = Dense(dim, activation='relu')(encoded)
        encoder_output = Dense(self.dim, activation='relu')(encoded)

        # decoder
        decoded = Dense(intermediate_dim[-1], activation='relu')(encoder_output)
        for dim in intermediate_dim[-2::-1]:
            decoded = Dense(dim, activation='relu')(decoded)
        decoder_output = Dense(input_dim, activation='sigmoid')(decoded)

        self.auto_encoder = Model(inputs=img, outputs=decoder_output)
        self.encoder = Model(inputs=img, outputs=encoder_output)

        self.auto_encoder.compile(optimizer='adam', loss='binary_crossentropy')

    def fit(self, X_train, epochs=50, batch_size=128, validation_data=None):
        self.auto_encoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size,
                              shuffle=True, validation_data=(validation_data, validation_data))

    def predict(self, X_train, X_test):
        return self.encoder.predict(X_train), self.encoder.predict(X_test)


class VAE(AE):
    def __init__(self, dim=2):
        super().__init__(dim)
        self.generator = None
        tf.compat.v1.disable_eager_execution()

    def build(self, input_dim, intermediate_dim=512, epsilon_std=1.0):
        x = Input(shape=(input_dim,))
        h = Dense(intermediate_dim, activation='relu')(x)
        # compute mean and variance of p(Z|X)
        z_mean = Dense(self.dim)(h)
        z_log_var = Dense(self.dim)(h)

        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = K.random_normal(shape=(self.dim,), stddev=epsilon_std)
            return z_mean + K.exp(z_log_sigma / 2) * epsilon

        z = Lambda(sampling, output_shape=(self.dim,))([z_mean, z_log_var])

        # decoder
        decoder_h = Dense(intermediate_dim, activation='relu')
        decoder_mean = Dense(input_dim, activation='sigmoid')
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)

        self.auto_encoder = Model(x, x_decoded_mean)
        self.encoder = Model(x, z_mean)

        # generator, from latent space to reconstructed inputs
        decoder_input = Input(shape=(self.dim,))
        _h_decoded = decoder_h(decoder_input)
        _x_decoded_mean = decoder_mean(_h_decoded)
        self.generator = Model(decoder_input, _x_decoded_mean)

        def vae_loss(x, x_decoded_mean):
            xent_loss = K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=-1)
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(xent_loss + kl_loss)

        self.auto_encoder.compile(optimizer='rmsprop', loss=vae_loss, experimental_run_tf_function=False)

    def fit(self, X_train, epochs=50, batch_size=128, validation_data=None):
        AE.fit(self, X_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)
