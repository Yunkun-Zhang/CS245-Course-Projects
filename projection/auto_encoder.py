import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.losses import binary_crossentropy


class AE:
    def __init__(self, dim=2):
        self.dim = dim
        self.auto_encoder = None
        self.encoder = None

    def build(self, input_shape):
        img = Input(shape=input_shape)
        shape = input_shape // 2

        # encoder
        encoded = Dense(shape, activation='relu')(img)
        shape //= 4
        while shape > self.dim:
            encoded = Dense(shape, activation='relu')(encoded)
            shape //= 4
        encoder_output = Dense(self.dim, activation='relu')(encoded)

        # decoder
        shape *= 4
        decoded = Dense(shape, activation='relu')(encoder_output)
        shape *= 4
        while shape < input_shape:
            decoded = Dense(shape, activation='relu')(decoded)
            shape *= 4
        decoder_output = Dense(input_shape, activation='sigmoid')(decoded)

        self.auto_encoder = Model(inputs=img, outputs=decoder_output)
        self.encoder = Model(inputs=img, outputs=encoder_output)

        self.auto_encoder.compile(optimizer='adam', loss='binary_crossentropy')

    def fit(self, X_train, epochs=40, batch_size=128, validation_data=None):
        self.auto_encoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size,
                              shuffle=True, validation_data=(validation_data, validation_data))

    def predict(self, X_train, X_test):
        return self.encoder.predict(X_train), self.encoder.predict(X_test)


class VAE:
    def __init__(self, dim=2, batch_size=128):
        self.dim = dim
        self.batch_size = batch_size
        self.auto_encoder = None
        self.encoder = None
        self.generator = None

    def build(self, original_dim, intermediate_dim=512, epsilon_std=1.0):
        x = Input(batch_shape=(self.batch_size, original_dim))
        h = Dense(intermediate_dim, activation='relu')(x)
        z_mean = Dense(self.dim)(h)
        z_log_sigma = Dense(self.dim)(h)

        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = K.random_normal(shape=(self.batch_size, self.dim), stddev=epsilon_std)
            return z_mean + K.exp(z_log_sigma) * epsilon

        z = Lambda(sampling, output_shape=(self.dim,))([z_mean, z_log_sigma])

        decoder_h = Dense(intermediate_dim, activation='relu')
        decoder_mean = Dense(original_dim, activation='sigmoid')
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
            xent_loss = binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
            return xent_loss + kl_loss

        self.auto_encoder.compile(optimizer='rmsprop', loss=vae_loss, experimental_run_tf_function=False)

    def fit(self, X_train, epochs=40, validation_data=None):
        self.auto_encoder.fit(X_train, X_train, epochs=epochs, batch_size=self.batch_size,
                              shuffle=True, validation_data=(validation_data, validation_data))

    def predict(self, X_train, X_test):
        return self.encoder.predict(X_train), self.encoder.predict(X_test)
