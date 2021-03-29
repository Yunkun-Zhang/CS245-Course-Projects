from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


class AE:
    def __init__(self, dim=2):
        self.dim = dim
        self.auto_encoder = None
        self.encoder = None

    def build(self, input_shape):
        img = Input(shape=input_shape)
        shape = input_shape // 2

        encoded = Dense(shape, activation='relu')(img)
        shape //= 4
        while shape > self.dim:
            encoded = Dense(shape, activation='relu')(encoded)
            shape //= 4
        encoder_output = Dense(self.dim)(encoded)

        shape *= 4
        decoded = Dense(shape, activation='relu')(encoder_output)
        shape *= 4
        while shape < input_shape:
            decoded = Dense(shape, activation='relu')(decoded)
            shape *= 4
        decoder_output = Dense(input_shape)(decoded)

        self.auto_encoder = Model(inputs=img, outputs=decoder_output)
        self.encoder = Model(inputs=img, outputs=encoder_output)

        self.auto_encoder.compile(optimizer='adam', loss='mse')

    def fit(self, X_train, y_train, epochs=40, batch_size=256, validation_data=None):
        self.auto_encoder.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                              validation_data=validation_data)

    def predict(self, X_train, X_test):
        return self.encoder.predict(X_train), self.encoder.predict(X_test)


if __name__ == '__main__':
    pass
