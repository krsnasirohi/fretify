from tensorflow.keras import layers, models


def build_model(input_shape=(252, 9, 1)):
    """CQT window (252 bins x 9 frames) -> (6 strings, 25 frets) probabilities."""
    return models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPool2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPool2D((2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.BatchNormalization(),

        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.BatchNormalization(),

        layers.Dense(6 * 25, activation='sigmoid'),
        layers.Reshape((6, 25)),
    ])
