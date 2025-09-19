from tensorflow.keras import layers, models, optimizers


# --------------------------
# Resistance Model
# >>>> Architecture might need to be revised. 
# --------------------------
class ResistanceModel:
    def __init__(self, max_len, num_classes, hidden_size=64, model_type="lstm"):
        """
        max_len: padded protein length
        num_classes: number of unique classes
        model_type: "lstm" or "gru"
        """
        inputs = layers.Input(shape=(max_len, 20))
        x = layers.Masking(mask_value=0.0)(inputs)

        if model_type.lower() == "lstm":
            x = layers.Bidirectional(layers.LSTM(hidden_size))(x)
        elif model_type.lower() == "gru":
            x = layers.Bidirectional(layers.GRU(hidden_size))(x)
        else:
            raise ValueError("Unsupported model_type. Choose 'lstm' or 'gru'.")

        # Hidden dense layer
        x = layers.Dense(128, activation="relu")(x)

        # Regularization + normalization
        x = layers.BatchNormalization()(x)   # normalize activations
        x = layers.Dropout(0.3)(x)           # prevent overfitting

        # Another hidden layer
        x = layers.Dense(64, activation="relu")(x)

        # Optional extra features
        x = layers.GaussianNoise(0.1)(x)     # robustness
        x = layers.Dense(32, activation="relu")(x)

        outputs = layers.Dense(num_classes, activation="softmax")(x)

        self.model = models.Model(inputs, outputs)
        self.model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=optimizers.Adam(1e-3),
            metrics=["accuracy"]
        )

    def train(self, X, y, epochs=10, batch_size=32, validation_split=0.1):
        return self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )

    def predict(self, X):
        return self.model.predict(X)

